#!/usr/bin/env python3
"""Fit reliable module-wise activation corrections and write a new checkpoint.

The weight-side H-SVDQuant state is kept byte-for-byte equivalent at the tensor
level: D, the full-precision branch, residual codes, and scales are never
updated.  Only an optional ``correction`` dictionary is added to admitted
modules.  Admission uses disjoint calibration batches and a cost ratio measured
relative to the module's original GEMM.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from common import _dtype_from_name, _move_tree, load_experiment_model, make_calibration
from analyze_structured_correction import (
    EPS,
    _gain,
    _mse,
    load_states,
    module_type,
    parse_layers,
    quantize_activation_with_codes,
    sparse_prediction,
    wanted_module,
)
from hsvdquant import (
    HSVQuantLinear,
    _decoder_hidden,
    _dequantize_codes,
    _set_submodule,
    capture_first_layer_inputs,
)


def _run_layer(
    layer: torch.nn.Module,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device,
) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
        result = layer(hidden.to(device), **_move_tree(kwargs, device))
        outputs.append(_decoder_hidden(result).detach().to("cpu"))
    return outputs


def collect_split_inputs(
    layer: torch.nn.Module,
    layer_index: int,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device,
    max_tokens_per_split: int,
    module_filters: set[str],
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], list[torch.Tensor]]:
    """Capture whole calibration batches into disjoint fit/held-out splits."""

    buffers: dict[str, dict[str, list[torch.Tensor]]] = {}
    counts: dict[str, dict[str, int]] = {}
    handles = []
    split_state = {"name": "fit"}

    for name, module in layer.named_modules():
        if not isinstance(module, HSVQuantLinear):
            continue
        full_name = f"model.layers.{layer_index}.{name}"
        if not wanted_module(full_name, module_filters):
            continue
        buffers[full_name] = {"fit": [], "test": []}
        counts[full_name] = {"fit": 0, "test": 0}

        def hook(_module: torch.nn.Module, args: tuple[Any, ...], _output: Any, target=full_name) -> None:
            split = split_state["name"]
            if counts[target][split] >= max_tokens_per_split:
                return
            rows = args[0].detach().reshape(-1, args[0].shape[-1]).to("cpu", dtype=torch.float32)
            keep = min(rows.shape[0], max_tokens_per_split - counts[target][split])
            if keep:
                buffers[target][split].append(rows[:keep])
                counts[target][split] += keep

        handles.append(module.register_forward_hook(hook))

    next_hidden: list[torch.Tensor] = []
    try:
        for batch_index, (hidden, kwargs) in enumerate(zip(hidden_batches, layer_kwargs, strict=True)):
            split_state["name"] = "fit" if batch_index % 2 == 0 else "test"
            result = layer(hidden.to(device), **_move_tree(kwargs, device))
            next_hidden.append(_decoder_hidden(result).detach().to("cpu"))
    finally:
        for handle in handles:
            handle.remove()

    collected: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, parts in buffers.items():
        if parts["fit"] and parts["test"]:
            collected[name] = (torch.cat(parts["fit"], dim=0), torch.cat(parts["test"], dim=0))
    return collected, next_hidden


def _candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key != "correction"}


def fit_module(
    fit_x: torch.Tensor,
    test_x: torch.Tensor,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    d = state["d"].float()
    rhat = _dequantize_codes(
        state["codes"], state["scales"].float(), int(state["group_size"])
    ).T.contiguous().float()
    activation_bits = int(state.get("activation_bits", args.activation_bits))
    activation_group_size = int(state.get("activation_group_size", args.activation_group_size))

    fit_xt = fit_x.float() / d[None, :]
    test_xt = test_x.float() / d[None, :]
    fit_q, _fit_codes, _fit_scales = quantize_activation_with_codes(
        fit_xt, activation_bits, activation_group_size
    )
    test_q, _test_codes, _test_scales = quantize_activation_with_codes(
        test_xt, activation_bits, activation_group_size
    )
    fit_ea = fit_xt - fit_q
    test_ea = test_xt - test_q
    fit_target = fit_ea @ rhat
    test_target = test_ea @ rhat
    in_features, out_features = rhat.shape
    full_macs = max(float(in_features * out_features), 1.0)
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    candidates: list[dict[str, Any]] = []

    if "sparse" in families:
        for threshold in args.sparse_thresholds:
            fit_pred, fit_details = sparse_prediction(
                fit_ea, fit_xt, rhat, activation_group_size, threshold
            )
            test_pred, test_details = sparse_prediction(
                test_ea, test_xt, rhat, activation_group_size, threshold
            )
            fit_gain = _gain(fit_target, fit_target - fit_pred)
            test_gain = _gain(test_target, test_target - test_pred)
            cost_ratio = float(test_details["nonzero_rate"])
            candidates.append(
                {
                    "name": f"sparse_tau{threshold:g}",
                    "family": "sparse",
                    "calib_gain": fit_gain,
                    "test_gain": test_gain,
                    "cost_ratio": cost_ratio,
                    "gain_per_cost": test_gain / max(cost_ratio, EPS),
                    "reliability_ratio": test_gain / max(fit_gain, EPS),
                    "fit_nonzero_rate": float(fit_details["nonzero_rate"]),
                    "test_nonzero_rate": float(test_details["nonzero_rate"]),
                    "correction": {
                        "strategy": "sparse",
                        "sparse_threshold": float(threshold),
                    },
                }
            )

    ranks = sorted({int(rank) for rank in args.generic_ranks if int(rank) > 0})
    if "generic" in families and ranks:
        max_rank = min(max(ranks), fit_target.shape[0], fit_target.shape[1])
        q = min(max_rank + int(args.generic_oversample), fit_target.shape[0], fit_target.shape[1])
        if q > 0:
            _u, _s, output_basis = torch.pca_lowrank(
                fit_target.float(), q=q, center=False, niter=int(args.generic_power_iterations)
            )
            for rank in ranks:
                actual_rank = min(rank, output_basis.shape[1])
                if actual_rank <= 0:
                    continue
                basis = output_basis[:, :actual_rank].contiguous()
                left = (rhat @ basis).contiguous()
                right = basis.T.contiguous()
                fit_pred = (fit_ea @ left) @ right
                test_pred = (test_ea @ left) @ right
                fit_gain = _gain(fit_target, fit_target - fit_pred)
                test_gain = _gain(test_target, test_target - test_pred)
                cost_ratio = float(actual_rank * (in_features + out_features)) / full_macs
                storage_dtype = state["l1"].dtype
                candidates.append(
                    {
                        "name": f"generic_rank{actual_rank}",
                        "family": "generic",
                        "rank": actual_rank,
                        "calib_gain": fit_gain,
                        "test_gain": test_gain,
                        "cost_ratio": cost_ratio,
                        "gain_per_cost": test_gain / max(cost_ratio, EPS),
                        "reliability_ratio": test_gain / max(fit_gain, EPS),
                        "correction": {
                            "strategy": "generic",
                            "generic_left": left.to(dtype=storage_dtype),
                            "generic_right": right.to(dtype=storage_dtype),
                        },
                    }
                )

    admitted = [
        row
        for row in candidates
        if row["test_gain"] >= args.min_test_gain
        and row["cost_ratio"] <= args.max_cost_ratio
        and row["gain_per_cost"] >= args.min_gain_per_cost
        and row["reliability_ratio"] >= args.min_reliability_ratio
    ]
    chosen = max(admitted, key=lambda row: (row["test_gain"], row["gain_per_cost"]), default=None)
    public_candidates = [_candidate_public(row) for row in candidates]
    record = {
        "fit_tokens": int(fit_x.shape[0]),
        "test_tokens": int(test_x.shape[0]),
        "g_fit_mse": _mse(fit_target),
        "g_test_mse": _mse(test_target),
        "candidates": public_candidates,
        "chosen": None if chosen is None else _candidate_public(chosen),
        "admitted": chosen is not None,
    }
    if chosen is None:
        return None, record
    correction = dict(chosen["correction"])
    correction["admission"] = _candidate_public(chosen)
    return correction, record


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    records = payload["records"]
    admitted = [row for row in records if row["admitted"]]
    family_counts: dict[str, int] = {}
    for row in admitted:
        family = row["chosen"]["family"]
        family_counts[family] = family_counts.get(family, 0) + 1
    lines = [
        "# Structured Correction Fit",
        "",
        f"- source: `{payload['source_checkpoint']}`",
        f"- output: `{payload['output_checkpoint']}`",
        f"- modules tested: {len(records)}",
        f"- modules admitted: {len(admitted)}",
        f"- admitted families: {family_counts}",
        "",
        "## Admitted modules",
        "",
    ]
    for row in admitted:
        chosen = row["chosen"]
        lines.append(
            f"- `{row['module']}`: `{chosen['name']}`, test_gain={chosen['test_gain']:.4f}, "
            f"cost_ratio={chosen['cost_ratio']:.4f}, reliability={chosen['reliability_ratio']:.3f}"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(args.checkpoint)
    output = Path(args.output_checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output checkpoint: {output}")
    states = load_states(str(source))
    if not args.replace_existing and any(state.get("correction") for state in states.values()):
        raise RuntimeError("source already contains corrections; pass --replace-existing to refit")
    if args.replace_existing:
        for state in states.values():
            state.pop("correction", None)

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    _tokenizer, batches = make_calibration(
        args.model,
        args.dataset,
        args.nsamples,
        args.sequence_length,
        1,
        args.seed,
    )
    del _tokenizer
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=str(source),
        device=device,
        dtype=dtype,
    )
    hidden, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    del batches
    target_layers = parse_layers(args.layers, len(model.model.layers))
    module_filters = {item.strip() for item in args.modules.split(",") if item.strip()}
    records: list[dict[str, Any]] = []

    for layer_index, layer in enumerate(model.model.layers):
        selected_layer = target_layers is None or layer_index in target_layers
        filters = module_filters if selected_layer else {"__none__"}
        collected, base_next_hidden = collect_split_inputs(
            layer,
            layer_index,
            hidden,
            layer_kwargs,
            device,
            args.max_tokens_per_split,
            filters,
        )
        changed = False
        for full_name, (fit_x, test_x) in sorted(collected.items()):
            if full_name not in states:
                continue
            correction, record = fit_module(fit_x, test_x, states[full_name], args)
            record.update(
                {
                    "module": full_name,
                    "layer": layer_index,
                    "module_type": module_type(full_name),
                }
            )
            records.append(record)
            chosen_text = "none" if correction is None else correction["strategy"]
            chosen_gain = 0.0 if record["chosen"] is None else record["chosen"]["test_gain"]
            print(f"[fit] {full_name}: {chosen_text}, test_gain={chosen_gain:.4f}", flush=True)
            if correction is None:
                states[full_name].pop("correction", None)
                continue
            states[full_name]["correction"] = correction
            relative_name = full_name.split(f"model.layers.{layer_index}.", 1)[1]
            replacement = HSVQuantLinear(states[full_name], compute_dtype=dtype).to(device)
            _set_submodule(layer, relative_name, replacement)
            changed = True
        hidden = _run_layer(layer, hidden, layer_kwargs, device) if changed else base_next_hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    torch.save(states, output / "hsvdquant.pt")
    metadata_path = output / "hsvdquant_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    admitted = [row for row in records if row["admitted"]]
    metadata["structured_correction"] = {
        "format": "structured-correction-v1",
        "source_checkpoint": str(source),
        "families": args.families,
        "layers": args.layers,
        "modules": args.modules,
        "modules_tested": len(records),
        "modules_admitted": len(admitted),
        "max_cost_ratio": args.max_cost_ratio,
        "min_test_gain": args.min_test_gain,
        "min_gain_per_cost": args.min_gain_per_cost,
        "min_reliability_ratio": args.min_reliability_ratio,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    payload = {
        "source_checkpoint": str(source),
        "output_checkpoint": str(output),
        "runtime": runtime.__dict__,
        "args": vars(args),
        "seconds": time.perf_counter() - started,
        "records": records,
        "summary": {
            "modules_tested": len(records),
            "modules_admitted": len(admitted),
            "mean_admitted_test_gain": (
                sum(row["chosen"]["test_gain"] for row in admitted) / len(admitted)
                if admitted
                else 0.0
            ),
        },
    }
    report = Path(args.report) if args.report else output / "structured_correction_fit.json"
    _write_report(report, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--dataset", choices=["wikitext2", "c4", "synthetic"], default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="")
    parser.add_argument("--modules", default="q_proj,k_proj,v_proj,down_proj")
    parser.add_argument("--max-tokens-per-split", type=int, default=1024)
    parser.add_argument("--families", default="sparse,generic")
    parser.add_argument("--sparse-thresholds", type=float, nargs="*", default=[2.5, 3.0, 4.0])
    parser.add_argument("--generic-ranks", type=int, nargs="*", default=[2, 4, 8])
    parser.add_argument("--generic-oversample", type=int, default=4)
    parser.add_argument("--generic-power-iterations", type=int, default=2)
    parser.add_argument("--max-cost-ratio", type=float, default=0.03)
    parser.add_argument("--min-test-gain", type=float, default=0.02)
    parser.add_argument("--min-gain-per-cost", type=float, default=0.5)
    parser.add_argument("--min-reliability-ratio", type=float, default=0.5)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--activation-group-size", type=int, default=128)
    parser.add_argument("--replace-existing", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
