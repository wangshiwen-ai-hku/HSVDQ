#!/usr/bin/env python3
"""Calibration-only joint L1/L2 rotation under an empirical FW trust region.

The deployed operator is unchanged.  A low-rank direction estimated from
activation error is added to the existing branch, then retracted exactly back
to the original rank.  Only ``l1`` and ``l2`` are written to the new checkpoint.
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

from common import _dtype_from_name, load_experiment_model, make_calibration
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
from fit_structured_correction import _run_layer, collect_split_inputs
from hsvdquant import (
    HSVQuantLinear,
    _dequantize_codes,
    _set_submodule,
    capture_first_layer_inputs,
)


class OriginalWeightReader:
    """Lazy safetensors reader for the unquantized model."""

    def __init__(self, model_root: str) -> None:
        self.root = Path(model_root)
        index_path = self.root / "model.safetensors.index.json"
        if index_path.exists():
            self.weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        elif (self.root / "model.safetensors").exists():
            self.weight_map = None
        else:
            raise FileNotFoundError(f"cannot locate safetensors in {self.root}")

    def get(self, tensor_name: str) -> torch.Tensor:
        from safetensors import safe_open

        shard = "model.safetensors" if self.weight_map is None else self.weight_map[tensor_name]
        with safe_open(self.root / shard, framework="pt", device="cpu") as handle:
            return handle.get_tensor(tensor_name).float()


def _rank_factors(matrix: torch.Tensor, rank: int, niter: int) -> tuple[torch.Tensor, torch.Tensor]:
    actual = min(rank, matrix.shape[0], matrix.shape[1])
    q = min(actual + 4, matrix.shape[0], matrix.shape[1])
    u, s, v = torch.pca_lowrank(matrix.float(), q=q, center=False, niter=niter)
    root = s[:actual].clamp_min(0).sqrt()
    return u[:, :actual] * root[None, :], root[:, None] * v[:, :actual].T


def _ridge_feature_map(x: torch.Tensor, z: torch.Tensor, ridge_ratio: float) -> tuple[torch.Tensor, float]:
    """Solve min_J ||XJ-Z|| using the sample-space ridge system."""

    kernel = x @ x.T
    ridge = float(ridge_ratio) * float(torch.trace(kernel).item()) / max(1, kernel.shape[0])
    system = kernel + torch.eye(kernel.shape[0], device=x.device, dtype=x.dtype) * max(ridge, 1e-8)
    dual = torch.linalg.solve(system, z)
    return x.T @ dual, ridge


def _correction_factors(
    x: torch.Tensor,
    target: torch.Tensor,
    rank: int,
    ridge_ratio: float,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    actual = min(rank, target.shape[0], target.shape[1])
    q = min(actual + 4, target.shape[0], target.shape[1])
    u, s, v = torch.pca_lowrank(target.float(), q=q, center=False, niter=niter)
    z = u[:, :actual] * s[:actual][None, :]
    left, ridge = _ridge_feature_map(x, z, ridge_ratio)
    return left, v[:, :actual].T.contiguous(), ridge


def _retract_sum(
    l1: torch.Tensor,
    l2: torch.Tensor,
    delta_l1: torch.Tensor,
    delta_l2: torch.Tensor,
    alpha: float,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact best rank-r retraction of L1L2 + alpha*dL1*dL2.

    The SVD is performed on a matrix of at most 2r by 2r, avoiding a dense
    decomposition of the full weight matrix.
    """

    left = torch.cat((l1, delta_l1), dim=1)
    right = torch.cat((l2, delta_l2 * float(alpha)), dim=0)
    ql, rl = torch.linalg.qr(left, mode="reduced")
    qr, rr = torch.linalg.qr(right.T, mode="reduced")
    u, s, vh = torch.linalg.svd(rl @ rr.T, full_matrices=False)
    actual = min(rank, s.numel())
    root = s[:actual].clamp_min(0).sqrt()
    new_l1 = (ql @ u[:, :actual]) * root[None, :]
    new_l2 = root[:, None] * (vh[:actual, :] @ qr.T)
    return new_l1, new_l2


def _subspace_rotation(old_l1: torch.Tensor, new_l1: torch.Tensor) -> dict[str, float]:
    qo = torch.linalg.qr(old_l1, mode="reduced").Q
    qn = torch.linalg.qr(new_l1, mode="reduced").Q
    cosines = torch.linalg.svdvals(qo.T @ qn).clamp(0, 1)
    angles = torch.acos(cosines) * (180.0 / math.pi)
    return {
        "max_principal_angle_deg": float(angles.max().item()),
        "mean_principal_angle_deg": float(angles.mean().item()),
    }


def _metrics(
    x: torch.Tensor,
    weight_target: torch.Tensor,
    true_g: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    fw0: torch.Tensor,
    total0: torch.Tensor,
) -> dict[str, float]:
    fw = weight_target - (x @ l1) @ l2
    total = fw + true_g
    return {
        "fw_ratio": float(fw.square().sum().item()) / max(float(fw0.square().sum().item()), EPS),
        "fw_gain": _gain(fw0, fw),
        "g_gain": _gain(true_g, true_g - ((x @ l1) @ l2 - (weight_target - fw0))),
        "total_gain": _gain(total0, total),
        "total_mse": _mse(total),
    }


@torch.no_grad()
def fit_module(
    fit_x_raw: torch.Tensor,
    test_x_raw: torch.Tensor,
    state: dict[str, Any],
    original_weight: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor] | None, dict[str, Any]]:
    d = state["d"].to(device=device, dtype=torch.float32)
    old_l1 = state["l1"].to(device=device, dtype=torch.float32)
    old_l2 = state["l2"].to(device=device, dtype=torch.float32)
    rank = old_l1.shape[1]
    rhat = _dequantize_codes(
        state["codes"].to(device),
        state["scales"].to(device=device, dtype=torch.float32),
        int(state["group_size"]),
    ).T.contiguous().float()
    wtilde = d[:, None] * original_weight.T.to(device=device, dtype=torch.float32)
    weight_matrix = wtilde - rhat

    fit_x = fit_x_raw.to(device=device, dtype=torch.float32) / d[None, :]
    test_x = test_x_raw.to(device=device, dtype=torch.float32) / d[None, :]
    activation_bits = int(state["activation_bits"])
    activation_group_size = int(state.get("activation_group_size", 0))
    fit_q, _, _ = quantize_activation_with_codes(fit_x, activation_bits, activation_group_size)
    test_q, _, _ = quantize_activation_with_codes(test_x, activation_bits, activation_group_size)
    fit_ea = fit_x - fit_q
    test_ea = test_x - test_q
    fit_g = fit_ea @ rhat
    test_g = test_ea @ rhat
    fit_weight_target = fit_x @ weight_matrix
    test_weight_target = test_x @ weight_matrix
    fit_fw0 = fit_weight_target - (fit_x @ old_l1) @ old_l2
    test_fw0 = test_weight_target - (test_x @ old_l1) @ old_l2
    fit_total0 = fit_fw0 + fit_g
    test_total0 = test_fw0 + test_g

    family_targets: dict[str, tuple[torch.Tensor, torch.Tensor, dict[str, Any]]] = {}
    if "full" in args.families:
        family_targets["full_ea"] = (fit_g, test_g, {})
    if "sparse" in args.families:
        fit_sparse, fit_details = sparse_prediction(
            fit_ea, fit_x, rhat, activation_group_size, args.sparse_threshold
        )
        test_sparse, test_details = sparse_prediction(
            test_ea, test_x, rhat, activation_group_size, args.sparse_threshold
        )
        family_targets[f"sparse_tau{args.sparse_threshold:g}"] = (
            fit_sparse,
            test_sparse,
            {
                "fit_nonzero_rate": float(fit_details["nonzero_rate"]),
                "test_nonzero_rate": float(test_details["nonzero_rate"]),
                "fit_raw_gain": _gain(fit_g, fit_g - fit_sparse),
                "test_raw_gain": _gain(test_g, test_g - test_sparse),
            },
        )

    candidates: list[dict[str, Any]] = []
    tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for family, (fit_target, test_target, details) in family_targets.items():
        correction_rank = rank if args.correction_rank <= 0 else int(args.correction_rank)
        delta_l1, delta_l2, ridge = _correction_factors(
            fit_x,
            fit_target,
            correction_rank,
            args.ridge_ratio,
            args.power_iterations,
        )
        direct_fit = (fit_x @ delta_l1) @ delta_l2
        direct_test = (test_x @ delta_l1) @ delta_l2
        family_info = {
            **details,
            "projected_fit_target_gain": _gain(fit_target, fit_target - direct_fit),
            "projected_test_target_gain": _gain(test_target, test_target - direct_test),
            "ridge": ridge,
        }
        for alpha in args.alphas:
            new_l1, new_l2 = _retract_sum(
                old_l1, old_l2, delta_l1, delta_l2, alpha, rank
            )
            fit_metrics = _metrics(
                fit_x, fit_weight_target, fit_g, new_l1, new_l2, fit_fw0, fit_total0
            )
            test_metrics = _metrics(
                test_x, test_weight_target, test_g, new_l1, new_l2, test_fw0, test_total0
            )
            rotation = _subspace_rotation(old_l1, new_l1)
            feasible = (
                fit_metrics["fw_ratio"] <= 1.0 + args.fw_epsilon + args.fw_tolerance
                and test_metrics["fw_ratio"] <= 1.0 + args.fw_epsilon + args.fw_tolerance
            )
            admitted = feasible and test_metrics["total_gain"] >= args.min_test_gain
            row = {
                "family": family,
                "alpha": float(alpha),
                "feasible": feasible,
                "admitted": admitted,
                "fit": fit_metrics,
                "test": test_metrics,
                "rotation": rotation,
                "family_diagnostics": family_info,
            }
            tensors[len(candidates)] = (new_l1, new_l2)
            candidates.append(row)

    chosen_index = max(
        (index for index, row in enumerate(candidates) if row["admitted"]),
        key=lambda index: candidates[index]["test"]["total_gain"],
        default=None,
    )
    record = {
        "rank": int(rank),
        "fit_tokens": int(fit_x.shape[0]),
        "test_tokens": int(test_x.shape[0]),
        "baseline": {
            "fit_fw_mse": _mse(fit_fw0),
            "test_fw_mse": _mse(test_fw0),
            "fit_g_mse": _mse(fit_g),
            "test_g_mse": _mse(test_g),
            "fit_total_mse": _mse(fit_total0),
            "test_total_mse": _mse(test_total0),
        },
        "candidates": candidates,
        "admitted": chosen_index is not None,
        "chosen": None if chosen_index is None else candidates[chosen_index],
    }
    if chosen_index is None:
        return None, record
    new_l1, new_l2 = tensors[chosen_index]
    storage_dtype = state["l1"].dtype
    return {
        "l1": new_l1.detach().to(device="cpu", dtype=storage_dtype),
        "l2": new_l2.detach().to(device="cpu", dtype=storage_dtype),
    }, record


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    admitted = [row for row in payload["records"] if row["admitted"]]
    lines = [
        "# Joint L1/L2 Rotation",
        "",
        f"- source: `{payload['source_checkpoint']}`",
        f"- output: `{payload['output_checkpoint']}`",
        f"- FW epsilon: {payload['args']['fw_epsilon']}",
        f"- tested/admitted: {len(payload['records'])}/{len(admitted)}",
        "",
        "| module | family | alpha | test FW ratio | test total gain | max angle |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in admitted:
        chosen = row["chosen"]
        lines.append(
            f"| `{row['module']}` | `{chosen['family']}` | {chosen['alpha']:.4g} | "
            f"{chosen['test']['fw_ratio']:.5f} | {chosen['test']['total_gain']:.5f} | "
            f"{chosen['rotation']['max_principal_angle_deg']:.2f} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    source = Path(args.checkpoint)
    output = Path(args.output_checkpoint)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    states = load_states(str(source))
    if any(state.get("correction") for state in states.values()):
        raise RuntimeError("runtime correction state is not supported by this calibration-only fit")

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    _, batches = make_calibration(
        args.model, args.dataset, args.nsamples, args.sequence_length, 1, args.seed
    )
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model, checkpoint=str(source), device=device, dtype=dtype
    )
    hidden, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    del batches
    reader = OriginalWeightReader(args.model)
    target_layers = parse_layers(args.layers, len(model.model.layers))
    filters = {item.strip() for item in args.modules.split(",") if item.strip()}
    records: list[dict[str, Any]] = []

    for layer_index, layer in enumerate(model.model.layers):
        selected = target_layers is None or layer_index in target_layers
        collected, base_next_hidden = collect_split_inputs(
            layer,
            layer_index,
            hidden,
            layer_kwargs,
            device,
            args.max_tokens_per_split,
            filters if selected else {"__none__"},
        )
        changed = False
        for full_name, (fit_x, test_x) in sorted(collected.items()):
            state = states.get(full_name)
            if state is None:
                continue
            update, record = fit_module(
                fit_x,
                test_x,
                state,
                reader.get(full_name + ".weight"),
                args,
                device,
            )
            record.update(
                {"module": full_name, "layer": layer_index, "module_type": module_type(full_name)}
            )
            records.append(record)
            if update is not None:
                state["l1"] = update["l1"]
                state["l2"] = update["l2"]
                state["joint_rotation"] = {
                    "family": record["chosen"]["family"],
                    "alpha": record["chosen"]["alpha"],
                    "fw_epsilon": args.fw_epsilon,
                }
                relative = full_name.split(f"model.layers.{layer_index}.", 1)[1]
                _set_submodule(layer, relative, HSVQuantLinear(state, compute_dtype=dtype).to(device))
                changed = True
            chosen = record["chosen"]
            summary = "none" if chosen is None else (
                f"{chosen['family']} a={chosen['alpha']:.3g} "
                f"gain={chosen['test']['total_gain']:.4f} fw={chosen['test']['fw_ratio']:.4f}"
            )
            print(f"[rotate] {full_name}: {summary}", flush=True)
        hidden = _run_layer(layer, hidden, layer_kwargs, device) if changed else base_next_hidden
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output)
    torch.save(states, output / "hsvdquant.pt")
    metadata_path = output / "hsvdquant_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    admitted = [row for row in records if row["admitted"]]
    metadata["joint_rotation"] = {
        "format": "calibration-only-joint-l1-l2-v1",
        "source_checkpoint": str(source),
        "fw_epsilon": args.fw_epsilon,
        "families": sorted(args.families),
        "sparse_threshold": args.sparse_threshold,
        "modules_tested": len(records),
        "modules_admitted": len(admitted),
        "rank_unchanged": True,
        "runtime_operator_unchanged": True,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    payload = {
        "source_checkpoint": str(source),
        "output_checkpoint": str(output),
        "runtime": runtime.__dict__,
        "args": {**vars(args), "families": sorted(args.families)},
        "seconds": time.perf_counter() - started,
        "records": records,
        "summary": {
            "modules_tested": len(records),
            "modules_admitted": len(admitted),
            "mean_test_total_gain": (
                sum(row["chosen"]["test"]["total_gain"] for row in admitted) / len(admitted)
                if admitted else 0.0
            ),
            "max_test_fw_ratio": max(
                (row["chosen"]["test"]["fw_ratio"] for row in admitted), default=1.0
            ),
            "mean_max_principal_angle_deg": (
                sum(row["chosen"]["rotation"]["max_principal_angle_deg"] for row in admitted)
                / len(admitted) if admitted else 0.0
            ),
        },
    }
    report = Path(args.report) if args.report else output / "joint_rotation_fit.json"
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
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-tokens-per-split", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="")
    parser.add_argument("--modules", default="q_proj,k_proj,v_proj,down_proj")
    parser.add_argument("--families", nargs="+", choices=["full", "sparse"], default=["full", "sparse"])
    parser.add_argument("--sparse-threshold", type=float, default=2.5)
    parser.add_argument("--correction-rank", type=int, default=0)
    parser.add_argument("--ridge-ratio", type=float, default=1e-3)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0])
    parser.add_argument("--fw-epsilon", type=float, default=0.01)
    parser.add_argument("--fw-tolerance", type=float, default=1e-4)
    parser.add_argument("--min-test-gain", type=float, default=0.001)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
