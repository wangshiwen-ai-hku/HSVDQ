#!/usr/bin/env python3
"""Oracle admission tests for energy-conditioned structured correction.

The script reads an existing H-SVDQuant checkpoint and measures how much of
the deployed activation-output error

    G = (X_tilde - Q0(X_tilde)) @ R_hat

is predictable from runtime-available feature families.  It is intentionally
diagnostic-only: no checkpoint is modified and no calibration activations are
stored for inference.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from common import (
    REPO_ROOT,
    _dtype_from_name,
    _move_tree,
    environment_metadata,
    load_experiment_model,
    make_calibration,
    write_json,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import (  # noqa: E402
    HSVQuantLinear,
    _decoder_hidden,
    _dequantize_codes,
    _get_submodule,
    capture_first_layer_inputs,
)


EPS = 1e-30
LAYER_RE = re.compile(r"model\.layers\.(?P<layer>\d+)\.(?P<module>.+)")


@dataclass
class StageResult:
    name: str
    calib_gain: float
    test_gain: float
    calib_mse: float
    test_mse: float
    cost_units: float
    gain_per_cost: float
    details: dict[str, Any]


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _sumsq(x: torch.Tensor) -> float:
    return float(x.float().square().sum().item())


def _mse(x: torch.Tensor) -> float:
    return float(x.float().square().mean().item())


def _gain(target: torch.Tensor, residual: torch.Tensor) -> float:
    return 1.0 - _sumsq(residual) / max(_sumsq(target), EPS)


def _split_rows(rows: int, holdout_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if rows < 4:
        index = torch.arange(rows)
        return index, index
    holdout = max(1, int(round(rows * holdout_fraction)))
    holdout = min(holdout, rows - 2)
    test = torch.linspace(0, rows - 1, holdout).round().long().unique()
    mask = torch.ones(rows, dtype=torch.bool)
    mask[test] = False
    train = mask.nonzero(as_tuple=False).flatten()
    if train.numel() < 2 or test.numel() < 1:
        mid = rows // 2
        return torch.arange(mid), torch.arange(mid, rows)
    return train, test


def _safe_condition(features: torch.Tensor) -> float | None:
    if features.numel() == 0:
        return None
    x = features.float()
    x = x - x.mean(dim=0, keepdim=True)
    gram = x @ x.T if x.shape[1] > x.shape[0] else x.T @ x
    if gram.numel() == 0:
        return None
    eig = torch.linalg.eigvalsh((gram + gram.T) * 0.5).clamp_min(0)
    positive = eig[eig > eig[-1].clamp_min(EPS) * 1e-8]
    if positive.numel() == 0:
        return None
    return float((positive[-1] / positive[0].clamp_min(EPS)).item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    if a.shape != b.shape or a.numel() == 0:
        return None
    lhs = a.float().flatten()
    rhs = b.float().flatten()
    denom = lhs.norm() * rhs.norm()
    if float(denom.item()) <= EPS:
        return None
    return float((lhs @ rhs / denom).item())


def ridge_fit(features: torch.Tensor, target: torch.Tensor, ridge: float) -> torch.Tensor:
    """Fit C for min ||Y - Phi C||^2 + ridge ||C||^2.

    Uses the dual form when feature dimension exceeds sample count.  This keeps
    LUT histograms affordable for wide MLP projections.
    """

    phi = features.float()
    y = target.float()
    rows, cols = phi.shape
    if rows == 0 or cols == 0:
        return phi.new_zeros((cols, y.shape[1]))
    if cols <= rows:
        gram = phi.T @ phi
        gram = gram + torch.eye(cols, device=phi.device, dtype=phi.dtype) * float(ridge)
        return torch.linalg.solve(gram, phi.T @ y)
    gram = phi @ phi.T
    gram = gram + torch.eye(rows, device=phi.device, dtype=phi.dtype) * float(ridge)
    alpha = torch.linalg.solve(gram, y)
    return phi.T @ alpha


def split_coefficient_cosine(features: torch.Tensor, target: torch.Tensor, ridge: float) -> float | None:
    if features.shape[0] < 4:
        return None
    even = torch.arange(0, features.shape[0], 2)
    odd = torch.arange(1, features.shape[0], 2)
    if even.numel() < 2 or odd.numel() < 2:
        return None
    left = ridge_fit(features[even], target[even], ridge)
    right = ridge_fit(features[odd], target[odd], ridge)
    return _cosine(left, right)


def fitted_stage(
    name: str,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_target: torch.Tensor,
    test_target: torch.Tensor,
    ridge: float,
    cost_units: float,
    extra: dict[str, Any] | None = None,
    coeff_rank: int | None = None,
) -> tuple[StageResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    coeff = ridge_fit(train_features, train_target, ridge)
    if coeff_rank is not None and 0 < coeff_rank < min(coeff.shape):
        u, s, vh = torch.linalg.svd(coeff.float(), full_matrices=False)
        coeff = (u[:, :coeff_rank] * s[:coeff_rank]) @ vh[:coeff_rank]
    train_pred = train_features.float() @ coeff
    test_pred = test_features.float() @ coeff
    train_residual = train_target.float() - train_pred
    test_residual = test_target.float() - test_pred
    test_gain = _gain(test_target, test_residual)
    details = {
        "features": int(train_features.shape[1]),
        "feature_condition": _finite(_safe_condition(train_features) or float("nan")),
        "split_coeff_cosine": _finite(split_coefficient_cosine(train_features, train_target, ridge) or float("nan")),
    }
    if extra:
        details.update(extra)
    result = StageResult(
        name=name,
        calib_gain=_gain(train_target, train_residual),
        test_gain=test_gain,
        calib_mse=_mse(train_residual),
        test_mse=_mse(test_residual),
        cost_units=float(cost_units),
        gain_per_cost=test_gain / max(float(cost_units), EPS),
        details=details,
    )
    return result, train_pred, test_pred, coeff


def quantize_activation_with_codes(
    inputs: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if bits >= 16:
        codes = torch.zeros_like(inputs, dtype=torch.int16)
        scales = torch.ones(*inputs.shape[:-1], 1, dtype=inputs.dtype, device=inputs.device)
        return inputs, codes, scales
    qmax = float(2 ** (bits - 1) - 1)
    columns = inputs.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    lead = inputs.shape[:-1]
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = inputs if pad == 0 else F.pad(inputs, (0, pad))
    grouped = padded.reshape(*lead, num_groups, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    codes = torch.round(grouped / scales).clamp(-qmax, qmax)
    quantized = (codes * scales).reshape(*lead, num_groups * group_size)
    codes = codes.reshape(*lead, num_groups * group_size).to(torch.int16)
    if pad:
        quantized = quantized[..., :columns]
        codes = codes[..., :columns]
    return quantized, codes, scales.squeeze(-1)


def group_centers(x: torch.Tensor, group_size: int) -> torch.Tensor:
    columns = x.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = x if pad == 0 else F.pad(x, (0, pad))
    return padded.reshape(x.shape[0], num_groups, group_size).mean(dim=-1)


def code_histograms(codes: torch.Tensor, bits: int, group_size: int) -> tuple[torch.Tensor, dict[str, Any]]:
    rows, columns = codes.shape
    qmax = int(2 ** (bits - 1) - 1)
    levels = 2 * qmax + 1
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = codes if pad == 0 else F.pad(codes, (0, pad), value=0)
    grouped = padded.reshape(rows, num_groups, group_size).long() + qmax
    hist = torch.zeros((rows, num_groups, levels), dtype=torch.float32, device=codes.device)
    hist.scatter_add_(2, grouped.clamp(0, levels - 1), torch.ones_like(grouped, dtype=torch.float32))
    support = hist.sum(dim=0)
    active = support[support > 0]
    details = {
        "groups": int(num_groups),
        "levels": int(levels),
        "active_routes": int(active.numel()),
        "total_routes": int(num_groups * levels),
        "min_route_count": float(active.min().item()) if active.numel() else 0.0,
        "median_route_count": float(active.median().item()) if active.numel() else 0.0,
    }
    return hist.reshape(rows, num_groups * levels), details


def sparse_prediction(
    ea: torch.Tensor,
    x_tilde: torch.Tensor,
    rhat: torch.Tensor,
    group_size: int,
    threshold: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    columns = x_tilde.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = x_tilde if pad == 0 else F.pad(x_tilde, (0, pad))
    grouped = padded.reshape(x_tilde.shape[0], num_groups, group_size)
    rms = grouped.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
    normalized = grouped.abs() / rms
    mask = normalized > float(threshold)
    mask = mask.reshape(x_tilde.shape[0], num_groups * group_size)
    if pad:
        mask = mask[:, :columns]
    pred = (ea * mask.to(ea.dtype)) @ rhat
    details = {
        "threshold": float(threshold),
        "nonzero": int(mask.sum().item()),
        "nonzero_rate": float(mask.float().mean().item()),
    }
    return pred, details


def low_rank_stage(
    train_target: torch.Tensor,
    test_target: torch.Tensor,
    rank: int,
) -> tuple[StageResult, torch.Tensor, torch.Tensor]:
    max_rank = min(rank, train_target.shape[0], train_target.shape[1])
    if max_rank <= 0:
        zero_train = torch.zeros_like(train_target)
        zero_test = torch.zeros_like(test_target)
        result = StageResult("generic_rank0", 0.0, 0.0, _mse(train_target), _mse(test_target), 0.0, 0.0, {})
        return result, zero_train, zero_test
    _u, _s, vh = torch.linalg.svd(train_target.float(), full_matrices=False)
    basis = vh[:max_rank].T.contiguous()
    train_pred = train_target.float() @ basis @ basis.T
    test_pred = test_target.float() @ basis @ basis.T
    train_residual = train_target.float() - train_pred
    test_residual = test_target.float() - test_pred
    test_gain = _gain(test_target, test_residual)
    result = StageResult(
        name=f"generic_rank{max_rank}",
        calib_gain=_gain(train_target, train_residual),
        test_gain=test_gain,
        calib_mse=_mse(train_residual),
        test_mse=_mse(test_residual),
        cost_units=float(max_rank),
        gain_per_cost=test_gain / max(float(max_rank), EPS),
        details={"rank": int(max_rank), "basis": "train_output_svd"},
    )
    return result, train_pred, test_pred


def load_states(checkpoint: str) -> dict[str, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint)
    path = checkpoint_dir / "hsvdquant.pt" if checkpoint_dir.is_dir() else checkpoint_dir
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    return payload.get("states", payload) if isinstance(payload, dict) else {}


def parse_layers(value: str, max_layers: int) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(item))
    return {idx for idx in result if 0 <= idx < max_layers}


def module_type(full_name: str) -> str:
    return full_name.rsplit(".", 1)[-1]


def wanted_module(full_name: str, module_filters: set[str]) -> bool:
    if not module_filters or "all" in module_filters:
        return True
    short = module_type(full_name)
    return short in module_filters or full_name in module_filters


def collect_inputs_for_layer(
    layer: torch.nn.Module,
    layer_index: int,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device,
    max_tokens: int,
    module_filters: set[str],
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    buffers: dict[str, list[torch.Tensor]] = {}
    counts: dict[str, int] = {}
    handles = []

    for name, module in layer.named_modules():
        if not isinstance(module, HSVQuantLinear):
            continue
        full_name = f"model.layers.{layer_index}.{name}"
        if not wanted_module(full_name, module_filters):
            continue
        buffers[full_name] = []
        counts[full_name] = 0

        def hook(_module: torch.nn.Module, args: tuple[Any, ...], _output: Any, target=full_name) -> None:
            if counts[target] >= max_tokens:
                return
            rows = args[0].detach().reshape(-1, args[0].shape[-1]).to("cpu", dtype=torch.float32)
            keep = min(rows.shape[0], max_tokens - counts[target])
            if keep > 0:
                buffers[target].append(rows[:keep])
                counts[target] += keep

        handles.append(module.register_forward_hook(hook))

    next_hidden: list[torch.Tensor] = []
    try:
        for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
            output = layer(hidden.to(device), **_move_tree(kwargs, device))
            next_hidden.append(_decoder_hidden(output).detach().to("cpu"))
    finally:
        for handle in handles:
            handle.remove()

    collected = {
        name: torch.cat(parts, dim=0)
        for name, parts in buffers.items()
        if parts
    }
    return collected, next_hidden


def analyze_module(
    full_name: str,
    x: torch.Tensor,
    state: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device("cpu")
    d = state["d"].float().to(device)
    rhat = _dequantize_codes(
        state["codes"].to(device),
        state["scales"].float().to(device),
        int(state["group_size"]),
    ).T.contiguous()
    activation_bits = int(state.get("activation_bits", args.activation_bits))
    activation_group_size = int(state.get("activation_group_size", args.activation_group_size))
    x_tilde = x.float().to(device) / d[None, :]
    qx, codes, _scales = quantize_activation_with_codes(x_tilde, activation_bits, activation_group_size)
    ea = x_tilde - qx
    target = ea @ rhat
    train_index, test_index = _split_rows(target.shape[0], args.holdout_fraction)
    train_target = target[train_index]
    test_target = target[test_index]
    train_ea = ea[train_index]
    test_ea = ea[test_index]
    train_xt = x_tilde[train_index]
    test_xt = x_tilde[test_index]
    train_codes = codes[train_index]
    test_codes = codes[test_index]

    stages: list[StageResult] = [
        StageResult(
            name="base",
            calib_gain=0.0,
            test_gain=0.0,
            calib_mse=_mse(train_target),
            test_mse=_mse(test_target),
            cost_units=0.0,
            gain_per_cost=0.0,
            details={},
        )
    ]

    dc_train = group_centers(train_xt, activation_group_size)
    dc_test = group_centers(test_xt, activation_group_size)
    dc_result, _dc_train_pred, _dc_test_pred, _dc_coeff = fitted_stage(
        "dc_group_center",
        dc_train,
        dc_test,
        train_target,
        test_target,
        args.ridge,
        float(dc_train.shape[1]),
        extra={"groups": int(dc_train.shape[1])},
    )
    stages.append(dc_result)

    lut_train, lut_details = code_histograms(train_codes, activation_bits, activation_group_size)
    lut_test, _ = code_histograms(test_codes, activation_bits, activation_group_size)
    lut_full, _train_pred, _test_pred, lut_coeff = fitted_stage(
        "lut_full",
        lut_train,
        lut_test,
        train_target,
        test_target,
        args.ridge,
        float(lut_train.shape[1]),
        extra=lut_details,
    )
    stages.append(lut_full)
    for rank in args.lut_modes:
        rank = min(int(rank), min(lut_coeff.shape))
        if rank <= 0:
            continue
        result, _tp, _vp, _c = fitted_stage(
            f"lut_rank{rank}",
            lut_train,
            lut_test,
            train_target,
            test_target,
            args.ridge,
            float(rank),
            extra={**lut_details, "coefficient_rank": int(rank)},
            coeff_rank=rank,
        )
        stages.append(result)

    for threshold in args.sparse_thresholds:
        train_pred, train_sparse = sparse_prediction(train_ea, train_xt, rhat, activation_group_size, threshold)
        test_pred, test_sparse = sparse_prediction(test_ea, test_xt, rhat, activation_group_size, threshold)
        train_residual = train_target - train_pred
        test_residual = test_target - test_pred
        test_gain = _gain(test_target, test_residual)
        cost = max(float(test_sparse["nonzero_rate"]), EPS)
        stages.append(
            StageResult(
                name=f"sparse_tau{threshold:g}",
                calib_gain=_gain(train_target, train_residual),
                test_gain=test_gain,
                calib_mse=_mse(train_residual),
                test_mse=_mse(test_residual),
                cost_units=cost,
                gain_per_cost=test_gain / cost,
                details={**train_sparse, **{f"test_{k}": v for k, v in test_sparse.items()}},
            )
        )

    for rank in args.generic_ranks:
        result, _train_pred, _test_pred = low_rank_stage(train_target, test_target, int(rank))
        stages.append(result)

    best = max(stages[1:], key=lambda row: row.test_gain, default=stages[0])
    match = LAYER_RE.match(full_name)
    return {
        "module": full_name,
        "layer": int(match.group("layer")) if match else -1,
        "module_type": module_type(full_name),
        "tokens": int(x.shape[0]),
        "train_tokens": int(train_index.numel()),
        "test_tokens": int(test_index.numel()),
        "in_features": int(x.shape[1]),
        "out_features": int(rhat.shape[1]),
        "activation_bits": activation_bits,
        "activation_group_size": activation_group_size,
        "g_mse": _mse(target),
        "g_energy": _sumsq(target),
        "ea_mse": _mse(ea),
        "stages": [asdict(stage) for stage in stages],
        "best_stage": asdict(best),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[float]] = {}
    for record in records:
        for stage in record["stages"]:
            by_stage.setdefault(stage["name"], []).append(float(stage["test_gain"]))
    stage_summary = {
        name: {
            "count": len(values),
            "mean_test_gain": sum(values) / max(1, len(values)),
            "max_test_gain": max(values),
            "positive_rate": sum(1.0 for value in values if value > 0) / max(1, len(values)),
        }
        for name, values in sorted(by_stage.items())
    }
    best_records = sorted(records, key=lambda row: row["best_stage"]["test_gain"], reverse=True)
    return {
        "modules": len(records),
        "stage_summary": stage_summary,
        "top_modules": [
            {
                "module": row["module"],
                "g_mse": row["g_mse"],
                "best_stage": row["best_stage"]["name"],
                "best_test_gain": row["best_stage"]["test_gain"],
                "best_gain_per_cost": row["best_stage"]["gain_per_cost"],
            }
            for row in best_records[:10]
        ],
    }


def write_markdown(output: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Structured Correction Oracle",
        "",
        f"- checkpoint: `{payload['args']['checkpoint']}`",
        f"- model: `{payload['args']['model']}`",
        f"- modules analyzed: {payload['summary']['modules']}",
        "",
        "## Stage Summary",
        "",
        "| stage | count | mean test gain | max test gain | positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in payload["summary"]["stage_summary"].items():
        lines.append(
            f"| `{name}` | {row['count']} | {row['mean_test_gain']:.4f} | "
            f"{row['max_test_gain']:.4f} | {row['positive_rate']:.2f} |"
        )
    lines.extend(["", "## Top Modules", ""])
    for row in payload["summary"]["top_modules"]:
        lines.append(
            f"- `{row['module']}`: best `{row['best_stage']}`, "
            f"test_gain={row['best_test_gain']:.4f}, gain_per_cost={row['best_gain_per_cost']:.4f}"
        )
    (output / "structured_correction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    tokenizer, batches = make_calibration(
        args.model,
        args.dataset,
        args.nsamples,
        args.sequence_length,
        args.batch_size,
        args.seed,
    )
    del tokenizer
    model, _tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    del _tokenizer
    states = load_states(args.checkpoint)
    hidden, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    del batches

    layer_count = len(model.model.layers)
    target_layers = parse_layers(args.layers, layer_count)
    module_filters = {item.strip() for item in args.modules.split(",") if item.strip()}
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.model.layers):
        should_collect_layer = target_layers is None or layer_index in target_layers
        filters = module_filters if should_collect_layer else {"__none__"}
        collected, next_hidden = collect_inputs_for_layer(
            layer,
            layer_index,
            hidden,
            layer_kwargs,
            device,
            args.max_tokens_per_module,
            filters,
        )
        hidden = next_hidden
        for full_name, x in sorted(collected.items()):
            if full_name not in states:
                continue
            print(f"[structured] {full_name} tokens={x.shape[0]}", flush=True)
            records.append(analyze_module(full_name, x, states[full_name], args))
        if args.max_layers > 0 and layer_index + 1 >= args.max_layers:
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "runtime": runtime.__dict__,
        "args": vars(args),
        "records": records,
        "summary": summarize(records),
        "environment": environment_metadata(),
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "structured_correction_oracle.json", payload)
    write_markdown(output, payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--dataset", choices=["wikitext2", "c4", "synthetic"], default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=-1)
    parser.add_argument("--layers", default="", help="comma/range layer filter, e.g. 20-27,0")
    parser.add_argument("--modules", default="all", help="comma module filter, e.g. q_proj,k_proj,down_proj")
    parser.add_argument("--max-tokens-per-module", type=int, default=1024)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--activation-group-size", type=int, default=128)
    parser.add_argument("--lut-modes", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    parser.add_argument("--sparse-thresholds", type=float, nargs="*", default=[2.5, 3.0, 4.0, 6.0])
    parser.add_argument("--generic-ranks", type=int, nargs="*", default=[1, 2, 4, 8])
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
