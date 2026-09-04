#!/usr/bin/env python3
"""Benchmark W4A4/W4A16 crossover points on checkpoint Linear shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant_hybrid import HybridHSVQuantLinear, nunchaku_version  # noqa: E402


def load_states(checkpoint: Path) -> dict[str, dict[str, Any]]:
    path = checkpoint / "hsvdquant.pt"
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def representative_states(
    states: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    representatives: dict[tuple[int, int, int, int], tuple[str, dict[str, Any]]] = {}
    for name, state in states.items():
        key = (
            int(state["in_features"]),
            int(state["out_features"]),
            int(state["l1"].shape[1]),
            int(state["group_size"]),
        )
        representatives.setdefault(key, (name, state))
    return list(representatives.values())


@torch.no_grad()
def measure_ms(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    return values


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p50 = ordered[len(ordered) // 2]
    return {"mean_ms": statistics.fmean(values), "p50_ms": p50, "min_ms": min(values)}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--rows", default="1,4,16,64,128")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--allow-activation-group-remap", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hybrid Linear benchmark")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rows_to_test = [int(value) for value in args.rows.split(",") if value]
    if not rows_to_test or min(rows_to_test) <= 0:
        raise ValueError("--rows must contain positive comma-separated integers")

    states = load_states(Path(args.checkpoint))
    shapes: list[dict[str, Any]] = []
    crossovers: list[int] = []
    for name, state in representative_states(states):
        module = HybridHSVQuantLinear(
            state,
            dtype,
            allow_activation_group_remap=args.allow_activation_group_remap,
        ).to(device).eval()
        shape_result: dict[str, Any] = {
            "module": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "rank": module.rank,
            "cases": {},
        }
        crossover = None
        for rows in rows_to_test:
            inputs = torch.randn(rows, module.in_features, device=device, dtype=dtype)
            module.policy = "force_w4a16"
            w4a16 = measure_ms(lambda: module(inputs), args.warmup, args.iters)
            module.policy = "force_w4a4"
            w4a4 = measure_ms(lambda: module(inputs), args.warmup, args.iters)
            w4a16_stats = summarize(w4a16)
            w4a4_stats = summarize(w4a4)
            ratio = w4a16_stats["mean_ms"] / w4a4_stats["mean_ms"]
            shape_result["cases"][str(rows)] = {
                "w4a16": w4a16_stats,
                "w4a4": w4a4_stats,
                "w4a4_speedup_vs_w4a16": ratio,
            }
            if crossover is None and ratio >= 1.05:
                crossover = rows
        shape_result["w4a4_crossover_rows"] = crossover
        if crossover is not None:
            crossovers.append(crossover)
        shapes.append(shape_result)
        del module
        torch.cuda.empty_cache()

    recommendation = max(crossovers) if len(crossovers) == len(shapes) else None
    result = {
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": args.dtype,
        "nunchaku_version": nunchaku_version(),
        "activation_group_remap": args.allow_activation_group_remap,
        "required_margin": 1.05,
        "recommended_global_threshold": recommendation,
        "shapes": shapes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
