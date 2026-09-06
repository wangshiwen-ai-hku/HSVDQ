#!/usr/bin/env python3
"""Benchmark W4A4/W4A16 crossover points on checkpoint Linear shapes."""

from __future__ import annotations

import argparse
from collections import Counter
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
) -> list[tuple[str, dict[str, Any], int]]:
    representatives: dict[tuple[int, int, int, int], tuple[str, dict[str, Any], int]] = {}
    for name, state in states.items():
        key = (
            int(state["in_features"]),
            int(state["out_features"]),
            int(state["l1"].shape[1]),
            int(state["group_size"]),
        )
        if key in representatives:
            representative_name, representative_state, count = representatives[key]
            representatives[key] = (representative_name, representative_state, count + 1)
        else:
            representatives[key] = (name, state, 1)
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


def run_policy(
    module: HybridHSVQuantLinear,
    policy: str,
    inputs: torch.Tensor,
) -> torch.Tensor:
    module.policy = policy
    return module(inputs)


@torch.no_grad()
def profile_cuda_events(fn) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    events = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    names = Counter(event.name for event in events)
    return {
        "cuda_event_count": len(events),
        "cuda_event_names": dict(sorted(names.items())),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--rows", default="1,4,16,64,128,256,512,1024")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile-kernels", action="store_true")
    parser.add_argument("--profile-rows", default="1,256")
    parser.add_argument("--allow-activation-group-remap", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the hybrid Linear benchmark")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    rows_to_test = [int(value) for value in args.rows.split(",") if value]
    profile_rows = {int(value) for value in args.profile_rows.split(",") if value}
    if not rows_to_test or min(rows_to_test) <= 0:
        raise ValueError("--rows must contain positive comma-separated integers")
    if args.profile_kernels and (not profile_rows or min(profile_rows) <= 0):
        raise ValueError("--profile-rows must contain positive comma-separated integers")

    states = load_states(Path(args.checkpoint))
    shapes: list[dict[str, Any]] = []
    crossovers: list[int] = []
    weighted_by_rows: dict[str, dict[str, float]] = {
        str(rows): {"dense_ms": 0.0, "w4a4_ms": 0.0, "w4a16_ms": 0.0}
        for rows in rows_to_test
    }
    total_linear_layers = 0
    for name, state, occurrence_count in representative_states(states):
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
            "occurrence_count": occurrence_count,
            "cases": {},
        }
        total_linear_layers += occurrence_count
        crossover = None
        for rows in rows_to_test:
            inputs = torch.randn(rows, module.in_features, device=device, dtype=dtype)
            dense_weight = torch.randn(
                module.out_features,
                module.in_features,
                device=device,
                dtype=dtype,
            )
            dense = measure_ms(
                lambda: torch.nn.functional.linear(inputs, dense_weight),
                args.warmup,
                args.iters,
            )
            module.policy = "force_w4a16"
            w4a16 = measure_ms(lambda: module(inputs), args.warmup, args.iters)
            module.policy = "force_w4a4"
            w4a4 = measure_ms(lambda: module(inputs), args.warmup, args.iters)
            dense_stats = summarize(dense)
            w4a16_stats = summarize(w4a16)
            w4a4_stats = summarize(w4a4)
            ratio = w4a16_stats["mean_ms"] / w4a4_stats["mean_ms"]
            case_result: dict[str, Any] = {
                "dense_fp16": dense_stats,
                "w4a16": w4a16_stats,
                "w4a4": w4a4_stats,
                "w4a4_speedup_vs_w4a16": ratio,
                "w4a16_speedup_vs_dense": dense_stats["mean_ms"] / w4a16_stats["mean_ms"],
                "w4a4_speedup_vs_dense": dense_stats["mean_ms"] / w4a4_stats["mean_ms"],
            }
            if args.profile_kernels and rows in profile_rows:
                case_result["cuda_events"] = {
                    "dense_fp16": profile_cuda_events(
                        lambda: torch.nn.functional.linear(inputs, dense_weight)
                    ),
                    "w4a16": profile_cuda_events(
                        lambda: run_policy(module, "force_w4a16", inputs)
                    ),
                    "w4a4": profile_cuda_events(
                        lambda: run_policy(module, "force_w4a4", inputs)
                    ),
                }
            shape_result["cases"][str(rows)] = case_result
            weighted = weighted_by_rows[str(rows)]
            weighted["dense_ms"] += dense_stats["mean_ms"] * occurrence_count
            weighted["w4a4_ms"] += w4a4_stats["mean_ms"] * occurrence_count
            weighted["w4a16_ms"] += w4a16_stats["mean_ms"] * occurrence_count
            if crossover is None and ratio >= 1.05:
                crossover = rows
            del dense_weight
        shape_result["w4a4_crossover_rows"] = crossover
        if crossover is not None:
            crossovers.append(crossover)
        shapes.append(shape_result)
        del module
        torch.cuda.empty_cache()

    recommendation = max(crossovers) if len(crossovers) == len(shapes) else None
    for weighted in weighted_by_rows.values():
        weighted["w4a4_speedup_vs_dense"] = weighted["dense_ms"] / weighted["w4a4_ms"]
        weighted["w4a16_speedup_vs_dense"] = weighted["dense_ms"] / weighted["w4a16_ms"]

    result = {
        "metric_scope": "complete Linear operator only; excludes other Transformer operations",
        "checkpoint": args.checkpoint,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "dtype": args.dtype,
        "nunchaku_version": nunchaku_version(),
        "activation_group_remap": args.allow_activation_group_remap,
        "required_margin": 1.05,
        "profiled_rows": sorted(profile_rows) if args.profile_kernels else [],
        "total_linear_layers": total_linear_layers,
        "weighted_operator_latency_by_rows": weighted_by_rows,
        "recommended_global_threshold": recommendation,
        "shapes": shapes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
