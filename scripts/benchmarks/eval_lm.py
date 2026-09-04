#!/usr/bin/env python3
"""Run lm-eval with selectable runtime backend and per-task resume shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import _dtype_from_name, load_experiment_model
from hsvdquant import run_lm_eval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-8B")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--tasks", default="mmlu,gsm8k,arc_challenge,arc_easy,hellaswag,piqa")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--runtime-backend",
        choices=["eager", "hsvdq_cuda", "w4a16", "nunchaku", "hybrid"],
        default="eager",
    )
    parser.add_argument("--hybrid-policy", choices=["auto", "force_w4a4", "force_w4a16"], default="auto")
    parser.add_argument("--hybrid-threshold", type=int, default=128)
    parser.add_argument("--allow-activation-group-remap", action="store_true")
    parser.add_argument("--hybrid-profile-stats", action="store_true")
    parser.add_argument("--no-cpu-offload-layers", action="store_true", help="alias kept for scripts; offload is opt-in")
    parser.add_argument(
        "--cpu-offload-layers",
        action="store_true",
        help="page decoder blocks through GPU (lower peak, much slower; for 14B+)",
    )
    parser.add_argument(
        "--persist-qweight",
        action="store_true",
        help="eager only: dequant residual once and keep FP16 GEMM weights on GPU",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = _dtype_from_name(args.dtype)
    cpu_offload = args.cpu_offload_layers and not args.no_cpu_offload_layers
    model, tokenizer, _runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint or None,
        device=device,
        dtype=dtype,
        cpu_offload_layers=cpu_offload,
        runtime_backend=args.runtime_backend,
        persist_qweight=args.persist_qweight,
        hybrid_policy=args.hybrid_policy,
        hybrid_threshold=args.hybrid_threshold,
        allow_activation_group_remap=args.allow_activation_group_remap,
        hybrid_profile_stats=args.hybrid_profile_stats,
    )
    results = run_lm_eval(
        model,
        tokenizer,
        args.tasks.split(","),
        args.batch_size,
        args.limit,
        Path(args.output),
        device=device,
        resume=not args.no_resume,
    )
    print(results.get("results", {}))


if __name__ == "__main__":
    main()
