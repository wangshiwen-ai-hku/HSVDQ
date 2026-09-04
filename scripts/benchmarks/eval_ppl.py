#!/usr/bin/env python3
"""Evaluate Qwen3 perplexity on wikitext2 or C4."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    _dtype_from_name,
    compute_ppl,
    get_eval_input_ids,
    load_experiment_model,
    result_payload,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], required=True)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--runtime-backend",
        choices=["eager", "hsvdq_cuda", "w4a16", "nunchaku", "hybrid"],
        default="eager",
    )
    parser.add_argument("--hybrid-policy", choices=["auto", "force_w4a4", "force_w4a16"], default="auto")
    parser.add_argument("--hybrid-threshold", type=int, default=128)
    parser.add_argument("--allow-activation-group-remap", action="store_true")
    parser.add_argument("--hybrid-profile-stats", action="store_true")
    parser.add_argument(
        "--persist-qweight",
        action="store_true",
        help="eager only: dequant residual once and keep FP16 GEMM weights on GPU",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = _dtype_from_name(args.dtype)
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint or None,
        device=device,
        dtype=dtype,
        runtime_backend=args.runtime_backend,
        persist_qweight=args.persist_qweight,
        hybrid_policy=args.hybrid_policy,
        hybrid_threshold=args.hybrid_threshold,
        allow_activation_group_remap=args.allow_activation_group_remap,
        hybrid_profile_stats=args.hybrid_profile_stats,
    )
    input_ids = get_eval_input_ids(
        args.dataset,
        tokenizer,
        args.seqlen,
        None if args.max_tokens <= 0 else args.max_tokens,
    )
    metrics = compute_ppl(
        model,
        input_ids,
        args.seqlen,
        device,
        None if args.max_samples <= 0 else args.max_samples,
    )
    metrics["dataset"] = args.dataset
    metrics["seqlen"] = args.seqlen
    if args.runtime_backend == "hybrid" and args.hybrid_profile_stats:
        from hsvdquant_hybrid import collect_hybrid_runtime_stats

        metrics["hybrid_runtime"] = collect_hybrid_runtime_stats(model)
    payload = result_payload(runtime, args, metrics)
    write_json(Path(args.output), payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
