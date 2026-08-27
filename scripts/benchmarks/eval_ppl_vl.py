#!/usr/bin/env python3
"""Evaluate text-only perplexity for Qwen2.5-VL FP or HSVDQ checkpoints."""

from __future__ import annotations

import argparse

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
    parser.add_argument("--model", default="models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint or None,
        device=device,
        dtype=dtype,
    )
    input_ids = get_eval_input_ids(args.dataset, tokenizer, args.seqlen, None)
    metrics = compute_ppl(
        model,
        input_ids,
        args.seqlen,
        device,
        None if args.max_samples <= 0 else args.max_samples,
    )
    metrics.update(dataset=args.dataset, seqlen=args.seqlen, model_family="qwen2_5_vl")
    write_json(args.output, result_payload(runtime, metrics))
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
