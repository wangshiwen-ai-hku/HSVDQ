#!/usr/bin/env python3
"""Measure Qwen3 memory peaks for loading, PPL, prefill, and decode."""

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
    set_reproducible,
    write_json,
)


def memory_snapshot(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    index = torch.cuda.current_device() if device.index is None else device.index
    return {
        "allocated_gb": torch.cuda.memory_allocated(index) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(index) / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated(index) / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved(index) / 1024**3,
    }


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.reset_peak_memory_stats(index)
        torch.cuda.synchronize(index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--runtime-backend", choices=["eager", "nunchaku"], default="eager")
    parser.add_argument("--allow-activation-group-remap", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    set_reproducible(args.seed)
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    reset_peak(device)
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint or None,
        device=device,
        dtype=dtype,
        runtime_backend=args.runtime_backend,
        allow_activation_group_remap=args.allow_activation_group_remap,
    )
    metrics: dict[str, object] = {"load": memory_snapshot(device)}
    vocab = len(tokenizer)

    reset_peak(device)
    prompt = torch.randint(0, vocab, (args.batch_size, args.prompt_len), device=device)
    model(prompt, use_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["prefill"] = memory_snapshot(device)

    reset_peak(device)
    model.config.use_cache = True
    out = model(prompt, use_cache=True)
    decode = torch.randint(0, vocab, (args.batch_size, args.decode_len), device=device)
    model(decode, past_key_values=out.past_key_values, use_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["decode"] = memory_snapshot(device)

    reset_peak(device)
    input_ids = get_eval_input_ids(args.dataset, tokenizer, args.seqlen, args.seqlen * args.max_samples)
    ppl_metrics = compute_ppl(model, input_ids, args.seqlen, device, args.max_samples)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics["ppl"] = {**memory_snapshot(device), **ppl_metrics}

    payload = result_payload(runtime, args, metrics)
    write_json(Path(args.output), payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
