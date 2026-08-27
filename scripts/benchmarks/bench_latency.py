#!/usr/bin/env python3
"""Benchmark Qwen3 prefill/decode/generate latency."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from common import _dtype_from_name, load_experiment_model, result_payload, set_reproducible, write_json


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def summarize(values: list[float], tokens: int) -> dict[str, float]:
    mean = statistics.fmean(values)
    return {
        "mean_ms": mean * 1000.0,
        "p50_ms": percentile(values, 0.50) * 1000.0,
        "p95_ms": percentile(values, 0.95) * 1000.0,
        "p99_ms": percentile(values, 0.99) * 1000.0,
        "tokens_per_s": tokens / mean if mean > 0 else float("inf"),
    }


def timed_loop(fn, warmup: int, iters: int, device: torch.device) -> list[float]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - start)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    set_reproducible(args.seed)
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint or None,
        device=device,
        dtype=dtype,
    )
    vocab = len(tokenizer)
    input_ids = torch.randint(0, vocab, (args.batch_size, args.prompt_len), device=device)
    model.config.use_cache = True

    def prefill():
        model(input_ids, use_cache=True)

    prefill_values = timed_loop(prefill, args.warmup, args.iters, device)
    with torch.inference_mode():
        cache = model(input_ids, use_cache=True).past_key_values
    decode_ids = torch.randint(0, vocab, (args.batch_size, 1), device=device)

    def decode_one():
        model(decode_ids, past_key_values=cache, use_cache=True)

    decode_values = timed_loop(decode_one, args.warmup, args.iters, device)

    def generate_once():
        model.generate(
            input_ids,
            max_new_tokens=args.decode_len,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Full generate is expensive; keep the requested iteration count but users can lower it.
    generate_values = timed_loop(generate_once, max(1, args.warmup // 2), max(1, args.iters // 10), device)
    metrics = {
        "prefill": summarize(prefill_values, args.batch_size * args.prompt_len),
        "decode_one_token": summarize(decode_values, args.batch_size),
        "generate": summarize(generate_values, args.batch_size * args.decode_len),
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "batch_size": args.batch_size,
    }
    payload = result_payload(runtime, args, metrics)
    write_json(Path(args.output), payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
