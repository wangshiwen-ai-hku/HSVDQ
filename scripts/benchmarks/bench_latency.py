#!/usr/bin/env python3
"""Benchmark Qwen3 prefill/decode/generate latency."""

from __future__ import annotations

import argparse
from collections import Counter
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


def timed_generation(
    fn,
    warmup: int,
    iters: int,
    device: torch.device,
    prompt_tokens: int,
) -> tuple[list[float], list[int]]:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values: list[float] = []
    generated_tokens: list[int] = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - start)
        generated_tokens.append(int(output.numel() - prompt_tokens))
    return values, generated_tokens


def profile_cuda_work(
    fn,
    calls: int,
    tokens: int,
    scope: str,
) -> dict[str, object]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            fn()
    torch.cuda.synchronize()

    averages = prof.key_averages()
    total_cuda_us = sum(float(item.self_cuda_time_total) for item in averages)
    cuda_events = [
        event
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    event_names = Counter(event.name for event in cuda_events)
    top_ops = sorted(
        (
            {
                "name": item.key,
                "self_cuda_ms": float(item.self_cuda_time_total) / 1000.0,
                "calls": int(item.count),
            }
            for item in averages
            if item.self_cuda_time_total > 0
        ),
        key=lambda item: item["self_cuda_ms"],
        reverse=True,
    )[:20]
    total_cuda_ms = total_cuda_us / 1000.0
    if total_cuda_ms <= 0:
        raise RuntimeError(
            "CUDA profiler recorded no device time; verify that CUPTI is available"
        )
    return {
        "scope": scope,
        "profile_calls": calls,
        "tokens": tokens,
        "total_cuda_ms": total_cuda_ms,
        "mean_call_ms": total_cuda_ms / calls,
        "ms_per_token": total_cuda_ms / tokens,
        "tokens_per_s": tokens * 1000.0 / total_cuda_ms,
        "cuda_event_count": len(cuda_events),
        "cuda_events_per_token": len(cuda_events) / tokens,
        "cuda_event_names": dict(sorted(event_names.items())),
        "top_cuda_ops": top_ops,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--window-repeats", type=int, default=3)
    parser.add_argument("--profile-cuda", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=16)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    if args.window_repeats <= 0:
        raise ValueError("--window-repeats must be positive")
    if args.profile_steps <= 0:
        raise ValueError("--profile-steps must be positive")
    set_reproducible(args.seed)
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
        hybrid_policy=args.hybrid_policy,
        hybrid_threshold=args.hybrid_threshold,
        allow_activation_group_remap=args.allow_activation_group_remap,
        hybrid_profile_stats=args.hybrid_profile_stats,
    )
    vocab = len(tokenizer)
    input_ids = torch.randint(0, vocab, (args.batch_size, args.prompt_len), device=device)
    model.config.use_cache = True

    def prefill():
        model(input_ids, use_cache=True)

    prefill_values = timed_loop(prefill, args.warmup, args.iters, device)
    decode_ids = torch.randint(0, vocab, (args.batch_size, 1), device=device)

    with torch.inference_mode():
        cache = model(input_ids, use_cache=True).past_key_values

    def decode_one():
        nonlocal cache
        output = model(decode_ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values

    for _ in range(args.warmup):
        decode_one()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    # Reset after warmup, then measure one intentional continuous decode window.
    cache = model(input_ids, use_cache=True).past_key_values
    decode_values = timed_loop(decode_one, 0, args.iters, device)

    # This window synchronizes once around several decode calls. It includes
    # Python/dispatcher gaps without adding one explicit synchronization per token.
    decode_window_values: list[float] = []
    for _ in range(args.window_repeats):
        cache = model(input_ids, use_cache=True).past_key_values
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(args.iters):
            decode_one()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_window_values.append(time.perf_counter() - started)

    prefill_gpu_metrics = None
    model_gpu_metrics = None
    if args.profile_cuda:
        if device.type != "cuda":
            raise ValueError("--profile-cuda requires a CUDA device")
        prefill_gpu_metrics = profile_cuda_work(
            prefill,
            1,
            args.batch_size * args.prompt_len,
            "all CUDA work in one prefill; excludes CPU gaps and offload",
        )
        cache = model(input_ids, use_cache=True).past_key_values
        model_gpu_metrics = profile_cuda_work(
            decode_one,
            args.profile_steps,
            args.profile_steps * args.batch_size,
            "all CUDA work in a continuous cached decode window; excludes prefill and CPU gaps",
        )

    def generate_once():
        return model.generate(
            input_ids,
            max_new_tokens=args.decode_len,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Full generate is expensive; keep the requested iteration count but users can lower it.
    generate_values, generated_tokens = timed_generation(
        generate_once,
        max(1, args.warmup // 2),
        max(1, args.iters // 10),
        device,
        input_ids.numel(),
    )
    generate_metrics = summarize(generate_values, 1)
    generate_metrics["tokens_per_s"] = sum(generated_tokens) / sum(generate_values)
    generate_metrics["generated_tokens_mean"] = statistics.fmean(generated_tokens)
    metrics = {
        "prefill": summarize(prefill_values, args.batch_size * args.prompt_len),
        "decode_one_token": summarize(decode_values, args.batch_size),
        "decode_wall_window": {
            **summarize(
                decode_window_values,
                args.batch_size * args.iters,
            ),
            "scope": "continuous cached decode wall time with one synchronization per window",
            "steps_per_window": args.iters,
            "window_repeats": args.window_repeats,
            "ms_per_token": statistics.fmean(decode_window_values)
            * 1000.0
            / (args.batch_size * args.iters),
        },
        "generate": generate_metrics,
        "prompt_len": args.prompt_len,
        "decode_len": args.decode_len,
        "batch_size": args.batch_size,
        "decode_context_start": args.prompt_len,
        "decode_context_end": args.prompt_len + args.iters,
    }
    if prefill_gpu_metrics is not None:
        metrics["prefill_model_gpu"] = prefill_gpu_metrics
    if model_gpu_metrics is not None:
        metrics["decode_model_gpu"] = model_gpu_metrics
    if args.runtime_backend == "hybrid" and args.hybrid_profile_stats:
        from hsvdquant_hybrid import collect_hybrid_runtime_stats

        metrics["hybrid_runtime"] = collect_hybrid_runtime_stats(model)
    payload = result_payload(runtime, args, metrics)
    write_json(Path(args.output), payload)
    print(payload["metrics"])


if __name__ == "__main__":
    main()
