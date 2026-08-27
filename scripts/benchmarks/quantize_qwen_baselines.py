#!/usr/bin/env python3
"""Quantize Qwen3 with GPTQ, base-SVDQuant, or GANQ baselines."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from common import (
    ActivationQuantLinear,
    REPO_ROOT,
    _dtype_from_name,
    _get_submodule,
    _load_model,
    _qwen_sequential_groups,
    _set_submodule,
    advance_hidden_batches,
    collect_layer_stats,
    environment_metadata,
    make_calibration,
    set_reproducible,
    write_json,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import (  # noqa: E402
    HSVQuantLinear,
    QuantConfig,
    _state_error,
    gptq_quantize_residual,
    save_quant_checkpoint,
    weighted_low_rank,
    capture_first_layer_inputs,
)
from lut_quant import LUTQuant  # noqa: E402


def make_compact_state(
    module: torch.nn.Linear,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    config: QuantConfig,
    method: str,
    scale_grid: list[float],
    scale_clip: float,
) -> dict[str, Any]:
    weight = module.weight.detach().T.float()
    if method == "gptq":
        d = torch.ones(weight.shape[0], device=weight.device)
        l1 = weight.new_zeros((weight.shape[0], 0))
        l2 = weight.new_zeros((0, weight.shape[1]))
        quantized, codes, scales = gptq_quantize_residual(weight, hessian, config)
        error = _state_error(weight, hessian, cached_x, d, l1, l2, quantized, config.activation_bits)
    elif method in {"base_svdquant", "smoothquant", "awq"}:
        d, l1, l2, quantized, codes, scales, error = search_smoothed_state(
            weight,
            hessian,
            cached_x,
            config,
            method,
            scale_grid,
            scale_clip,
        )
    else:
        raise ValueError(f"unsupported compact method: {method}")
    original_dtype = module.weight.dtype
    return {
        "d": d.detach().cpu().float(),
        "l1": l1.detach().cpu().to(original_dtype),
        "l2": l2.detach().cpu().to(original_dtype),
        "codes": codes.detach().cpu(),
        "scales": scales.detach().cpu().to(original_dtype),
        "bias": None if module.bias is None else module.bias.detach().cpu().to(original_dtype),
        "in_features": module.in_features,
        "out_features": module.out_features,
        "group_size": config.group_size,
        "bits": config.bits,
        "activation_bits": config.activation_bits,
        "error": error,
        "outer_iteration": 0,
        "history": [error],
    }


def _normalize_scale(scale: torch.Tensor, clip: float) -> torch.Tensor:
    scale = scale.float().clamp_min(1e-6)
    scale = scale / scale.log().mean().exp().clamp_min(1e-6)
    if clip > 1:
        scale = scale.clamp(1.0 / clip, clip)
        scale = scale / scale.log().mean().exp().clamp_min(1e-6)
    return scale


def _scale_candidates(
    weight: torch.Tensor,
    cached_x: torch.Tensor | None,
    method: str,
    scale_grid: list[float],
    scale_clip: float,
) -> list[tuple[float, torch.Tensor]]:
    if cached_x is None:
        x_abs = torch.ones(weight.shape[0], device=weight.device)
    else:
        x_abs = cached_x.to(device=weight.device, dtype=torch.float32).abs().amax(dim=0).clamp_min(1e-6)
    w_abs = weight.abs().amax(dim=1).clamp_min(1e-6)
    candidates: list[tuple[float, torch.Tensor]] = []
    for alpha in scale_grid:
        if method == "awq":
            raw = x_abs.pow(alpha)
        else:
            raw = x_abs.pow(alpha) / w_abs.pow(1.0 - alpha)
        candidates.append((alpha, _normalize_scale(raw, scale_clip).to(weight.device)))
    return candidates


def symmetric_group_quantize(
    matrix: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Symmetric per-output-row group quantization for [in, out] matrices."""

    weight = matrix.T.contiguous().float()
    out_features, in_features = weight.shape
    qmax = 2 ** (bits - 1) - 1
    group_size = in_features if group_size <= 0 else group_size
    num_groups = math.ceil(in_features / group_size)
    codes = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty((out_features, num_groups), device=weight.device, dtype=torch.float32)
    dequant = torch.empty_like(weight)
    for group in range(num_groups):
        start = group * group_size
        end = min(start + group_size, in_features)
        scale = weight[:, start:end].abs().amax(dim=1).clamp_min(1e-8) / float(qmax)
        code = torch.round(weight[:, start:end] / scale[:, None]).clamp(-qmax, qmax).to(torch.int8)
        codes[:, start:end] = code
        scales[:, group] = scale
        dequant[:, start:end] = code.float() * scale[:, None]
    return dequant.T.contiguous(), codes, scales


def search_smoothed_state(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    config: QuantConfig,
    method: str,
    scale_grid: list[float],
    scale_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    best: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float] | None = None
    base_config = QuantConfig(**{**config.__dict__, "beta": 0.0, "d_mode": "closed_form", "d_steps": 0})
    for alpha, d in _scale_candidates(weight, cached_x, method, scale_grid, scale_clip):
        smoothed_weight = d[:, None] * weight
        if method == "base_svdquant":
            l1, l2 = weighted_low_rank(hessian, smoothed_weight, base_config)
            residual = smoothed_weight - l1 @ l2
        else:
            l1 = weight.new_zeros((weight.shape[0], 0))
            l2 = weight.new_zeros((0, weight.shape[1]))
            residual = smoothed_weight
        quantized, codes, scales = symmetric_group_quantize(residual, config.bits, config.group_size)
        error = _state_error(weight, hessian, cached_x, d, l1, l2, quantized, config.activation_bits)
        if best is None or error < best[-1]:
            best = (d, l1, l2, quantized, codes, scales, error)
    assert best is not None
    return best


def make_ganq_state(
    module: torch.nn.Linear,
    hessian: torch.Tensor,
    bits: int,
    activation_bits: int,
    max_epoch: int,
    pre_process: bool,
) -> dict[str, Any]:
    quant = LUTQuant(
        bits=bits,
        W=module.weight.detach().float(),
        XXt=hessian.to(module.weight.device).float(),
        max_epoch=max_epoch,
        model_type="llama",
        pre_process=pre_process,
    )
    qweight = quant.quantization().to(module.weight.dtype).detach().cpu()
    return {
        "weight": qweight,
        "bias": None if module.bias is None else module.bias.detach().cpu().to(module.weight.dtype),
        "activation_bits": activation_bits,
        "in_features": module.in_features,
        "out_features": module.out_features,
    }


@torch.no_grad()
def quantize_baseline(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducible(args.seed)
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    tokenizer, batches = make_calibration(
        args.model,
        args.calib_dataset,
        args.nsamples,
        args.sequence_length,
        args.calib_batch_size,
        args.seed,
    )
    model = _load_model(args.model, device, dtype)
    hidden_batches, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    del batches
    config = QuantConfig(
        bits=args.bits,
        activation_bits=args.activation_bits,
        rank=args.rank,
        beta=args.beta,
        p=args.p,
        group_size=args.group_size,
        block_size=args.block_size,
        outer_iters=1,
        d_mode="closed_form",
        d_steps=0,
        damp=args.damp,
        svd_mode=args.svd_mode,
        svd_oversample=args.svd_oversample,
        svd_niter=args.svd_niter,
    )
    states: dict[str, dict[str, Any]] = {}
    layer_count = len(model.model.layers) if args.max_layers < 0 else min(args.max_layers, len(model.model.layers))
    started = time.time()
    for layer_index in range(layer_count):
        layer = model.model.layers[layer_index]
        print(f"[{args.method}] layer {layer_index + 1}/{layer_count}", flush=True)
        stats_by_name = collect_layer_stats(
            model,
            hidden_batches,
            layer_kwargs,
            layer_index,
            device,
            args.activation_cache_tokens,
            args.hessian_block_size,
            args.seed,
        )
        for group in _qwen_sequential_groups(layer):
            for name in group:
                module = _get_submodule(layer, name)
                hessian, cached_x = stats_by_name[name].finalize()
                full_name = f"model.layers.{layer_index}.{name}"
                tick = time.time()
                if args.method in {"gptq", "base_svdquant", "smoothquant", "awq"}:
                    state = make_compact_state(
                        module,
                        hessian,
                        cached_x,
                        config,
                        args.method,
                        args.scale_grid,
                        args.scale_clip,
                    )
                    replacement = HSVQuantLinear(state, compute_dtype=module.weight.dtype).to(device)
                elif args.method == "ganq":
                    state = make_ganq_state(
                        module,
                        hessian,
                        args.bits,
                        args.activation_bits,
                        args.ganq_epochs,
                        args.ganq_preprocess,
                    )
                    replacement = ActivationQuantLinear(
                        state["weight"].to(device=device, dtype=module.weight.dtype),
                        None if state["bias"] is None else state["bias"].to(device=device, dtype=module.weight.dtype),
                        args.activation_bits,
                    )
                else:
                    raise ValueError(f"unsupported method: {args.method}")
                _set_submodule(layer, name, replacement)
                states[full_name] = state
                stats_by_name[name].free()
                print(f"  {name}: time={time.time() - tick:.1f}s", flush=True)
        hidden_batches = advance_hidden_batches(model, hidden_batches, layer_kwargs, layer_index, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    elapsed = time.time() - started
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.method in {"gptq", "base_svdquant", "smoothquant", "awq"}:
        save_args = SimpleNamespace(
            calib_dataset=args.calib_dataset,
            nsamples=args.nsamples,
            sequence_length=args.sequence_length,
            seed=args.seed,
        )
        save_quant_checkpoint(output_dir, args.model, tokenizer, states, config, save_args)
        meta_path = output_dir / "hsvdquant_config.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "method": args.method,
                "scale_grid": args.scale_grid,
                "scale_clip": args.scale_clip,
                "elapsed_seconds": elapsed,
                "environment": environment_metadata(),
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    else:
        tokenizer.save_pretrained(output_dir)
        torch.save(states, output_dir / "baseline_quant.pt")
        write_json(
            output_dir / "baseline_quant_config.json",
            {
                "format": "qwen-baseline-quant-v1",
                "method": args.method,
                "base_model": args.model,
                "quant_config": {
                    "bits": args.bits,
                    "activation_bits": args.activation_bits,
                    "ganq_epochs": args.ganq_epochs,
                    "ganq_preprocess": args.ganq_preprocess,
                    "scale_grid": args.scale_grid,
                    "scale_clip": args.scale_clip,
                },
                "calibration": {
                    "dataset": args.calib_dataset,
                    "nsamples": args.nsamples,
                    "sequence_length": args.sequence_length,
                    "seed": args.seed,
                },
                "modules": list(states),
                "elapsed_seconds": elapsed,
                "environment": environment_metadata(),
            },
        )
    return {"method": args.method, "modules": len(states), "elapsed_seconds": elapsed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["gptq", "base_svdquant", "smoothquant", "awq", "ganq"], required=True)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--calib-dataset", choices=["wikitext2", "c4", "synthetic"], default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=4)
    parser.add_argument("--activation-cache-tokens", type=int, default=2048)
    parser.add_argument("--hessian-block-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=-1)
    parser.add_argument("--bits", type=int, choices=[3, 4], default=4)
    parser.add_argument("--activation-bits", type=int, default=4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--svd-mode", choices=["exact", "lowrank"], default="lowrank")
    parser.add_argument("--svd-oversample", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument("--ganq-epochs", type=int, default=3)
    parser.add_argument("--ganq-preprocess", action="store_true")
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--scale-clip", type=float, default=16.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = quantize_baseline(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
