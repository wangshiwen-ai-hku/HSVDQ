#!/usr/bin/env python3
"""Probe peak VRAM for H-SVDQuant layer-offload on a Qwen3-family model.

Loads weights on CPU, keeps only one decoder block on GPU, then measures:
  - single-block residency
  - ActivationStats Hessian accumulation for down_proj
  - GPTQ metric factorization peak (Cholesky path)

Does not run full calibration; intended to estimate the largest model that fits
a given GPU under --cpu-offload-layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hsvdquant import (  # noqa: E402
    ActivationStats,
    QuantConfig,
    _decoder_layers,
    _dtype_from_name,
    _get_submodule,
    _prepare_gptq_metric,
    _qwen_sequential_groups,
    gptq_quantize_residual,
    joint_quantize_linear,
)


def mem_gb(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    idx = device.index or 0
    return {
        "allocated_gb": torch.cuda.memory_allocated(idx) / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved(idx) / 1024**3,
        "max_allocated_gb": torch.cuda.max_memory_allocated(idx) / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved(idx) / 1024**3,
        "total_gb": torch.cuda.get_device_properties(idx).total_memory / 1024**3,
    }


def reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)


def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--cache-tokens", type=int, default=2048)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(args.model)
    print(
        f"[probe] model={args.model} type={cfg.model_type} "
        f"h={cfg.hidden_size} inter={getattr(cfg, 'intermediate_size', None)} "
        f"L={cfg.num_hidden_layers}",
        flush=True,
    )

    reset(device)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval()
    model.config.use_cache = False
    # Keep on CPU; only move one block.
    model.to("cpu")
    params_b = count_params(model)
    layers = _decoder_layers(model)
    layer = layers[0]
    layer.to(device)
    reset(device)
    layer.to(device)
    torch.cuda.synchronize(device)
    snapshot = {"after_one_block": mem_gb(device), "params_b": params_b, "load_s": time.time() - t0}

    # Find down_proj (or largest linear) in the block.
    groups = _qwen_sequential_groups(layer)
    names = [n for g in groups for n in g]
    downs = [n for n in names if n.endswith("down_proj")]
    target_name = downs[0] if downs else names[-1]
    module = _get_submodule(layer, target_name)
    assert isinstance(module, nn.Linear)
    print(
        f"[probe] target={target_name} in={module.in_features} out={module.out_features}",
        flush=True,
    )

    # Synthetic activations on CPU reservoir path (matches calibration).
    reset(device)
    stats = ActivationStats(
        module.in_features,
        device,
        args.cache_tokens,
        hessian_block_size=4096,
        seed=0,
    )
    gen = torch.Generator(device="cpu").manual_seed(0)
    for _ in range(args.nsamples):
        x = torch.randn(args.seqlen, module.in_features, generator=gen, dtype=torch.float16)
        stats.add_batch(x.unsqueeze(0))
    hessian, cached_x = stats.finalize()
    snapshot["after_hessian"] = mem_gb(device)
    print(f"[probe] hessian device={hessian.device} shape={tuple(hessian.shape)}", flush=True)

    # Move module weight path: joint_quantize keeps weight on module.device.
    reset(device)
    config = QuantConfig(
        bits=4,
        activation_bits=4,
        activation_group_size=128,
        rank=4,
        code_objective="joint",
        joint_code_iters=1,
        outer_iters=1,
        d_mode="closed_form",
        d_steps=0,
        activation_weight=0.25,
        group_size=128,
        block_size=128,
        svd_mode="lowrank",
    )
    try:
        state = joint_quantize_linear(module, hessian, cached_x, config)
        torch.cuda.synchronize(device)
        snapshot["after_joint_quantize"] = mem_gb(device)
        snapshot["joint_ok"] = True
        snapshot["best_mse"] = float(state["error"])
    except Exception as exc:  # noqa: BLE001
        snapshot["joint_ok"] = False
        snapshot["joint_error"] = f"{type(exc).__name__}: {exc}"
        snapshot["after_joint_quantize"] = mem_gb(device)
        print(f"[probe] joint_quantize FAILED: {snapshot['joint_error']}", flush=True)

    # Isolated GPTQ peak (upper-bound stress).
    reset(device)
    try:
        h = hessian.to(device=device, dtype=torch.float32)
        w = module.weight.detach().T.float()
        _, upper = _prepare_gptq_metric(h, config)
        _ = gptq_quantize_residual(w, h, config, prepared_upper=upper)
        torch.cuda.synchronize(device)
        snapshot["after_gptq_isolated"] = mem_gb(device)
        snapshot["gptq_ok"] = True
    except Exception as exc:  # noqa: BLE001
        snapshot["gptq_ok"] = False
        snapshot["gptq_error"] = f"{type(exc).__name__}: {exc}"
        snapshot["after_gptq_isolated"] = mem_gb(device)
        print(f"[probe] gptq FAILED: {snapshot['gptq_error']}", flush=True)

    payload = {
        "model": args.model,
        "device": str(device),
        "dtype": args.dtype,
        "config": {
            "hidden_size": cfg.hidden_size,
            "intermediate_size": getattr(cfg, "intermediate_size", None),
            "num_hidden_layers": cfg.num_hidden_layers,
            "model_type": cfg.model_type,
            "num_experts": getattr(cfg, "num_experts", None),
        },
        "target_linear": target_name,
        "in_features": module.in_features,
        "out_features": module.out_features,
        "metrics": snapshot,
        "fits_24gb": float(snapshot.get("after_joint_quantize", {}).get("max_allocated_gb", 99)) < 23.5
        and snapshot.get("joint_ok", False),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2), flush=True)
    print(f"[probe] wrote {out} fits_24gb={payload['fits_24gb']}", flush=True)


if __name__ == "__main__":
    main()
