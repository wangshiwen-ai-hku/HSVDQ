#!/usr/bin/env python3
"""Small calibration-only L2 projection oracle for one H-SVDQuant module."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from common import _dtype_from_name, load_experiment_model, make_calibration
from analyze_structured_correction import (
    EPS,
    _gain,
    _mse,
    load_states,
    quantize_activation_with_codes,
    sparse_prediction,
)
from fit_structured_correction import _run_layer, collect_split_inputs
from hsvdquant import _dequantize_codes, capture_first_layer_inputs


def load_original_weight(model_root: str, tensor_name: str) -> torch.Tensor:
    root = Path(model_root)
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        from safetensors.torch import load_file

        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard = index["weight_map"][tensor_name]
        return load_file(root / shard, device="cpu")[tensor_name].float()
    single = root / "model.safetensors"
    if single.exists():
        from safetensors.torch import load_file

        return load_file(single, device="cpu")[tensor_name].float()
    raise FileNotFoundError(f"cannot locate safetensors for {model_root}")


def fit_delta(a: torch.Tensor, target: torch.Tensor, ridge_ratio: float) -> tuple[torch.Tensor, float]:
    gram = a.T.float() @ a.float()
    ridge = float(ridge_ratio) * float(torch.trace(gram).item()) / max(1, gram.shape[0])
    system = gram + torch.eye(gram.shape[0], dtype=gram.dtype) * ridge
    delta = torch.linalg.solve(system, a.T.float() @ target.float())
    return delta, ridge


def budget_alpha(
    weight_residual: torch.Tensor,
    correction: torch.Tensor,
    epsilon: float | None,
) -> float:
    if epsilon is None:
        return 1.0
    fw0 = float(weight_residual.float().square().sum().item())
    direction = float((weight_residual.float() * correction.float()).sum().item())
    curvature = float(correction.float().square().sum().item())
    if curvature <= EPS:
        return 0.0
    discriminant = max(0.0, direction * direction + curvature * float(epsilon) * fw0)
    root = (direction + math.sqrt(discriminant)) / curvature
    return min(1.0, max(0.0, root))


def evaluate_target(
    name: str,
    train_target: torch.Tensor,
    test_target: torch.Tensor,
    train_true_g: torch.Tensor,
    test_true_g: torch.Tensor,
    train_a: torch.Tensor,
    test_a: torch.Tensor,
    train_weight_residual: torch.Tensor,
    test_weight_residual: torch.Tensor,
    ridge_ratio: float,
    epsilons: list[float | None],
) -> dict[str, Any]:
    delta, ridge = fit_delta(train_a, train_target, ridge_ratio)
    train_corr = train_a.float() @ delta
    test_corr = test_a.float() @ delta
    fw0_train = float(train_weight_residual.float().square().sum().item())
    fw0_test = float(test_weight_residual.float().square().sum().item())
    total0_train = train_weight_residual.float() + train_true_g.float()
    total0_test = test_weight_residual.float() + test_true_g.float()
    normal_cross = float((train_weight_residual.float() * train_corr).sum().item())
    normal_cosine = normal_cross / max(
        float(train_weight_residual.float().norm().item() * train_corr.float().norm().item()), EPS
    )
    rows = []
    for epsilon in epsilons:
        alpha = budget_alpha(train_weight_residual, train_corr, epsilon)
        train_fw = train_weight_residual.float() - alpha * train_corr
        test_fw = test_weight_residual.float() - alpha * test_corr
        train_total = total0_train - alpha * train_corr
        test_total = total0_test - alpha * test_corr
        rows.append(
            {
                "epsilon": "inf" if epsilon is None else epsilon,
                "alpha": alpha,
                "train_fw_ratio": float(train_fw.square().sum().item()) / max(fw0_train, EPS),
                "test_fw_ratio": float(test_fw.square().sum().item()) / max(fw0_test, EPS),
                "train_true_g_gain": _gain(train_true_g, train_true_g.float() - alpha * train_corr),
                "test_true_g_gain": _gain(test_true_g, test_true_g.float() - alpha * test_corr),
                "train_target_gain": _gain(train_target, train_target.float() - alpha * train_corr),
                "test_target_gain": _gain(test_target, test_target.float() - alpha * test_corr),
                "train_total_gain": _gain(total0_train, train_total),
                "test_total_gain": _gain(total0_test, test_total),
            }
        )
    return {
        "name": name,
        "ridge": ridge,
        "normal_equation_cosine": normal_cosine,
        "delta_norm": float(delta.norm().item()),
        "unscaled_train_correction_energy": float(train_corr.square().sum().item()),
        "unscaled_test_correction_energy": float(test_corr.square().sum().item()),
        "budgets": rows,
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    _tok, batches = make_calibration(
        args.model, args.dataset, args.nsamples, args.sequence_length, 1, args.seed
    )
    del _tok
    model, _tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    del _tokenizer
    states = load_states(args.checkpoint)
    hidden, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    del batches

    full_name = f"model.layers.{args.layer}.{args.module}"
    collected = None
    for layer_index, layer in enumerate(model.model.layers):
        if layer_index == args.layer:
            captured, _next = collect_split_inputs(
                layer,
                layer_index,
                hidden,
                layer_kwargs,
                device,
                args.max_tokens_per_split,
                {args.module, args.module.rsplit(".", 1)[-1]},
            )
            collected = captured.get(full_name)
            break
        hidden = _run_layer(layer, hidden, layer_kwargs, device)
    if collected is None:
        raise RuntimeError(f"did not capture {full_name}")
    train_x, test_x = collected
    state = states[full_name]
    d = state["d"].float()
    l1 = state["l1"].float()
    l2 = state["l2"].float()
    rhat = _dequantize_codes(
        state["codes"], state["scales"].float(), int(state["group_size"])
    ).T.contiguous().float()
    original_weight = load_original_weight(args.model, full_name + ".weight")
    wtilde = d[:, None] * original_weight.T.contiguous()

    train_xt = train_x.float() / d[None, :]
    test_xt = test_x.float() / d[None, :]
    activation_bits = int(state["activation_bits"])
    activation_group_size = int(state.get("activation_group_size", 0))
    train_q, _train_codes, _train_scales = quantize_activation_with_codes(
        train_xt, activation_bits, activation_group_size
    )
    test_q, _test_codes, _test_scales = quantize_activation_with_codes(
        test_xt, activation_bits, activation_group_size
    )
    train_ea = train_xt - train_q
    test_ea = test_xt - test_q
    train_g = train_ea @ rhat
    test_g = test_ea @ rhat
    train_sp, train_sp_details = sparse_prediction(
        train_ea, train_xt, rhat, activation_group_size, args.sparse_threshold
    )
    test_sp, test_sp_details = sparse_prediction(
        test_ea, test_xt, rhat, activation_group_size, args.sparse_threshold
    )
    train_a = train_xt @ l1
    test_a = test_xt @ l1
    train_weight_target = train_xt @ (wtilde - rhat)
    test_weight_target = test_xt @ (wtilde - rhat)
    train_weight_residual = train_weight_target - train_a @ l2
    test_weight_residual = test_weight_target - test_a @ l2
    epsilons: list[float | None] = [0.0, 0.01, 0.05, 0.1, None]

    payload = {
        "runtime": runtime.__dict__,
        "args": vars(args),
        "module": full_name,
        "train_tokens": int(train_x.shape[0]),
        "test_tokens": int(test_x.shape[0]),
        "activation": {
            "g_train_mse": _mse(train_g),
            "g_test_mse": _mse(test_g),
            "sparse_train_nonzero_rate": train_sp_details["nonzero_rate"],
            "sparse_test_nonzero_rate": test_sp_details["nonzero_rate"],
            "sparse_train_raw_gain": _gain(train_g, train_g - train_sp),
            "sparse_test_raw_gain": _gain(test_g, test_g - test_sp),
        },
        "targets": [
            evaluate_target(
                "full_ea",
                train_g,
                test_g,
                train_g,
                test_g,
                train_a,
                test_a,
                train_weight_residual,
                test_weight_residual,
                args.ridge_ratio,
                epsilons,
            ),
            evaluate_target(
                f"sparse_tau{args.sparse_threshold:g}",
                train_sp,
                test_sp,
                train_g,
                test_g,
                train_a,
                test_a,
                train_weight_residual,
                test_weight_residual,
                args.ridge_ratio,
                epsilons,
            ),
        ],
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "l2_projection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Calibration-only L2 Projection",
        "",
        f"- module: `{full_name}`",
        f"- train/test tokens: {train_x.shape[0]}/{test_x.shape[0]}",
        f"- sparse tau: {args.sparse_threshold}",
        f"- sparse test nonzero rate: {test_sp_details['nonzero_rate']:.4f}",
        f"- sparse raw test gain: {_gain(test_g, test_g - test_sp):.4f}",
        "",
        "| target | epsilon | alpha | test FW ratio | test G gain | test total gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for target in payload["targets"]:
        for row in target["budgets"]:
            lines.append(
                f"| `{target['name']}` | {row['epsilon']} | {row['alpha']:.4f} | "
                f"{row['test_fw_ratio']:.4f} | {row['test_true_g_gain']:.4f} | "
                f"{row['test_total_gain']:.4f} |"
            )
    (output / "l2_projection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen2.5-7B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--module", default="self_attn.v_proj")
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--max-tokens-per-split", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse-threshold", type=float, default=2.5)
    parser.add_argument("--ridge-ratio", type=float, default=1e-6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
