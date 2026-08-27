#!/usr/bin/env python3
"""Held-out decomposition of uniform activation error.

The analysis-only oracle keeps the deployed per-token/per-group normalization
but replaces the signed uniform grid by train-fitted 1-D Lloyd-Max codebooks.
It measures

    e_u = e_red + e_irr,
    e_red = (Q_* - Q_u) R_hat,
    e_irr = (X_tilde - Q_*) R_hat,

without modifying the checkpoint or the runtime quantizer.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from analyze_structured_correction import (
    collect_inputs_for_layer,
    load_states,
    parse_layers,
    quantize_activation_with_codes,
)
from common import (
    _dtype_from_name,
    environment_metadata,
    load_experiment_model,
    make_calibration,
    write_json,
)
from hsvdquant import _dequantize_codes, capture_first_layer_inputs


EPS = 1e-30


def split_rows(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    index = torch.arange(rows)
    return index[0::2], index[1::2]


def normalized_groups(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    columns = x.shape[-1]
    group_size = columns if group_size <= 0 or group_size >= columns else group_size
    groups = (columns + group_size - 1) // group_size
    pad = groups * group_size - columns
    padded = x if pad == 0 else F.pad(x, (0, pad))
    values = padded.reshape(x.shape[0], groups, group_size)
    scale = values.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    return values / scale, scale, pad


def fit_lloyd_codebooks(
    x: torch.Tensor,
    group_size: int,
    levels: int,
    iterations: int,
) -> torch.Tensor:
    normalized, _scale, _pad = normalized_groups(x, group_size)
    books = []
    quantiles = torch.linspace(0.0, 1.0, levels, device=x.device)
    for group in range(normalized.shape[1]):
        values = normalized[:, group, :].reshape(-1).float()
        centers = torch.quantile(values, quantiles).sort().values
        for _ in range(iterations):
            assignment = (values[:, None] - centers[None, :]).abs().argmin(dim=1)
            sums = torch.zeros_like(centers).scatter_add_(0, assignment, values)
            counts = torch.zeros_like(centers).scatter_add_(
                0, assignment, torch.ones_like(values)
            )
            proposal = torch.where(counts > 0, sums / counts.clamp_min(1), centers)
            if torch.max(torch.abs(proposal - centers)) < 1e-6:
                centers = proposal
                break
            centers = proposal
        books.append(centers.sort().values)
    return torch.stack(books)


def apply_codebooks(x: torch.Tensor, group_size: int, books: torch.Tensor) -> torch.Tensor:
    normalized, scale, pad = normalized_groups(x, group_size)
    quantized = torch.empty_like(normalized)
    for group in range(normalized.shape[1]):
        values = normalized[:, group, :]
        centers = books[group]
        assignment = (values[..., None] - centers).abs().argmin(dim=-1)
        quantized[:, group, :] = centers[assignment]
    result = (quantized * scale).reshape(x.shape[0], -1)
    return result[:, : x.shape[1]] if pad else result


def energy(x: torch.Tensor) -> float:
    return float(x.float().square().mean().item())


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    lhs = a.float().flatten()
    rhs = b.float().flatten()
    return float((lhs @ rhs / (lhs.norm() * rhs.norm()).clamp_min(EPS)).item())


def pearson_log(a: torch.Tensor, b: torch.Tensor) -> float:
    x = torch.log(a.float().clamp_min(1e-20))
    y = torch.log(b.float().clamp_min(1e-20))
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y / (x.norm() * y.norm()).clamp_min(EPS)).item())


def bias_fraction(error: torch.Tensor) -> float:
    mean_energy = error.float().mean(dim=0).square().sum()
    row_energy = error.float().square().sum(dim=1).mean()
    return float((mean_energy / row_energy.clamp_min(EPS)).item())


def analyze_module(
    name: str,
    x: torch.Tensor,
    state: dict[str, Any],
    levels: int,
    lloyd_iters: int,
) -> dict[str, Any]:
    d = state["d"].float()
    rhat = _dequantize_codes(
        state["codes"], state["scales"].float(), int(state["group_size"])
    ).T.float()
    bits = int(state.get("activation_bits", 4))
    group_size = int(state.get("activation_group_size", 128))
    xt = x.float() / d[None, :]
    train, test = split_rows(xt.shape[0])
    books = fit_lloyd_codebooks(xt[train], group_size, levels, lloyd_iters)

    q_uniform, _codes, _scales = quantize_activation_with_codes(xt, bits, group_size)
    q_oracle = apply_codebooks(xt, group_size, books)
    e_uniform = (xt - q_uniform) @ rhat
    raw_oracle_residual = (xt - q_oracle) @ rhat
    oracle_direction = (q_oracle - q_uniform) @ rhat
    # The raw Lloyd-Max endpoint need not be the L2 projection in the output
    # metric.  Project the uniform error onto its train-fitted direction; this
    # is the one-dimensional conditional-mean surrogate whose residual is
    # orthogonal on the fit split and testable without refitting on holdout.
    alpha = float(
        ((e_uniform[train] * oracle_direction[train]).sum()
         / oracle_direction[train].square().sum().clamp_min(EPS)).clamp(0.0, 1.0).item()
    )
    e_reducible = oracle_direction * alpha
    e_irreducible = e_uniform - e_reducible
    identity_rel = float(
        ((e_uniform - e_reducible - e_irreducible).norm() / e_uniform.norm().clamp_min(EPS)).item()
    )

    row_rms = xt.square().mean(dim=1).sqrt()
    row_outlier = xt.abs().amax(dim=1) / row_rms.clamp_min(1e-20)
    uniform_row_energy = e_uniform.square().mean(dim=1)
    oracle_row_energy = e_irreducible.square().mean(dim=1)
    uniform_relative_energy = uniform_row_energy / row_rms.square().clamp_min(1e-20)
    oracle_relative_energy = oracle_row_energy / row_rms.square().clamp_min(1e-20)
    red_train_profile = e_reducible[train].square().mean(dim=0)
    red_test_profile = e_reducible[test].square().mean(dim=0)

    fu_train, fu_test = energy(e_uniform[train]), energy(e_uniform[test])
    fi_train, fi_test = energy(e_irreducible[train]), energy(e_irreducible[test])
    fr_train, fr_test = energy(e_reducible[train]), energy(e_reducible[test])
    raw_train, raw_test = energy(raw_oracle_residual[train]), energy(raw_oracle_residual[test])
    cross_train = float(
        (2.0 * (e_reducible[train] * e_irreducible[train]).mean() / max(fu_train, EPS)).item()
    )
    cross_test = float(
        (2.0 * (e_reducible[test] * e_irreducible[test]).mean() / max(fu_test, EPS)).item()
    )
    return {
        "module": name,
        "tokens": int(xt.shape[0]),
        "activation_bits": bits,
        "activation_group_size": group_size,
        "levels": levels,
        "projection_alpha": alpha,
        "identity_relative_error": identity_rel,
        "train": {
            "uniform_mse": fu_train,
            "oracle_residual_mse": fi_train,
            "raw_lloyd_residual_mse": raw_train,
            "reducible_mse": fr_train,
            "oracle_gain": 1.0 - fi_train / max(fu_train, EPS),
            "raw_lloyd_gain": 1.0 - raw_train / max(fu_train, EPS),
            "normalized_cross_term": cross_train,
        },
        "test": {
            "uniform_mse": fu_test,
            "oracle_residual_mse": fi_test,
            "raw_lloyd_residual_mse": raw_test,
            "reducible_mse": fr_test,
            "oracle_gain": 1.0 - fi_test / max(fu_test, EPS),
            "raw_lloyd_gain": 1.0 - raw_test / max(fu_test, EPS),
            "normalized_cross_term": cross_test,
            "reducible_profile_cosine": cosine(red_train_profile, red_test_profile),
            "uniform_logerr_rms_corr": pearson_log(uniform_row_energy[test], row_rms[test]),
            "oracle_logerr_rms_corr": pearson_log(oracle_row_energy[test], row_rms[test]),
            "uniform_logerr_outlier_corr": pearson_log(uniform_row_energy[test], row_outlier[test]),
            "oracle_logerr_outlier_corr": pearson_log(oracle_row_energy[test], row_outlier[test]),
            "uniform_logrelerr_rms_corr": pearson_log(uniform_relative_energy[test], row_rms[test]),
            "oracle_logrelerr_rms_corr": pearson_log(oracle_relative_energy[test], row_rms[test]),
            "uniform_logrelerr_outlier_corr": pearson_log(uniform_relative_energy[test], row_outlier[test]),
            "oracle_logrelerr_outlier_corr": pearson_log(oracle_relative_energy[test], row_outlier[test]),
            "uniform_bias_fraction": bias_fraction(e_uniform[test]),
            "oracle_bias_fraction": bias_fraction(e_irreducible[test]),
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [float(record["test"][key]) for record in records]
        return sum(values) / max(1, len(values))

    return {
        "modules": len(records),
        "mean_test_oracle_gain": mean("oracle_gain"),
        "mean_test_raw_lloyd_gain": mean("raw_lloyd_gain"),
        "mean_projection_alpha": sum(float(record["projection_alpha"]) for record in records) / max(1, len(records)),
        "mean_test_normalized_cross_term": mean("normalized_cross_term"),
        "mean_reducible_profile_cosine": mean("reducible_profile_cosine"),
        "mean_uniform_logerr_rms_corr": mean("uniform_logerr_rms_corr"),
        "mean_oracle_logerr_rms_corr": mean("oracle_logerr_rms_corr"),
        "mean_uniform_logerr_outlier_corr": mean("uniform_logerr_outlier_corr"),
        "mean_oracle_logerr_outlier_corr": mean("oracle_logerr_outlier_corr"),
        "mean_uniform_logrelerr_rms_corr": mean("uniform_logrelerr_rms_corr"),
        "mean_oracle_logrelerr_rms_corr": mean("oracle_logrelerr_rms_corr"),
        "mean_uniform_logrelerr_outlier_corr": mean("uniform_logrelerr_outlier_corr"),
        "mean_oracle_logrelerr_outlier_corr": mean("oracle_logrelerr_outlier_corr"),
        "mean_uniform_bias_fraction": mean("uniform_bias_fraction"),
        "mean_oracle_bias_fraction": mean("oracle_bias_fraction"),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    _tokenizer, batches = make_calibration(
        args.model, args.dataset, args.nsamples, args.sequence_length,
        args.batch_size, args.seed,
    )
    model, _tokenizer, runtime = load_experiment_model(
        model_name=args.model, checkpoint=args.checkpoint, device=device, dtype=dtype,
    )
    states = load_states(args.checkpoint)
    hidden, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    layers = parse_layers(args.layers, len(model.model.layers))
    modules = {item.strip() for item in args.modules.split(",") if item.strip()}
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.model.layers):
        filters = modules if layers is None or layer_index in layers else {"__none__"}
        collected, hidden = collect_inputs_for_layer(
            layer, layer_index, hidden, layer_kwargs, device,
            args.max_tokens_per_module, filters,
        )
        for name, x in sorted(collected.items()):
            if name not in states:
                continue
            print(f"[decompose] {name} tokens={x.shape[0]}", flush=True)
            records.append(analyze_module(name, x, states[name], args.levels, args.lloyd_iters))
        if layers is not None and layer_index >= max(layers):
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "args": vars(args),
        "runtime": runtime.__dict__,
        "records": records,
        "summary": summarize(records),
        "environment": environment_metadata(),
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "reducible_activation_error.json", payload)
    lines = ["# Reducible Activation Error", "", f"Modules: {len(records)}", "", "```json", json.dumps(payload["summary"], indent=2), "```", ""]
    (output / "reducible_activation_error.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4", "synthetic"])
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="0,25")
    parser.add_argument("--modules", default="q_proj,v_proj")
    parser.add_argument("--max-tokens-per-module", type=int, default=512)
    parser.add_argument("--levels", type=int, default=15)
    parser.add_argument("--lloyd-iters", type=int, default=20)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
