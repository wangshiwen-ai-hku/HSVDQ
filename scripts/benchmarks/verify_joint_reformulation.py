#!/usr/bin/env python3
"""Verify the joint-reformulation reducible / irreducible split on a checkpoint.

Prediction 5 of docs/joint_reformulation.pdf:
  - the uniform→oracle gap should reproduce across calibration splits
  - the oracle residual should have near-zero conditional mean and weaker
    dependence on activation RMS / outlier statistics
  - only the reproducible gap is a legal calibration target

This script also summarizes fixed-code F_A^red refine admissions stored in a
reducible-objective checkpoint (fw trust region, line scales, reliability).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from analyze_reducible_activation_error import analyze_module, summarize
from analyze_structured_correction import (
    collect_inputs_for_layer,
    load_states,
    parse_layers,
)
from common import (
    _dtype_from_name,
    environment_metadata,
    load_experiment_model,
    make_calibration,
    write_json,
)
from hsvdquant import capture_first_layer_inputs


def summarize_admissions(states: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, int] = {}
    reliabilities: list[float] = []
    oracle_gains: list[float] = []
    accepted_modules = 0
    eligible_scales = 0
    total_scales = 0
    best_gains: list[float] = []
    best_fw_ratios: list[float] = []
    line_scales: list[float] = []
    for state in states.values():
        source = str(state.get("reducible_source", "missing"))
        sources[source] = sources.get(source, 0) + 1
        teacher = state.get("reducible_teacher_diagnostics") or {}
        if "reducible_oracle_gain" in teacher:
            oracle_gains.append(float(teacher["reducible_oracle_gain"]))
        if "reducible_reliability" in state:
            reliabilities.append(float(state["reducible_reliability"]))
        accepted = int(state.get("reducible_accepted_updates", 0) or 0)
        if accepted > 0:
            accepted_modules += 1
            if "reducible_line_scale" in state:
                line_scales.append(float(state["reducible_line_scale"]))
        history = state.get("reducible_refine_history") or []
        total_scales += len(history)
        module_best_gain = None
        module_best_fw = None
        for row in history:
            eligible_scales += int(float(row.get("fw_eligible", 0.0)) > 0.5)
            gain = row.get("reducible_heldout_gain")
            if gain is None:
                continue
            gain = float(gain)
            if module_best_gain is None or gain > module_best_gain:
                module_best_gain = gain
                module_best_fw = float(row.get("fw_ratio", 1.0))
        if module_best_gain is not None:
            best_gains.append(module_best_gain)
            if module_best_fw is not None:
                best_fw_ratios.append(module_best_fw)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "modules": len(states),
        "sources": sources,
        "accepted_modules": accepted_modules,
        "accepted_fraction": accepted_modules / max(1, len(states)),
        "eligible_line_scales": eligible_scales,
        "total_line_scales": total_scales,
        "mean_oracle_gain": mean(oracle_gains),
        "mean_reliability": mean(reliabilities),
        "mean_best_heldout_red_gain": mean(best_gains),
        "mean_best_fw_ratio": mean(best_fw_ratios),
        "mean_accepted_line_scale": mean(line_scales),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    _tokenizer, batches = make_calibration(
        args.model,
        args.dataset,
        args.nsamples,
        args.sequence_length,
        args.batch_size,
        args.seed,
    )
    model, _tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    states = load_states(args.checkpoint)
    admission = summarize_admissions(states)
    hidden, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    layers = parse_layers(args.layers, len(model.model.layers))
    modules = {item.strip() for item in args.modules.split(",") if item.strip()}
    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(model.model.layers):
        filters = modules if layers is None or layer_index in layers else {"__none__"}
        collected, hidden = collect_inputs_for_layer(
            layer,
            layer_index,
            hidden,
            layer_kwargs,
            device,
            args.max_tokens_per_module,
            filters,
        )
        for name, x in sorted(collected.items()):
            if name not in states:
                continue
            print(f"[pred5] {name} tokens={x.shape[0]}", flush=True)
            records.append(
                analyze_module(name, x, states[name], args.levels, args.lloyd_iters)
            )
        if layers is not None and layer_index >= max(layers):
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    decomposition = summarize(records)
    # Prediction-5 pass/fail style checks on the held-out split.
    checks = {
        "cross_term_near_zero": abs(float(decomposition["mean_test_normalized_cross_term"]))
        < args.cross_tol,
        "profile_reproducible": float(decomposition["mean_reducible_profile_cosine"])
        >= args.profile_cosine_min,
        "oracle_gain_positive": float(decomposition["mean_test_oracle_gain"])
        >= args.min_oracle_gain,
        "relative_outlier_corr_drops": (
            float(decomposition["mean_oracle_logrelerr_outlier_corr"])
            <= float(decomposition["mean_uniform_logrelerr_outlier_corr"]) + args.corr_slack
        ),
    }
    payload = {
        "args": vars(args),
        "runtime": runtime.__dict__,
        "admission": admission,
        "decomposition_summary": decomposition,
        "prediction5_checks": checks,
        "prediction5_pass": all(checks.values()),
        "records": records,
        "environment": environment_metadata(),
        "seconds": time.perf_counter() - started,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "joint_reformulation_verify.json", payload)
    lines = [
        "# Joint Reformulation Verification",
        "",
        f"Prediction-5 pass: **{payload['prediction5_pass']}**",
        "",
        "## Admission",
        "",
        "```json",
        json.dumps(admission, indent=2),
        "```",
        "",
        "## Decomposition",
        "",
        "```json",
        json.dumps(decomposition, indent=2),
        "```",
        "",
        "## Checks",
        "",
        "```json",
        json.dumps(checks, indent=2),
        "```",
        "",
    ]
    (output / "joint_reformulation_verify.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "prediction5_pass": payload["prediction5_pass"],
        "checks": checks,
        "admission": admission,
        "decomposition": decomposition,
    }, indent=2), flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4", "synthetic"])
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", default="0,14,27")
    parser.add_argument("--modules", default="q_proj,v_proj,down_proj")
    parser.add_argument("--max-tokens-per-module", type=int, default=512)
    parser.add_argument("--levels", type=int, default=15)
    parser.add_argument("--lloyd-iters", type=int, default=20)
    parser.add_argument("--cross-tol", type=float, default=0.05)
    parser.add_argument("--profile-cosine-min", type=float, default=0.9)
    parser.add_argument("--min-oracle-gain", type=float, default=0.2)
    parser.add_argument("--corr-slack", type=float, default=0.05)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
