#!/usr/bin/env python3
"""Toy ablation for V3-OAR on activation-channel tail mismatch.

The experiment is intentionally local and checkpoint-free.  It compares the
native V2/V2-plus geometry against amplitude sorting, output-aware grouping,
tail-aware D, and a smaller-group reference on held-out synthetic activations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v3_outlier_routing import (  # noqa: E402
    V3OARConfig,
    activation_output_loss_for_order,
    amax_permutation,
    block_hadamard_right,
    block_hadamard_weight,
    exact_error_terms,
    fixed_code_fp_route,
    optimize_tail_smoothing,
    quantize_activations,
    quantize_weights,
    refine_output_aware_permutation,
    state_metrics,
    tail_token_weights,
    weighted_rank_projection,
)


@dataclass
class LayerState:
    order: torch.Tensor
    d: torch.Tensor
    branch: torch.Tensor
    l1: torch.Tensor
    l2: torch.Tensor
    qweight: torch.Tensor
    qinputs_train: torch.Tensor
    route_diagnostics: list[dict[str, float]]
    hadamard_signs: torch.Tensor | None = None


def _randn(shape: tuple[int, ...], generator: torch.Generator, scale: float = 1.0) -> torch.Tensor:
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * float(scale)


def _inject_tail(
    inputs: torch.Tensor,
    channels: list[int],
    generator: torch.Generator,
    probability: float,
    amplitudes: torch.Tensor,
    shared_tokens: bool,
) -> None:
    rows = inputs.shape[0]
    if shared_tokens:
        mask = torch.rand(rows, generator=generator) < probability
        signs = torch.where(
            torch.rand(rows, generator=generator) < 0.5,
            -torch.ones(rows),
            torch.ones(rows),
        )
        for index, channel in enumerate(channels):
            jitter = 0.85 + 0.3 * torch.rand(rows, generator=generator)
            inputs[:, channel] += mask * signs * amplitudes[index] * jitter
        return
    for index, channel in enumerate(channels):
        mask = torch.rand(rows, generator=generator) < probability
        signs = torch.where(
            torch.rand(rows, generator=generator) < 0.5,
            -torch.ones(rows),
            torch.ones(rows),
        )
        jitter = 0.85 + 0.3 * torch.rand(rows, generator=generator)
        inputs[:, channel] += mask * signs * amplitudes[index] * jitter


def make_case(
    name: str,
    seed: int,
    rows_fit: int,
    rows_holdout: int,
    rows_test: int,
    channels: int,
    outputs: int,
    group_size: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    total = rows_fit + rows_holdout + rows_test
    channel_scale = torch.exp(torch.linspace(-0.35, 0.35, channels))
    inputs = _randn((total, channels), generator, 0.32) * channel_scale
    low_left = _randn((channels, rank), generator, 0.45)
    low_right = _randn((rank, outputs), generator, 0.45)
    residual = _randn((channels, outputs), generator, 0.16)

    scattered = list(range(2, channels, group_size))
    amplitudes = torch.linspace(10.0, 34.0, len(scattered))
    metadata: dict[str, Any] = {
        "tail_channels_train": scattered,
        "tail_channels_test": scattered,
        "description": "",
    }
    if name == "stable_scattered_tail":
        _inject_tail(inputs, scattered, generator, 0.035, amplitudes, shared_tokens=False)
        residual[scattered] *= 2.4
        metadata["description"] = (
            "one persistent high-sensitivity tail channel per native group; "
            "range clustering should isolate pollution"
        )
    elif name == "sensitivity_mismatch":
        _inject_tail(inputs, scattered, generator, 0.04, amplitudes * 1.35, shared_tokens=False)
        residual[scattered] *= 0.12
        sensitive = [min(channel + 4, channels - 1) for channel in scattered]
        moderate = torch.linspace(4.0, 8.0, len(sensitive))
        _inject_tail(inputs, sensitive, generator, 0.07, moderate, shared_tokens=True)
        residual[sensitive] *= 6.0
        metadata["sensitive_moderate_channels"] = sensitive
        metadata["description"] = (
            "largest-amplitude rows have low output sensitivity while moderate "
            "tails multiply large residual rows; amax alone is mis-specified"
        )
    elif name == "unstable_tail_location":
        _inject_tail(
            inputs[:rows_fit], scattered, generator, 0.04, amplitudes, shared_tokens=False
        )
        shifted = [int((channel + group_size // 2) % channels) for channel in scattered]
        _inject_tail(
            inputs[rows_fit:], shifted, generator, 0.04, amplitudes, shared_tokens=False
        )
        residual[scattered] *= 2.4
        residual[shifted] *= 2.4
        metadata["tail_channels_test"] = shifted
        metadata["description"] = (
            "tail identity shifts between grouping-fit and admission/test tokens; "
            "negative control for the held-out admission requirement"
        )
    else:
        raise ValueError(f"unknown case: {name}")

    weight = low_left @ low_right + residual
    fit = inputs[:rows_fit]
    holdout = inputs[rows_fit : rows_fit + rows_holdout]
    test = inputs[rows_fit + rows_holdout :]
    return weight, fit, holdout, test, metadata


def _calibrate_state(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    d: torch.Tensor,
    order: torch.Tensor,
    config: V3OARConfig,
    route_iters: int,
    hadamard_signs: torch.Tensor | None = None,
) -> LayerState:
    permuted_inputs = (inputs / d).index_select(1, order)
    permuted_weight = (d[:, None] * weight).index_select(0, order)
    hessian = permuted_inputs.T @ permuted_inputs / float(inputs.shape[0])
    reference = hessian.diagonal().mean().clamp_min(1e-12)
    hessian = hessian + torch.eye(hessian.shape[0]) * (config.damp * reference)
    branch, l1, l2 = weighted_rank_projection(
        permuted_weight, hessian, None, config.rank
    )
    route_diagnostics: list[dict[str, float]] = []
    quantizer_inputs = (
        permuted_inputs
        if hadamard_signs is None
        else block_hadamard_right(
            permuted_inputs, config.activation_group_size, hadamard_signs
        )
    )
    qinputs, _activation_codes, _activation_scales = quantize_activations(
        quantizer_inputs, config.activation_bits, config.activation_group_size
    )
    qweight = torch.empty_like(permuted_weight)
    for _ in range(route_iters + 1):
        residual = permuted_weight - branch
        quantizer_residual = (
            residual
            if hadamard_signs is None
            else block_hadamard_weight(
                residual, config.activation_group_size, hadamard_signs
            )
        )
        qweight, _weight_codes, _weight_scales = quantize_weights(
            quantizer_residual, config.weight_bits, config.weight_group_size
        )
        if len(route_diagnostics) >= route_iters:
            break
        activation_error = (quantizer_inputs - qinputs) @ qweight
        token_weights = tail_token_weights(
            activation_error, config.tail_fraction, config.tail_weight
        )
        branch, l1, l2, diagnostics = fixed_code_fp_route(
            permuted_inputs,
            qinputs,
            permuted_weight,
            qweight,
            config.rank,
            token_weights=token_weights,
            baseline_branch=branch,
            damp=config.damp,
            fw_epsilon=config.fw_epsilon,
        )
        route_diagnostics.append(diagnostics)
    return LayerState(
        order=order,
        d=d,
        branch=branch,
        l1=l1,
        l2=l2,
        qweight=qweight,
        qinputs_train=qinputs,
        route_diagnostics=route_diagnostics,
        hadamard_signs=hadamard_signs,
    )


def _evaluate_state(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    state: LayerState,
    config: V3OARConfig,
) -> dict[str, float]:
    permuted_inputs = (inputs / state.d).index_select(1, state.order)
    permuted_weight = (state.d[:, None] * weight).index_select(0, state.order)
    quantizer_inputs = (
        permuted_inputs
        if state.hadamard_signs is None
        else block_hadamard_right(
            permuted_inputs,
            config.activation_group_size,
            state.hadamard_signs,
        )
    )
    qinputs, _codes, scales = quantize_activations(
        quantizer_inputs, config.activation_bits, config.activation_group_size
    )
    metrics = state_metrics(
        permuted_inputs,
        permuted_weight,
        state.branch,
        qinputs,
        state.qweight,
        tail_fraction=config.tail_fraction,
        quantizer_inputs=quantizer_inputs,
    )
    residual_target = permuted_weight - state.branch
    if state.hadamard_signs is not None:
        residual_target = block_hadamard_weight(
            residual_target,
            config.activation_group_size,
            state.hadamard_signs,
        )
    terms = exact_error_terms(
        quantizer_inputs,
        qinputs,
        residual_target,
        state.qweight,
    )
    grouped = quantizer_inputs.reshape(
        quantizer_inputs.shape[0],
        quantizer_inputs.shape[1] // config.activation_group_size,
        config.activation_group_size,
    )
    group_rms = grouped.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    pollution = grouped.abs().amax(dim=-1) / group_rms
    metrics.update(
        {
            "scale_mean": float(scales.mean().item()),
            "scale_p99": float(torch.quantile(scales.reshape(-1), 0.99).item()),
            "pollution_mean": float(pollution.mean().item()),
            "pollution_p99": float(torch.quantile(pollution.reshape(-1), 0.99).item()),
            "loss_fw": terms["fw"],
            "loss_cross": terms["cross"],
            "loss_fa": terms["fa"],
            "cross_over_total": terms["cross"] / max(abs(terms["direct"]), 1e-30),
            "identity_error": terms["relative_identity_error"],
        }
    )
    return metrics


@torch.no_grad()
def _select_hadamard_signs(
    fit: torch.Tensor,
    holdout: torch.Tensor,
    weight: torch.Tensor,
    base_state: LayerState,
    config: V3OARConfig,
    trials: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Select a static block H-D transform using calibration holdout loss."""

    if trials < 1:
        raise ValueError("Hadamard trials must be at least one")
    generator = torch.Generator().manual_seed(seed)
    channels = fit.shape[1]
    fit_inputs = (fit / base_state.d).index_select(1, base_state.order)
    holdout_inputs = (holdout / base_state.d).index_select(1, base_state.order)
    permuted_weight = (base_state.d[:, None] * weight).index_select(
        0, base_state.order
    )
    residual = permuted_weight - base_state.branch
    teacher = holdout_inputs @ permuted_weight
    losses: list[float] = []
    candidates: list[torch.Tensor] = []
    for trial in range(trials):
        signs = (
            torch.ones(channels)
            if trial == 0
            else torch.where(
                torch.rand(channels, generator=generator) < 0.5,
                -torch.ones(channels),
                torch.ones(channels),
            )
        )
        transformed_weight = block_hadamard_weight(
            residual, config.activation_group_size, signs
        )
        qweight, _codes, _scales = quantize_weights(
            transformed_weight, config.weight_bits, config.weight_group_size
        )
        transformed_fit = block_hadamard_right(
            fit_inputs, config.activation_group_size, signs
        )
        transformed_holdout = block_hadamard_right(
            holdout_inputs, config.activation_group_size, signs
        )
        # Quantize fit as well so a candidate with a lucky holdout-only range is
        # mildly penalized.  The held-out term remains the dominant selector.
        qfit, _fit_codes, _fit_scales = quantize_activations(
            transformed_fit, config.activation_bits, config.activation_group_size
        )
        qholdout, _holdout_codes, _holdout_scales = quantize_activations(
            transformed_holdout, config.activation_bits, config.activation_group_size
        )
        fit_error = fit_inputs @ permuted_weight - (
            fit_inputs @ base_state.branch + qfit @ qweight
        )
        holdout_error = teacher - (
            holdout_inputs @ base_state.branch + qholdout @ qweight
        )
        loss = 0.2 * fit_error.square().mean() + holdout_error.square().mean()
        candidates.append(signs)
        losses.append(float(loss.item()))
    selected = min(range(len(losses)), key=losses.__getitem__)
    return candidates[selected], {
        "selected_trial": selected,
        "selection_losses": losses,
    }


def _initial_v2_smoothing(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    config: V3OARConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    order = torch.arange(inputs.shape[1])
    hessian = inputs.T @ inputs / float(inputs.shape[0])
    branch, _l1, _l2 = weighted_rank_projection(weight, hessian, None, config.rank)
    residual = weight - branch
    v2_config = replace(config, tail_weight=0.0)
    return optimize_tail_smoothing(inputs, residual, order, v2_config)


def _select_grouping(
    fit: torch.Tensor,
    holdout: torch.Tensor,
    v2_state: LayerState,
    config: V3OARConfig,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    physical_qweight = torch.empty_like(v2_state.qweight)
    physical_qweight[v2_state.order] = v2_state.qweight
    fit_smoothed = fit / v2_state.d
    holdout_smoothed = holdout / v2_state.d
    identity = torch.arange(fit.shape[1])
    sort_order = amax_permutation(fit_smoothed)
    baseline_qinputs, _codes, _scales = quantize_activations(
        fit_smoothed, config.activation_bits, config.activation_group_size
    )
    token_weights = tail_token_weights(
        (fit_smoothed - baseline_qinputs) @ physical_qweight,
        config.tail_fraction,
        config.tail_weight,
    )
    refined, refine_diagnostics = refine_output_aware_permutation(
        fit_smoothed,
        physical_qweight,
        sort_order,
        config,
        token_weights=token_weights,
        seed=seed,
    )
    candidates = {"identity": identity, "sort": sort_order, "cost": refined}
    fit_losses: dict[str, float] = {}
    holdout_losses: dict[str, float] = {}
    for label, order in candidates.items():
        fit_losses[label] = float(
            activation_output_loss_for_order(
                fit_smoothed,
                physical_qweight,
                order,
                config.activation_bits,
                config.activation_group_size,
                token_weights,
            ).item()
        )
        holdout_losses[label] = float(
            activation_output_loss_for_order(
                holdout_smoothed,
                physical_qweight,
                order,
                config.activation_bits,
                config.activation_group_size,
            ).item()
        )
    selected = min(holdout_losses, key=holdout_losses.get)
    return candidates, {
        "selected": selected,
        "fit_losses": fit_losses,
        "holdout_losses": holdout_losses,
        "refinement": refine_diagnostics,
    }


def run_case(
    name: str,
    case_index: int,
    args: argparse.Namespace,
    config: V3OARConfig,
) -> dict[str, Any]:
    weight, fit, holdout, test, metadata = make_case(
        name,
        args.seed + case_index * 1009,
        args.fit_tokens,
        args.holdout_tokens,
        args.test_tokens,
        args.channels,
        args.outputs,
        args.group_size,
        args.rank,
    )
    calibration = torch.cat((fit, holdout), dim=0)
    identity = torch.arange(args.channels)
    d_v2, v2_smoothing = _initial_v2_smoothing(calibration, weight, config)
    v2 = _calibrate_state(calibration, weight, d_v2, identity, config, route_iters=0)
    v2_plus = _calibrate_state(
        calibration, weight, d_v2, identity, config, route_iters=args.route_iters
    )
    candidates, grouping = _select_grouping(
        fit, holdout, v2_plus, config, args.seed + case_index * 37
    )

    sort_state = _calibrate_state(
        calibration,
        weight,
        d_v2,
        candidates["sort"],
        config,
        route_iters=args.route_iters,
    )
    cost_state = _calibrate_state(
        calibration,
        weight,
        d_v2,
        candidates["cost"],
        config,
        route_iters=args.route_iters,
    )
    selected_order = candidates[str(grouping["selected"])]
    physical_branch = torch.empty_like(v2_plus.branch)
    physical_branch[v2_plus.order] = v2_plus.branch
    branch_effective = physical_branch / d_v2[:, None]
    residual_unsmoothed = weight - branch_effective
    d_v3, v3_smoothing = optimize_tail_smoothing(
        calibration,
        residual_unsmoothed,
        selected_order,
        config,
        initial=d_v2,
    )
    full_candidate = _calibrate_state(
        calibration,
        weight,
        d_v3,
        selected_order,
        config,
        route_iters=args.route_iters,
    )
    baseline_holdout = _evaluate_state(holdout, weight, v2_plus, config)
    candidate_holdout = _evaluate_state(holdout, weight, full_candidate, config)
    full_accepted = candidate_holdout["weighted_loss"] < baseline_holdout["weighted_loss"]
    full_state = full_candidate if full_accepted else v2_plus
    hadamard_signs, hadamard_selection = _select_hadamard_signs(
        fit,
        holdout,
        weight,
        full_state,
        config,
        args.hadamard_trials,
        args.seed + case_index * 53,
    )
    hadamard_candidate = _calibrate_state(
        calibration,
        weight,
        full_state.d,
        full_state.order,
        config,
        route_iters=args.route_iters,
        hadamard_signs=hadamard_signs,
    )
    full_holdout = _evaluate_state(holdout, weight, full_state, config)
    hadamard_holdout = _evaluate_state(holdout, weight, hadamard_candidate, config)
    hadamard_accepted = (
        hadamard_holdout["weighted_loss"] < full_holdout["weighted_loss"]
    )
    hadamard_state = hadamard_candidate if hadamard_accepted else full_state
    small_config = replace(
        config,
        activation_group_size=args.group_size // 2,
        weight_group_size=args.group_size,
    )
    small_state = _calibrate_state(
        calibration,
        weight,
        d_v2,
        identity,
        small_config,
        route_iters=args.route_iters,
    )

    methods = {
        "v2": (v2, config),
        "v2_plus": (v2_plus, config),
        "v3_sort": (sort_state, config),
        "v3_cost": (cost_state, config),
        "v3_full": (full_state, config),
        "v3_block_hadamard": (hadamard_state, config),
        "g_half_reference": (small_state, small_config),
    }
    rows: list[dict[str, Any]] = []
    for method, (state, method_config) in methods.items():
        metrics = _evaluate_state(test, weight, state, method_config)
        rows.append(
            {
                "case": name,
                "method": method,
                **metrics,
                "route_last": state.route_diagnostics[-1]
                if state.route_diagnostics
                else {},
            }
        )
    return {
        "case": name,
        "metadata": metadata,
        "grouping": grouping,
        "v2_smoothing": v2_smoothing,
        "v3_smoothing": v3_smoothing,
        "full_admission": {
            "accepted": full_accepted,
            "baseline_holdout_loss": baseline_holdout["weighted_loss"],
            "candidate_holdout_loss": candidate_holdout["weighted_loss"],
        },
        "hadamard": {
            **hadamard_selection,
            "accepted": hadamard_accepted,
            "baseline_holdout_loss": full_holdout["weighted_loss"],
            "candidate_holdout_loss": hadamard_holdout["weighted_loss"],
        },
        "selected_order": selected_order.tolist(),
        "rows": rows,
    }


def _write_csv(cases: list[dict[str, Any]], path: Path) -> None:
    flat: list[dict[str, Any]] = []
    for case in cases:
        for row in case["rows"]:
            item = {key: value for key, value in row.items() if key != "route_last"}
            item["route_gain"] = row["route_last"].get("a4_gain", 0.0)
            item["route_scale"] = row["route_last"].get("accepted_scale", 0.0)
            flat.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(flat[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(flat)


def _write_report(cases: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# V3-OAR activation-tail toy experiment",
        "",
        "All metrics are evaluated on unseen synthetic tokens. Lower is better for NMSE/CVaR.",
        "",
    ]
    for case in cases:
        lines.extend(
            [
                f"## {case['case']}",
                "",
                str(case["metadata"]["description"]),
                "",
                f"Grouping selected on calibration holdout: `{case['grouping']['selected']}`.",
                f"Full V3 held-out admission: `{case['full_admission']['accepted']}`.",
                "Block Hadamard held-out admission: "
                f"`{case['hadamard']['accepted']}` "
                f"(sign trial `{case['hadamard']['selected_trial']}`).",
                "",
                "| method | NMSE | CVaR | activation MSE | A16 MSE | cross / total | scale p99 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in case["rows"]:
            lines.append(
                "| {method} | {nmse:.4e} | {cvar:.4e} | {activation_mse:.4e} | "
                "{a16_mse:.4e} | {cross_over_total:.2%} | {scale_p99:.4e} |".format(**row)
            )
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _plot(cases: list[dict[str, Any]], path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    methods = [row["method"] for row in cases[0]["rows"]]
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.2), squeeze=False)
    for axis, case in zip(axes[0], cases, strict=True):
        values = [next(row["nmse"] for row in case["rows"] if row["method"] == method) for method in methods]
        colors = ["#777777", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
        axis.bar(range(len(methods)), values, color=colors[: len(methods)])
        axis.set_yscale("log")
        axis.set_xticks(range(len(methods)), methods, rotation=35, ha="right")
        axis.set_title(case["case"])
        axis.set_ylabel("held-out output NMSE")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["stable_scattered_tail", "sensitivity_mismatch", "unstable_tail_location"],
    )
    parser.add_argument("--output", default="hsvdquant/toy/results/v3_outlier_routing")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--outputs", type=int, default=48)
    parser.add_argument("--group-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--fit-tokens", type=int, default=384)
    parser.add_argument("--holdout-tokens", type=int, default=384)
    parser.add_argument("--test-tokens", type=int, default=1024)
    parser.add_argument("--route-iters", type=int, default=2)
    parser.add_argument("--hadamard-trials", type=int, default=8)
    parser.add_argument("--grouping-passes", type=int, default=2)
    parser.add_argument("--grouping-candidates", type=int, default=1200)
    parser.add_argument("--smoothing-steps", type=int, default=70)
    parser.add_argument("--tail-fraction", type=float, default=0.02)
    parser.add_argument("--tail-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.group_size % 2:
        raise ValueError("group-size must be even for the g/2 reference")
    if args.group_size <= 0 or args.group_size & (args.group_size - 1):
        raise ValueError("group-size must be a power of two for block Hadamard")
    config = V3OARConfig(
        activation_group_size=args.group_size,
        weight_group_size=args.group_size,
        rank=args.rank,
        tail_fraction=args.tail_fraction,
        tail_weight=args.tail_weight,
        grouping_passes=args.grouping_passes,
        grouping_candidates=args.grouping_candidates,
        smoothing_steps=args.smoothing_steps,
        smoothing_lr=0.04,
        activation_weight=0.35,
        fw_epsilon=0.08,
    )
    config.validate(args.channels)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [run_case(name, index, args, config) for index, name in enumerate(args.cases)]
    payload = {"config": asdict(config), "arguments": vars(args), "cases": cases}
    (output / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_csv(cases, output / "metrics.csv")
    _write_report(cases, output / "report.md")
    plotted = _plot(cases, output / "nmse.png")
    print(json.dumps({"output": str(output), "cases": len(cases), "plot": plotted}, indent=2))
    for case in cases:
        best = min(case["rows"], key=lambda row: row["nmse"])
        print(
            f"{case['case']}: selected={case['grouping']['selected']} "
            f"best={best['method']} nmse={best['nmse']:.4e}"
        )


if __name__ == "__main__":
    main()
