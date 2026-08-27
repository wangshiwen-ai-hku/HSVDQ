#!/usr/bin/env python3
"""Summarize teacher-student trajectory correction quality from checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


LABEL_RE = re.compile(
    r"^(?P<calibration>wikitext2|c4)_w(?P<bits>\d+)a4_r\d+_"
    r"(?P<variant>v3|v2v3)(?:_|$)"
)
LAYER_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.")
EPS = 1e-20


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def module_type(name: str) -> str:
    for kind in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if name.endswith(kind):
            return kind
    return name.rsplit(".", 1)[-1]


def failure_mode(
    upstream: float,
    projected: float,
    quantized: float,
    holdout_gain: float | None,
    split_cosine: float | None,
    cap_headroom: float | None,
    quantized_acceptance: float | None,
) -> str:
    if upstream <= EPS:
        return "no_upstream_gap"
    projection_gain = 1.0 - projected / upstream
    net_gain = 1.0 - quantized / upstream
    if projection_gain <= 0:
        return "irreducible_or_bad_projection"
    if holdout_gain is not None and holdout_gain <= 0:
        return "holdout_overfit"
    if split_cosine is not None and split_cosine < 0.2:
        return "unstable_direction"
    if cap_headroom is not None and cap_headroom > 1.0 and projection_gain < 0.05:
        return "cap_limited"
    if quantized_acceptance is not None and quantized_acceptance <= 0:
        return "quantized_rejected"
    if net_gain <= 0:
        return "realization_failure"
    return "accepted"


def load_rows(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for checkpoint in sorted((result_root / "checkpoints").glob("*/hsvdquant.pt")):
        match = LABEL_RE.match(checkpoint.parent.name)
        if not match:
            continue
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        states = payload.get("states", payload) if isinstance(payload, dict) else {}
        for name, state in states.items():
            if not isinstance(state, dict):
                continue
            diagnostics = state.get("objective_diagnostics") or {}
            upstream = finite(diagnostics.get("upstream_mse"))
            projected = finite(diagnostics.get("projected_mse"))
            quantized = finite(state.get("error"))
            if upstream is None or projected is None or quantized is None:
                continue
            layer_match = LAYER_RE.search(name)
            norm_ratio = finite(diagnostics.get("correction_norm_ratio")) or 0.0
            raw_norm_ratio = finite(diagnostics.get("raw_correction_norm_ratio")) or 0.0
            line_scale = finite(diagnostics.get("correction_line_scale")) or 0.0
            trust_scale = finite(diagnostics.get("correction_trust_scale"))
            trust_scale = 1.0 if trust_scale is None else trust_scale
            block = state.get("block_trajectory_diagnostics") or {}
            valid_denominator = upstream > EPS
            projection_gain = 1.0 - projected / upstream if valid_denominator else None
            realization_gap = (quantized - projected) / upstream if valid_denominator else None
            net_gain = 1.0 - quantized / upstream if valid_denominator else None
            oracle_projected = finite(diagnostics.get("oracle_projected_mse"))
            line_projected = finite(diagnostics.get("line_projected_mse"))
            holdout_upstream = finite(diagnostics.get("holdout_upstream_mse"))
            holdout_projected = finite(diagnostics.get("holdout_projected_mse"))
            holdout_gain = finite(diagnostics.get("holdout_gain"))
            cap_headroom = finite(diagnostics.get("cap_headroom"))
            split_cosine = finite(diagnostics.get("split_direction_cosine"))
            quantized_acceptance = finite(diagnostics.get("trajectory_quantized_acceptance"))
            rows.append(
                {
                    **match.groupdict(),
                    "bits": int(match.group("bits")),
                    "checkpoint": checkpoint.parent.name,
                    "module": name,
                    "layer": int(layer_match.group("layer")) if layer_match else -1,
                    "module_type": module_type(name),
                    "upstream_mse": upstream,
                    "oracle_projected_mse": oracle_projected,
                    "line_projected_mse": line_projected,
                    "projected_mse": projected,
                    "quantized_target_mse": quantized,
                    "projection_gain": projection_gain,
                    "oracle_projection_gain": finite(diagnostics.get("oracle_projection_gain")),
                    "oracle_to_projected_gap": finite(diagnostics.get("oracle_to_projected_gap")),
                    "stabilization_gap": finite(diagnostics.get("stabilization_gap")),
                    "realization_gap": realization_gap,
                    "net_gain": net_gain,
                    "holdout_upstream_mse": holdout_upstream,
                    "holdout_projected_mse": holdout_projected,
                    "holdout_gain": holdout_gain,
                    "holdout_best_scale": finite(diagnostics.get("holdout_best_scale")),
                    "correction_norm_ratio": norm_ratio,
                    "raw_correction_norm_ratio": raw_norm_ratio,
                    "cap_headroom": cap_headroom,
                    "line_scale": line_scale,
                    "trust_scale": trust_scale,
                    "trust_hit": int(trust_scale < 1.0 - 1e-6),
                    "zero_fallback": int(valid_denominator and norm_ratio <= 1e-12),
                    "split_direction_cosine": split_cosine,
                    "split_norm_ratio": finite(diagnostics.get("split_norm_ratio")),
                    "effective_rank": finite(diagnostics.get("effective_rank")),
                    "hessian_condition": finite(diagnostics.get("hessian_condition")),
                    "weak_subspace_energy_share": finite(diagnostics.get("weak_subspace_energy_share")),
                    "weak_subspace_dim": finite(diagnostics.get("weak_subspace_dim")),
                    "spectral_kept_dim": finite(diagnostics.get("spectral_kept_dim")),
                    "trajectory_reliability_reject": int(
                        (finite(diagnostics.get("trajectory_reliability_reject")) or 0.0) > 0
                    ),
                    "trajectory_quantized_acceptance": quantized_acceptance,
                    "trajectory_quantized_reverted": int(
                        (finite(diagnostics.get("trajectory_quantized_reverted")) or 0.0) > 0
                    ),
                    "failure_mode": failure_mode(
                        upstream,
                        projected,
                        quantized,
                        holdout_gain,
                        split_cosine,
                        cap_headroom,
                        quantized_acceptance,
                    ),
                    "block_input_mse": finite(block.get("input_mse")),
                    "block_output_mse": finite(block.get("output_mse")),
                    "block_input_nmse": finite(block.get("input_nmse")),
                    "block_output_nmse": finite(block.get("output_nmse")),
                    "block_input_teacher_energy": finite(block.get("input_teacher_energy")),
                    "block_output_teacher_energy": finite(block.get("output_teacher_energy")),
                    "block_input_student_energy": finite(block.get("input_student_energy")),
                    "block_output_student_energy": finite(block.get("output_student_energy")),
                    "block_error_delta": finite(block.get("error_delta")),
                    "block_correction_gain": finite(block.get("correction_gain")),
                }
            )
    return rows


def mean(values: Iterable[float]) -> float | None:
    data = [value for value in values if value is not None]
    return statistics.fmean(data) if data else None


def median(values: Iterable[float]) -> float | None:
    data = [value for value in values if value is not None]
    return statistics.median(data) if data else None


def summarize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for group_key, group in sorted(groups.items()):
        valid = [row for row in group if row["projection_gain"] is not None]
        upstream_sum = sum(row["upstream_mse"] for row in valid)
        projected_sum = sum(row["projected_mse"] for row in valid)
        quantized_sum = sum(row["quantized_target_mse"] for row in valid)
        entry: dict[str, Any] = dict(zip(keys, group_key))
        entry.update(
            {
                "modules": len(group),
                "correctable_modules": len(valid),
                "upstream_mse_mean": mean(row["upstream_mse"] for row in group),
                "projection_gain_weighted": (
                    1.0 - projected_sum / upstream_sum if upstream_sum > EPS else None
                ),
                "net_gain_weighted": (
                    1.0 - quantized_sum / upstream_sum if upstream_sum > EPS else None
                ),
                "projection_gain_median": median(
                    row["projection_gain"] for row in valid
                ),
                "realization_gap_median": median(
                    row["realization_gap"] for row in valid
                ),
                "net_gain_median": median(row["net_gain"] for row in valid),
                "positive_projection_rate": mean(
                    float(row["projection_gain"] > 0.0) for row in valid
                ),
                "positive_net_rate": mean(float(row["net_gain"] > 0.0) for row in valid),
                "correction_norm_ratio_median": median(
                    row["correction_norm_ratio"] for row in valid
                ),
                "raw_correction_norm_ratio_median": median(
                    row["raw_correction_norm_ratio"] for row in valid
                ),
                "line_scale_median": median(row["line_scale"] for row in valid),
                "trust_hit_rate": mean(float(row["trust_hit"]) for row in valid),
                "zero_fallback_rate": mean(float(row["zero_fallback"]) for row in valid),
                "holdout_gain_median": median(row["holdout_gain"] for row in valid),
                "split_direction_cosine_median": median(row["split_direction_cosine"] for row in valid),
                "cap_headroom_median": median(row["cap_headroom"] for row in valid),
                "effective_rank_median": median(row["effective_rank"] for row in valid),
                "weak_subspace_energy_share_median": median(
                    row["weak_subspace_energy_share"] for row in valid
                ),
                "quantized_acceptance_rate": mean(
                    float(row["trajectory_quantized_acceptance"] > 0)
                    for row in valid
                    if row["trajectory_quantized_acceptance"] is not None
                ),
                "reliability_reject_rate": mean(
                    float(row["trajectory_reliability_reject"]) for row in valid
                ),
                "quantized_revert_rate": mean(
                    float(row["trajectory_quantized_reverted"]) for row in valid
                ),
            }
        )
        counts: dict[str, int] = defaultdict(int)
        for row in valid:
            counts[row["failure_mode"]] += 1
        entry["dominant_failure_mode"] = max(counts, key=counts.get) if counts else ""
        output.append(entry)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.4f}"


def write_report(path: Path, overall: list[dict[str, Any]]) -> None:
    lines = [
        "# Trajectory correction quality",
        "",
        "- `projection_gain = 1 - projected_mse / upstream_mse`: teacher trajectory error that the damped/trusted correction can remove before quantization.",
        "- `realization_gap = (quantized_target_mse - projected_mse) / upstream_mse`: correction lost when W/A quantization realizes the target; lower is better.",
        "- `net_gain = 1 - quantized_target_mse / upstream_mse`: correction actually retained after quantization; positive is required.",
        "- `holdout_gain`, `split_direction_cosine`, and `cap_headroom` explain whether a correction is statistically stable before quantization.",
        "- `dominant_failure_mode` groups failures into bad projection, holdout overfit, unstable direction, cap limitation, or realization failure.",
        "",
        "| calib | W bits | variant | modules | projection gain | realization gap | net gain | holdout gain | split cos | cap headroom | positive net | dominant failure |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in overall:
        lines.append(
            f"| {row['calibration']} | {row['bits']} | {row['variant']} | {row['modules']} | "
            f"{fmt(row['projection_gain_weighted'])} | {fmt(row['realization_gap_median'])} | "
            f"{fmt(row['net_gain_weighted'])} | {fmt(row['holdout_gain_median'])} | "
            f"{fmt(row['split_direction_cosine_median'])} | {fmt(row['cap_headroom_median'])} | "
            f"{fmt(row['positive_net_rate'])} | {row.get('dominant_failure_mode', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(output_dir: Path, rows: list[dict[str, Any]], by_depth: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid = [row for row in rows if row["projection_gain"] is not None and row["net_gain"] is not None]
    if valid:
        figure, axis = plt.subplots(figsize=(8, 6))
        types = sorted({row["module_type"] for row in valid})
        for kind in types:
            subset = [row for row in valid if row["module_type"] == kind]
            axis.scatter(
                [row["projection_gain"] for row in subset],
                [row["net_gain"] for row in subset],
                s=18,
                alpha=0.7,
                label=kind,
            )
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.axline((0, 0), slope=1, color="gray", linewidth=1, linestyle=":")
        axis.set_xlabel("projection gain")
        axis.set_ylabel("net gain after quantization")
        axis.set_title("Trajectory projection gain vs realized quantized gain")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "projection_vs_net_gain.png", dpi=180)
        plt.close(figure)

    blocks = {
        (row["calibration"], row["bits"], row["variant"], row["layer"]): row
        for row in rows
        if row["block_output_mse"] is not None
    }
    block_rows = [blocks[key] for key in sorted(blocks)]
    if block_rows:
        figure, axis1 = plt.subplots(figsize=(10, 5))
        layers = [row["layer"] + 1 for row in block_rows]
        axis1.plot(layers, [row["block_output_mse"] for row in block_rows], marker="o", label="absolute MSE")
        axis1.set_xlabel("decoder block")
        axis1.set_ylabel("absolute MSE")
        axis1.grid(True, alpha=0.25)
        axis2 = axis1.twinx()
        axis2.plot(
            layers,
            [row["block_output_nmse"] for row in block_rows],
            marker="s",
            color="tab:orange",
            label="NMSE",
        )
        axis2.set_ylabel("NMSE")
        axis1.set_title("Block trajectory error: absolute MSE vs NMSE")
        lines = axis1.get_lines() + axis2.get_lines()
        axis1.legend(lines, [line.get_label() for line in lines])
        figure.tight_layout()
        figure.savefig(output_dir / "block_mse_vs_nmse.png", dpi=180)
        plt.close(figure)

    depth_valid = [row for row in by_depth if row.get("cap_headroom_median") is not None]
    if depth_valid:
        figure, axis = plt.subplots(figsize=(10, 5))
        for key in sorted({(row["calibration"], row["bits"], row["variant"]) for row in depth_valid}):
            subset = [row for row in depth_valid if (row["calibration"], row["bits"], row["variant"]) == key]
            axis.plot(
                [row["layer"] + 1 for row in subset],
                [row["cap_headroom_median"] for row in subset],
                marker="o",
                label=f"{key[0]} W{key[1]} {key[2]}",
            )
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_xlabel("decoder block")
        axis.set_ylabel("median raw_norm / cap")
        axis.set_title("Trajectory cap saturation by depth")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "cap_headroom_by_depth.png", dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir) if args.output_dir else result_root / "trajectory_diagnostics"
    rows = load_rows(result_root)
    if not rows:
        raise SystemExit(f"no stable V3/V2+V3 checkpoint diagnostics found below {result_root}")
    overall = summarize(rows, ("calibration", "bits", "variant"))
    by_depth = summarize(rows, ("calibration", "bits", "variant", "layer"))
    by_type = summarize(rows, ("calibration", "bits", "variant", "module_type"))
    blocks_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["block_output_mse"] is not None:
            key = (row["checkpoint"], row["layer"])
            blocks_by_key.setdefault(
                key,
                {
                    name: row[name]
                    for name in (
                        "calibration",
                        "bits",
                        "variant",
                        "checkpoint",
                        "layer",
                        "block_input_mse",
                        "block_output_mse",
                        "block_input_nmse",
                        "block_output_nmse",
                        "block_input_teacher_energy",
                        "block_output_teacher_energy",
                        "block_input_student_energy",
                        "block_output_student_energy",
                        "block_error_delta",
                        "block_correction_gain",
                    )
                },
            )
    block_rows = list(blocks_by_key.values())
    write_csv(output_dir / "per_module.csv", rows)
    write_csv(output_dir / "overall.csv", overall)
    write_csv(output_dir / "by_depth.csv", by_depth)
    write_csv(output_dir / "by_module_type.csv", by_type)
    write_csv(output_dir / "per_block.csv", block_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_report(output_dir / "report.md", overall)
    write_plots(output_dir, rows, by_depth)
    print(f"wrote {len(rows)} module rows to {output_dir}")


if __name__ == "__main__":
    main()
