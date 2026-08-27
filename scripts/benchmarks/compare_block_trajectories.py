#!/usr/bin/env python3
"""Compare paired FP-teacher/student trajectories for V1 and V3 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch


def load_blocks(checkpoint: Path) -> dict[int, dict[str, float]]:
    payload = torch.load(checkpoint / "hsvdquant.pt", map_location="cpu", weights_only=False)
    states = payload.get("states", payload)
    blocks: dict[int, dict[str, float]] = {}
    for name, state in states.items():
        diagnostics = state.get("block_trajectory_diagnostics") if isinstance(state, dict) else None
        if not diagnostics:
            continue
        parts = name.split(".")
        layer = int(parts[parts.index("layers") + 1])
        blocks.setdefault(layer, {key: float(value) for key, value in diagnostics.items()})
    if not blocks:
        raise ValueError(f"{checkpoint} has no block trajectory diagnostics; quantize with --trajectory-diagnostics")
    return blocks


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 1e-30 else None


def fmt(value: Any) -> str:
    return "" if value is None else f"{float(value):.6g}"


def write_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    layers = [row["layer"] + 1 for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(layers, [row["v1_output_nmse"] for row in rows], marker="o", label="V1")
    axes[0].plot(layers, [row["v3_output_nmse"] for row in rows], marker="o", label="V3")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("teacher-student NMSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[1].plot(layers, [row["v1_contraction"] for row in rows], marker="o", label="V1")
    axes[1].plot(layers, [row["v3_contraction"] for row in rows], marker="o", label="V3")
    axes[1].set_xlabel("block output depth")
    axes[1].set_ylabel(r"contraction $g_{l+1}/g_l$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--v3-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    v1 = load_blocks(Path(args.v1_checkpoint))
    v3 = load_blocks(Path(args.v3_checkpoint))
    if set(v1) != set(v3):
        raise ValueError(f"V1/V3 layer sets differ: {sorted(v1)} vs {sorted(v3)}")
    rows: list[dict[str, Any]] = []
    for layer in sorted(v1):
        v1_in, v1_out = v1[layer]["input_nmse"], v1[layer]["output_nmse"]
        v3_in, v3_out = v3[layer]["input_nmse"], v3[layer]["output_nmse"]
        rows.append(
            {
                "layer": layer,
                "v1_input_nmse": v1_in,
                "v1_output_nmse": v1_out,
                "v1_contraction": safe_ratio(v1_out, v1_in),
                "v1_output_cosine": v1[layer]["output_cosine"],
                "v3_input_nmse": v3_in,
                "v3_output_nmse": v3_out,
                "v3_contraction": safe_ratio(v3_out, v3_in),
                "v3_output_cosine": v3[layer]["output_cosine"],
                "v3_gap_reduction_vs_v1": 1.0 - safe_ratio(v3_out, v1_out),
            }
        )
    valid_v1 = [row["v1_contraction"] for row in rows if row["v1_contraction"] is not None]
    valid_v3 = [row["v3_contraction"] for row in rows if row["v3_contraction"] is not None]
    summary = {
        "layers": len(rows),
        "v1_gap_auc": sum(row["v1_output_nmse"] for row in rows) / len(rows),
        "v3_gap_auc": sum(row["v3_output_nmse"] for row in rows) / len(rows),
        "gap_auc_reduction": 1.0
        - sum(row["v3_output_nmse"] for row in rows)
        / max(sum(row["v1_output_nmse"] for row in rows), 1e-30),
        "v1_final_gap": rows[-1]["v1_output_nmse"],
        "v3_final_gap": rows[-1]["v3_output_nmse"],
        "final_gap_reduction": 1.0
        - rows[-1]["v3_output_nmse"] / max(rows[-1]["v1_output_nmse"], 1e-30),
        "v1_contracting_block_rate": sum(value < 1.0 for value in valid_v1) / max(len(valid_v1), 1),
        "v3_contracting_block_rate": sum(value < 1.0 for value in valid_v3) / max(len(valid_v3), 1),
        "v3_better_gap_block_rate": sum(
            row["v3_output_nmse"] < row["v1_output_nmse"] for row in rows
        )
        / len(rows),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "trajectory_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# V1 vs V3 teacher-student trajectory",
        "",
        "A contraction below 1 means the block reduces its incoming normalized teacher-student gap.",
        "",
        f"- Gap AUC reduction: {summary['gap_auc_reduction']:.2%}",
        f"- Final gap reduction: {summary['final_gap_reduction']:.2%}",
        f"- Contracting blocks: V1 {summary['v1_contracting_block_rate']:.2%}, V3 {summary['v3_contracting_block_rate']:.2%}",
        f"- V3 has lower gap than V1 after {summary['v3_better_gap_block_rate']:.2%} of blocks",
        "",
        "| block | V1 gap | V3 gap | V1 contraction | V3 contraction | V3 gap reduction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['layer'] + 1} | {fmt(row['v1_output_nmse'])} | {fmt(row['v3_output_nmse'])} | "
            f"{fmt(row['v1_contraction'])} | {fmt(row['v3_contraction'])} | "
            f"{fmt(row['v3_gap_reduction_vs_v1'])} |"
        )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_plot(output / "trajectory_comparison.png", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
