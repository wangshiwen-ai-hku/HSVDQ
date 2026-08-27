#!/usr/bin/env python3
"""Summarize rank/lambda sweeps and reliability proxies from checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


LABEL_RE = re.compile(
    r"(?P<calib>wikitext2|c4)_w(?P<bits>\d+)a(?P<abits>\d+)_"
    r"r(?P<rank>\d+)_(?P<variant>v\d(?:v\d)?)_lam(?P<lam>[0-9p]+)_s(?P<seed>\d+)"
)
BASELINE_RE = re.compile(
    r"(?P<calib>wikitext2|c4)_w(?P<bits>\d+)a(?P<abits>\d+)_"
    r"r(?P<rank>\d+)_(?P<variant>v1)(?:_.*)?_s(?P<seed>\d+)"
)


def parse_lambda(text: str) -> float:
    return float(text.replace("p", "."))


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mean(values: Iterable[float | None]) -> float | None:
    data = [value for value in values if value is not None]
    return sum(data) / len(data) if data else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    num = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    den_x = math.sqrt(sum((x - x_mean) ** 2 for x, _ in pairs))
    den_y = math.sqrt(sum((y - y_mean) ** 2 for _, y in pairs))
    if den_x <= 1e-30 or den_y <= 1e-30:
        return None
    return num / (den_x * den_y)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_states(checkpoint: Path) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint / "hsvdquant.pt", map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint / "hsvdquant.pt", map_location="cpu")
    return payload.get("states", payload) if isinstance(payload, dict) else {}


def checkpoint_summary(checkpoint: Path, metrics_dir: Path) -> dict[str, Any] | None:
    match = LABEL_RE.match(checkpoint.name)
    baseline_match = None if match else BASELINE_RE.match(checkpoint.name)
    if baseline_match:
        match = baseline_match
    if not match:
        return None
    states = load_states(checkpoint)
    sigma_means: list[float] = []
    sigma_maxes: list[float] = []
    fa_values: list[float] = []
    fw_values: list[float] = []
    split_cosines: list[float] = []
    holdout_gains: list[float] = []
    for state in states.values():
        if not isinstance(state, dict):
            continue
        sigma_mean = finite(state.get("sigma_a_mean"))
        sigma_max = finite(state.get("sigma_a_max"))
        if sigma_mean is not None:
            sigma_means.append(sigma_mean)
        if sigma_max is not None:
            sigma_maxes.append(sigma_max)
        diagnostics = state.get("joint_diagnostics") or []
        if diagnostics:
            last = diagnostics[-1]
            fa = finite(last.get("fa"))
            fw = finite(last.get("fw"))
            if fa is not None:
                fa_values.append(fa)
            if fw is not None:
                fw_values.append(fw)
        objective = state.get("objective_diagnostics") or {}
        split = finite(objective.get("split_direction_cosine"))
        holdout = finite(objective.get("holdout_gain"))
        if split is not None:
            split_cosines.append(split)
        if holdout is not None:
            holdout_gains.append(holdout)

    metric_payloads = {}
    for metric_file in (metrics_dir / checkpoint.name).glob("ppl_*.json"):
        payload = read_json(metric_file)
        if payload:
            dataset = payload.get("metrics", {}).get("dataset") or metric_file.stem.replace("ppl_", "")
            metric_payloads[dataset] = finite(payload.get("metrics", {}).get("ppl"))

    groups = match.groupdict()
    sigma_mean_value = mean(sigma_means)
    sigma_max_value = mean(sigma_maxes)
    return {
        "checkpoint": checkpoint.name,
        "calibration": groups["calib"],
        "bits": int(groups["bits"]),
        "activation_bits": int(groups["abits"]),
        "rank": int(groups["rank"]),
        "variant": groups["variant"],
        "lambda": parse_lambda(groups["lam"]) if "lam" in groups and groups.get("lam") else 0.0,
        "seed": int(groups["seed"]),
        "ppl": metric_payloads,
        "modules": len(states),
        "sigma_a_mean": sigma_mean_value,
        "sigma_a_max": sigma_max_value,
        "sigma_a_max_over_mean": (
            sigma_max_value / max(sigma_mean_value, 1e-30)
            if sigma_mean_value is not None and sigma_max_value is not None
            else None
        ),
        "fw_mean": mean(fw_values),
        "fa_mean": mean(fa_values),
        "fa_over_fw": (
            mean(fa_values) / max(mean(fw_values) or 0.0, 1e-30)
            if mean(fa_values) is not None and mean(fw_values) is not None
            else None
        ),
        "split_direction_cosine_mean": mean(split_cosines),
        "holdout_gain_mean": mean(holdout_gains),
        "has_split_reliability": bool(split_cosines or holdout_gains),
    }


def summarize(rows: list[dict[str, Any]], target_rank: int, target_lambdas: list[float]) -> dict[str, Any]:
    rank_rows = [row for row in rows if row["rank"] == target_rank]
    missing: dict[str, list[float]] = {}
    for calibration in sorted({row["calibration"] for row in rank_rows}):
        present = {round(row["lambda"], 8) for row in rank_rows if row["calibration"] == calibration}
        missing[calibration] = [lam for lam in target_lambdas if round(lam, 8) not in present]

    best: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        for dataset, ppl in row["ppl"].items():
            if ppl is not None:
                grouped[(row["calibration"], dataset)].append({**row, "eval": dataset, "eval_ppl": ppl})
    for (calibration, dataset), group in sorted(grouped.items()):
        winner = min(group, key=lambda row: row["eval_ppl"])
        best.append(
            {
                "calibration": calibration,
                "eval": dataset,
                "best_lambda": winner["lambda"],
                "best_ppl": winner["eval_ppl"],
                "available_lambdas": sorted({row["lambda"] for row in group}),
            }
        )

    correlations: dict[str, Any] = {}
    for dataset in sorted({dataset for row in rank_rows for dataset in row["ppl"]}):
        metric_rows = [row for row in rank_rows if row["ppl"].get(dataset) is not None]
        correlations[dataset] = {
            "lambda_vs_ppl": pearson(
                [row["lambda"] for row in metric_rows],
                [row["ppl"][dataset] for row in metric_rows],
            ),
            "sigma_a_mean_vs_ppl": pearson(
                [row["sigma_a_mean"] or float("nan") for row in metric_rows],
                [row["ppl"][dataset] for row in metric_rows],
            ),
            "fa_over_fw_vs_ppl": pearson(
                [row["fa_over_fw"] or float("nan") for row in metric_rows],
                [row["ppl"][dataset] for row in metric_rows],
            ),
            "split_reliability_vs_ppl": pearson(
                [row["split_direction_cosine_mean"] or float("nan") for row in metric_rows],
                [row["ppl"][dataset] for row in metric_rows],
            ),
        }

    return {
        "target_rank": target_rank,
        "target_lambdas": target_lambdas,
        "missing_target_lambdas": missing,
        "best_by_calibration_eval": best,
        "correlations": correlations,
        "split_reliability_available": any(row["has_split_reliability"] for row in rank_rows),
    }


def write_markdown(output: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Lambda Reliability Summary",
        "",
        f"- result root: `{payload['args']['root']}`",
        f"- target rank: {summary['target_rank']}",
        f"- split reliability available: {summary['split_reliability_available']}",
        "",
        "## Best Lambda",
        "",
    ]
    for row in summary["best_by_calibration_eval"]:
        lines.append(
            f"- calib `{row['calibration']}` eval `{row['eval']}`: "
            f"best λ={row['best_lambda']}, ppl={row['best_ppl']:.4f}, "
            f"available={row['available_lambdas']}"
        )
    lines.extend(["", "## Missing Target Lambdas", ""])
    for calibration, missing in summary["missing_target_lambdas"].items():
        lines.append(f"- `{calibration}`: {missing}")
    lines.extend(["", "## Correlations", ""])
    for dataset, row in summary["correlations"].items():
        lines.append(
            f"- `{dataset}`: lambda_vs_ppl={row['lambda_vs_ppl']}, "
            f"sigma_a_mean_vs_ppl={row['sigma_a_mean_vs_ppl']}, "
            f"fa_over_fw_vs_ppl={row['fa_over_fw_vs_ppl']}, "
            f"split_reliability_vs_ppl={row['split_reliability_vs_ppl']}"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/v2_lambda_grid_w4a4_g128")
    parser.add_argument("--extra-root", action="append", default=[], help="additional result roots with checkpoints/metrics")
    parser.add_argument("--output", default="results/structured_correction/lambda_reliability")
    parser.add_argument("--target-rank", type=int, default=4)
    parser.add_argument("--target-lambdas", type=float, nargs="*", default=[0.0, 0.0625, 0.125, 0.25, 0.5])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    roots = [Path(args.root), *[Path(root) for root in args.extra_root]]
    rows = []
    for root in roots:
        for checkpoint in sorted((root / "checkpoints").glob("*")):
            if not checkpoint.is_dir():
                continue
            row = checkpoint_summary(checkpoint, root / "metrics")
            if row is not None:
                row["root"] = str(root)
                rows.append(row)
    payload = {
        "args": vars(args),
        "rows": rows,
        "summary": summarize(rows, args.target_rank, args.target_lambdas),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "lambda_reliability.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(output / "lambda_reliability.md", payload)
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
