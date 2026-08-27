#!/usr/bin/env python3
"""Extract per-module smooth-D stats and quantization MSE from hsvdquant checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

LAYER_RE = re.compile(r"layers\.(?P<layer>\d+)\.")
LABEL_RE = re.compile(
    r"^(?P<calibration>wikitext2|c4)_w(?P<bits>\d+)a4_r(?P<rank>\d+)_"
    r"(?P<variant>v\d+)_(?P<block_input>quantized|reference)_"
    r"(?P<intra>sequential|fp_independent)_o(?P<outer>\d+)_s(?P<seed>\d+)$"
)
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
D_CLIP = 16.0
EPS = 1e-12


def module_kind(name: str) -> str:
    for kind in KINDS:
        if name.endswith(kind):
            return kind
    return name.rsplit(".", 1)[-1]


def d_stats(d: torch.Tensor) -> dict[str, float]:
    values = d.detach().float().flatten()
    log_d = values.clamp_min(EPS).log()
    clip_hi = (values >= D_CLIP * 0.99).float().mean().item()
    clip_lo = (values <= (1.0 / D_CLIP) * 1.01).float().mean().item()
    return {
        "d_mean": float(values.mean().item()),
        "d_std": float(values.std(unbiased=False).item()),
        "d_min": float(values.min().item()),
        "d_max": float(values.max().item()),
        "d_p10": float(values.quantile(0.10).item()),
        "d_p50": float(values.quantile(0.50).item()),
        "d_p90": float(values.quantile(0.90).item()),
        "d_log_std": float(log_d.std(unbiased=False).item()),
        "d_dyn_range": float((values.max() / values.clamp_min(EPS).min()).item()),
        "d_clip_hi_frac": float(clip_hi),
        "d_clip_lo_frac": float(clip_lo),
        "d_clip_frac": float(clip_hi + clip_lo),
    }


def last_joint(state: dict[str, Any]) -> dict[str, float]:
    history = state.get("joint_diagnostics") or []
    if not history:
        return {}
    last = history[-1]
    out: dict[str, float] = {}
    for key in ("fw", "fa", "weighted_fa", "joint", "state_mse", "target_norm_ratio"):
        value = last.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def parse_label(name: str) -> dict[str, Any] | None:
    match = LABEL_RE.match(name)
    if not match:
        return None
    data = match.groupdict()
    data["rank"] = int(data["rank"])
    data["outer"] = int(data["outer"])
    data["bits"] = int(data["bits"])
    data["seed"] = int(data["seed"])
    data["lambda"] = 0.25 if data["variant"] == "v2" else 0.0
    return data


def summarize_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row)

    summaries = []
    numeric = [
        "mse",
        "sigma_a_mean",
        "sigma_a_max",
        "d_mean",
        "d_std",
        "d_min",
        "d_max",
        "d_p10",
        "d_p50",
        "d_p90",
        "d_log_std",
        "d_dyn_range",
        "d_clip_frac",
        "fw",
        "fa",
        "weighted_fa",
        "joint",
        "target_norm_ratio",
    ]
    for key, group in sorted(buckets.items()):
        item = dict(zip(keys, key))
        item["n"] = len(group)
        for field in numeric:
            values = [float(r[field]) for r in group if r.get(field) is not None]
            if not values:
                continue
            values.sort()
            item[f"{field}_mean"] = sum(values) / len(values)
            item[f"{field}_median"] = values[len(values) // 2]
            item[f"{field}_min"] = values[0]
            item[f"{field}_max"] = values[-1]
        summaries.append(item)
    return summaries


def layer_series(rows: list[dict[str, Any]], pred) -> dict[str, list[float]]:
    selected = [r for r in rows if pred(r)]
    layers = sorted({r["layer"] for r in selected})
    series: dict[str, list[float]] = {"layer": [float(i) for i in layers]}
    for kind in KINDS:
        by_layer = {r["layer"]: r for r in selected if r["kind"] == kind}
        series[f"{kind}_mse"] = [
            math.log10(max(by_layer[i]["mse"], EPS)) if i in by_layer else float("nan")
            for i in layers
        ]
        series[f"{kind}_d_log_std"] = [
            by_layer[i]["d_log_std"] if i in by_layer else float("nan") for i in layers
        ]
        series[f"{kind}_d_dyn"] = [
            math.log10(max(by_layer[i]["d_dyn_range"], 1.0)) if i in by_layer else float("nan")
            for i in layers
        ]
        series[f"{kind}_sigma"] = [
            math.log10(max(by_layer[i]["sigma_a_mean"], EPS)) if i in by_layer else float("nan")
            for i in layers
        ]
        series[f"{kind}_clip"] = [
            by_layer[i]["d_clip_frac"] if i in by_layer else float("nan") for i in layers
        ]
    return series


def extract_checkpoint(path: Path) -> list[dict[str, Any]]:
    meta = parse_label(path.parent.name)
    if meta is None:
        return []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("states", payload) if isinstance(payload, dict) else {}
    rows: list[dict[str, Any]] = []
    for name, state in states.items():
        if not isinstance(state, dict) or "d" not in state:
            continue
        layer_match = LAYER_RE.search(name)
        if layer_match is None:
            continue
        stats = d_stats(state["d"])
        joint = last_joint(state)
        mse = state.get("error")
        if not isinstance(mse, (int, float)) or not math.isfinite(float(mse)):
            continue
        rows.append(
            {
                **meta,
                "checkpoint": path.parent.name,
                "module": name,
                "layer": int(layer_match.group("layer")),
                "kind": module_kind(name),
                "mse": float(mse),
                "sigma_a_mean": float(state.get("sigma_a_mean") or 0.0),
                "sigma_a_max": float(state.get("sigma_a_max") or 0.0),
                "in_features": int(state.get("in_features") or 0),
                "out_features": int(state.get("out_features") or 0),
                **stats,
                **joint,
            }
        )
    del payload, states
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/propagation_grid_w4a4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--select",
        nargs="*",
        default=None,
        help="Substring filters; if omitted, load a focused representative subset.",
    )
    args = parser.parse_args()

    default_needles = [
        "r4_v1_quantized_sequential_o2",
        "r8_v1_quantized_sequential_o2",
        "r16_v1_quantized_sequential_o2",
        "r4_v2_quantized_sequential_o2",
        "r8_v2_quantized_sequential_o2",
        "r16_v2_quantized_sequential_o2",
        "r8_v3_quantized_sequential_o2",
        "r8_v1_reference_sequential_o2",
        "r8_v2_reference_sequential_o2",
        "r8_v1_quantized_fp_independent_o2",
        "r8_v2_quantized_fp_independent_o2",
        "r16_v2_quantized_sequential_o5",
        "r8_v2_quantized_sequential_o5",
        "r4_v2_quantized_sequential_o5",
    ]
    needles = args.select or default_needles
    checkpoints = sorted((args.root / "checkpoints").glob("*/hsvdquant.pt"))
    chosen = []
    for ckpt in checkpoints:
        if any(n in ckpt.parent.name for n in needles):
            chosen.append(ckpt)

    rows: list[dict[str, Any]] = []
    for ckpt in chosen:
        print(f"loading {ckpt.parent.name}", flush=True)
        rows.extend(extract_checkpoint(ckpt))

    def pred_r8_v1_qseq(r):
        return (
            r["rank"] == 8
            and r["variant"] == "v1"
            and r["block_input"] == "quantized"
            and r["intra"] == "sequential"
            and r["outer"] == 2
        )

    def pred_r8_v2_qseq(r):
        return (
            r["rank"] == 8
            and r["variant"] == "v2"
            and r["block_input"] == "quantized"
            and r["intra"] == "sequential"
            and r["outer"] == 2
        )

    def pred_r8_v1_ref(r):
        return (
            r["rank"] == 8
            and r["variant"] == "v1"
            and r["block_input"] == "reference"
            and r["intra"] == "sequential"
            and r["outer"] == 2
        )

    def pred_r8_v2_ref(r):
        return (
            r["rank"] == 8
            and r["variant"] == "v2"
            and r["block_input"] == "reference"
            and r["intra"] == "sequential"
            and r["outer"] == 2
        )

    payload = {
        "n_rows": len(rows),
        "checkpoints": sorted({r["checkpoint"] for r in rows}),
        "by_kind_setting": summarize_rows(
            rows,
            ("variant", "rank", "block_input", "intra", "outer", "lambda", "kind"),
        ),
        "by_setting": summarize_rows(
            rows,
            ("variant", "rank", "block_input", "intra", "outer", "lambda"),
        ),
        "layer_r8_v1_quantized_seq_o2": layer_series(rows, pred_r8_v1_qseq),
        "layer_r8_v2_quantized_seq_o2": layer_series(rows, pred_r8_v2_qseq),
        "layer_r8_v1_reference_seq_o2": layer_series(rows, pred_r8_v1_ref),
        "layer_r8_v2_reference_seq_o2": layer_series(rows, pred_r8_v2_ref),
    }

    # rank x kind for v1/v2 quantized sequential o2
    rank_table = []
    for variant in ("v1", "v2"):
        for rank in (4, 8, 16):
            for kind in KINDS:
                group = [
                    r
                    for r in rows
                    if r["variant"] == variant
                    and r["rank"] == rank
                    and r["kind"] == kind
                    and r["block_input"] == "quantized"
                    and r["intra"] == "sequential"
                    and r["outer"] == 2
                ]
                if not group:
                    continue
                mses = sorted(r["mse"] for r in group)
                dlogs = [r["d_log_std"] for r in group]
                clips = [r["d_clip_frac"] for r in group]
                sigmas = [r["sigma_a_mean"] for r in group]
                rank_table.append(
                    {
                        "variant": variant,
                        "rank": rank,
                        "kind": kind,
                        "mse_median": mses[len(mses) // 2],
                        "mse_mean": sum(mses) / len(mses),
                        "d_log_std_mean": sum(dlogs) / len(dlogs),
                        "d_clip_mean": sum(clips) / len(clips),
                        "sigma_a_mean": sum(sigmas) / len(sigmas),
                    }
                )
    payload["rank_by_kind"] = rank_table

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output} rows={len(rows)} ckpts={len(payload['checkpoints'])}")


if __name__ == "__main__":
    main()
