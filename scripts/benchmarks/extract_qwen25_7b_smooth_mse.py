#!/usr/bin/env python3
"""Extract D / MSE summaries from Qwen2.5-7B v2 checkpoints (large .pt)."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
LAYER_RE = re.compile(r"layers\.(?P<layer>\d+)\.")
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
    return {
        "d_mean": float(values.mean()),
        "d_std": float(values.std(unbiased=False)),
        "d_min": float(values.min()),
        "d_max": float(values.max()),
        "d_p50": float(values.quantile(0.5)),
        "d_log_std": float(log_d.std(unbiased=False)),
        "d_dyn_range": float(values.max() / values.clamp_min(EPS).min()),
        "d_clip_frac": float(
            ((values >= D_CLIP * 0.99) | (values <= (1.0 / D_CLIP) * 1.01)).float().mean()
        ),
    }


def extract(path: Path, rank: int) -> dict[str, Any]:
    print(f"loading {path} ...", flush=True)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    rows: list[dict[str, Any]] = []
    for name, state in states.items():
        if not isinstance(state, dict) or "d" not in state:
            continue
        m = LAYER_RE.search(name)
        if m is None:
            continue
        mse = state.get("error")
        if not isinstance(mse, (int, float)) or not math.isfinite(float(mse)):
            continue
        kind = module_kind(name)
        stats = d_stats(state["d"])
        joint = {}
        hist = state.get("joint_diagnostics") or []
        if hist:
            last = hist[-1]
            for key in ("fw", "fa", "weighted_fa", "joint"):
                if isinstance(last.get(key), (int, float)):
                    joint[key] = float(last[key])
        rows.append(
            {
                "rank": rank,
                "layer": int(m.group("layer")),
                "kind": kind,
                "module": name,
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
    print(f"  extracted {len(rows)} modules", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)

    kind_summary = []
    for kind in KINDS:
        group = by_kind[kind]
        if not group:
            continue
        mses = sorted(r["mse"] for r in group)
        dlogs = [r["d_log_std"] for r in group]
        dyns = [r["d_dyn_range"] for r in group]
        sigs = [r["sigma_a_mean"] for r in group]
        kind_summary.append(
            {
                "kind": kind,
                "n": len(group),
                "mse_median": mses[len(mses) // 2],
                "mse_mean": sum(mses) / len(mses),
                "mse_min": mses[0],
                "mse_max": mses[-1],
                "d_log_std_mean": sum(dlogs) / len(dlogs),
                "d_dyn_mean": sum(dyns) / len(dyns),
                "sigma_a_mean": sum(sigs) / len(sigs),
                "in_features": group[0]["in_features"],
                "out_features": group[0]["out_features"],
            }
        )

    layers = sorted({r["layer"] for r in rows})
    layer_series: dict[str, list] = {"layer": layers}
    for kind in KINDS:
        by_l = {r["layer"]: r for r in rows if r["kind"] == kind}
        layer_series[f"{kind}_mse"] = [by_l[i]["mse"] if i in by_l else float("nan") for i in layers]
        layer_series[f"{kind}_mse_log10"] = [
            math.log10(max(by_l[i]["mse"], EPS)) if i in by_l else float("nan") for i in layers
        ]
        layer_series[f"{kind}_d_log_std"] = [
            by_l[i]["d_log_std"] if i in by_l else float("nan") for i in layers
        ]
        layer_series[f"{kind}_d_dyn"] = [
            by_l[i]["d_dyn_range"] if i in by_l else float("nan") for i in layers
        ]
        layer_series[f"{kind}_sigma"] = [
            by_l[i]["sigma_a_mean"] if i in by_l else float("nan") for i in layers
        ]

    # Q vs V D disagreement per layer
    qv = []
    for layer in layers:
        q = next(r for r in rows if r["layer"] == layer and r["kind"] == "q_proj")
        # need full d vectors for cos — skip if not stored; use dyn/log_std proxy
        qv.append(
            {
                "layer": layer,
                "d_log_q": q["d_log_std"],
                "mse_q": q["mse"],
            }
        )
    return {"by_kind": kind_summary, "layer_series": layer_series, "n_rows": len(rows)}


def main() -> None:
    out_dir = ROOT / "results/qwen25_7b_v2/summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": "Qwen2.5-7B", "variant": "v2", "ranks": {}}

    for rank in (4, 8):
        ckpt = (
            ROOT
            / f"results/qwen25_7b_v2/checkpoints/wikitext2_w4a4_r{rank}_v2_s0/hsvdquant.pt"
        )
        rows = extract(ckpt, rank)
        # also dump lightweight per-row csv-like json without full d
        slim_path = out_dir / f"module_rows_r{rank}.json"
        slim_path.write_text(json.dumps(rows))
        payload["ranks"][str(rank)] = summarize(rows)

        # keep D tensors only for r8 focus layers for later act analysis
        if rank == 8:
            print("reloading r8 for focus-layer D vectors...", flush=True)
            states = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(states, dict) and "states" in states:
                states = states["states"]
            focus = {}
            for L in (0, 7, 14, 21, 27):
                focus[str(L)] = {
                    kind: states[f"model.layers.{L}.self_attn.{kind}"]["d"].float().tolist()
                    if kind in ("q_proj", "k_proj", "v_proj")
                    else states[f"model.layers.{L}.mlp.{kind}"]["d"].float().tolist()
                    for kind in ("q_proj", "k_proj", "v_proj", "gate_proj", "down_proj")
                    if (
                        f"model.layers.{L}.self_attn.{kind}" in states
                        or f"model.layers.{L}.mlp.{kind}" in states
                    )
                }
                # fix keys
                packed = {}
                for kind in ("q_proj", "k_proj", "v_proj"):
                    packed[kind] = states[f"model.layers.{L}.self_attn.{kind}"]["d"].float().tolist()
                for kind in ("gate_proj", "down_proj"):
                    packed[kind] = states[f"model.layers.{L}.mlp.{kind}"]["d"].float().tolist()
                focus[str(L)] = packed
            (out_dir / "r8_focus_d.json").write_text(json.dumps(focus))
            del states

    # rank sensitivity table
    sens = []
    for kind in KINDS:
        a = next(x for x in payload["ranks"]["4"]["by_kind"] if x["kind"] == kind)
        b = next(x for x in payload["ranks"]["8"]["by_kind"] if x["kind"] == kind)
        sens.append(
            {
                "kind": kind,
                "mse_med_r4": a["mse_median"],
                "mse_med_r8": b["mse_median"],
                "drop_r4_to_r8": 1.0 - b["mse_median"] / a["mse_median"],
                "d_log_r4": a["d_log_std_mean"],
                "d_log_r8": b["d_log_std_mean"],
                "d_dyn_r4": a["d_dyn_mean"],
                "d_dyn_r8": b["d_dyn_mean"],
            }
        )
    payload["rank_sensitivity"] = sens

    # PPL table
    metrics_dir = ROOT / "results/qwen25_7b_v2/metrics"
    ppl = {}
    for name in ("fp_baseline", "r4", "r8"):
        for ds in ("wikitext2", "c4"):
            path = metrics_dir / f"{name}_ppl_{ds}.json"
            if path.exists():
                ppl[f"{name}_{ds}"] = json.loads(path.read_text())["metrics"]["ppl"]
    payload["ppl"] = ppl

    out = out_dir / "smooth_mse_summary.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)
    print("\n=== rank sensitivity ===")
    for s in sens:
        print(
            f"{s['kind']:10} r4={s['mse_med_r4']:.4e} r8={s['mse_med_r8']:.4e} "
            f"drop={100*s['drop_r4_to_r8']:.1f}% dlog={s['d_log_r4']:.3f}->{s['d_log_r8']:.3f} "
            f"dyn={s['d_dyn_r4']:.1f}->{s['d_dyn_r8']:.1f}"
        )
    print("\n=== PPL ===")
    for k, v in ppl.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
