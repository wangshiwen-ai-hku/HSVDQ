#!/usr/bin/env python3
"""Per-layer activation-error diagnostic for H-SVDQuant (W4A4).

Reuses the real quantization walk: for every linear module we build the true
H-SVDQuant state, then measure how the layer-output error splits between the
weight channel F_W and the activation channel F_A (Lemma 1 of hsvdquant.tex).
No pipeline behaviour is changed; modules are replaced exactly as in the real
run so downstream layers see quantized inputs.

Outputs, under --output:
  per_module.json   one record per linear module
  by_type.json      aggregates grouped by projection type
  by_depth.json     aggregates grouped by layer index
  scatter_fa.png    surrogate F_A vs measured F_A (the "falsify" plot)
  phi_concentration.png   channel-mass concentration by module type
  g128_gain.png     measured F_A reduction from per-group activation quant
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from common import (
    REPO_ROOT,
    _dtype_from_name,
    _get_submodule,
    _load_model,
    _qwen_sequential_groups,
    _set_submodule,
    advance_hidden_batches,
    collect_layer_stats,
    environment_metadata,
    make_calibration,
    set_reproducible,
    write_json,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import (  # noqa: E402
    HSVQuantLinear,
    QuantConfig,
    capture_first_layer_inputs,
    joint_quantize_linear,
)


def _module_type(name: str) -> str:
    return name.split(".")[-1]


def _quantize_activation_grouped(inputs: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Per-token, per-group symmetric uniform activation quantization."""
    if bits >= 16:
        return inputs
    qmax = float(2 ** (bits - 1) - 1)
    c = inputs.shape[-1]
    if group <= 0 or group >= c:
        scale = inputs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        return torch.round(inputs / scale).clamp(-qmax, qmax) * scale
    lead = inputs.shape[:-1]
    ng = (c + group - 1) // group
    pad = ng * group - c
    x = inputs
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    x = x.reshape(*lead, ng, group)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax) * scale
    q = q.reshape(*lead, ng * group)
    if pad:
        q = q[..., :c]
    return q


@torch.no_grad()
def _analyze_module(
    module: torch.nn.Linear,
    hessian: torch.Tensor,
    cached_x: torch.Tensor | None,
    config: QuantConfig,
    group_sizes: list[int],
) -> tuple[HSVQuantLinear, dict[str, Any]]:
    """Build the true H-SVDQuant state and measure the F_W / F_A split.

    Returns the replacement module (so the walk can swap it in) and a metrics
    record. All errors are output-space MSE on the cached activation reservoir,
    which is exactly the object F = ||XW - Yhat||^2 / (m*n) of Lemma 1.
    """
    device = module.weight.device
    dtype = module.weight.dtype
    state = joint_quantize_linear(module, hessian, cached_x, config)
    replacement = HSVQuantLinear(state, compute_dtype=dtype).to(device)

    record: dict[str, Any] = {
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "rank": int(state["l1"].shape[1]),
        "state_error": float(state["error"]),
    }
    if cached_x is None:
        record["note"] = "no cached activations"
        return replacement, record

    x = cached_x.to(device=device, dtype=torch.float32)
    m, n = x.shape[0], module.out_features
    d = state["d"].to(device=device, dtype=torch.float32)
    l1 = state["l1"].to(device=device, dtype=torch.float32)
    l2 = state["l2"].to(device=device, dtype=torch.float32)
    from hsvdquant import _dequantize_codes

    qweight = _dequantize_codes(
        state["codes"].to(device), state["scales"].to(device=device, dtype=torch.float32), config.group_size
    )  # [out, in] dequantized residual R_hat (F.linear layout)
    rq = qweight.t()  # [in, out] for X @ R_hat
    smoothed = x / d  # Xtilde
    target = x.to(torch.float32) @ module.weight.detach().t().float()  # XW, exact
    branch = (smoothed @ l1) @ l2 if l1.shape[1] else torch.zeros_like(target)

    def out_mse(pred: torch.Tensor) -> float:
        return float((pred - target).square().mean().item())

    # A16: activation quantization OFF -> isolates the weight channel F_W.
    pred_a16 = branch + smoothed @ rq
    fw = out_mse(pred_a16)

    # A4 with the current (global-max, per-token) activation quantizer.
    from hsvdquant import _quantize_activation

    qa_global = _quantize_activation(smoothed, config.activation_bits)
    pred_a4 = branch + qa_global @ rq
    total = out_mse(pred_a4)
    # F_A measured as the increment from turning activation quant on.
    fa_global = max(total - fw, 0.0)

    # Surrogate F_A from Eq. (FAdyn): (sum_t max_i Xti^2/di^2)(sum_i di^2 ||P_i||^2)/(12 kappa^2 m n)
    residual_unsmoothed = module.weight.detach().T.float() - (l1 @ l2) / d[:, None]
    kappa = float(2 ** (config.activation_bits - 1) - 1)
    token_step2 = (x.square() / d.square()).amax(dim=1).sum()
    energy = (d.square() * residual_unsmoothed.square().sum(dim=1)).sum()
    fa_surrogate = float((token_step2 * energy / (12.0 * kappa * kappa) / (m * n)).item())

    # Per-channel contribution phi_i = a_i^2 ||P_i,:||^2 (Prop.1 static form).
    a = x.abs().amax(dim=0)  # max_t |X_ti|, unsmoothed
    p_rows = residual_unsmoothed  # [in, out]; P = W - B in unsmoothed coords
    phi = (a.square() * p_rows.square().sum(dim=1)).cpu()
    phi_sorted, _ = torch.sort(phi, descending=True)
    phi_total = float(phi_sorted.sum().clamp_min(1e-30))
    top_mass = {
        f"top{k}": float(phi_sorted[:k].sum().item() / phi_total)
        for k in (1, 4, 8, 16, 32, 64)
        if k <= phi.numel()
    }

    # Lever A counterfactual: per-group activation quant, measured (not modeled).
    g_gain = {}
    for g in group_sizes:
        qa_g = _quantize_activation_grouped(smoothed, config.activation_bits, g)
        fa_g = max(out_mse(branch + qa_g @ rq) - fw, 0.0)
        g_gain[f"g{g}"] = fa_g

    record.update(
        {
            "fw": fw,
            "fa_global": fa_global,
            "total": total,
            "fa_surrogate": fa_surrogate,
            "fa_fraction": float(fa_global / max(total, 1e-30)),
            "phi_top_mass": top_mass,
            "fa_grouped": g_gain,
        }
    )
    return replacement, record


def run_diagnostic(args: argparse.Namespace) -> list[dict[str, Any]]:
    set_reproducible(args.seed)
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    tokenizer, batches = make_calibration(
        args.model, args.calib_dataset, args.nsamples, args.sequence_length, args.calib_batch_size, args.seed
    )
    del tokenizer
    model = _load_model(args.model, device, dtype)
    hidden_batches, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    del batches
    config = QuantConfig(
        bits=args.bits,
        activation_bits=args.activation_bits,
        rank=args.rank,
        beta=args.beta,
        p=args.p,
        group_size=args.group_size,
        outer_iters=args.outer_iters,
        d_mode=args.d_mode,
        d_steps=args.d_steps,
        damp=args.damp,
    )
    group_sizes = [int(g) for g in args.act_group_sizes]
    records: list[dict[str, Any]] = []
    layer_count = len(model.model.layers) if args.max_layers < 0 else min(args.max_layers, len(model.model.layers))
    for layer_index in range(layer_count):
        layer = model.model.layers[layer_index]
        print(f"[diagnose] layer {layer_index + 1}/{layer_count}", flush=True)
        stats_by_name = collect_layer_stats(
            model, hidden_batches, layer_kwargs, layer_index, device,
            args.activation_cache_tokens, args.hessian_block_size, args.seed,
        )
        for group in _qwen_sequential_groups(layer):
            for name in group:
                module = _get_submodule(layer, name)
                hessian, cached_x = stats_by_name[name].finalize()
                replacement, record = _analyze_module(module, hessian, cached_x, config, group_sizes)
                record["module"] = f"model.layers.{layer_index}.{name}"
                record["layer"] = layer_index
                record["type"] = _module_type(name)
                records.append(record)
                _set_submodule(layer, name, replacement)
                stats_by_name[name].free()
                fa = record.get("fa_global", float("nan"))
                print(f"  {name}: fw={record.get('fw', float('nan')):.3e} "
                      f"fa={fa:.3e} frac={record.get('fa_fraction', float('nan')):.2f}", flush=True)
        hidden_batches = advance_hidden_batches(model, hidden_batches, layer_kwargs, layer_index, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if x == x]  # drop NaN
    return float(sum(xs) / len(xs)) if xs else float("nan")


def aggregate(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if "fa_global" not in r:
            continue
        by_type[r["type"]].append(r)
        by_depth[r["layer"]].append(r)

    def summarize(rs: list[dict[str, Any]]) -> dict[str, Any]:
        out = {
            "count": len(rs),
            "fw_mean": _mean([r["fw"] for r in rs]),
            "fa_mean": _mean([r["fa_global"] for r in rs]),
            "fa_fraction_mean": _mean([r["fa_fraction"] for r in rs]),
            "fa_total_share": float(sum(r["fa_global"] for r in rs)),
            "phi_top8_mean": _mean([r["phi_top_mass"].get("top8", float("nan")) for r in rs]),
            "phi_top32_mean": _mean([r["phi_top_mass"].get("top32", float("nan")) for r in rs]),
        }
        if rs and rs[0].get("fa_grouped"):
            for g in rs[0]["fa_grouped"]:
                out[f"fa_{g}_mean"] = _mean([r["fa_grouped"][g] for r in rs])
        return out

    type_summary = {t: summarize(rs) for t, rs in sorted(by_type.items())}
    depth_summary = {str(d): summarize(rs) for d, rs in sorted(by_depth.items())}
    return type_summary, depth_summary


def make_plots(records: list[dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[diagnose] matplotlib unavailable, skipping plots: {exc}", flush=True)
        return
    rs = [r for r in records if "fa_global" in r]
    types = sorted({r["type"] for r in rs})
    cmap = {t: c for t, c in zip(types, plt.cm.tab10.colors)}

    # 1) surrogate vs measured F_A (the falsify plot)
    fig, ax = plt.subplots(figsize=(5, 5))
    for t in types:
        xs = [r["fa_surrogate"] for r in rs if r["type"] == t]
        ys = [r["fa_global"] for r in rs if r["type"] == t]
        ax.scatter(xs, ys, s=18, alpha=0.7, label=t, color=cmap[t])
    both = [v for r in rs for v in (r["fa_surrogate"], r["fa_global"]) if v > 0]
    if both:
        lo, hi = min(both), max(both)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("surrogate F_A  (Eq. FAdyn)")
    ax.set_ylabel("measured F_A  (A4 - A16 output MSE)")
    ax.set_title("Activation-error surrogate vs measured")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "scatter_fa.png", dpi=130); plt.close(fig)

    # 2) phi concentration: fraction of F_A mass in top-k channels, by type
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = [1, 4, 8, 16, 32, 64]
    for t in types:
        rows = [r for r in rs if r["type"] == t and r.get("phi_top_mass")]
        if not rows:
            continue
        ys = [_mean([r["phi_top_mass"].get(f"top{k}", float("nan")) for r in rows]) for k in ks]
        ax.plot(ks, ys, marker="o", label=t, color=cmap[t])
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels(ks)
    ax.set_xlabel("k (top channels kept)")
    ax.set_ylabel("fraction of F_A channel mass")
    ax.set_title("Channel-mass concentration (Lever C sizing)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "phi_concentration.png", dpi=130); plt.close(fig)

    # 3) g128 gain: measured F_A vs group size, by type
    if rs and rs[0].get("fa_grouped"):
        gkeys = list(rs[0]["fa_grouped"].keys())
        gvals = [int(k[1:]) for k in gkeys]
        fig, ax = plt.subplots(figsize=(6, 4))
        for t in types:
            rows = [r for r in rs if r["type"] == t and r.get("fa_grouped")]
            if not rows:
                continue
            base = _mean([r["fa_global"] for r in rows])
            ys = [_mean([r["fa_grouped"][k] for r in rows]) / max(base, 1e-30) for k in gkeys]
            ax.plot(gvals, ys, marker="s", label=t, color=cmap[t])
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("activation group size")
        ax.set_ylabel("F_A(group) / F_A(global)")
        ax.set_title("Per-group activation quant gain (Lever A)")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "g128_gain.png", dpi=130); plt.close(fig)
    print(f"[diagnose] plots written to {out}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--calib-dataset", choices=["wikitext2", "c4", "synthetic"], default="wikitext2")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--calib-batch-size", type=int, default=4)
    p.add_argument("--activation-cache-tokens", type=int, default=2048)
    p.add_argument("--hessian-block-size", type=int, default=4096)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--activation-bits", type=int, default=4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--p", type=float, default=2.0)
    p.add_argument("--outer-iters", type=int, default=2)
    p.add_argument("--d-mode", default="cached")
    p.add_argument("--d-steps", type=int, default=20)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--max-layers", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--act-group-sizes", nargs="+", default=[32, 64, 128, 256])
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = run_diagnostic(args)
    write_json(out / "per_module.json", {"records": records, "config": vars(args)})
    type_summary, depth_summary = aggregate(records)
    write_json(out / "by_type.json", type_summary)
    write_json(out / "by_depth.json", depth_summary)
    make_plots(records, out)

    total_fa = sum(r.get("fa_global", 0.0) for r in records)
    total_fw = sum(r.get("fw", 0.0) for r in records)
    print("\n===== summary =====", flush=True)
    print(f"sum F_W = {total_fw:.4e}   sum F_A = {total_fa:.4e}   "
          f"F_A share = {total_fa / max(total_fa + total_fw, 1e-30):.1%}", flush=True)
    print("F_A by type (measured, descending):", flush=True)
    for t, s in sorted(type_summary.items(), key=lambda kv: -kv[1]["fa_total_share"]):
        line = (f"  {t:14s} fa_mean={s['fa_mean']:.3e} frac={s['fa_fraction_mean']:.2f} "
                f"phi_top8={s['phi_top8_mean']:.2f}")
        if "fa_g128_mean" in s:
            line += f" g128/global={s['fa_g128_mean'] / max(s['fa_mean'], 1e-30):.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()



