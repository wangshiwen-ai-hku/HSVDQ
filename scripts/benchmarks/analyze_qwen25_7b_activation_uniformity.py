#!/usr/bin/env python3
"""Activation uniformity / relative A-error for Qwen2.5-7B v2 r8."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hsvdquant import (  # noqa: E402
    _decoder_hidden,
    _dtype_from_name,
    _load_model,
    _make_calibration_batches,
    _move_tree,
    _quantize_activation,
    capture_first_layer_inputs,
)


def gini(values: torch.Tensor) -> float:
    v = values.float().flatten().clamp_min(0)
    if v.numel() == 0 or float(v.sum()) <= 0:
        return float("nan")
    v = torch.sort(v).values
    n = v.numel()
    idx = torch.arange(1, n + 1, dtype=v.dtype)
    return float((2.0 * (idx * v).sum() / (n * v.sum()) - (n + 1) / n).item())


def pearson(xs: list[float], ys: list[float]) -> float:
    a = torch.tensor(xs, dtype=torch.float64)
    b = torch.tensor(ys, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    if float(den) < 1e-20:
        return float("nan")
    return float((a * b).sum() / den)


@torch.no_grad()
def a_quant_mse(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> dict[str, float]:
    xf = x.float()
    q = _quantize_activation(xf, bits, group_size)
    err = (q - xf).square().mean()
    energy = xf.square().mean().clamp_min(1e-20)
    return {
        "a_mse": float(err),
        "a_rel_mse": float(err / energy),
        "x_energy": float(energy),
        "x_rms": float(energy.sqrt()),
    }


@torch.no_grad()
def asymmetric_a_quant_mse(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> dict[str, float]:
    xf = x.float()
    qmax = float(2 ** (bits - 1) - 1)
    qmin = -qmax
    columns = xf.shape[-1]
    lead = xf.shape[:-1]
    num_groups = (columns + group_size - 1) // group_size
    pad = num_groups * group_size - columns
    padded = xf if not pad else F.pad(xf, (0, pad))
    grouped = padded.reshape(*lead, num_groups, group_size)
    xmin = grouped.amin(dim=-1, keepdim=True)
    xmax = grouped.amax(dim=-1, keepdim=True)
    scale = ((xmax - xmin) / (qmax - qmin)).clamp_min(1e-8)
    zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
    q = (torch.round(grouped / scale) + zp).clamp(qmin, qmax)
    deq = (q - zp) * scale
    deq = deq.reshape(*lead, num_groups * group_size)
    if pad:
        deq = deq[..., :columns]
    err = (deq - xf).square().mean()
    energy = xf.square().mean().clamp_min(1e-20)
    return {"a_mse_affine": float(err), "a_rel_mse_affine": float(err / energy)}


def uniformity(x: torch.Tensor) -> dict[str, float]:
    xf = x.float()
    rms = xf.pow(2).mean(-1).sqrt().clamp_min(1e-12)
    amax = xf.abs().amax(-1)
    ch_rms = xf.pow(2).mean(0).sqrt()
    ch_amax = xf.abs().amax(0)
    return {
        "token_max_over_rms_mean": float((amax / rms).mean()),
        "ch_energy_gini": gini(ch_rms.square()),
        "ch_max_over_rms_mean": float((ch_amax / ch_rms.clamp_min(1e-12)).mean()),
        "ch_rms_dyn": float(ch_rms.max() / ch_rms.clamp_min(1e-12).min()),
        "asym_mean_over_std": float(xf.mean() / xf.std(unbiased=False).clamp_min(1e-12)),
    }


def load_d_map(ckpt: Path) -> dict[str, torch.Tensor]:
    print(f"loading D from {ckpt} ...", flush=True)
    states = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(states, dict) and "states" in states:
        states = states["states"]
    d_map = {}
    mse_map = {}
    for name, state in states.items():
        if isinstance(state, dict) and "d" in state:
            d_map[name] = state["d"].float().clone()
            mse_map[name] = float(state.get("error") or 0.0)
    del states
    print(f"  kept {len(d_map)} D vectors", flush=True)
    return d_map, mse_map


def main() -> None:
    out_dir = ROOT / "results/qwen25_7b_v2/summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    progress = out_dir / "activation_uniformity.log"
    ckpt = ROOT / "results/qwen25_7b_v2/checkpoints/wikitext2_w4a4_r8_v2_s0/hsvdquant.pt"
    model_path = ROOT / "models/Qwen/Qwen2.5-7B"

    # Prefer free L40: try cuda:0 then cuda:1
    if torch.cuda.is_available():
        # pick GPU with most free memory
        best = 0
        best_free = -1
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            if free > best_free:
                best_free = free
                best = i
        device = torch.device(f"cuda:{best}")
    else:
        device = torch.device("cpu")
    progress.write_text(f"device={device}\n")
    print(f"device={device}", flush=True)

    d_map, mse_map = load_d_map(ckpt)
    model = _load_model(str(model_path), device=device, dtype=_dtype_from_name("bfloat16"))
    _, batches = _make_calibration_batches(
        str(model_path),
        dataset_name="wikitext2",
        nsamples=32,
        sequence_length=512,
        batch_size=2,
        seed=0,
    )
    hidden_batches, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    kwargs_dev = [_move_tree(k, device) for k in layer_kwargs]
    running = [h.to(device) for h in hidden_batches]

    bits, group_size, max_tokens = 4, 128, 4096
    rows: list[dict[str, Any]] = []
    for layer_index in range(len(model.model.layers)):
        layer = model.model.layers[layer_index]
        xs: list[torch.Tensor] = []
        next_running: list[torch.Tensor] = []
        with torch.no_grad():
            for hidden, kwargs in zip(running, kwargs_dev):
                hn = layer.input_layernorm(hidden)
                xs.append(hn.detach().float().reshape(-1, hn.shape[-1]).cpu())
                out = layer(hidden, **kwargs)
                next_running.append(_decoder_hidden(out).detach())
        running = next_running
        x_full = torch.cat(xs, dim=0)
        step = max(1, x_full.shape[0] // max_tokens)
        x = x_full[::step][:max_tokens]

        q_name = f"model.layers.{layer_index}.self_attn.q_proj"
        d_q = d_map[q_name]
        x_sm = x / d_q[None, :]
        raw_u = uniformity(x)
        sm_u = uniformity(x_sm)
        raw_a = a_quant_mse(x.to(device), bits, group_size)
        sm_a = a_quant_mse(x_sm.to(device), bits, group_size)
        sm_aff = asymmetric_a_quant_mse(x_sm.to(device), bits, group_size)
        const = torch.full_like(x, float(x.square().mean().sqrt()))
        const_a = a_quant_mse(const.to(device), bits, group_size)
        recon = mse_map[q_name]

        row = {
            "layer": layer_index,
            "recon_mse_q": recon,
            "recon_rel_mse_q": recon / max(raw_a["x_energy"], 1e-20),
            "d_q_log_std": float(d_q.clamp_min(1e-12).log().std(unbiased=False)),
            "d_q_dyn": float(d_q.max() / d_q.min()),
            **{f"raw_{k}": v for k, v in {**raw_u, **raw_a}.items()},
            **{f"smq_{k}": v for k, v in {**sm_u, **sm_a, **sm_aff}.items()},
            "const_a_rel_mse": const_a["a_rel_mse"],
            "rel_a_reduction_by_d_q": 1.0 - sm_a["a_rel_mse"] / max(raw_a["a_rel_mse"], 1e-20),
            "affine_gain_smq": 1.0 - sm_aff["a_rel_mse_affine"] / max(sm_a["a_rel_mse"], 1e-20),
        }
        rows.append(row)
        line = (
            f"L{layer_index:02d} rms={raw_a['x_rms']:.3f} relA={raw_a['a_rel_mse']:.4f}->"
            f"{sm_a['a_rel_mse']:.4f} gini={raw_u['ch_energy_gini']:.3f}->{sm_u['ch_energy_gini']:.3f} "
            f"peak={raw_u['token_max_over_rms_mean']:.2f}->{sm_u['token_max_over_rms_mean']:.2f} "
            f"aff={row['affine_gain_smq']:.3f} recon_rel={row['recon_rel_mse_q']:.4f}\n"
        )
        print(line, end="", flush=True)
        with progress.open("a") as fh:
            fh.write(line)

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    corr = {
        "corr(abs_A_mse, energy)": pearson(col("raw_a_mse"), col("raw_x_energy")),
        "corr(rel_A_mse, energy)": pearson(col("raw_a_rel_mse"), col("raw_x_energy")),
        "corr(rel_A_mse, token_peak)": pearson(
            col("raw_a_rel_mse"), col("raw_token_max_over_rms_mean")
        ),
        "corr(rel_A_mse, gini)": pearson(col("raw_a_rel_mse"), col("raw_ch_energy_gini")),
        "corr(smq_rel_A, smq_peak)": pearson(
            col("smq_a_rel_mse"), col("smq_token_max_over_rms_mean")
        ),
        "corr(smq_rel_A, smq_gini)": pearson(col("smq_a_rel_mse"), col("smq_ch_energy_gini")),
        "corr(recon_mse, energy)": pearson(col("recon_mse_q"), col("raw_x_energy")),
        "corr(recon_rel_mse, energy)": pearson(col("recon_rel_mse_q"), col("raw_x_energy")),
    }
    summary = {
        "raw_a_rel_mse_mean": sum(col("raw_a_rel_mse")) / len(rows),
        "smq_a_rel_mse_mean": sum(col("smq_a_rel_mse")) / len(rows),
        "const_a_rel_mse_mean": sum(col("const_a_rel_mse")) / len(rows),
        "affine_gain_smq_mean": sum(col("affine_gain_smq")) / len(rows),
        "rel_a_reduction_by_d_mean": sum(col("rel_a_reduction_by_d_q")) / len(rows),
        "energy_L0": rows[0]["raw_x_energy"],
        "energy_L27": rows[-1]["raw_x_energy"],
        "energy_ratio_L27_L0": rows[-1]["raw_x_energy"] / max(rows[0]["raw_x_energy"], 1e-20),
        "abs_A_ratio_L27_L0": rows[-1]["raw_a_mse"] / max(rows[0]["raw_a_mse"], 1e-20),
        "relA_ratio_L27_L0": rows[-1]["raw_a_rel_mse"] / max(rows[0]["raw_a_rel_mse"], 1e-20),
        "recon_mse_L0": rows[0]["recon_mse_q"],
        "recon_mse_L27": rows[-1]["recon_mse_q"],
        "recon_ratio_L27_L0": rows[-1]["recon_mse_q"] / max(rows[0]["recon_mse_q"], 1e-20),
        "recon_rel_L0": rows[0]["recon_rel_mse_q"],
        "recon_rel_L27": rows[-1]["recon_rel_mse_q"],
    }
    out = {
        "model": "Qwen2.5-7B",
        "checkpoint": str(ckpt),
        "correlations": corr,
        "summary": summary,
        "layers": rows,
    }
    out_path = out_dir / "activation_uniformity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("\n=== correlations ===")
    for k, v in corr.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.6g}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
