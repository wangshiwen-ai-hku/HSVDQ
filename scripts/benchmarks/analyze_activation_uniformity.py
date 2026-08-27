#!/usr/bin/env python3
"""Test whether activation error tracks energy or distribution non-uniformity.

Computes per-layer:
  - raw / smoothed activation uniformity (max/rms, Gini, kurtosis, entropy)
  - pure A4 quantization MSE (absolute and relative to E[X^2])
  - how much D reduces relative A-error vs residual non-uniformity
  - asymmetry / zero-point (shift) opportunity under the current symmetric quantizer
"""

from __future__ import annotations

import json
import math
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
    """Gini of non-negative vector (0=uniform, 1=one-hot)."""
    v = values.float().flatten().clamp_min(0)
    if v.numel() == 0 or float(v.sum()) <= 0:
        return float("nan")
    v = torch.sort(v).values
    n = v.numel()
    idx = torch.arange(1, n + 1, dtype=v.dtype)
    return float((2.0 * (idx * v).sum() / (n * v.sum()) - (n + 1) / n).item())


def shannon_entropy_bits(values: torch.Tensor, bins: int = 64) -> float:
    """Histogram entropy of |values| in bits (higher = more uniform mass)."""
    v = values.float().abs().flatten()
    if v.numel() == 0:
        return float("nan")
    hist = torch.histc(v, bins=bins, min=0.0, max=float(v.max().clamp_min(1e-8)))
    p = hist / hist.sum().clamp_min(1e-12)
    p = p[p > 0]
    return float(-(p * p.log2()).sum().item())


def channel_energy_gini(x: torch.Tensor) -> float:
    return gini(x.float().pow(2).mean(0))


def token_peakiness(x: torch.Tensor) -> dict[str, float]:
    """Per-token max/rms and fraction of channels near the token amax."""
    xf = x.float()
    rms = xf.pow(2).mean(-1).sqrt().clamp_min(1e-12)
    amax = xf.abs().amax(-1)
    near = (xf.abs() > 0.5 * amax.unsqueeze(-1)).float().mean(-1)
    return {
        "token_max_over_rms_mean": float((amax / rms).mean()),
        "token_max_over_rms_p90": float(torch.quantile(amax / rms, 0.9)),
        "token_frac_near_amax_mean": float(near.mean()),
        "token_frac_near_amax_p10": float(torch.quantile(near, 0.1)),
    }


def kurtosis_excess(x: torch.Tensor) -> float:
    v = x.float().flatten()
    mu = v.mean()
    centered = v - mu
    m2 = centered.pow(2).mean().clamp_min(1e-20)
    m4 = centered.pow(4).mean()
    return float((m4 / m2.pow(2) - 3.0).item())


def asymmetry_stats(x: torch.Tensor) -> dict[str, float]:
    v = x.float().flatten()
    mean = float(v.mean())
    std = float(v.std(unbiased=False).clamp_min(1e-12))
    # skewness
    skew = float(((v - mean) / std).pow(3).mean())
    # mass on positive vs negative
    pos = float((v > 0).float().mean())
    # optimal symmetric vs affine (shift) scale on a subsample for speed
    sample = v[:: max(1, v.numel() // 200_000)].contiguous()
    return {
        "mean": mean,
        "std": std,
        "mean_over_std": mean / std,
        "skew": skew,
        "frac_positive": pos,
        "p01": float(torch.quantile(sample, 0.01)),
        "p99": float(torch.quantile(sample, 0.99)),
        "asymmetry_range": float(
            abs(torch.quantile(sample, 0.99) - torch.quantile(sample, 0.01))
            / (2.0 * sample.abs().amax().clamp_min(1e-12))
        ),
    }


@torch.no_grad()
def a_quant_mse(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> dict[str, float]:
    """Pure activation quantization MSE in input space."""
    xf = x.float()
    q = _quantize_activation(xf, bits, group_size)
    err = (q - xf).square().mean()
    energy = xf.square().mean().clamp_min(1e-20)
    # also report relative to per-token energy
    token_energy = xf.square().mean(-1).clamp_min(1e-20)
    token_err = (q - xf).square().mean(-1)
    rel_token = (token_err / token_energy).mean()
    return {
        "a_mse": float(err),
        "a_rel_mse": float(err / energy),  # = 1 - SNR-ish; scale-invariant if Q is homogeneous
        "a_rel_mse_token_mean": float(rel_token),
        "x_energy": float(energy),
        "x_rms": float(energy.sqrt()),
    }


@torch.no_grad()
def asymmetric_a_quant_mse(x: torch.Tensor, bits: int = 4, group_size: int = 128) -> dict[str, float]:
    """Oracle per-token (or per-group) affine quant: scale + zero-point.

    Compares the gain of a shift against the current symmetric quantizer.
    """
    xf = x.float()
    qmax = float(2 ** (bits - 1) - 1)
    qmin = -qmax
    columns = xf.shape[-1]
    if group_size <= 0 or group_size >= columns:
        xmin = xf.amin(dim=-1, keepdim=True)
        xmax = xf.amax(dim=-1, keepdim=True)
        scale = ((xmax - xmin) / (qmax - qmin)).clamp_min(1e-8)
        zp = torch.round(qmin - xmin / scale).clamp(qmin, qmax)
        q = (torch.round(xf / scale) + zp).clamp(qmin, qmax)
        deq = (q - zp) * scale
    else:
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


def uniformity_bundle(x: torch.Tensor) -> dict[str, float]:
    xf = x.float()
    ch_rms = xf.pow(2).mean(0).sqrt()
    ch_amax = xf.abs().amax(0)
    out = {
        **token_peakiness(xf),
        "ch_energy_gini": channel_energy_gini(xf),
        "ch_max_over_rms_mean": float((ch_amax / ch_rms.clamp_min(1e-12)).mean()),
        "ch_max_over_rms_p99": float(torch.quantile(ch_amax / ch_rms.clamp_min(1e-12), 0.99)),
        "ch_rms_dyn": float(ch_rms.max() / ch_rms.clamp_min(1e-12).min()),
        "kurtosis_excess": kurtosis_excess(xf),
        "abs_entropy_bits": shannon_entropy_bits(xf),
        **{f"asym_{k}": v for k, v in asymmetry_stats(xf).items()},
    }
    return out


def pearson(xs: list[float], ys: list[float]) -> float:
    a = torch.tensor(xs, dtype=torch.float64)
    b = torch.tensor(ys, dtype=torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    if float(den) < 1e-20:
        return float("nan")
    return float((a * b).sum() / den)


def main() -> None:
    ckpt = ROOT / (
        "results/propagation_grid_w4a4/checkpoints/"
        "wikitext2_w4a4_r8_v2_quantized_sequential_o2_s0/hsvdquant.pt"
    )
    model_path = ROOT / "models/Qwen/Qwen3-0.6B"
    out_path = ROOT / "results/propagation_grid_w4a4/summary/activation_uniformity.json"
    progress_path = ROOT / "results/propagation_grid_w4a4/summary/activation_uniformity.log"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    progress_path.write_text(f"device={device}\n")
    print(f"device={device}", flush=True)
    states = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = _load_model(str(model_path), device=device, dtype=_dtype_from_name("bfloat16"))
    _, batches = _make_calibration_batches(
        str(model_path),
        dataset_name="wikitext2",
        nsamples=32,
        sequence_length=512,
        batch_size=4,
        seed=0,
    )
    hidden_batches, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    kwargs_dev = [_move_tree(k, device) for k in layer_kwargs]

    running = [h.to(device) for h in hidden_batches]
    rows: list[dict[str, Any]] = []
    bits = 4
    group_size = 128
    max_tokens = 4096  # subsample for stats; enough for stable means

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
        # fixed stride subsample for reproducibility
        step = max(1, x_full.shape[0] // max_tokens)
        x = x_full[::step][:max_tokens]

        d_q = states[f"model.layers.{layer_index}.self_attn.q_proj"]["d"].float()
        d_v = states[f"model.layers.{layer_index}.self_attn.v_proj"]["d"].float()
        mse_q = float(states[f"model.layers.{layer_index}.self_attn.q_proj"]["error"])
        x_sm_q = x / d_q[None, :]
        x_sm_v = x / d_v[None, :]

        raw_u = uniformity_bundle(x)
        sm_q_u = uniformity_bundle(x_sm_q)
        # run A-quant on GPU if available
        quant_device = device if device.type == "cuda" else torch.device("cpu")
        raw_a = a_quant_mse(x.to(quant_device), bits, group_size)
        sm_q_a = a_quant_mse(x_sm_q.to(quant_device), bits, group_size)
        sm_v_a = a_quant_mse(x_sm_v.to(quant_device), bits, group_size)
        raw_aff = asymmetric_a_quant_mse(x.to(quant_device), bits, group_size)
        sm_q_aff = asymmetric_a_quant_mse(x_sm_q.to(quant_device), bits, group_size)

        # Constant-activation sanity: energy large, uniformity perfect → A error ~0
        const = torch.full_like(x, float(x.square().mean().sqrt()))
        const_a = a_quant_mse(const.to(quant_device), bits, group_size)

        row = {
            "layer": layer_index,
            "recon_mse_q": mse_q,
            "recon_rel_mse_q": mse_q / max(raw_a["x_energy"], 1e-20),
            "d_q_log_std": float(d_q.clamp_min(1e-12).log().std(unbiased=False)),
            "d_v_log_std": float(d_v.clamp_min(1e-12).log().std(unbiased=False)),
            "d_q_dyn": float(d_q.max() / d_q.min()),
            **{f"raw_{k}": v for k, v in {**raw_u, **raw_a, **raw_aff}.items()},
            **{f"smq_{k}": v for k, v in {**sm_q_u, **sm_q_a, **sm_q_aff}.items()},
            "smv_a_rel_mse": sm_v_a["a_rel_mse"],
            "smv_a_mse": sm_v_a["a_mse"],
            "const_a_rel_mse": const_a["a_rel_mse"],
            "const_a_mse": const_a["a_mse"],
            # gains
            "rel_a_reduction_by_d_q": 1.0 - sm_q_a["a_rel_mse"] / max(raw_a["a_rel_mse"], 1e-20),
            "affine_gain_raw": 1.0 - raw_aff["a_rel_mse_affine"] / max(raw_a["a_rel_mse"], 1e-20),
            "affine_gain_smq": 1.0 - sm_q_aff["a_rel_mse_affine"] / max(sm_q_a["a_rel_mse"], 1e-20),
        }
        rows.append(row)
        line = (
            f"L{layer_index:02d} energy={raw_a['x_rms']:.3f} "
            f"raw_relA={raw_a['a_rel_mse']:.4f} smq_relA={sm_q_a['a_rel_mse']:.4f} "
            f"gini={raw_u['ch_energy_gini']:.3f}->{sm_q_u['ch_energy_gini']:.3f} "
            f"peak={raw_u['token_max_over_rms_mean']:.2f}->{sm_q_u['token_max_over_rms_mean']:.2f} "
            f"aff_gain={row['affine_gain_smq']:.3f} recon_rel={row['recon_rel_mse_q']:.4f}\n"
        )
        print(line, end="", flush=True)
        with progress_path.open("a") as fh:
            fh.write(line)

    # correlations across layers
    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows]

    corr_table = {
        "corr(abs_A_mse, energy)": pearson(col("raw_a_mse"), col("raw_x_energy")),
        "corr(rel_A_mse, energy)": pearson(col("raw_a_rel_mse"), col("raw_x_energy")),
        "corr(rel_A_mse, ch_gini)": pearson(col("raw_a_rel_mse"), col("raw_ch_energy_gini")),
        "corr(rel_A_mse, token_peak)": pearson(
            col("raw_a_rel_mse"), col("raw_token_max_over_rms_mean")
        ),
        "corr(rel_A_mse, kurtosis)": pearson(col("raw_a_rel_mse"), col("raw_kurtosis_excess")),
        "corr(smq_rel_A, smq_gini)": pearson(col("smq_a_rel_mse"), col("smq_ch_energy_gini")),
        "corr(smq_rel_A, smq_peak)": pearson(
            col("smq_a_rel_mse"), col("smq_token_max_over_rms_mean")
        ),
        "corr(smq_rel_A, energy)": pearson(col("smq_a_rel_mse"), col("raw_x_energy")),
        "corr(recon_mse, energy)": pearson(col("recon_mse_q"), col("raw_x_energy")),
        "corr(recon_rel_mse, energy)": pearson(col("recon_rel_mse_q"), col("raw_x_energy")),
        "corr(recon_rel_mse, smq_rel_A)": pearson(col("recon_rel_mse_q"), col("smq_a_rel_mse")),
        "corr(recon_rel_mse, smq_peak)": pearson(
            col("recon_rel_mse_q"), col("smq_token_max_over_rms_mean")
        ),
        "corr(|mean|/std, affine_gain)": pearson(
            [abs(r["raw_asym_mean_over_std"]) for r in rows], col("affine_gain_smq")
        ),
    }

    payload = {
        "bits": bits,
        "group_size": group_size,
        "n_layers": len(rows),
        "correlations": corr_table,
        "layers": rows,
        "summary": {
            "raw_a_rel_mse_mean": sum(col("raw_a_rel_mse")) / len(rows),
            "smq_a_rel_mse_mean": sum(col("smq_a_rel_mse")) / len(rows),
            "const_a_rel_mse_mean": sum(col("const_a_rel_mse")) / len(rows),
            "affine_gain_smq_mean": sum(col("affine_gain_smq")) / len(rows),
            "affine_gain_smq_max": max(col("affine_gain_smq")),
            "rel_a_reduction_by_d_mean": sum(col("rel_a_reduction_by_d_q")) / len(rows),
            "energy_L0": rows[0]["raw_x_energy"],
            "energy_L25": rows[25]["raw_x_energy"],
            "raw_relA_L0": rows[0]["raw_a_rel_mse"],
            "raw_relA_L25": rows[25]["raw_a_rel_mse"],
            "smq_relA_L0": rows[0]["smq_a_rel_mse"],
            "smq_relA_L25": rows[25]["smq_a_rel_mse"],
            "abs_A_mse_L0": rows[0]["raw_a_mse"],
            "abs_A_mse_L25": rows[25]["raw_a_mse"],
            "abs_A_ratio_L25_L0": rows[25]["raw_a_mse"] / max(rows[0]["raw_a_mse"], 1e-20),
            "energy_ratio_L25_L0": rows[25]["raw_x_energy"] / max(rows[0]["raw_x_energy"], 1e-20),
            "relA_ratio_L25_L0": rows[25]["raw_a_rel_mse"] / max(rows[0]["raw_a_rel_mse"], 1e-20),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("\n==== correlations ====")
    for k, v in corr_table.items():
        print(f"  {k}: {v:.4f}")
    print("\n==== summary ====")
    for k, v in payload["summary"].items():
        print(f"  {k}: {v:.6g}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
