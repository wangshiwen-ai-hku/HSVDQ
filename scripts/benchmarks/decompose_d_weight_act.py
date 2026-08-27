#!/usr/bin/env python3
"""Decompose per-channel D against weight and activation stats for QKV."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hsvdquant import (  # noqa: E402
    _decoder_hidden,
    _dtype_from_name,
    _load_model,
    _make_calibration_batches,
    _move_tree,
    capture_first_layer_inputs,
)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    den = a.norm() * b.norm()
    if float(den) < 1e-20:
        return float("nan")
    return float((a * b).sum() / den)


def summarize(name: str, v: torch.Tensor) -> dict[str, float]:
    v = v.float().flatten()
    return {
        f"{name}_mean": float(v.mean()),
        f"{name}_std": float(v.std(unbiased=False)),
        f"{name}_min": float(v.min()),
        f"{name}_max": float(v.max()),
        f"{name}_p50": float(v.quantile(0.5)),
        f"{name}_p90": float(v.quantile(0.9)),
        f"{name}_p99": float(v.quantile(0.99)),
        f"{name}_dyn": float(v.max() / v.clamp_min(1e-12).min()),
    }


def main() -> None:
    ckpt_path = ROOT / (
        "results/propagation_grid_w4a4/checkpoints/"
        "wikitext2_w4a4_r8_v2_quantized_sequential_o2_s0/hsvdquant.pt"
    )
    model_path = ROOT / "models/Qwen/Qwen3-0.6B"
    out_path = ROOT / "results/propagation_grid_w4a4/summary/d_weight_act_decompose.json"
    focus = [0, 5, 14, 25]
    kinds = ("q_proj", "k_proj", "v_proj")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    states = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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

    print("collecting attn inputs...", flush=True)
    running = [h.to(device) for h in hidden_batches]
    attn_inputs: dict[int, torch.Tensor] = {}
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
        if layer_index in focus:
            attn_inputs[layer_index] = torch.cat(xs, dim=0)
            print(f"  L{layer_index} X={tuple(attn_inputs[layer_index].shape)}", flush=True)

    results: dict[str, Any] = {"focus": focus, "layers": {}}
    for layer_index in focus:
        x = attn_inputs[layer_index]
        x_rms = x.pow(2).mean(0).sqrt()
        x_amax = x.abs().amax(0)
        x_max_over_rms = x_amax / x_rms.clamp_min(1e-12)
        x_p99 = torch.quantile(x.abs(), 0.99, dim=0)
        layer_pack: dict[str, Any] = {
            "x_global": {
                **summarize("x_rms", x_rms),
                **summarize("x_amax", x_amax),
                **summarize("x_max_over_rms", x_max_over_rms),
                "x_token_rms_mean": float(x.pow(2).mean().sqrt()),
                "x_global_amax": float(x.abs().max()),
                "x_frac_gt_5_global_rms": float(
                    (x.abs() > 5.0 * x.pow(2).mean().sqrt()).float().mean()
                ),
            },
            "modules": {},
        }
        tensors: dict[str, dict[str, torch.Tensor]] = {}
        layer = model.model.layers[layer_index]
        for kind in kinds:
            name = f"model.layers.{layer_index}.self_attn.{kind}"
            state = states[name]
            d = state["d"].float()
            weight = getattr(layer.self_attn, kind).weight.detach().float().cpu()
            w_col_rms = weight.pow(2).mean(0).sqrt()
            w_col_amax = weight.abs().amax(0)
            w_col_l2 = weight.pow(2).sum(0).sqrt()
            w_col_max_over_rms = w_col_amax / w_col_rms.clamp_min(1e-12)

            residual = weight.T.contiguous()
            omega = residual.abs().pow(2.0).sum(1).clamp_min(1e-12)
            d_w_only = ((torch.zeros_like(omega) - omega.log()) / 4.0).exp()
            d_w_only = d_w_only / d_w_only.log().mean().exp()
            d_x_only = x_amax.clamp_min(1e-12)
            d_x_only = d_x_only / d_x_only.log().mean().exp()

            x_sm = x / d[None, :]
            x_sm_amax = x_sm.abs().amax(0)
            x_sm_rms = x_sm.pow(2).mean(0).sqrt()
            weight_eff = weight * d[None, :]
            weff_max_over_rms = weight_eff.abs().amax(0) / weight_eff.pow(2).mean(0).sqrt().clamp_min(
                1e-12
            )

            mod = {
                "mse": float(state["error"]),
                "sigma_a_mean": float(state.get("sigma_a_mean") or 0.0),
                "sigma_a_max": float(state.get("sigma_a_max") or 0.0),
                **summarize("d", d),
                "d_log_std": float(d.clamp_min(1e-12).log().std(unbiased=False)),
                "w_fro": float(weight.pow(2).sum().sqrt()),
                "w_abs_max": float(weight.abs().max()),
                "w_col_rms_mean": float(w_col_rms.mean()),
                "w_col_rms_max": float(w_col_rms.max()),
                "w_col_amax_max": float(w_col_amax.max()),
                "w_col_max_over_rms_mean": float(w_col_max_over_rms.mean()),
                "w_col_max_over_rms_p99": float(w_col_max_over_rms.quantile(0.99)),
                "weff_col_max_over_rms_mean": float(weff_max_over_rms.mean()),
                "corr_d_x_amax": corr(d, x_amax),
                "corr_d_x_rms": corr(d, x_rms),
                "corr_d_x_max_over_rms": corr(d, x_max_over_rms),
                "corr_d_w_col_rms": corr(d, w_col_rms),
                "corr_d_w_col_amax": corr(d, w_col_amax),
                "corr_d_w_col_max_over_rms": corr(d, w_col_max_over_rms),
                "corr_d_inv_w_col_l2": corr(d, 1.0 / w_col_l2.clamp_min(1e-12)),
                "corr_d_d_x_only": corr(d, d_x_only),
                "corr_d_d_w_only": corr(d, d_w_only),
                "corr_logd_logxamax": corr(d.log(), x_amax.log()),
                "corr_logd_logwrms": corr(d.log(), w_col_rms.log()),
                "x_sm_amax_mean": float(x_sm_amax.mean()),
                "x_sm_amax_max": float(x_sm_amax.max()),
                "x_sm_max_over_rms_mean": float(
                    (x_sm_amax / x_sm_rms.clamp_min(1e-12)).mean()
                ),
            }
            layer_pack["modules"][kind] = mod
            tensors[kind] = {
                "d": d,
                "w_col_rms": w_col_rms,
                "w_col_max_over_rms": w_col_max_over_rms,
            }

        dq = tensors["q_proj"]["d"]
        dk = tensors["k_proj"]["d"]
        dv = tensors["v_proj"]["d"]
        top = torch.topk((dq.log() - dv.log()).abs(), k=12)
        disagree = []
        for idx, val in zip(top.indices.tolist(), top.values.tolist()):
            disagree.append(
                {
                    "ch": idx,
                    "abs_log_ratio_qv": float(val),
                    "log_ratio_qv": float((dq[idx] / dv[idx]).log()),
                    "d_q": float(dq[idx]),
                    "d_k": float(dk[idx]),
                    "d_v": float(dv[idx]),
                    "x_amax": float(x_amax[idx]),
                    "x_rms": float(x_rms[idx]),
                    "x_max_over_rms": float(x_max_over_rms[idx]),
                    "w_q_col_rms": float(tensors["q_proj"]["w_col_rms"][idx]),
                    "w_k_col_rms": float(tensors["k_proj"]["w_col_rms"][idx]),
                    "w_v_col_rms": float(tensors["v_proj"]["w_col_rms"][idx]),
                    "w_q_max_over_rms": float(tensors["q_proj"]["w_col_max_over_rms"][idx]),
                    "w_v_max_over_rms": float(tensors["v_proj"]["w_col_max_over_rms"][idx]),
                }
            )
        layer_pack["qv_disagree_top"] = disagree
        layer_pack["pearson_dq_dv"] = corr(dq, dv)
        layer_pack["pearson_dq_dk"] = corr(dq, dk)
        results["layers"][str(layer_index)] = layer_pack
        print(
            f"L{layer_index} mse_q={layer_pack['modules']['q_proj']['mse']:.4e} "
            f"x_rms={layer_pack['x_global']['x_token_rms_mean']:.4f}",
            flush=True,
        )

    a = results["layers"]["0"]["modules"]["q_proj"]
    b = results["layers"]["25"]["modules"]["q_proj"]
    x0 = results["layers"]["0"]["x_global"]
    x25 = results["layers"]["25"]["x_global"]
    results["L25_over_L0_q"] = {
        "mse_ratio": b["mse"] / a["mse"],
        "x_token_rms_ratio": x25["x_token_rms_mean"] / x0["x_token_rms_mean"],
        "x_global_amax_ratio": x25["x_global_amax"] / x0["x_global_amax"],
        "x_rms_mean_ratio": x25["x_rms_mean"] / x0["x_rms_mean"],
        "x_max_over_rms_mean_ratio": x25["x_max_over_rms_mean"] / x0["x_max_over_rms_mean"],
        "w_fro_ratio": b["w_fro"] / a["w_fro"],
        "w_abs_max_ratio": b["w_abs_max"] / a["w_abs_max"],
        "w_col_max_over_rms_mean_ratio": b["w_col_max_over_rms_mean"] / a["w_col_max_over_rms_mean"],
        "w_col_amax_max_ratio": b["w_col_amax_max"] / a["w_col_amax_max"],
        "d_log_std_ratio": b["d_log_std"] / a["d_log_std"],
        "d_dyn_ratio": b["d_dyn"] / a["d_dyn"],
        "sigma_a_ratio": b["sigma_a_mean"] / max(a["sigma_a_mean"], 1e-20),
        "x_sm_amax_max_ratio": b["x_sm_amax_max"] / max(a["x_sm_amax_max"], 1e-20),
        "mse_explained_by_xrms_sq": (x25["x_token_rms_mean"] / x0["x_token_rms_mean"]) ** 2,
        "mse_explained_by_xrms_wfro_sq": (
            (x25["x_token_rms_mean"] / x0["x_token_rms_mean"]) * (b["w_fro"] / a["w_fro"])
        )
        ** 2,
        "relative_mse_ratio": (b["mse"] / a["mse"])
        / max((x25["x_token_rms_mean"] / x0["x_token_rms_mean"]) ** 2, 1e-20),
    }

    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path}", flush=True)

    print("\n==== D flatter meaning ====")
    print(
        "Large D_i shrinks X_i and grows W[:,i]. Spikier D = stronger per-channel "
        "transfer for activation outliers. Flatter D = less channel-wise rebalancing "
        "(milder act outliers and/or weight side refuses the transfer)."
    )

    for layer_index in focus:
        pack = results["layers"][str(layer_index)]
        xg = pack["x_global"]
        print(f"\n======== Layer {layer_index} ========")
        print(
            f"  X token_rms={xg['x_token_rms_mean']:.4f} amax={xg['x_global_amax']:.3f} "
            f"ch max/rms mean={xg['x_max_over_rms_mean']:.2f} p99={xg['x_max_over_rms_p99']:.2f} "
            f"frac>|5rms|={xg['x_frac_gt_5_global_rms']:.4f}"
        )
        for kind in kinds:
            m = pack["modules"][kind]
            print(
                f"  {kind}: mse={m['mse']:.4e} d_log_std={m['d_log_std']:.3f} d_dyn={m['d_dyn']:.1f} "
                f"corr(d,x_amax)={m['corr_d_x_amax']:+.3f} corr(d,w_rms)={m['corr_d_w_col_rms']:+.3f} "
                f"corr(logd,logx)={m['corr_logd_logxamax']:+.3f} corr(logd,logw)={m['corr_logd_logwrms']:+.3f} "
                f"corr(d,dx)={m['corr_d_d_x_only']:+.3f} corr(d,dw)={m['corr_d_d_w_only']:+.3f} "
                f"w_max/rms={m['w_col_max_over_rms_mean']:.2f} w_amax_max={m['w_col_amax_max']:.4f} "
                f"sigma_a={m['sigma_a_mean']:.3e}"
            )
        print("  Top Q-vs-V D disagree channels:")
        for row in pack["qv_disagree_top"][:8]:
            print(
                f"    ch{row['ch']:4d} d_q/k/v={row['d_q']:.3f}/{row['d_k']:.3f}/{row['d_v']:.3f} "
                f"log(dq/dv)={row['log_ratio_qv']:+.3f} x_amax={row['x_amax']:.3f} "
                f"x_max/rms={row['x_max_over_rms']:.2f} "
                f"w_rms q/k/v={row['w_q_col_rms']:.4f}/{row['w_k_col_rms']:.4f}/{row['w_v_col_rms']:.4f} "
                f"w_out q/v={row['w_q_max_over_rms']:.2f}/{row['w_v_max_over_rms']:.2f}"
            )

    print("\n==== L25 / L0 for q_proj ====")
    for key, value in results["L25_over_L0_q"].items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()
