#!/usr/bin/env python3
"""Decompose FP teacher residual-stream energy into attention and MLP branches."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hsvdquant import (
    _decoder_hidden,
    _dtype_from_name,
    _load_model,
    _make_calibration_batches,
    _move_tree,
    capture_first_layer_inputs,
)


def energy(tensor: torch.Tensor) -> float:
    return float(tensor.float().square().mean().item())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    lhs, rhs = left.float(), right.float()
    return float((lhs * rhs).sum().item()) / max(
        math.sqrt(float(lhs.square().sum().item()) * float(rhs.square().sum().item())),
        1e-30,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dataset", default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--layers", type=int, default=5)
    args = parser.parse_args()
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    _, batches = _make_calibration_batches(
        args.model, args.dataset, args.nsamples, args.seqlen, 4, 0
    )
    model = _load_model(args.model, device, dtype)
    hidden_batches, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    for layer_index, layer in enumerate(model.model.layers[: args.layers]):
        totals: dict[str, list[float]] = {
            key: []
            for key in (
                "input_e",
                "attn_input_e",
                "attn_e",
                "post_attn_e",
                "mlp_input_e",
                "gate_e",
                "up_e",
                "swiglu_e",
                "mlp_e",
                "output_e",
                "attn_cos",
                "mlp_cos",
                "swiglu_absmax",
                "mlp_absmax",
                "mlp_token_rms_max",
            )
        }
        next_hidden = []
        for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
            x = hidden.to(device)
            normalized = layer.input_layernorm(x)
            attn_output, _ = layer.self_attn(
                hidden_states=normalized, **_move_tree(kwargs, device)
            )
            post_attn = x + attn_output
            mlp_input = layer.post_attention_layernorm(post_attn)
            gate = layer.mlp.gate_proj(mlp_input)
            up = layer.mlp.up_proj(mlp_input)
            swiglu = layer.mlp.act_fn(gate) * up
            mlp_output = layer.mlp.down_proj(swiglu)
            output = post_attn + mlp_output
            totals["input_e"].append(energy(x))
            totals["attn_input_e"].append(energy(normalized))
            totals["attn_e"].append(energy(attn_output))
            totals["post_attn_e"].append(energy(post_attn))
            totals["mlp_input_e"].append(energy(mlp_input))
            totals["gate_e"].append(energy(gate))
            totals["up_e"].append(energy(up))
            totals["swiglu_e"].append(energy(swiglu))
            totals["mlp_e"].append(energy(mlp_output))
            totals["output_e"].append(energy(output))
            totals["attn_cos"].append(cosine(x, attn_output))
            totals["mlp_cos"].append(cosine(post_attn, mlp_output))
            totals["swiglu_absmax"].append(float(swiglu.float().abs().max().item()))
            totals["mlp_absmax"].append(float(mlp_output.float().abs().max().item()))
            totals["mlp_token_rms_max"].append(
                float(mlp_output.float().square().mean(dim=-1).sqrt().max().item())
            )
            next_hidden.append(output.detach().cpu())
        means = {key: sum(values) / len(values) for key, values in totals.items()}
        print(
            f"block={layer_index + 1} input_E={means['input_e']:.6g} "
            f"norm1_E={means['attn_input_e']:.6g} attn_E={means['attn_e']:.6g} "
            f"post_attn_E={means['post_attn_e']:.6g} attn_cos={means['attn_cos']:.6g} "
            f"norm2_E={means['mlp_input_e']:.6g} mlp_E={means['mlp_e']:.6g} "
            f"output_E={means['output_e']:.6g} mlp_cos={means['mlp_cos']:.6g} "
            f"gate_E={means['gate_e']:.6g} up_E={means['up_e']:.6g} "
            f"swiglu_E={means['swiglu_e']:.6g} swiglu_max={means['swiglu_absmax']:.6g} "
            f"mlp_max={means['mlp_absmax']:.6g} token_rms_max={means['mlp_token_rms_max']:.6g}"
        )
        hidden_batches = next_hidden


if __name__ == "__main__":
    main()
