#!/usr/bin/env python3
"""Measure per-decoder-block error caused by dynamic activation quantization.

For a quantized checkpoint, this runs two streams with the same quantized
weights:

* reference stream: all quantized Linear modules in the current block use A16
  (activation quantization disabled)
* quantized stream: those modules use their checkpoint activation bit-width

The local metric isolates the current block by feeding the same reference input
to A16 and A4.  The propagated metric feeds each stream's own hidden state and
therefore includes accumulated upstream drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from common import (
    REPO_ROOT,
    _dtype_from_name,
    _move_tree,
    environment_metadata,
    load_experiment_model,
    make_calibration,
    write_json,
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import _decoder_hidden, capture_first_layer_inputs  # noqa: E402


def _activation_modules(module: torch.nn.Module) -> list[torch.nn.Module]:
    return [child for child in module.modules() if hasattr(child, "activation_bits")]


def _set_activation_bits(module: torch.nn.Module, bits: int) -> list[int]:
    modules = _activation_modules(module)
    old = [int(child.activation_bits) for child in modules]
    for child in modules:
        child.activation_bits = int(bits)
    return old


def _restore_activation_bits(module: torch.nn.Module, bits: list[int]) -> None:
    modules = _activation_modules(module)
    if len(modules) != len(bits):
        raise RuntimeError("activation module count changed while restoring bits")
    for child, old in zip(modules, bits, strict=True):
        child.activation_bits = int(old)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def _run_layer(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    kwargs: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    output = layer(hidden.to(device), **_move_tree(kwargs, device))
    return _decoder_hidden(output).detach().to("cpu")


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    tokenizer, batches = make_calibration(
        args.model,
        args.dataset,
        args.nsamples,
        args.sequence_length,
        args.batch_size,
        args.seed,
    )
    del tokenizer
    model, _tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    del _tokenizer
    hidden_ref, layer_kwargs = capture_first_layer_inputs(model, batches, device)
    hidden_quant = [hidden.clone() for hidden in hidden_ref]
    del batches

    layer_count = len(model.model.layers) if args.max_layers < 0 else min(args.max_layers, len(model.model.layers))
    records: list[dict[str, Any]] = []
    for layer_index in range(layer_count):
        layer = model.model.layers[layer_index]
        original_bits = _set_activation_bits(layer, 16)
        ref_next = [
            _run_layer(layer, hidden, kwargs, device)
            for hidden, kwargs in zip(hidden_ref, layer_kwargs, strict=True)
        ]

        _restore_activation_bits(layer, original_bits)
        local_next = [
            _run_layer(layer, hidden, kwargs, device)
            for hidden, kwargs in zip(hidden_ref, layer_kwargs, strict=True)
        ]
        quant_next = [
            _run_layer(layer, hidden, kwargs, device)
            for hidden, kwargs in zip(hidden_quant, layer_kwargs, strict=True)
        ]

        input_mse_values: list[float] = []
        local_mse_values: list[float] = []
        propagated_mse_values: list[float] = []
        ref_energy_values: list[float] = []
        for h_ref, h_quant, y_ref, y_local, y_quant in zip(
            hidden_ref,
            hidden_quant,
            ref_next,
            local_next,
            quant_next,
            strict=True,
        ):
            input_mse_values.append(float((h_quant.float() - h_ref.float()).square().mean().item()))
            local_mse_values.append(float((y_local.float() - y_ref.float()).square().mean().item()))
            propagated_mse_values.append(float((y_quant.float() - y_ref.float()).square().mean().item()))
            ref_energy_values.append(float(y_ref.float().square().mean().item()))

        ref_energy = _mean(ref_energy_values)
        record = {
            "layer": layer_index,
            "activation_bits": sorted(set(original_bits)),
            "activation_modules": len(original_bits),
            "input_mse": _mean(input_mse_values),
            "local_activation_mse": _mean(local_mse_values),
            "propagated_mse": _mean(propagated_mse_values),
            "local_relative_mse": _mean(local_mse_values) / max(ref_energy, 1e-30),
            "propagated_relative_mse": _mean(propagated_mse_values) / max(ref_energy, 1e-30),
            "ref_output_energy": ref_energy,
        }
        records.append(record)
        print(
            f"[block] {layer_index:02d} input={record['input_mse']:.3e} "
            f"local={record['local_activation_mse']:.3e} "
            f"prop={record['propagated_mse']:.3e} "
            f"rel={record['propagated_relative_mse']:.3e}",
            flush=True,
        )
        hidden_ref, hidden_quant = ref_next, quant_next
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "runtime": runtime.__dict__,
        "args": vars(args),
        "records": records,
        "environment": environment_metadata(),
    }
    write_json(Path(args.output) / "per_block.json", payload)
    total_local = sum(row["local_activation_mse"] for row in records)
    total_prop = sum(row["propagated_mse"] for row in records)
    summary = {
        "layers": len(records),
        "sum_local_activation_mse": total_local,
        "sum_propagated_mse": total_prop,
        "mean_local_relative_mse": _mean([row["local_relative_mse"] for row in records]),
        "mean_propagated_relative_mse": _mean([row["propagated_relative_mse"] for row in records]),
        "top_local_layers": sorted(records, key=lambda row: row["local_activation_mse"], reverse=True)[: args.top_k],
        "top_propagated_layers": sorted(records, key=lambda row: row["propagated_mse"], reverse=True)[: args.top_k],
    }
    write_json(Path(args.output) / "summary.json", summary)
    print("\n===== block activation summary =====", flush=True)
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--dataset", choices=["wikitext2", "c4", "synthetic"], default="wikitext2")
    parser.add_argument("--nsamples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=-1)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    diagnose(build_parser().parse_args())


if __name__ == "__main__":
    main()
