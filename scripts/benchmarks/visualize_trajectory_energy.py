#!/usr/bin/env python3
"""Visualize FP teacher vs quantized student hidden-state energy trajectories.

For each decoder block output Y (or Y_hat), we plot token x channel energy maps
E[t, c] = Y[t, c]^2 and depth-wise summaries that highlight block-to-block jumps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hsvdquant import (  # noqa: E402
    _decoder_hidden,
    _dtype_from_name,
    _load_model,
    _make_calibration_batches,
    _move_tree,
    capture_first_layer_inputs,
    load_quant_checkpoint,
)


def collect_block_outputs(
    model: torch.nn.Module,
    hidden_batches: list[torch.Tensor],
    layer_kwargs: list[dict[str, Any]],
    device: torch.device,
    max_layers: int = -1,
) -> list[torch.Tensor]:
    """Return per-block hidden outputs averaged over calibration batches."""

    layers = model.model.layers
    layer_count = len(layers) if max_layers < 0 else min(max_layers, len(layers))
    accumulators: list[torch.Tensor | None] = [None] * layer_count
    counts = 0
    for hidden, kwargs in zip(hidden_batches, layer_kwargs, strict=True):
        current = hidden.to(device)
        for layer_index in range(layer_count):
            current = _decoder_hidden(
                layers[layer_index](current, **_move_tree(kwargs, device))
            )
            block = current.detach().float().cpu()
            if accumulators[layer_index] is None:
                accumulators[layer_index] = block
            else:
                accumulators[layer_index] = accumulators[layer_index] + block
        counts += 1
    return [tensor / float(counts) for tensor in accumulators if tensor is not None]


def energy_map(hidden: torch.Tensor) -> np.ndarray:
    """Mean over batch -> [tokens, channels] energy."""

    values = hidden.float()
    if values.ndim == 3:
        values = values.mean(dim=0)
    return values.square().numpy()


def per_token_energy(hidden: torch.Tensor) -> np.ndarray:
    values = hidden.float()
    if values.ndim == 3:
        values = values.mean(dim=0)
    return values.square().mean(dim=-1).numpy()


def per_channel_energy(hidden: torch.Tensor) -> np.ndarray:
    values = hidden.float()
    if values.ndim == 3:
        values = values.mean(dim=0)
    return values.square().mean(dim=0).numpy()


def mean_energy(hidden: torch.Tensor) -> float:
    return float(hidden.float().square().mean().item())


def block_jump_ratio(energies: list[float]) -> list[float | None]:
    ratios: list[float | None] = [None]
    for index in range(1, len(energies)):
        prev = energies[index - 1]
        ratios.append(energies[index] / prev if prev > 1e-30 else None)
    return ratios


def save_heatmap(path: Path, matrix: np.ndarray, title: str, vmax_percentile: float = 99.5) -> None:
    import matplotlib.pyplot as plt

    positive = matrix[matrix > 0]
    vmax = float(np.percentile(positive, vmax_percentile)) if positive.size else 1.0
    vmax = max(vmax, 1e-12)
    figure, axis = plt.subplots(figsize=(10, 4))
    image = axis.imshow(
        np.log10(matrix + 1e-12),
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=math.log10(1e-12),
        vmax=math.log10(vmax),
    )
    axis.set_title(title)
    axis.set_xlabel("channel")
    axis.set_ylabel("token")
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02, label="log10 energy")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_depth_curves(
    path: Path,
    series: dict[str, list[float]],
    title: str,
    ylabel: str,
    highlight_layers: tuple[int, ...] = (1, 2, 3),
) -> None:
    import matplotlib.pyplot as plt

    layers = list(range(len(next(iter(series.values())))))
    figure, axis = plt.subplots(figsize=(10, 5))
    for label, values in series.items():
        axis.plot([layer + 1 for layer in layers], values, marker="o", label=label)
    for layer in highlight_layers:
        axis.axvline(layer + 1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("decoder block")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_token_channel_panel(
    path: Path,
    teacher: np.ndarray,
    students: dict[str, np.ndarray],
    block_index: int,
    token_limit: int,
    channel_limit: int,
) -> None:
    import matplotlib.pyplot as plt

    panels = [("FP16 teacher", teacher)] + list(students.items())
    figure, axes = plt.subplots(len(panels), 2, figsize=(12, 2.6 * len(panels)), sharex="col")
    if len(panels) == 1:
        axes = np.array([axes])
    token_slice = slice(0, min(token_limit, teacher.shape[0]))
    channel_slice = slice(0, min(channel_limit, teacher.shape[1]))
    for row, (label, matrix) in enumerate(panels):
        cropped = matrix[token_slice, channel_slice]
        positive = cropped[cropped > 0]
        vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
        vmax = max(vmax, 1e-12)
        image = axes[row, 0].imshow(
            np.log10(cropped + 1e-12),
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=math.log10(1e-12),
            vmax=math.log10(vmax),
        )
        axes[row, 0].set_ylabel(label)
        axes[row, 1].plot(matrix[token_slice, :].mean(axis=0), linewidth=0.8)
        axes[row, 1].set_yscale("log")
        axes[row, 1].grid(True, alpha=0.2)
        if row == 0:
            axes[row, 0].set_title(f"block {block_index + 1}: log10 E[token, channel]")
            axes[row, 1].set_title("channel-mean energy")
        if row == len(panels) - 1:
            axes[row, 0].set_xlabel("channel")
            axes[row, 1].set_xlabel("channel")
    figure.colorbar(image, ax=axes[:, 0], fraction=0.02, pad=0.02, label="log10 energy")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def configure_data_env(
    *,
    hf_endpoint: str,
    c4_train: str,
    c4_validation: str,
) -> None:
    """Prefer mirror endpoints and local C4 shards when available."""

    os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if c4_train:
        os.environ["HSVDQ_C4_TRAIN"] = str(Path(c4_train).resolve())
    if c4_validation:
        os.environ["HSVDQ_C4_VALIDATION"] = str(Path(c4_validation).resolve())


def default_c4_shard(name: str) -> str:
    candidates = [
        ROOT / "results" / "trajectory_ablation_r16_lam025" / "data" / name,
        ROOT / "data" / name,
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    parser.add_argument("--nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--c4-train", default=default_c4_shard("c4-train.00000-of-01024.json.gz"))
    parser.add_argument(
        "--c4-validation",
        default=default_c4_shard("c4-validation.00000-of-00008.json.gz"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--v1-checkpoint", default="")
    parser.add_argument("--v2-checkpoint", default="")
    parser.add_argument("--v3-checkpoint", default="")
    parser.add_argument("--focus-blocks", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--token-limit", type=int, default=256)
    parser.add_argument("--channel-limit", type=int, default=256)
    args = parser.parse_args()
    configure_data_env(
        hf_endpoint=args.hf_endpoint,
        c4_train=args.c4_train,
        c4_validation=args.c4_validation,
    )

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, batches = _make_calibration_batches(
        args.model, args.dataset, args.nsamples, args.seqlen, min(4, args.nsamples), 0
    )
    fp_model = _load_model(args.model, device, dtype)
    hidden_batches, layer_kwargs = capture_first_layer_inputs(fp_model, batches, device)
    del batches

    teacher_blocks = collect_block_outputs(fp_model, hidden_batches, layer_kwargs, device)
    del fp_model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    variant_checkpoints = {
        "V1": args.v1_checkpoint,
        "V2": args.v2_checkpoint,
        "V3": args.v3_checkpoint,
    }
    student_blocks: dict[str, list[torch.Tensor]] = {}
    for label, checkpoint in variant_checkpoints.items():
        if not checkpoint:
            continue
        model, _, _ = load_quant_checkpoint(Path(checkpoint), device, dtype)
        student_blocks[label] = collect_block_outputs(model, hidden_batches, layer_kwargs, device)
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    teacher_energy_maps = [energy_map(block) for block in teacher_blocks]
    student_energy_maps = {
        label: [energy_map(block) for block in blocks]
        for label, blocks in student_blocks.items()
    }

    teacher_mean = [mean_energy(block) for block in teacher_blocks]
    student_mean = {
        label: [mean_energy(block) for block in blocks]
        for label, blocks in student_blocks.items()
    }
    teacher_token = [per_token_energy(block) for block in teacher_blocks]
    student_token = {
        label: [per_token_energy(block) for block in blocks]
        for label, blocks in student_blocks.items()
    }

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "dataset": args.dataset,
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "c4_train": os.environ.get("HSVDQ_C4_TRAIN", ""),
        "c4_validation": os.environ.get("HSVDQ_C4_VALIDATION", ""),
        "teacher_mean_energy": teacher_mean,
        "teacher_block_jump": block_jump_ratio(teacher_mean),
        "variants": {
            label: {
                "checkpoint": variant_checkpoints[label],
                "mean_energy": student_mean[label],
                "block_jump": block_jump_ratio(student_mean[label]),
            }
            for label in student_blocks
        },
    }
    (output_dir / "energy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    save_depth_curves(
        output_dir / "mean_energy_vs_block.png",
        {"FP16 teacher": teacher_mean, **student_mean},
        "Mean hidden-state energy vs decoder block",
        "mean(Y^2)",
        highlight_layers=tuple(args.focus_blocks),
    )
    save_depth_curves(
        output_dir / "block_energy_jump.png",
        {
            "FP16 teacher": [value for value in block_jump_ratio(teacher_mean) if value is not None],
            **{
                label: [value for value in block_jump_ratio(values) if value is not None]
                for label, values in student_mean.items()
            },
        },
        "Block-to-block mean-energy ratio E_l / E_{l-1}",
        "energy ratio",
        highlight_layers=tuple(args.focus_blocks),
    )

    for block_index in args.focus_blocks:
        if block_index < 0 or block_index >= len(teacher_blocks):
            continue
        save_heatmap(
            output_dir / f"teacher_block{block_index + 1:02d}_energy.png",
            teacher_energy_maps[block_index],
            f"FP16 teacher block {block_index + 1}: log10 E[token, channel]",
        )
        for label, maps in student_energy_maps.items():
            save_heatmap(
                output_dir / f"{label.lower()}_block{block_index + 1:02d}_energy.png",
                maps[block_index],
                f"{label} block {block_index + 1}: log10 E[token, channel]",
            )
        save_token_channel_panel(
            output_dir / f"compare_block{block_index + 1:02d}_panel.png",
            teacher_energy_maps[block_index],
            {label: maps[block_index] for label, maps in student_energy_maps.items()},
            block_index,
            args.token_limit,
            args.channel_limit,
        )

    np.savez_compressed(
        output_dir / "energy_arrays.npz",
        teacher=np.stack(teacher_energy_maps, axis=0),
        **{
            f"{label.lower()}_yhat": np.stack(maps, axis=0)
            for label, maps in student_energy_maps.items()
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
