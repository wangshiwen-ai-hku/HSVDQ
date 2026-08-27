#!/usr/bin/env python3
"""Connect V3 trajectory corrections to logits, NLL, and layer rollback oracles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from common import get_eval_input_ids, make_tokenizer, write_json  # noqa: E402
from hsvdquant import (  # noqa: E402
    HSVQuantLinear,
    _dtype_from_name,
    _load_model,
    _set_submodule,
    load_quant_checkpoint,
)


def load_states(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads((checkpoint / "hsvdquant_config.json").read_text(encoding="utf-8"))
    try:
        states = torch.load(checkpoint / "hsvdquant.pt", map_location="cpu", weights_only=True)
    except TypeError:
        states = torch.load(checkpoint / "hsvdquant.pt", map_location="cpu")
    return metadata, states.get("states", states) if isinstance(states, dict) else states


def layer_index(name: str) -> int:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1])


def replace_layers_with_states(
    model: nn.Module,
    states: dict[str, Any],
    layers: set[int],
    dtype: torch.dtype,
) -> None:
    for name, state in states.items():
        if layer_index(name) in layers:
            _set_submodule(model, name, HSVQuantLinear(state, compute_dtype=dtype))


@torch.no_grad()
def logits_and_hidden_metrics(
    *,
    teacher: nn.Module,
    student: nn.Module,
    input_ids: torch.Tensor,
    device: torch.device,
    max_samples: int,
    seqlen: int,
) -> dict[str, Any]:
    loss_fct = nn.CrossEntropyLoss(reduction="sum")
    nll_teacher = 0.0
    nll_student = 0.0
    kl_sum = 0.0
    top_flip = 0
    tokens = 0
    hidden_rows: dict[int, dict[str, float]] = {}
    nsamples = input_ids.numel() // seqlen
    if max_samples > 0:
        nsamples = min(nsamples, max_samples)
    for index in range(nsamples):
        batch = input_ids[:, index * seqlen : (index + 1) * seqlen].to(device)
        teacher_out = teacher(batch, output_hidden_states=True)
        student_out = student(batch, output_hidden_states=True)
        teacher_logits = teacher_out.logits[:, :-1, :].float()
        student_logits = student_out.logits[:, :-1, :].float()
        labels = batch[:, 1:]
        nll_teacher += float(loss_fct(teacher_logits.reshape(-1, teacher_logits.shape[-1]), labels.reshape(-1)).item())
        nll_student += float(loss_fct(student_logits.reshape(-1, student_logits.shape[-1]), labels.reshape(-1)).item())
        teacher_logp = F.log_softmax(teacher_logits, dim=-1)
        student_logp = F.log_softmax(student_logits, dim=-1)
        teacher_prob = teacher_logp.exp()
        kl_sum += float((teacher_prob * (teacher_logp - student_logp)).sum().item())
        top_flip += int((teacher_logits.argmax(dim=-1) != student_logits.argmax(dim=-1)).sum().item())
        tokens += int(labels.numel())
        for block, (teacher_hidden, student_hidden) in enumerate(
            zip(teacher_out.hidden_states[1:], student_out.hidden_states[1:]),
            start=0,
        ):
            lhs = student_hidden.float()
            rhs = teacher_hidden.float()
            delta = lhs - rhs
            row = hidden_rows.setdefault(
                block,
                {"sse": 0.0, "teacher_energy": 0.0, "student_energy": 0.0, "elements": 0.0},
            )
            row["sse"] += float(delta.square().sum().item())
            row["teacher_energy"] += float(rhs.square().sum().item())
            row["student_energy"] += float(lhs.square().sum().item())
            row["elements"] += float(delta.numel())
    block_rows = []
    for block, row in sorted(hidden_rows.items()):
        elements = max(row["elements"], 1.0)
        mse = row["sse"] / elements
        teacher_energy = row["teacher_energy"] / elements
        block_rows.append(
            {
                "layer": block,
                "mse": mse,
                "teacher_energy": teacher_energy,
                "student_energy": row["student_energy"] / elements,
                "nmse": mse / max(teacher_energy, 1e-30),
            }
        )
    denom = max(tokens, 1)
    return {
        "tokens": tokens,
        "teacher_nll": nll_teacher / denom,
        "student_nll": nll_student / denom,
        "delta_nll": (nll_student - nll_teacher) / denom,
        "teacher_ppl": math.exp(nll_teacher / denom),
        "student_ppl": math.exp(nll_student / denom),
        "kl_to_teacher": kl_sum / denom,
        "top_flip_rate": top_flip / denom,
        "blocks": block_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--v1-checkpoint", required=True)
    parser.add_argument("--v3-checkpoint", required=True)
    parser.add_argument("--dataset", choices=["wikitext2", "c4"], default="wikitext2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--leave-one-layer-out", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    device = torch.device(args.device)
    dtype = _dtype_from_name(args.dtype)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tokenizer(args.model)
    input_ids = get_eval_input_ids(args.dataset, tokenizer, args.seqlen)

    teacher = _load_model(args.model, device, dtype)
    v1_model, _, _ = load_quant_checkpoint(Path(args.v1_checkpoint), device, dtype)
    v3_model, _, _ = load_quant_checkpoint(Path(args.v3_checkpoint), device, dtype)
    _, v1_states = load_states(Path(args.v1_checkpoint))

    model_rows = []
    block_rows = []
    for label, model in (("v1", v1_model), ("v3", v3_model)):
        metrics = logits_and_hidden_metrics(
            teacher=teacher,
            student=model,
            input_ids=input_ids,
            device=device,
            max_samples=args.max_samples,
            seqlen=args.seqlen,
        )
        model_rows.append({key: value for key, value in metrics.items() if key != "blocks"} | {"model": label})
        for row in metrics["blocks"]:
            block_rows.append({"model": label, **row})

    rollback_rows = []
    if args.leave_one_layer_out:
        layers = sorted({layer_index(name) for name in v1_states})
        for layer in layers:
            mixed_model, _, _ = load_quant_checkpoint(Path(args.v3_checkpoint), device, dtype)
            replace_layers_with_states(mixed_model, v1_states, {layer}, dtype)
            metrics = logits_and_hidden_metrics(
                teacher=teacher,
                student=mixed_model.to(device).eval(),
                input_ids=input_ids,
                device=device,
                max_samples=args.max_samples,
                seqlen=args.seqlen,
            )
            rollback_rows.append(
                {key: value for key, value in metrics.items() if key != "blocks"}
                | {"rollback_layer": layer}
            )
            del mixed_model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    write_csv(output / "logit_metrics.csv", model_rows)
    write_csv(output / "block_hidden_metrics.csv", block_rows)
    write_csv(output / "leave_one_layer_out.csv", rollback_rows)
    write_json(
        output / "summary.json",
        {
            "args": vars(args),
            "models": model_rows,
            "leave_one_layer_out": rollback_rows,
        },
    )
    print(json.dumps({"models": model_rows, "rollback_layers": len(rollback_rows)}, indent=2))


if __name__ == "__main__":
    main()
