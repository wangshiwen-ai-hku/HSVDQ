#!/usr/bin/env python3
"""Load an HSVDQ checkpoint and evaluate PPL with overridden activation bits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import (
    compute_ppl,
    environment_metadata,
    get_eval_input_ids,
    load_experiment_model,
    write_json,
)
from hsvdquant import HSVQuantLinear


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--infer-activation-bits", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, runtime = load_experiment_model(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device,
        dtype=torch.bfloat16,
    )
    n = 0
    for module in model.modules():
        if isinstance(module, HSVQuantLinear):
            module.activation_bits = int(args.infer_activation_bits)
            n += 1
    print(f"forced A{args.infer_activation_bits} on {n} modules", flush=True)
    input_ids = get_eval_input_ids("wikitext2", tokenizer, args.seqlen, None)
    metrics = compute_ppl(model, input_ids, args.seqlen, device, None)
    metrics.update(
        dataset="wikitext2",
        seqlen=args.seqlen,
        inference_activation_bits=args.infer_activation_bits,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        out,
        {
            "checkpoint": args.checkpoint,
            "runtime": runtime.__dict__,
            "metrics": metrics,
            "environment": environment_metadata(),
        },
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
