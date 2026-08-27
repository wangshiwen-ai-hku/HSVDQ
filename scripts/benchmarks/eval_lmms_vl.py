#!/usr/bin/env python3
"""Run lmms-eval for Qwen2.5-VL FP or HSVDQ language-tower checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoProcessor, AutoTokenizer

from common import _dtype_from_name, environment_metadata, load_quant_checkpoint, write_json
from lmms_eval import evaluator
from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.tasks import TaskManager


@register_model("hsvdq_qwen2_5_vl")
class HSVDQQwen2_5_VL(Qwen2_5_VL):
    """Qwen2.5-VL lmms-eval backend that can restore HSVDQ checkpoints."""

    def __init__(
        self,
        pretrained: str = "models/Qwen/Qwen2.5-VL-7B-Instruct",
        checkpoint: str = "",
        device: str = "cuda:0",
        device_map: str | None = None,
        batch_size: int | str = 1,
        dtype: str = "bfloat16",
        use_cache: bool = True,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        max_num_frames: int = 32,
        fps: float | None = None,
        system_prompt: str | None = "You are a helpful assistant.",
        interleave_visuals: bool = False,
        reasoning_prompt: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ) -> None:
        if not checkpoint:
            super().__init__(
                pretrained=pretrained,
                device=device,
                device_map=device_map,
                batch_size=batch_size,
                use_cache=use_cache,
                attn_implementation=attn_implementation,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_num_frames=max_num_frames,
                fps=fps,
                system_prompt=system_prompt,
                interleave_visuals=interleave_visuals,
                reasoning_prompt=reasoning_prompt,
                **kwargs,
            )
            return

        if kwargs:
            raise ValueError(f"unexpected model kwargs for HSVDQ checkpoint: {kwargs}")
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            raise RuntimeError("HSVDQ Qwen2.5-VL wrapper currently supports single-process eval only")
        self.accelerator = accelerator
        self._device = torch.device(device)
        self.device_map = device_map if device_map else device
        model, tokenizer, _metadata = load_quant_checkpoint(
            Path(checkpoint),
            self._device,
            _dtype_from_name(dtype),
        )
        model.config.use_cache = bool(use_cache)
        self._model = model.eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(pretrained)
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_num_frames = max_num_frames
        self.fps = fps
        self.system_prompt = system_prompt
        self.interleave_visuals = interleave_visuals
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self._rank = 0
        self._world_size = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--tasks", default="mmbench_en_dev,mmmu_val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--gen-kwargs", default="temperature=0,top_p=1,num_beams=1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import lmms_eval

    task_root = Path(lmms_eval.__file__).resolve().parent / "tasks"
    task_manager = TaskManager(
        include_path=[str(task_root / "mmbench"), str(task_root / "mmmu")],
        include_defaults=False,
    )
    model_args = {
        "pretrained": args.model,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "device_map": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
    }
    results = evaluator.simple_evaluate(
        model="hsvdq_qwen2_5_vl",
        model_args=model_args,
        tasks=[task.strip() for task in args.tasks.split(",") if task.strip()],
        batch_size=args.batch_size,
        device=args.device,
        limit=None if args.limit <= 0 else args.limit,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity="INFO",
    )
    payload = {
        "runtime": {
            "model": args.model,
            "checkpoint": args.checkpoint or None,
            "tasks": args.tasks,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "limit": None if args.limit <= 0 else args.limit,
        },
        "results": results,
        "environment": environment_metadata(),
    }
    write_json(args.output, payload)
    print(json.dumps(results.get("results", {}), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
