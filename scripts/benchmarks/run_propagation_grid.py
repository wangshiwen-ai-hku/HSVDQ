#!/usr/bin/env python3
"""Grid over block propagation x intra-block x outer_iters x V1/V2/V3 x rank."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent

BLOCK_INPUTS = ("quantized", "reference")
INTRA_BLOCKS = ("sequential", "fp_independent")
OUTER_ITERS = (2, 5)
VARIANTS = ("v1", "v2", "v3")
RANKS = (4, 8, 16)
V2_LAMBDA = 0.25


@dataclass(frozen=True)
class GridJob:
    calibration: str
    block_input_mode: str
    intra_block_mode: str
    outer_iters: int
    variant: str
    rank: int
    device: str


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def job_label(job: GridJob, seed: int) -> str:
    return (
        f"{job.calibration}_w4a4_r{job.rank}_{job.variant}_"
        f"{job.block_input_mode}_{job.intra_block_mode}_o{job.outer_iters}_s{seed}"
    )


def checkpoint_dir(root: Path, job: GridJob, seed: int) -> Path:
    return root / "checkpoints" / job_label(job, seed)


def metrics_dir(root: Path, job: GridJob, seed: int) -> Path:
    return root / "metrics" / job_label(job, seed)


def variant_settings(variant: str) -> dict[str, Any]:
    if variant == "v1":
        return {
            "ablation_mode": "custom",
            "code_objective": "fw",
            "joint_code_iters": 1,
            "linear_objective": "local",
            "activation_weight": 0.0,
        }
    if variant == "v2":
        return {
            "ablation_mode": "custom",
            "code_objective": "joint",
            "joint_code_iters": 2,
            "linear_objective": "local",
            "activation_weight": V2_LAMBDA,
        }
    if variant == "v3":
        return {
            "ablation_mode": "custom",
            "code_objective": "fw",
            "joint_code_iters": 1,
            "linear_objective": "cumulative",
            "activation_weight": 0.0,
        }
    raise ValueError(f"unknown variant: {variant}")


def quantize_command(job: GridJob, root: Path, args: argparse.Namespace) -> list[str]:
    settings = variant_settings(job.variant)
    command = [
        sys.executable,
        str(REPO_ROOT / "hsvdquant.py"),
        "quantize",
        "--model",
        args.model,
        "--output",
        str(checkpoint_dir(root, job, args.seed)),
        "--device",
        job.device,
        "--dtype",
        args.dtype,
        "--calib-dataset",
        job.calibration,
        "--nsamples",
        str(args.nsamples),
        "--sequence-length",
        str(args.calib_seqlen),
        "--calib-batch-size",
        str(args.calib_batch_size),
        "--activation-cache-tokens",
        str(args.activation_cache_tokens),
        "--max-layers",
        str(args.max_layers),
        "--bits",
        "4",
        "--activation-bits",
        "4",
        "--activation-group-size",
        str(args.activation_group_size),
        "--d-fa-group-size",
        "-1",
        "--rank",
        str(job.rank),
        "--rank-a",
        "0",
        "--ablation-mode",
        settings["ablation_mode"],
        "--code-objective",
        settings["code_objective"],
        "--joint-code-iters",
        str(settings["joint_code_iters"]),
        "--linear-objective",
        settings["linear_objective"],
        "--activation-weight",
        str(settings["activation_weight"]),
        "--block-input-mode",
        job.block_input_mode,
        "--intra-block-mode",
        job.intra_block_mode,
        "--trajectory-damp",
        str(args.trajectory_damp),
        "--trajectory-max-norm-ratio",
        str(args.trajectory_max_norm_ratio),
        "--trajectory-spectral-floor",
        str(args.trajectory_spectral_floor),
        "--trajectory-module-filter",
        args.trajectory_module_filter,
        "--beta",
        str(args.beta),
        "--p",
        str(args.p),
        "--group-size",
        str(args.group_size),
        "--block-size",
        str(args.block_size),
        "--outer-iters",
        str(job.outer_iters),
        "--d-mode",
        "cached",
        "--d-steps",
        str(args.d_steps),
        "--d-lr",
        str(args.d_lr),
        "--d-clip",
        str(args.d_clip),
        "--damp",
        str(args.damp),
        "--seed",
        str(args.seed),
    ]
    if job.variant == "v3" and args.trajectory_quantized_gate:
        command.append("--trajectory-quantized-gate")
    return command


def eval_command(job: GridJob, dataset: str, root: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(BENCH_DIR / "eval_ppl.py"),
        "--model",
        args.model,
        "--checkpoint",
        str(checkpoint_dir(root, job, args.seed)),
        "--dataset",
        dataset,
        "--seqlen",
        str(args.ppl_seqlen),
        "--max-samples",
        str(args.ppl_max_samples),
        "--device",
        job.device,
        "--dtype",
        args.dtype,
        "--output",
        str(metrics_dir(root, job, args.seed) / f"ppl_{dataset}.json"),
    ]


def run_logged(command: list[str], log_path: Path, env: dict[str, str], dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"+ {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {rendered}\n")
        handle.flush()
        subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_job(job: GridJob, root: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    label = job_label(job, args.seed)
    log_path = root / "logs" / f"{label}.log"
    checkpoint = checkpoint_dir(root, job, args.seed)
    if args.force or not (checkpoint / "hsvdquant_config.json").exists():
        run_logged(quantize_command(job, root, args), log_path, env, args.dry_run)
    if args.no_eval:
        return
    for dataset in args.eval_datasets:
        output = metrics_dir(root, job, args.seed) / f"ppl_{dataset}.json"
        if args.force or not output.exists():
            run_logged(eval_command(job, dataset, root, args), log_path, env, args.dry_run)


def read_ppl(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", {}).get("ppl")


def aggregate(root: Path, jobs: list[GridJob], args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        label = job_label(job, args.seed)
        row: dict[str, Any] = {
            "label": label,
            "calibration": job.calibration,
            "block_input_mode": job.block_input_mode,
            "intra_block_mode": job.intra_block_mode,
            "outer_iters": job.outer_iters,
            "variant": job.variant,
            "rank": job.rank,
            "lambda": V2_LAMBDA if job.variant == "v2" else 0.0,
            "device": job.device,
            "checkpoint": str(checkpoint_dir(root, job, args.seed)),
            "log": str(root / "logs" / f"{label}.log"),
        }
        for dataset in args.eval_datasets:
            row[f"ppl_{dataset}"] = read_ppl(metrics_dir(root, job, args.seed) / f"ppl_{dataset}.json")
        rows.append(row)

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with (summary_dir / "grid_ppl.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    completed = sum(
        1
        for job in jobs
        if all(
            (metrics_dir(root, job, args.seed) / f"ppl_{dataset}.json").exists()
            for dataset in args.eval_datasets
        )
    )
    write_json(
        summary_dir / "grid_status.json",
        {
            "total_jobs": len(jobs),
            "completed_jobs": completed,
            "pending_jobs": len(jobs) - completed,
        },
    )

    # Best per (eval_dataset, rank, variant) across propagation settings
    for dataset in args.eval_datasets:
        best_rows: list[dict[str, Any]] = []
        key = f"ppl_{dataset}"
        for rank in RANKS:
            for variant in VARIANTS:
                candidates = [row for row in rows if row["rank"] == rank and row["variant"] == variant]
                scored = [row for row in candidates if row.get(key) is not None]
                if not scored:
                    continue
                best = min(scored, key=lambda row: row[key])
                best_rows.append(
                    {
                        "eval_dataset": dataset,
                        "rank": rank,
                        "variant": variant,
                        "best_ppl": best[key],
                        "block_input_mode": best["block_input_mode"],
                        "intra_block_mode": best["intra_block_mode"],
                        "outer_iters": best["outer_iters"],
                        "label": best["label"],
                    }
                )
        if best_rows:
            with (summary_dir / f"best_by_variant_{dataset}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(best_rows[0].keys()))
                writer.writeheader()
                writer.writerows(best_rows)

    print(f"Wrote {summary_dir / 'grid_ppl.csv'} ({completed}/{len(jobs)} complete)", flush=True)


def is_heavy_job(job: GridJob) -> bool:
    return (
        job.variant in {"v2", "v3"}
        or job.outer_iters > 2
        or job.intra_block_mode == "fp_independent"
    )


def assign_device(
    job: GridJob,
    devices: list[str],
    l40_devices: list[str],
    heavy_counts: dict[str, int],
    light_counts: dict[str, int],
) -> str:
    """Prefer L40 for V2/V3, outer=5, and fp_independent jobs."""
    if is_heavy_job(job):
        pool = l40_devices
        counts = heavy_counts
    else:
        pool = devices
        counts = light_counts
    device = min(pool, key=lambda name: counts[name])
    counts[device] += 1
    return device


def build_jobs(args: argparse.Namespace) -> list[GridJob]:
    base: list[tuple[str, str, str, int, str, int]] = []
    for calibration in args.calib_datasets:
        for block_input_mode in BLOCK_INPUTS:
            for intra_block_mode in INTRA_BLOCKS:
                for outer_iters in OUTER_ITERS:
                    for variant in VARIANTS:
                        for rank in RANKS:
                            base.append(
                                (calibration, block_input_mode, intra_block_mode, outer_iters, variant, rank)
                            )
    devices = list(args.devices)
    l40_devices = list(args.l40_devices)
    heavy_counts = {device: 0 for device in l40_devices}
    light_counts = {device: 0 for device in devices}
    jobs: list[GridJob] = []
    for calibration, block_input_mode, intra_block_mode, outer_iters, variant, rank in base:
        job = GridJob(calibration, block_input_mode, intra_block_mode, outer_iters, variant, rank, "")
        device = assign_device(job, devices, l40_devices, heavy_counts, light_counts)
        jobs.append(
            GridJob(calibration, block_input_mode, intra_block_mode, outer_iters, variant, rank, device)
        )
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/propagation_grid_w4a4")
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    # cuda:0,cuda:1 = L40; cuda:2,cuda:3 = RTX 3090
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument(
        "--l40-devices",
        nargs="+",
        default=["cuda:0", "cuda:1"],
        help="L40 GPUs used preferentially for V2/V3, outer=5, and fp_independent jobs",
    )
    parser.add_argument("--calib-datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--eval-datasets", nargs="+", default=["wikitext2", "c4"])
    parser.add_argument("--trajectory-damp", type=float, default=0.1)
    parser.add_argument("--trajectory-max-norm-ratio", type=float, default=0.025)
    parser.add_argument("--trajectory-spectral-floor", type=float, default=1e-4)
    parser.add_argument("--trajectory-module-filter", default="all")
    parser.add_argument("--trajectory-quantized-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-group-size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=4)
    parser.add_argument("--activation-cache-tokens", type=int, default=2048)
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-max-samples", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--d-steps", type=int, default=20)
    parser.add_argument("--d-lr", type=float, default=0.05)
    parser.add_argument("--d-clip", type=float, default=16.0)
    parser.add_argument("--damp", type=float, default=0.01)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-layers", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-hsvdq")
    c4_train = REPO_ROOT / "results/trajectory_ablation_r16_lam025/data/c4-train.00000-of-01024.json.gz"
    c4_val = REPO_ROOT / "results/trajectory_ablation_r16_lam025/data/c4-validation.00000-of-00008.json.gz"
    if c4_train.exists():
        env.setdefault("HSVDQ_C4_TRAIN", str(c4_train))
    if c4_val.exists():
        env.setdefault("HSVDQ_C4_VALIDATION", str(c4_val))

    jobs = build_jobs(args)
    write_json(
        root / "experiment_manifest.json",
        {
            "arguments": vars(args),
            "grid": {
                "block_input_modes": BLOCK_INPUTS,
                "intra_block_modes": INTRA_BLOCKS,
                "outer_iters": OUTER_ITERS,
                "variants": VARIANTS,
                "ranks": RANKS,
                "v2_lambda": V2_LAMBDA,
                "devices": list(args.devices),
                "l40_devices": list(args.l40_devices),
            },
            "jobs": [asdict(job) | {"label": job_label(job, args.seed)} for job in jobs],
        },
    )

    if not args.aggregate_only:
        failures: list[tuple[str, str]] = []
        lock = threading.Lock()

        def worker(device: str) -> None:
            for job in [candidate for candidate in jobs if candidate.device == device]:
                try:
                    run_job(job, root, args, env)
                except Exception as error:
                    with lock:
                        failures.append((job_label(job, args.seed), repr(error)))
                    print(f"FAILED {job_label(job, args.seed)}: {error}", flush=True)

        threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in args.devices]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            write_json(root / "failures.json", failures)

    aggregate(root, jobs, args)


if __name__ == "__main__":
    main()
