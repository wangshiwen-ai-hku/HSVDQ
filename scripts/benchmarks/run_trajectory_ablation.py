#!/usr/bin/env python3
"""Run the isolated V1/V2/V3/V2+V3 WikiText-2/C4 PPL ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Variant:
    name: str
    activation_bits: int
    ablation_mode: str
    description: str


VARIANTS = {
    "a16": Variant("a16", 16, "v1", "same-weight-bit A16 baseline"),
    "v1": Variant("v1", 4, "v1", "original local reconstruction"),
    "v2": Variant("v2", 4, "v2", "local F_W + lambda F_A"),
    "v3": Variant("v3", 4, "v3", "teacher-student trajectory correction only"),
    "v2v3": Variant("v2v3", 4, "v2v3", "joint objective plus trajectory correction"),
}


@dataclass(frozen=True)
class Job:
    calibration: str
    bits: int
    variant: str
    device: str
    rank: int
    activation_weight: float


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def lambda_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def job_label(job: Job, args: argparse.Namespace) -> str:
    variant = VARIANTS[job.variant]
    suffix = f"_lam{lambda_tag(job.activation_weight)}" if job.variant in {"v2", "v2v3"} else ""
    if job.variant in {"v3", "v2v3"}:
        suffix += (
            f"_td{lambda_tag(args.trajectory_damp)}"
            f"_tc{lambda_tag(args.trajectory_max_norm_ratio)}"
        )
        if args.trajectory_module_filter != "all":
            suffix += f"_{args.trajectory_module_filter}"
        if args.trajectory_spectral_floor > 0:
            suffix += f"_sf{lambda_tag(args.trajectory_spectral_floor)}"
        if args.trajectory_holdout_fraction > 0:
            suffix += f"_ho{lambda_tag(args.trajectory_holdout_fraction)}"
        if args.trajectory_holdout_backtracking:
            suffix += "_bt"
        if args.trajectory_quantized_gate:
            suffix += "_qgate"
        if args.trajectory_start_layer:
            suffix += f"_start{args.trajectory_start_layer}"
        if args.trajectory_rebase:
            suffix += "_rebase"
    return (
        f"{job.calibration}_w{job.bits}a{variant.activation_bits}_r{job.rank}_"
        f"{job.variant}{suffix}_s{args.seed}"
    )


def checkpoint_dir(root: Path, job: Job, args: argparse.Namespace) -> Path:
    return root / "checkpoints" / job_label(job, args)


def metrics_dir(root: Path, job: Job, args: argparse.Namespace) -> Path:
    return root / "metrics" / job_label(job, args)


def quantize_command(job: Job, root: Path, args: argparse.Namespace) -> list[str]:
    variant = VARIANTS[job.variant]
    command = [
        sys.executable,
        "hsvdquant.py",
        "quantize",
        "--model",
        args.model,
        "--output",
        str(checkpoint_dir(root, job, args)),
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
        str(job.bits),
        "--activation-bits",
        str(variant.activation_bits),
        "--activation-group-size",
        str(args.activation_group_size),
        "--d-fa-group-size",
        "-1",
        "--rank",
        str(job.rank),
        "--rank-a",
        "0",
        "--ablation-mode",
        variant.ablation_mode,
        "--joint-code-iters",
        str(args.joint_code_iters),
        "--activation-weight",
        str(job.activation_weight),
        "--trajectory-damp",
        str(args.trajectory_damp),
        "--trajectory-max-norm-ratio",
        str(args.trajectory_max_norm_ratio),
        "--trajectory-scale",
        str(args.trajectory_scale),
        "--trajectory-start-layer",
        str(args.trajectory_start_layer),
        "--trajectory-holdout-fraction",
        str(args.trajectory_holdout_fraction),
        "--trajectory-backtrack-scales",
        *[str(scale) for scale in args.trajectory_backtrack_scales],
        "--trajectory-spectral-floor",
        str(args.trajectory_spectral_floor),
        "--trajectory-min-holdout-gain",
        str(args.trajectory_min_holdout_gain),
        "--trajectory-min-direction-cosine",
        str(args.trajectory_min_direction_cosine),
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
        str(args.outer_iters),
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
    if args.trajectory_diagnostics:
        command.append("--trajectory-diagnostics")
    if args.trajectory_rebase:
        command.append("--trajectory-rebase")
    if args.trajectory_holdout_backtracking:
        command.append("--trajectory-holdout-backtracking")
    if args.trajectory_quantized_gate:
        command.append("--trajectory-quantized-gate")
    if args.trajectory_oracle_diagnostics:
        command.append("--trajectory-oracle-diagnostics")
    return command


def eval_command(job: Job, dataset: str, root: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(BENCH_DIR / "eval_ppl.py"),
        "--model",
        args.model,
        "--checkpoint",
        str(checkpoint_dir(root, job, args)),
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
        str(metrics_dir(root, job, args) / f"ppl_{dataset}.json"),
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


def download_once(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    print(f"Downloading {url} -> {destination}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)


def prepare_c4(root: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    if args.dry_run or args.no_prefetch:
        return
    needs_train = "c4" in args.calib_datasets
    needs_validation = "c4" in args.eval_datasets
    endpoint = env["HF_ENDPOINT"].rstrip("/")
    cache = root / "data"
    if needs_train:
        train = cache / "c4-train.00000-of-01024.json.gz"
        download_once(f"{endpoint}/datasets/allenai/c4/resolve/main/en/{train.name}", train)
        env["HSVDQ_C4_TRAIN"] = str(train.resolve())
    if needs_validation:
        validation = cache / "c4-validation.00000-of-00008.json.gz"
        download_once(f"{endpoint}/datasets/allenai/c4/resolve/main/en/{validation.name}", validation)
        env["HSVDQ_C4_VALIDATION"] = str(validation.resolve())


def run_job(job: Job, root: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    label = job_label(job, args)
    log_path = root / "logs" / f"{label}.log"
    checkpoint = checkpoint_dir(root, job, args)
    if args.force or not (checkpoint / "hsvdquant_config.json").exists():
        run_logged(quantize_command(job, root, args), log_path, env, args.dry_run)
    if args.no_eval:
        return
    for dataset in args.eval_datasets:
        output = metrics_dir(root, job, args) / f"ppl_{dataset}.json"
        if args.force or not output.exists():
            run_logged(eval_command(job, dataset, root, args), log_path, env, args.dry_run)


def read_ppl(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics", {}).get("ppl")


def aggregate(root: Path, jobs: list[Job], args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, int, str, str, int, float], float] = {}
    for job in jobs:
        for dataset in args.eval_datasets:
            metric_path = metrics_dir(root, job, args) / f"ppl_{dataset}.json"
            ppl = read_ppl(metric_path)
            if ppl is None and args.baseline_root:
                fallback = metrics_dir(Path(args.baseline_root), job, args) / f"ppl_{dataset}.json"
                ppl = read_ppl(fallback)
            row = {
                "calibration": job.calibration,
                "eval_dataset": dataset,
                "bits": job.bits,
                "activation_bits": VARIANTS[job.variant].activation_bits,
                "rank": job.rank,
                "variant": job.variant,
                "lambda": job.activation_weight if job.variant in {"v2", "v2v3"} else 0.0,
                "ppl": ppl,
            }
            rows.append(row)
            if ppl is not None:
                lookup[(job.calibration, job.bits, dataset, job.variant, job.rank, job.activation_weight)] = ppl

    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "ppl_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    effects: list[dict[str, Any]] = []
    for calibration in args.calib_datasets:
        for bits in args.bits_list:
            for dataset in args.eval_datasets:
                for rank in args.rank_list:
                    for activation_weight in args.activation_weight_list:
                        values = {
                            name: lookup.get((calibration, bits, dataset, name, rank, activation_weight))
                            for name in ["a16", "v1", "v2", "v3", "v2v3"]
                        }
                        row: dict[str, Any] = {
                            "calibration": calibration,
                            "eval_dataset": dataset,
                            "bits": bits,
                            "rank": rank,
                            "lambda": activation_weight,
                            **{f"ppl_{name}": value for name, value in values.items()},
                        }
                        if values["a16"] is not None:
                            for name in ["v1", "v2", "v3", "v2v3"]:
                                if values[name] is not None:
                                    row[f"delta_{name}_vs_a16"] = values[name] - values["a16"]
                        if values["v1"] is not None:
                            for name in ["v2", "v3", "v2v3"]:
                                if values[name] is not None:
                                    row[f"effect_{name}_vs_v1"] = values[name] - values["v1"]
                        if all(values[name] is not None for name in ["v1", "v2", "v3", "v2v3"]):
                            row["v2_v3_interaction"] = (
                                values["v2v3"] - values["v2"] - values["v3"] + values["v1"]
                            )
                        if all(values[name] is not None for name in ["a16", "v1", "v3"]):
                            activation_gap = values["v1"] - values["a16"]
                            if activation_gap > 0:
                                row["v3_activation_gap_recovery"] = (
                                    values["v1"] - values["v3"]
                                ) / activation_gap
                        if all(values[name] is not None for name in ["a16", "v2", "v2v3"]):
                            residual_gap = values["v2"] - values["a16"]
                            if residual_gap > 0:
                                row["v2v3_residual_gap_recovery"] = (
                                    values["v2"] - values["v2v3"]
                                ) / residual_gap
                        effects.append(row)

    columns = sorted({key for row in effects for key in row})
    with (summary_dir / "factorial_effects.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(effects)

    report = [
        "# Teacher-Student Trajectory Ablation",
        "",
        "Negative effects mean lower (better) perplexity.",
        "",
        "| calib | eval | W bits | A16 | V1 | V2 | V3 | V2+V3 | V3 gap recovery | V2+V3 gap recovery | interaction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in effects:
        def fmt(key: str) -> str:
            value = row.get(key)
            return "" if value is None else f"{value:.4f}"

        report.append(
            f"| {row['calibration']} | {row['eval_dataset']} | {row['bits']} | "
            f"{fmt('ppl_a16')} | {fmt('ppl_v1')} | {fmt('ppl_v2')} | {fmt('ppl_v3')} | "
            f"{fmt('ppl_v2v3')} | {fmt('v3_activation_gap_recovery')} | "
            f"{fmt('v2v3_residual_gap_recovery')} | {fmt('v2_v3_interaction')} |"
        )
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--output-root", default="results/trajectory_ablation")
    parser.add_argument(
        "--baseline-root",
        default="",
        help="fallback result root for unchanged A16/V1/V2 rows when rerunning only trajectory variants",
    )
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    parser.add_argument("--calib-datasets", nargs="+", choices=["wikitext2", "c4"], default=["wikitext2", "c4"])
    parser.add_argument("--eval-datasets", nargs="+", choices=["wikitext2", "c4"], default=["wikitext2", "c4"])
    parser.add_argument("--bits-list", nargs="+", type=int, default=[4, 3])
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--rank-list", nargs="+", type=int, default=[])
    parser.add_argument("--activation-weight", type=float, default=0.25)
    parser.add_argument(
        "--activation-weight-list",
        nargs="+",
        type=float,
        default=[],
        help="lambda grid for V2/V2+V3; defaults to --activation-weight when unset",
    )
    parser.add_argument("--activation-group-size", type=int, default=128)
    parser.add_argument("--joint-code-iters", type=int, default=2)
    parser.add_argument("--trajectory-damp", type=float, default=0.1)
    parser.add_argument("--trajectory-max-norm-ratio", type=float, default=0.25)
    parser.add_argument("--trajectory-scale", type=float, default=1.0)
    parser.add_argument("--trajectory-diagnostics", action="store_true")
    parser.add_argument("--trajectory-start-layer", type=int, default=0)
    parser.add_argument("--trajectory-rebase", action="store_true")
    parser.add_argument("--trajectory-holdout-fraction", type=float, default=0.0)
    parser.add_argument("--trajectory-holdout-backtracking", action="store_true")
    parser.add_argument(
        "--trajectory-backtrack-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--trajectory-spectral-floor", type=float, default=0.0)
    parser.add_argument("--trajectory-min-holdout-gain", type=float, default=0.0)
    parser.add_argument("--trajectory-min-direction-cosine", type=float, default=-1.0)
    parser.add_argument("--trajectory-quantized-gate", action="store_true")
    parser.add_argument(
        "--trajectory-module-filter",
        choices=["all", "attention", "mlp", "down_proj"],
        default="all",
    )
    parser.add_argument("--trajectory-oracle-diagnostics", action="store_true")
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
    parser.add_argument("--outer-iters", type=int, default=2)
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
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rank_list = args.rank_list or [args.rank]
    activation_weight_list = args.activation_weight_list or [args.activation_weight]
    if any(weight <= 0 for weight in activation_weight_list) and any(
        name in {"v2", "v2v3"} for name in args.variants
    ):
        raise ValueError("V2 and V2+V3 require all activation weights > 0")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    prepare_c4(root, args, env)

    base_jobs: list[tuple[str, int, str, int, float]] = []
    for calibration in args.calib_datasets:
        for bits in args.bits_list:
            for variant in args.variants:
                for rank in rank_list:
                    weights = (
                        activation_weight_list
                        if variant in {"v2", "v2v3"}
                        else [activation_weight_list[0]]
                    )
                    for activation_weight in weights:
                        base_jobs.append((calibration, bits, variant, rank, activation_weight))
    jobs = [
        Job(calibration, bits, variant, args.devices[index % len(args.devices)], rank, activation_weight)
        for index, (calibration, bits, variant, rank, activation_weight) in enumerate(base_jobs)
    ]
    args.rank_list = rank_list
    args.activation_weight_list = activation_weight_list
    write_json(
        root / "experiment_manifest.json",
        {
            "arguments": vars(args),
            "variants": {name: asdict(variant) for name, variant in VARIANTS.items()},
            "jobs": [asdict(job) | {"label": job_label(job, args)} for job in jobs],
        },
    )
    if not args.aggregate_only:
        failures: list[tuple[str, str]] = []
        lock = threading.Lock()

        def worker(device: str) -> None:
            for job in [candidate for candidate in jobs if candidate.device == device]:
                try:
                    run_job(job, root, args, env)
                except Exception as error:  # preserve other GPUs' progress
                    with lock:
                        failures.append((job_label(job, args), repr(error)))
                    print(f"FAILED {job_label(job, args)}: {error}", flush=True)

        threads = [threading.Thread(target=worker, args=(device,), daemon=False) for device in args.devices]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            write_json(root / "failures.json", failures)
            raise RuntimeError(f"{len(failures)} ablation jobs failed; see {root / 'failures.json'}")
    summary_jobs = [
        Job(calibration, bits, variant, args.devices[index % len(args.devices)], rank, activation_weight)
        for index, (calibration, bits, variant, rank, activation_weight) in enumerate(
            (c, b, v, r, w)
            for c in args.calib_datasets
            for b in args.bits_list
            for v in VARIANTS
            for r in rank_list
            for w in (activation_weight_list if v in {"v2", "v2v3"} else [activation_weight_list[0]])
        )
    ]
    aggregate(root, summary_jobs, args)


if __name__ == "__main__":
    main()
