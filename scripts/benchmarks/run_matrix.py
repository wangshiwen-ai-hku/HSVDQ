#!/usr/bin/env python3
"""Run and aggregate the Qwen3 formal quantization matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent


def run_command(command: list[str], env: dict[str, str], skip: bool = False) -> None:
    print("+ " + " ".join(command), flush=True)
    if skip:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def checkpoint_dir(root: Path, method: str, bits: int, activation_bits: int, rank: int | None, calib: str) -> Path:
    rank_part = "norank" if rank is None else f"r{rank}"
    return root / "checkpoints" / method / f"{calib}_w{bits}a{activation_bits}_{rank_part}"


def result_dir(root: Path, label: str) -> Path:
    return root / "metrics" / label


def build_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.smoke:
        rows = [
            {"method": "hsvdquant", "bits": 4, "activation_bits": 4, "rank": 4, "calib": "synthetic"},
            {"method": "base_svdquant", "bits": 4, "activation_bits": 4, "rank": 4, "calib": "synthetic"},
            {"method": "smoothquant", "bits": 4, "activation_bits": 4, "rank": None, "calib": "synthetic"},
            {"method": "awq", "bits": 4, "activation_bits": 4, "rank": None, "calib": "synthetic"},
            {"method": "gptq", "bits": 4, "activation_bits": 4, "rank": None, "calib": "synthetic"},
            {"method": "ganq", "bits": 4, "activation_bits": 4, "rank": None, "calib": "synthetic"},
        ]
        return [row for row in rows if not args.methods or row["method"] in args.methods]
    calibs = ["wikitext2", "c4"]
    bits = args.bits_list
    ranks = args.ranks
    rows: list[dict[str, Any]] = []
    for calib in calibs:
        for bit in bits:
            for activation_bits in args.activation_bits_list:
                for rank in ranks:
                    rows.append(
                        {
                            "method": "hsvdquant",
                            "bits": bit,
                            "activation_bits": activation_bits,
                            "rank": rank,
                            "calib": calib,
                        }
                    )
                    rows.append(
                        {
                            "method": "base_svdquant",
                            "bits": bit,
                            "activation_bits": activation_bits,
                            "rank": rank,
                            "calib": calib,
                        }
                    )
                rows.append({"method": "smoothquant", "bits": bit, "activation_bits": activation_bits, "rank": None, "calib": calib})
                rows.append({"method": "awq", "bits": bit, "activation_bits": activation_bits, "rank": None, "calib": calib})
                rows.append({"method": "gptq", "bits": bit, "activation_bits": activation_bits, "rank": None, "calib": calib})
                rows.append({"method": "ganq", "bits": bit, "activation_bits": activation_bits, "rank": None, "calib": calib})
    return [row for row in rows if not args.methods or row["method"] in args.methods]


def quantize_command(row: dict[str, Any], out_dir: Path, args: argparse.Namespace) -> list[str]:
    common = [
        "--model",
        args.model,
        "--output",
        str(out_dir),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--calib-dataset",
        row["calib"],
        "--nsamples",
        str(args.nsamples),
        "--sequence-length",
        str(args.calib_seqlen),
        "--calib-batch-size",
        str(args.calib_batch_size),
        "--activation-cache-tokens",
        str(args.activation_cache_tokens),
        "--bits",
        str(row["bits"]),
        "--activation-bits",
        str(row["activation_bits"]),
        "--rank",
        str(row["rank"] or 0),
        "--group-size",
        str(args.group_size),
        "--seed",
        str(args.seed),
    ]
    if args.max_layers > 0:
        common.extend(["--max-layers", str(args.max_layers)])
    if row["method"] == "hsvdquant":
        return [
            sys.executable,
            "hsvdquant.py",
            "quantize",
            *common,
            "--beta",
            str(args.beta),
            "--p",
            str(args.p),
            "--outer-iters",
            str(args.outer_iters),
            "--d-mode",
            "cached",
            "--d-steps",
            str(args.d_steps),
        ]
    return [
        sys.executable,
        str(BENCH_DIR / "quantize_qwen_baselines.py"),
        "--method",
        row["method"],
        *common,
        "--ganq-epochs",
        str(args.ganq_epochs),
        "--scale-grid",
        *[str(value) for value in args.scale_grid],
        "--scale-clip",
        str(args.scale_clip),
    ]


def run_metrics(row: dict[str, Any], ckpt: Path, root: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    label = f"{row['method']}_{row['calib']}_w{row['bits']}a{row['activation_bits']}_{'norank' if row['rank'] is None else 'r' + str(row['rank'])}"
    out = result_dir(root, label)
    out.mkdir(parents=True, exist_ok=True)
    for dataset in ["wikitext2", "c4"]:
        path = out / f"ppl_{dataset}.json"
        if not (args.skip_existing and path.exists()):
            run_command(
                [
                    sys.executable,
                    str(BENCH_DIR / "eval_ppl.py"),
                    "--model",
                    args.model,
                    "--checkpoint",
                    str(ckpt),
                    "--dataset",
                    dataset,
                    "--seqlen",
                    str(args.ppl_seqlen),
                    "--max-samples",
                    str(args.ppl_max_samples),
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--output",
                    str(path),
                ],
                env,
                args.dry_run,
            )
    latency_path = out / "latency.json"
    if args.metrics == "ppl":
        return
    if not (args.skip_existing and latency_path.exists()):
        run_command(
            [
                sys.executable,
                str(BENCH_DIR / "bench_latency.py"),
                "--model",
                args.model,
                "--checkpoint",
                str(ckpt),
                "--prompt-len",
                str(args.prompt_len),
                "--decode-len",
                str(args.decode_len),
                "--warmup",
                str(args.latency_warmup),
                "--iters",
                str(args.latency_iters),
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--output",
                str(latency_path),
            ],
            env,
            args.dry_run,
        )
    memory_path = out / "memory.json"
    if not (args.skip_existing and memory_path.exists()):
        run_command(
            [
                sys.executable,
                str(BENCH_DIR / "bench_memory.py"),
                "--model",
                args.model,
                "--checkpoint",
                str(ckpt),
                "--dataset",
                "wikitext2",
                "--seqlen",
                str(args.ppl_seqlen),
                "--max-samples",
                "1",
                "--prompt-len",
                str(args.prompt_len),
                "--decode-len",
                "1",
                "--device",
                args.device,
                "--dtype",
                args.dtype,
                "--output",
                str(memory_path),
            ],
            env,
            args.dry_run,
        )


def run_fp_metrics(root: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    out = result_dir(root, "fp")
    out.mkdir(parents=True, exist_ok=True)
    for dataset in ["wikitext2", "c4"]:
        path = out / f"ppl_{dataset}.json"
        if not (args.skip_existing and path.exists()):
            run_command(
                [
                    sys.executable,
                    str(BENCH_DIR / "eval_ppl.py"),
                    "--model",
                    args.model,
                    "--dataset",
                    dataset,
                    "--seqlen",
                    str(args.ppl_seqlen),
                    "--max-samples",
                    str(args.ppl_max_samples),
                    "--device",
                    args.device,
                    "--dtype",
                    args.dtype,
                    "--output",
                    str(path),
                ],
                env,
                args.dry_run,
            )
    if args.metrics == "ppl":
        return
    for script, path_name in [("bench_latency.py", "latency.json"), ("bench_memory.py", "memory.json")]:
        path = out / path_name
        if args.skip_existing and path.exists():
            continue
        command = [
            sys.executable,
            str(BENCH_DIR / script),
            "--model",
            args.model,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--output",
            str(path),
        ]
        if script == "bench_latency.py":
            command.extend(
                [
                    "--prompt-len",
                    str(args.prompt_len),
                    "--decode-len",
                    str(args.decode_len),
                    "--warmup",
                    str(args.latency_warmup),
                    "--iters",
                    str(args.latency_iters),
                ]
            )
        else:
            command.extend(["--seqlen", str(args.ppl_seqlen), "--max-samples", "1", "--prompt-len", str(args.prompt_len)])
        run_command(command, env, args.dry_run)


def aggregate(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metric_dir in sorted((root / "metrics").glob("*")):
        if not metric_dir.is_dir():
            continue
        row: dict[str, Any] = {"label": metric_dir.name}
        for dataset in ["wikitext2", "c4"]:
            path = metric_dir / f"ppl_{dataset}.json"
            if path.exists():
                payload = read_json(path)
                row[f"{dataset}_ppl"] = payload["metrics"].get("ppl")
                runtime = payload.get("runtime", {})
                row.setdefault("method", runtime.get("method"))
        latency = metric_dir / "latency.json"
        if latency.exists():
            payload = read_json(latency)
            row["prefill_ms"] = payload["metrics"]["prefill"]["mean_ms"]
            row["decode_ms"] = payload["metrics"]["decode_one_token"]["mean_ms"]
            row["generate_ms"] = payload["metrics"]["generate"]["mean_ms"]
            row["decode_tokens_per_s"] = payload["metrics"]["decode_one_token"]["tokens_per_s"]
        memory = metric_dir / "memory.json"
        if memory.exists():
            payload = read_json(memory)
            row["load_peak_gb"] = payload["metrics"].get("load", {}).get("max_allocated_gb")
            row["prefill_peak_gb"] = payload["metrics"].get("prefill", {}).get("max_allocated_gb")
            row["decode_peak_gb"] = payload["metrics"].get("decode", {}).get("max_allocated_gb")
            row["ppl_peak_gb"] = payload["metrics"].get("ppl", {}).get("max_allocated_gb")
        rows.append(row)
    results_dir = root / "summary"
    results_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "label",
        "method",
        "wikitext2_ppl",
        "c4_ppl",
        "prefill_ms",
        "decode_ms",
        "generate_ms",
        "decode_tokens_per_s",
        "load_peak_gb",
        "prefill_peak_gb",
        "decode_peak_gb",
        "ppl_peak_gb",
    ]
    with (results_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    report = ["# Qwen3 Formal Quantization Evaluation", "", f"Rows aggregated: {len(rows)}", ""]
    report.append("| label | method | wikitext2 PPL | c4 PPL | prefill ms | decode ms | load GB | ppl GB |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        report.append(
            "| {label} | {method} | {wikitext2_ppl} | {c4_ppl} | {prefill_ms} | {decode_ms} | {load_peak_gb} | {ppl_peak_gb} |".format(
                **{key: row.get(key, "") for key in columns}
            )
        )
    (root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen/Qwen3-0.6B")
    parser.add_argument("--output-root", default="results/formal_qwen3_0.6b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=512)
    parser.add_argument("--calib-batch-size", type=int, default=4)
    parser.add_argument("--activation-cache-tokens", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--outer-iters", type=int, default=2)
    parser.add_argument("--d-steps", type=int, default=20)
    parser.add_argument("--ganq-epochs", type=int, default=3)
    parser.add_argument("--bits-list", type=int, nargs="+", default=[4, 3])
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--activation-bits-list", type=int, nargs="+", default=[4])
    parser.add_argument("--metrics", choices=["all", "ppl"], default="all")
    parser.add_argument("--scale-grid", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--scale-clip", type=float, default=16.0)
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-max-samples", type=int, default=0)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-iters", type=int, default=100)
    parser.add_argument("--max-layers", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-fp", action="store_true")
    parser.add_argument("--methods", nargs="+", default=[])
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.output_root)
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    write_json(root / "run_config.json", vars(args))
    if not args.aggregate_only:
        if not args.skip_fp:
            run_fp_metrics(root, args, env)
        for row in build_matrix(args):
            ckpt = checkpoint_dir(root, row["method"], row["bits"], row["activation_bits"], row["rank"], row["calib"])
            if not (args.skip_existing and (ckpt / "hsvdquant_config.json").exists()):
                if not (args.skip_existing and (ckpt / "baseline_quant_config.json").exists()):
                    run_command(quantize_command(row, ckpt, args), env, args.dry_run)
            run_metrics(row, ckpt, root, args, env)
    aggregate(root)


if __name__ == "__main__":
    main()
