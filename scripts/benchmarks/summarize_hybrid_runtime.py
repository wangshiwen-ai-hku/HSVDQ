#!/usr/bin/env python3
"""Summarize backend latency JSON files and compute speedups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.input_dir)
    files = sorted(root.glob("latency_*.json"))
    if not files:
        raise FileNotFoundError(f"no latency_*.json files found in {root}")

    rows: list[dict[str, Any]] = []
    for path in files:
        payload = read_result(path)
        metrics = payload["metrics"]
        backend = payload["runtime"]["backend"]
        if payload["runtime"]["source"] == "fp":
            backend = "dense"
        rows.append(
            {
                "backend": backend,
                "prefill_ms": metrics["prefill"]["mean_ms"],
                "decode_ms": metrics["decode_one_token"]["mean_ms"],
                "generate_tokens_per_s": metrics["generate"]["tokens_per_s"],
                "runtime": payload["runtime"],
                "hybrid_runtime": metrics.get("hybrid_runtime"),
            }
        )

    baseline = next((row for row in rows if row["backend"] == "dense"), None)
    if baseline is not None:
        for row in rows:
            row["prefill_speedup_vs_dense"] = baseline["prefill_ms"] / row["prefill_ms"]
            row["decode_speedup_vs_dense"] = baseline["decode_ms"] / row["decode_ms"]
            row["generate_speedup_vs_dense"] = (
                row["generate_tokens_per_s"] / baseline["generate_tokens_per_s"]
            )

    result = {"input_dir": str(root), "results": rows}
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
