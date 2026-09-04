#!/usr/bin/env python3
"""Validate hybrid dispatch, packing, and optional CUDA kernel numerics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import _dequantize_codes  # noqa: E402
from hsvdquant_cuda import unpack_signed_codes  # noqa: E402
from hsvdquant_hybrid import (  # noqa: E402
    HybridHSVQuantLinear,
    W4A16HSVQuantLinear,
    prepare_w4a16_state,
    select_hybrid_kernel,
)


def make_state(
    *,
    in_features: int = 256,
    out_features: int = 256,
    rank: int = 4,
    group_size: int = 64,
    activation_group_size: int = 64,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(29)
    groups = in_features // group_size
    return {
        "in_features": in_features,
        "out_features": out_features,
        "bits": 4,
        "activation_bits": 4,
        "group_size": group_size,
        "activation_group_size": activation_group_size,
        "d": (0.75 + 0.5 * torch.rand(in_features, generator=generator)).to(dtype),
        "codes": torch.randint(
            -7, 8, (out_features, in_features), generator=generator, dtype=torch.int8
        ),
        "scales": (0.005 + 0.02 * torch.rand(out_features, groups, generator=generator)).to(dtype),
        "l1": (0.02 * torch.randn(in_features, rank, generator=generator)).to(dtype),
        "l2": (0.02 * torch.randn(rank, out_features, generator=generator)).to(dtype),
        "bias": (0.01 * torch.randn(out_features, generator=generator)).to(dtype),
    }


def cpu_checks() -> dict[str, Any]:
    state = make_state(group_size=128, activation_group_size=128)
    packed = prepare_w4a16_state(state, torch.float16)
    torch.testing.assert_close(
        unpack_signed_codes(packed["qweight_decode"], 4),
        state["codes"],
        rtol=0,
        atol=0,
    )
    expected_dispatch = {
        "1": "w4a16",
        "127": "w4a16",
        "128": "w4a4",
        "256": "w4a4",
    }
    actual_dispatch = {
        rows: select_hybrid_kernel("auto", 128, int(rows)) for rows in expected_dispatch
    }
    if actual_dispatch != expected_dispatch:
        raise AssertionError(f"dispatch mismatch: {actual_dispatch}")
    assert select_hybrid_kernel("force_w4a4", 128, 1) == "w4a4"
    assert select_hybrid_kernel("force_w4a16", 128, 2048) == "w4a16"

    packed_bytes = sum(
        value.numel() * value.element_size() for value in packed.values() if torch.is_tensor(value)
    )
    dense_bytes = state["in_features"] * state["out_features"] * 2
    return {
        "status": "ok",
        "dispatch": actual_dispatch,
        "decode_packed_shape": list(packed["qweight_decode"].shape),
        "decode_state_bytes": packed_bytes,
        "dense_residual_bytes": dense_bytes,
        "decode_to_dense_residual_ratio": packed_bytes / dense_bytes,
    }


def w4a16_reference(inputs: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    residual = _dequantize_codes(
        state["codes"], state["scales"], int(state["group_size"])
    ).to(device=inputs.device, dtype=inputs.dtype)
    d = state["d"].to(device=inputs.device, dtype=inputs.dtype)
    l1 = state["l1"].to(device=inputs.device, dtype=inputs.dtype)
    l2 = state["l2"].to(device=inputs.device, dtype=inputs.dtype)
    bias = state["bias"].to(device=inputs.device, dtype=inputs.dtype)
    smoothed = inputs / d
    return F.linear(smoothed, residual, bias) + (smoothed @ l1) @ l2


@torch.no_grad()
def cuda_checks(dtype: torch.dtype, require_nunchaku: bool) -> dict[str, Any]:
    state = make_state(dtype=dtype)
    decode = W4A16HSVQuantLinear(state, dtype).to("cuda").eval()
    cases: dict[str, Any] = {}
    tolerance = 0.006 if dtype == torch.float16 else 0.025
    for rows in (1, 4, 17, 128):
        inputs = torch.randn(rows, state["in_features"], device="cuda", dtype=dtype)
        reference = w4a16_reference(inputs, state)
        actual = decode(inputs)
        torch.cuda.synchronize()
        error = (actual.float() - reference.float()).abs()
        relative_l2 = float(error.norm() / reference.float().norm().clamp_min(1e-8))
        if relative_l2 > tolerance:
            raise AssertionError(
                f"W4A16 rows={rows} relative L2 error is too high: {relative_l2:.6f}"
            )
        cases[str(rows)] = {
            "max_abs_error": float(error.max()),
            "relative_l2_error": relative_l2,
        }

    result: dict[str, Any] = {
        "status": "ok",
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "w4a16": cases,
    }
    try:
        hybrid = HybridHSVQuantLinear(
            state,
            dtype,
            threshold=128,
            profile_stats=True,
        ).to("cuda").eval()
    except RuntimeError as error:
        if require_nunchaku:
            raise
        result["hybrid"] = {"status": "skipped", "reason": str(error)}
        return result

    small = torch.randn(1, state["in_features"], device="cuda", dtype=dtype)
    large = torch.randn(128, state["in_features"], device="cuda", dtype=dtype)
    hybrid(small)
    hybrid(large)
    torch.cuda.synchronize()
    stats = hybrid.runtime_stats()
    if stats["kernel_calls"] != {"w4a4": 1, "w4a16": 1}:
        raise AssertionError(f"hybrid dispatch counters are wrong: {stats}")
    result["hybrid"] = {"status": "ok", "stats": stats}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-nunchaku", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {"cpu": cpu_checks()}
    if torch.cuda.is_available():
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        result["cuda"] = cuda_checks(dtype, args.require_nunchaku)
    elif args.require_cuda:
        raise RuntimeError("CUDA self-test was required but no CUDA device is available")
    else:
        result["cuda"] = {"status": "skipped", "reason": "CUDA is not available"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
