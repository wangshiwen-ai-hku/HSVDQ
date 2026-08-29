#!/usr/bin/env python3
"""Validate native H-SVDQuant packing on CPU and W4A4 numerics on CUDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hsvdquant import HSVQuantLinear  # noqa: E402
from hsvdquant_cuda import (  # noqa: E402
    HSVDCudaLinear,
    PackedQuantSpec,
    pack_signed_codes,
    prepare_hsvdq_cuda_state,
    resolve_kernel,
    unpack_signed_codes,
)


def make_state(
    *,
    in_features: int = 256,
    out_features: int = 256,
    rank: int = 4,
    group_size: int = 128,
    activation_group_size: int = 128,
    dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(17)
    groups = in_features // group_size
    return {
        "in_features": in_features,
        "out_features": out_features,
        "bits": 4,
        "activation_bits": 4,
        "group_size": group_size,
        "activation_group_size": activation_group_size,
        "d": (0.75 + 0.5 * torch.rand(in_features, generator=generator)).to(dtype),
        "codes": torch.randint(-7, 8, (out_features, in_features), generator=generator, dtype=torch.int8),
        "scales": (0.005 + 0.02 * torch.rand(out_features, groups, generator=generator)).to(dtype),
        "l1": (0.02 * torch.randn(in_features, rank, generator=generator)).to(dtype),
        "l2": (0.02 * torch.randn(rank, out_features, generator=generator)).to(dtype),
        "bias": (0.01 * torch.randn(out_features, generator=generator)).to(dtype),
    }


def cpu_checks() -> dict[str, Any]:
    generator = torch.Generator().manual_seed(3)
    roundtrips: dict[str, list[int]] = {}
    for bits in (2, 4, 8):
        qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        codes = torch.randint(qmin, qmax + 1, (7, 256), generator=generator, dtype=torch.int8)
        packed = pack_signed_codes(codes, bits)
        torch.testing.assert_close(unpack_signed_codes(packed, bits), codes, rtol=0, atol=0)
        roundtrips[f"int{bits}"] = list(packed.shape)

    state = make_state()
    packed = prepare_hsvdq_cuda_state(state, torch.float16)
    torch.testing.assert_close(
        unpack_signed_codes(packed["qweight"], 4), state["codes"], rtol=0, atol=0
    )
    assert packed["spec"].activation_group_size == 128
    assert packed["spec"].rank == 4
    assert packed["kernel_name"] == "w4a4_g128_wmma_sm75_89"

    permuted_state = dict(state)
    permuted_state["activation_permutation"] = torch.arange(
        state["in_features"] - 1, -1, -1
    )
    try:
        prepare_hsvdq_cuda_state(permuted_state, torch.float16)
    except ValueError as error:
        assert "does not yet fuse V3 activation permutations" in str(error)
    else:
        raise AssertionError("native packing silently accepted an unsupported V3 permutation")

    hadamard_state = dict(state)
    hadamard_state["activation_hadamard_group_size"] = 128
    hadamard_state["activation_hadamard_signs"] = torch.ones(state["in_features"])
    try:
        prepare_hsvdq_cuda_state(hadamard_state, torch.float16)
    except ValueError as error:
        assert "does not yet fuse V3 block Hadamard transforms" in str(error)
    else:
        raise AssertionError("native packing silently accepted an unsupported block Hadamard")

    try:
        resolve_kernel(PackedQuantSpec(4, 8, 128, 128, 4))
    except ValueError as error:
        assert "no native kernel registered" in str(error)
    else:
        raise AssertionError("unsupported W4A8 spec did not fail explicitly")

    tensor_bytes = sum(
        value.numel() * value.element_size() for value in packed.values() if torch.is_tensor(value)
    )
    dense_residual_bytes = state["in_features"] * state["out_features"] * 2
    return {
        "status": "ok",
        "roundtrip_shapes": roundtrips,
        "kernel": packed["kernel_name"],
        "packed_state_bytes": tensor_bytes,
        "dense_residual_bytes": dense_residual_bytes,
        "packed_to_dense_residual_ratio": tensor_bytes / dense_residual_bytes,
    }


@torch.no_grad()
def cuda_checks(dtype: torch.dtype) -> dict[str, Any]:
    state = make_state(dtype=dtype)
    eager = HSVQuantLinear(state, compute_dtype=dtype).to("cuda").eval()
    native = HSVDCudaLinear.from_state(state, dtype).to("cuda").eval()
    cases = {}
    for rows in (1, 17, 22):
        inputs = torch.randn(rows, state["in_features"], device="cuda", dtype=dtype)
        reference = eager(inputs)
        actual = native(inputs)
        torch.cuda.synchronize()
        error = (actual.float() - reference.float()).abs()
        relative_l2 = float(error.norm() / reference.float().norm().clamp_min(1e-8))
        if relative_l2 > 0.035:
            raise AssertionError(f"rows={rows}: native relative L2 error is too high: {relative_l2:.6f}")
        cases[str(rows)] = {
            "max_abs_error": float(error.max()),
            "relative_l2_error": relative_l2,
        }
    return {
        "status": "ok",
        "dtype": str(dtype),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result: dict[str, Any] = {"cpu": cpu_checks()}
    if torch.cuda.is_available():
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        result["cuda"] = cuda_checks(dtype)
    elif args.require_cuda:
        raise RuntimeError("CUDA self-test was required but no CUDA device is available")
    else:
        result["cuda"] = {"status": "skipped", "reason": "CUDA is not available"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
