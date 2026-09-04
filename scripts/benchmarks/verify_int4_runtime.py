#!/usr/bin/env python3
"""Validate packed H-SVDQuant state on CPU and Nunchaku numerics on CUDA."""

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
from hsvdquant_int4 import (  # noqa: E402
    NunchakuHSVQuantLinear,
    pack_nunchaku_lowrank,
    pack_nunchaku_scales,
    pack_nunchaku_weight,
    prepare_nunchaku_state,
    unpack_nunchaku_lowrank,
    unpack_nunchaku_scales,
    unpack_nunchaku_weight,
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
    state = make_state()
    codes = state["codes"]
    packed_weight = pack_nunchaku_weight(codes)
    torch.testing.assert_close(unpack_nunchaku_weight(packed_weight, codes.shape[1]), codes, rtol=0, atol=0)

    scales_g64 = state["scales"].repeat_interleave(2, dim=1)
    packed_scales = pack_nunchaku_scales(scales_g64)
    torch.testing.assert_close(unpack_nunchaku_scales(packed_scales), scales_g64, rtol=0, atol=0)

    down = torch.zeros(16, state["in_features"], dtype=torch.float16)
    down[:4] = (state["l1"] / state["d"][:, None]).T
    up = torch.zeros(state["out_features"], 16, dtype=torch.float16)
    up[:, :4] = state["l2"].T
    torch.testing.assert_close(
        unpack_nunchaku_lowrank(pack_nunchaku_lowrank(down, down=True), down=True),
        down,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        unpack_nunchaku_lowrank(pack_nunchaku_lowrank(up, down=False), down=False),
        up,
        rtol=0,
        atol=0,
    )

    packed = prepare_nunchaku_state(
        state,
        torch.float16,
        allow_activation_group_remap=True,
    )
    torch.testing.assert_close(
        unpack_nunchaku_weight(packed["qweight"], state["in_features"]),
        state["codes"],
        rtol=0,
        atol=0,
    )
    packed_bytes = sum(
        value.numel() * value.element_size() for value in packed.values() if torch.is_tensor(value)
    )
    dense_residual_bytes = state["in_features"] * state["out_features"] * 2
    return {
        "status": "ok",
        "packed_weight_shape": list(packed["qweight"].shape),
        "logical_rank": packed["logical_rank"],
        "physical_rank": packed["physical_rank"],
        "packed_state_bytes": packed_bytes,
        "dense_residual_bytes": dense_residual_bytes,
        "packed_to_dense_residual_ratio": packed_bytes / dense_residual_bytes,
    }


@torch.no_grad()
def cuda_checks(dtype: torch.dtype) -> dict[str, Any]:
    state = make_state(group_size=64, activation_group_size=64, dtype=dtype)
    eager = HSVQuantLinear(state, compute_dtype=dtype).to("cuda").eval()
    packed_state = prepare_nunchaku_state(state, dtype)
    packed = NunchakuHSVQuantLinear(packed_state, dtype).to("cuda").eval()
    inputs = torch.randn(2, 11, state["in_features"], device="cuda", dtype=dtype)
    reference = eager(inputs)
    actual = packed(inputs)
    torch.cuda.synchronize()
    error = (actual.float() - reference.float()).abs()
    relative_l2 = float(error.norm() / reference.float().norm().clamp_min(1e-8))
    if relative_l2 > 0.03:
        raise AssertionError(f"Nunchaku relative L2 error is too high: {relative_l2:.6f}")
    return {
        "status": "ok",
        "dtype": str(dtype),
        "max_abs_error": float(error.max()),
        "relative_l2_error": relative_l2,
        "gpu": torch.cuda.get_device_name(),
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
