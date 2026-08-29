#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

#include <cstdint>

namespace wmma = nvcuda::wmma;

template <typename scalar_t>
__device__ __forceinline__ float as_float(scalar_t value);

template <>
__device__ __forceinline__ float as_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value);

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename scalar_t>
__global__ void quantize_activation_lowrank_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ smooth,
    const scalar_t* __restrict__ l1,
    uint8_t* __restrict__ packed,
    scalar_t* __restrict__ scales,
    float* __restrict__ lowrank,
    int rows,
    int k,
    int groups,
    int group_size,
    int rank) {
  __shared__ float reduction[128];
  __shared__ float smoothed_values[128];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  float lowrank_sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int group = 0; group < groups; ++group) {
    const int column = group * group_size + lane;
    float value = 0.0f;
    if (row < rows) {
      value = as_float(input[row * k + column]) / as_float(smooth[column]);
      value = as_float(from_float<scalar_t>(value));
      for (int rank_index = 0; rank_index < rank; ++rank_index) {
        lowrank_sums[rank_index] += value * as_float(l1[column * rank + rank_index]);
      }
    }
    smoothed_values[lane] = value;
    reduction[lane] = fabsf(value);
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (lane < stride) {
        reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
      }
      __syncthreads();
    }
    const scalar_t scale = from_float<scalar_t>(fmaxf(reduction[0], 1.0e-8f) / 7.0f);
    if (lane == 0) {
      scales[row * groups + group] = scale;
    }
    if (lane < group_size / 2) {
      const float scale_float = as_float(scale);
      int q0 = 0;
      int q1 = 0;
      if (row < rows && scale_float > 0.0f) {
        q0 = __float2int_rn(smoothed_values[lane * 2] / scale_float);
        q1 = __float2int_rn(smoothed_values[lane * 2 + 1] / scale_float);
      }
      q0 = q0 < -7 ? -7 : (q0 > 7 ? 7 : q0);
      q1 = q1 < -7 ? -7 : (q1 > 7 ? 7 : q1);
      packed[row * (k / 2) + group * (group_size / 2) + lane] =
          static_cast<uint8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
    __syncthreads();
  }

  for (int rank_index = 0; rank_index < rank; ++rank_index) {
    reduction[lane] = lowrank_sums[rank_index];
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
      if (lane < stride) {
        reduction[lane] += reduction[lane + stride];
      }
      __syncthreads();
    }
    if (lane == 0) {
      lowrank[row * rank + rank_index] = reduction[0];
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void w4a4_wmma_kernel(
    const uint8_t* __restrict__ activation,
    const scalar_t* __restrict__ activation_scales,
    const uint8_t* __restrict__ weight,
    const scalar_t* __restrict__ weight_scales,
    const float* __restrict__ lowrank_activation,
    const scalar_t* __restrict__ l2,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int rows,
    int padded_rows,
    int n,
    int k,
    int groups,
    int group_size,
    int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750 && __CUDA_ARCH__ < 900
  __shared__ __align__(32) uint32_t activation_tile[32];
  __shared__ __align__(32) uint32_t weight_tile[32];
  __shared__ __align__(32) int32_t accumulator_tile[64];
  const int lane = threadIdx.x;
  const int row_start = blockIdx.y * 8;
  const int column_start = blockIdx.x * 8;
  float sums[2] = {0.0f, 0.0f};

  using signed4 = wmma::experimental::precision::s4;
  using matrix_a_fragment = wmma::fragment<wmma::matrix_a, 8, 8, 32, signed4, wmma::row_major>;
  using matrix_b_fragment = wmma::fragment<wmma::matrix_b, 8, 8, 32, signed4, wmma::col_major>;
  using accumulator_fragment = wmma::fragment<wmma::accumulator, 8, 8, 32, int>;

  for (int group = 0; group < groups; ++group) {
    accumulator_fragment accumulator;
    wmma::fill_fragment(accumulator, 0);
    for (int group_k = 0; group_k < group_size; group_k += 32) {
      const int tile_line = lane >> 2;
      const int word = lane & 3;
      const int row = row_start + tile_line;
      const int column = column_start + tile_line;
      const int packed_k = (group * group_size + group_k) / 2;
      activation_tile[lane] = reinterpret_cast<const uint32_t*>(
          activation + row * (k / 2) + packed_k)[word];
      weight_tile[lane] = column < n
          ? reinterpret_cast<const uint32_t*>(weight + column * (k / 2) + packed_k)[word]
          : 0u;
      __syncwarp();
      matrix_a_fragment a_fragment;
      matrix_b_fragment b_fragment;
      wmma::load_matrix_sync(a_fragment, reinterpret_cast<const signed4*>(activation_tile), 32);
      wmma::load_matrix_sync(b_fragment, reinterpret_cast<const signed4*>(weight_tile), 32);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
      __syncwarp();
    }
    wmma::store_matrix_sync(accumulator_tile, accumulator, 8, wmma::mem_row_major);
    __syncwarp();
    for (int item = 0; item < 2; ++item) {
      const int index = lane + item * 32;
      const int tile_row = index >> 3;
      const int tile_column = index & 7;
      const int row = row_start + tile_row;
      const int column = column_start + tile_column;
      if (row < rows && column < n) {
        sums[item] += static_cast<float>(accumulator_tile[index]) *
            as_float(activation_scales[row * groups + group]) *
            as_float(weight_scales[column * groups + group]);
      }
    }
    __syncwarp();
  }

  for (int item = 0; item < 2; ++item) {
    const int index = lane + item * 32;
    const int tile_row = index >> 3;
    const int tile_column = index & 7;
    const int row = row_start + tile_row;
    const int column = column_start + tile_column;
    if (row < rows && column < n) {
      float value = sums[item] + as_float(bias[column]);
      for (int rank_index = 0; rank_index < rank; ++rank_index) {
        value += lowrank_activation[row * rank + rank_index] *
            as_float(l2[rank_index * n + column]);
      }
      output[row * n + column] = from_float<scalar_t>(value);
    }
  }
#endif
}

template <typename scalar_t>
void launch_forward(
    const scalar_t* input,
    const uint8_t* qweight,
    const scalar_t* wscales,
    const scalar_t* smooth,
    const scalar_t* l1,
    const scalar_t* l2,
    const scalar_t* bias,
    scalar_t* output,
    int rows,
    int padded_rows,
    int n,
    int k,
    int groups,
    int group_size,
    int rank,
    uint8_t* qactivation,
    scalar_t* activation_scales,
    float* lowrank_activation,
    cudaStream_t stream) {
  quantize_activation_lowrank_kernel<<<padded_rows, 128, 0, stream>>>(
      input,
      smooth,
      l1,
      qactivation,
      activation_scales,
      lowrank_activation,
      rows,
      k,
      groups,
      group_size,
      rank);
  dim3 gemm_grid((n + 7) / 8, (rows + 7) / 8);
  w4a4_wmma_kernel<<<gemm_grid, 32, 0, stream>>>(
      qactivation,
      activation_scales,
      qweight,
      wscales,
      lowrank_activation,
      l2,
      bias,
      output,
      rows,
      padded_rows,
      n,
      k,
      groups,
      group_size,
      rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor hsvdq_w4a4_forward_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor wscales,
    torch::Tensor smooth,
    torch::Tensor l1,
    torch::Tensor l2,
    torch::Tensor bias,
    int64_t group_size) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(), "input must be contiguous [M, K]");
  TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "input must be float16 or bfloat16");
  TORCH_CHECK(qweight.is_cuda() && qweight.scalar_type() == at::kByte && qweight.is_contiguous(),
              "qweight must be contiguous CUDA uint8");
  TORCH_CHECK(wscales.is_cuda() && wscales.is_contiguous(), "wscales must be contiguous CUDA");
  TORCH_CHECK(smooth.is_cuda() && smooth.is_contiguous(), "smooth must be contiguous CUDA");
  TORCH_CHECK(l1.is_cuda() && l1.is_contiguous(), "l1 must be contiguous CUDA");
  TORCH_CHECK(l2.is_cuda() && l2.is_contiguous(), "l2 must be contiguous CUDA");
  TORCH_CHECK(bias.is_cuda() && bias.is_contiguous(), "bias must be contiguous CUDA");
  TORCH_CHECK(qweight.device() == input.device() && wscales.device() == input.device() &&
                  smooth.device() == input.device() && l1.device() == input.device() &&
                  l2.device() == input.device() && bias.device() == input.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(group_size == 128, "current W4A4 kernel requires group_size=128");
  TORCH_CHECK(wscales.scalar_type() == input.scalar_type() && smooth.scalar_type() == input.scalar_type() &&
                  l1.scalar_type() == input.scalar_type() && l2.scalar_type() == input.scalar_type() &&
                  bias.scalar_type() == input.scalar_type(),
              "all floating tensors must match input dtype");

  c10::cuda::CUDAGuard device_guard(input.device());
  cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, input.get_device()));
  const int capability = properties.major * 10 + properties.minor;
  TORCH_CHECK(capability >= 75 && capability < 90,
              "sub-byte WMMA kernel requires SM75..SM89; got SM", capability);

  const int rows = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(qweight.size(0));
  const int rank = static_cast<int>(l1.size(1));
  const int groups = k / static_cast<int>(group_size);
  const int padded_rows = ((rows + 7) / 8) * 8;
  TORCH_CHECK(rows > 0 && k % 128 == 0, "M must be positive and K divisible by 128");
  TORCH_CHECK(n % 8 == 0, "N must be divisible by 8");
  TORCH_CHECK(rank > 0 && rank <= 8, "rank must be in 1..8");
  TORCH_CHECK(qweight.dim() == 2 && qweight.size(0) == n && qweight.size(1) == k / 2,
              "qweight shape mismatch");
  TORCH_CHECK(wscales.dim() == 2 && wscales.size(0) == n && wscales.size(1) == groups,
              "wscales shape mismatch");
  TORCH_CHECK(smooth.numel() == k, "smooth shape mismatch");
  TORCH_CHECK(l1.dim() == 2 && l1.size(0) == k && l1.size(1) == rank, "l1 shape mismatch");
  TORCH_CHECK(l2.dim() == 2 && l2.size(0) == rank && l2.size(1) == n, "l2 shape mismatch");
  TORCH_CHECK(bias.numel() == n, "bias shape mismatch");

  auto output = torch::empty({rows, n}, input.options());
  auto qactivation = torch::zeros({padded_rows, k / 2}, input.options().dtype(at::kByte));
  auto activation_scales = torch::zeros({padded_rows, groups}, input.options());
  auto lowrank_activation = torch::zeros({padded_rows, rank}, input.options().dtype(at::kFloat));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  if (input.scalar_type() == at::kHalf) {
    launch_forward(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(wscales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(smooth.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(l1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(l2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        rows, padded_rows, n, k, groups, group_size, rank,
        qactivation.data_ptr<uint8_t>(),
        reinterpret_cast<__half*>(activation_scales.data_ptr<at::Half>()),
        lowrank_activation.data_ptr<float>(), stream);
  } else {
    launch_forward(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(wscales.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(smooth.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(l1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(l2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        rows, padded_rows, n, k, groups, group_size, rank,
        qactivation.data_ptr<uint8_t>(),
        reinterpret_cast<__nv_bfloat16*>(activation_scales.data_ptr<at::BFloat16>()),
        lowrank_activation.data_ptr<float>(), stream);
  }
  return output;
}
