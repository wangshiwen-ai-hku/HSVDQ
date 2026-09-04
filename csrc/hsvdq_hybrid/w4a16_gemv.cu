#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

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

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <typename scalar_t>
__global__ void lowrank_down_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ smooth,
    const scalar_t* __restrict__ l1,
    scalar_t* __restrict__ smoothed_input,
    float* __restrict__ lowrank,
    int rows,
    int k,
    int rank) {
  constexpr int kMaxRank = 8;
  __shared__ float warp_sums[8];
  const int row = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float sums[kMaxRank] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  for (int column = threadIdx.x; column < k; column += blockDim.x) {
    const scalar_t smoothed_value = from_float<scalar_t>(
        as_float(input[row * k + column]) / as_float(smooth[column]));
    smoothed_input[row * k + column] = smoothed_value;
    const float smoothed = as_float(smoothed_value);
#pragma unroll
    for (int rank_index = 0; rank_index < kMaxRank; ++rank_index) {
      if (rank_index < rank) {
        sums[rank_index] += smoothed * as_float(l1[column * rank + rank_index]);
      }
    }
  }

  for (int rank_index = 0; rank_index < rank; ++rank_index) {
    float value = warp_sum(sums[rank_index]);
    if (lane == 0) {
      warp_sums[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
      value = lane < (blockDim.x / 32) ? warp_sums[lane] : 0.0f;
      value = warp_sum(value);
      if (lane == 0) {
        lowrank[row * rank + rank_index] = value;
      }
    }
    __syncthreads();
  }
}

template <typename scalar_t>
__global__ void w4a16_gemv_kernel(
    const scalar_t* __restrict__ smoothed_input,
    const uint8_t* __restrict__ qweight,
    const scalar_t* __restrict__ wscales,
    const float* __restrict__ lowrank,
    const scalar_t* __restrict__ l2,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int rows,
    int n,
    int k,
    int groups,
    int group_size,
    int rank) {
  __shared__ float warp_sums[8];
  const int column_out = blockIdx.x;
  const int row = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float sum = 0.0f;

  for (int packed_column = threadIdx.x; packed_column < k / 2; packed_column += blockDim.x) {
    const uint8_t packed = qweight[column_out * (k / 2) + packed_column];
    int code0 = packed & 0x0f;
    int code1 = packed >> 4;
    code0 = code0 >= 8 ? code0 - 16 : code0;
    code1 = code1 >= 8 ? code1 - 16 : code1;
    const int column = packed_column * 2;
    const int group = column / group_size;
    const float scale = as_float(wscales[column_out * groups + group]);
    const float activation0 = as_float(smoothed_input[row * k + column]);
    const float activation1 = as_float(smoothed_input[row * k + column + 1]);
    sum += (activation0 * static_cast<float>(code0) +
            activation1 * static_cast<float>(code1)) * scale;
  }

  sum = warp_sum(sum);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    sum = lane < (blockDim.x / 32) ? warp_sums[lane] : 0.0f;
    sum = warp_sum(sum);
    if (lane == 0) {
      float value = sum + as_float(bias[column_out]);
      for (int rank_index = 0; rank_index < rank; ++rank_index) {
        value += lowrank[row * rank + rank_index] *
            as_float(l2[rank_index * n + column_out]);
      }
      output[row * n + column_out] = from_float<scalar_t>(value);
    }
  }
}

template <typename scalar_t>
void launch_w4a16(
    const scalar_t* input,
    const uint8_t* qweight,
    const scalar_t* wscales,
    const scalar_t* smooth,
    const scalar_t* l1,
    const scalar_t* l2,
    const scalar_t* bias,
    scalar_t* output,
    scalar_t* smoothed_input,
    float* lowrank,
    int rows,
    int n,
    int k,
    int groups,
    int group_size,
    int rank,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  lowrank_down_kernel<<<rows, kThreads, 0, stream>>>(
      input, smooth, l1, smoothed_input, lowrank, rows, k, rank);
  const dim3 grid(n, rows);
  w4a16_gemv_kernel<<<grid, kThreads, 0, stream>>>(
      smoothed_input,
      qweight,
      wscales,
      lowrank,
      l2,
      bias,
      output,
      rows,
      n,
      k,
      groups,
      group_size,
      rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor hsvdq_w4a16_forward_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor wscales,
    torch::Tensor smooth,
    torch::Tensor l1,
    torch::Tensor l2,
    torch::Tensor bias,
    torch::Tensor smoothed_workspace,
    torch::Tensor lowrank_workspace,
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
  TORCH_CHECK(smoothed_workspace.is_cuda() && smoothed_workspace.is_contiguous(),
              "smoothed workspace must be contiguous CUDA");
  TORCH_CHECK(lowrank_workspace.is_cuda() && lowrank_workspace.scalar_type() == at::kFloat &&
                  lowrank_workspace.is_contiguous(),
              "lowrank workspace must be contiguous CUDA float32");
  TORCH_CHECK(group_size > 0 && group_size % 2 == 0,
              "group_size must be positive and even");

  c10::cuda::CUDAGuard device_guard(input.device());
  const int device_index = input.get_device();
  cudaDeviceProp properties;
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device_index));
  TORCH_CHECK(properties.major >= 7, "W4A16 decode kernel requires SM70 or newer");
  if (input.scalar_type() == at::kBFloat16) {
    TORCH_CHECK(properties.major >= 8, "bfloat16 W4A16 decode requires SM80 or newer");
  }

  const int rows = static_cast<int>(input.size(0));
  const int k = static_cast<int>(input.size(1));
  const int n = static_cast<int>(qweight.size(0));
  const int rank = static_cast<int>(l1.size(1));
  TORCH_CHECK(rows > 0 && k > 0 && k % 2 == 0, "M and K must be positive; K must be even");
  TORCH_CHECK(k % group_size == 0, "K must be divisible by group_size");
  TORCH_CHECK(rank > 0 && rank <= 8, "rank must be in 1..8");
  const int groups = k / static_cast<int>(group_size);

  TORCH_CHECK(qweight.dim() == 2 && qweight.size(1) == k / 2, "qweight shape mismatch");
  TORCH_CHECK(wscales.dim() == 2 && wscales.size(0) == n && wscales.size(1) == groups,
              "wscales shape mismatch");
  TORCH_CHECK(smooth.numel() == k, "smooth shape mismatch");
  TORCH_CHECK(l1.dim() == 2 && l1.size(0) == k && l1.size(1) == rank, "l1 shape mismatch");
  TORCH_CHECK(l2.dim() == 2 && l2.size(0) == rank && l2.size(1) == n, "l2 shape mismatch");
  TORCH_CHECK(bias.numel() == n, "bias shape mismatch");
  TORCH_CHECK(lowrank_workspace.dim() == 2 && lowrank_workspace.size(0) >= rows &&
                  lowrank_workspace.size(1) == rank,
              "lowrank workspace shape mismatch");
  TORCH_CHECK(smoothed_workspace.dim() == 2 && smoothed_workspace.size(0) >= rows &&
                  smoothed_workspace.size(1) == k,
              "smoothed workspace shape mismatch");
  TORCH_CHECK(qweight.device() == input.device() && wscales.device() == input.device() &&
                  smooth.device() == input.device() && l1.device() == input.device() &&
                  l2.device() == input.device() && bias.device() == input.device() &&
                  smoothed_workspace.device() == input.device() &&
                  lowrank_workspace.device() == input.device(),
              "all tensors must be on the same CUDA device");
  TORCH_CHECK(wscales.scalar_type() == input.scalar_type() &&
                  smooth.scalar_type() == input.scalar_type() &&
                  l1.scalar_type() == input.scalar_type() &&
                  l2.scalar_type() == input.scalar_type() &&
                  bias.scalar_type() == input.scalar_type() &&
                  smoothed_workspace.scalar_type() == input.scalar_type(),
              "all floating tensors must match input dtype");

  auto output = torch::empty({rows, n}, input.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (input.scalar_type() == at::kHalf) {
    launch_w4a16(
        reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const __half*>(wscales.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(smooth.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(l1.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(l2.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(smoothed_workspace.data_ptr<at::Half>()),
        lowrank_workspace.data_ptr<float>(),
        rows, n, k, groups, static_cast<int>(group_size), rank, stream);
  } else {
    launch_w4a16(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        qweight.data_ptr<uint8_t>(),
        reinterpret_cast<const __nv_bfloat16*>(wscales.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(smooth.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(l1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(l2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(smoothed_workspace.data_ptr<at::BFloat16>()),
        lowrank_workspace.data_ptr<float>(),
        rows, n, k, groups, static_cast<int>(group_size), rank, stream);
  }
  return output;
}
