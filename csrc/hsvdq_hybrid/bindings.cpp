#include <torch/extension.h>

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
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "w4a16_forward",
      &hsvdq_w4a16_forward_cuda,
      "Packed H-SVDQuant W4A16 decode forward (CUDA)");
}
