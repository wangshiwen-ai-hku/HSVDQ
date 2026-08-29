#include <torch/extension.h>

torch::Tensor hsvdq_w4a4_forward_cuda(
    torch::Tensor input,
    torch::Tensor qweight,
    torch::Tensor wscales,
    torch::Tensor smooth,
    torch::Tensor l1,
    torch::Tensor l2,
    torch::Tensor bias,
    int64_t group_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "w4a4_forward",
      &hsvdq_w4a4_forward_cuda,
      "Fused H-SVDQuant W4A4 forward (CUDA)");
}
