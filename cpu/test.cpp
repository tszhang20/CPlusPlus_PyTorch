#include "test.h"

// 前向传播，两个Tensor相加
torch::Tensor Test_forward_cpu(const torch::Tensor& x, const torch::Tensor& y) {
    AT_ASSERTM(x.sizes() == y.sizes(), "x must be the same size as y");
    torch::Tensor z = torch::zeros(x.sizes());
    z = 2 * x + y;
    return z;
}
// 反向传播
std::vector<torch::Tensor> Test_backward_cpu(const torch::Tensor& gradOutput) {
    // z = 2 * x + y，对x和y的导数分别是2和1
    torch::Tensor gradOutputX = 2 * gradOutput * torch::ones(gradOutput.sizes());
    torch::Tensor gradOutputY = gradOutput * torch::ones(gradOutput.sizes());
    return {gradOutputX, gradOutputY};
}
// pybind11绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &Test_forward_cpu, "TEST forward");
    m.def("backward", &Test_backward_cpu, "TEST backward");
}
