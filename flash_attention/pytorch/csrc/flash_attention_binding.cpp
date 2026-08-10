#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <flash_attention_api.cuh>

namespace cuda_operator {
    at::Tensor flash_attention_cpu(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
        bool causal) {
        TORCH_CHECK(false,
            "flash_attention is a CUDA-only operator; move q, k and v to a CUDA device");
    }

    at::Tensor flash_attention_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
        bool causal) {
        TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
        TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
        TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
        TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
            "q, k and v must be on the same CUDA device");

        TORCH_CHECK(q.scalar_type() == at::kFloat,
            "q must have dtype torch.float32, got ", q.scalar_type());
        TORCH_CHECK(k.scalar_type() == at::kFloat,
            "k must have dtype torch.float32, got ", k.scalar_type());
        TORCH_CHECK(v.scalar_type() == at::kFloat,
            "v must have dtype torch.float32, got ", v.scalar_type());

        TORCH_CHECK(q.dim() == 4, "q must have shape [B, H, N, D], got ", q.sizes());
        TORCH_CHECK(k.dim() == 4, "k must have shape [B, H, N, D], got ", k.sizes());
        TORCH_CHECK(v.dim() == 4, "v must have shape [B, H, N, D], got ", v.sizes());
        TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
            "q, k and v must have the same shape, got q=", q.sizes(),
            ", k=", k.sizes(), ", v=", v.sizes());

        TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
        TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
        TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
        TORCH_CHECK(q.size(0) > 0, "batch size must be greater than zero");
        TORCH_CHECK(q.size(1) > 0, "head count must be greater than zero");
        TORCH_CHECK(q.size(2) > 0, "sequence length must be greater than zero");
        TORCH_CHECK(q.size(3) == 64,
            "flash_attention currently requires head_dim=64, got ", q.size(3));
        TORCH_CHECK(q.size(0) * q.size(1) <= 65535,
            "batch_size * head_num must not exceed CUDA grid.y limit 65535");
        TORCH_CHECK(q.size(0) <= INT_MAX && q.size(1) <= INT_MAX && q.size(2) <= INT_MAX,
            "tensor dimensions exceed the range supported by this kernel");
        TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(),
            "flash_attention is forward-only; pass detached tensors when gradients are not needed");

        const c10::cuda::CUDAGuard device_guard(q.device());
        at::Tensor output = at::empty_like(q);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();

        launch_flash_attention_v0(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            output.data_ptr<float>(),
            static_cast<int>(q.size(0)),
            static_cast<int>(q.size(1)),
            static_cast<int>(q.size(2)),
            static_cast<int>(q.size(3)),
            causal,
            stream
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        return output;
    }

}  // namespace cuda_operator

TORCH_LIBRARY(cuda_operator, m) {
    m.def("flash_attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_operator, CPU, m) {
    m.impl("flash_attention", &cuda_operator::flash_attention_cpu);
}

TORCH_LIBRARY_IMPL(cuda_operator, CUDA, m) {
    m.impl("flash_attention", &cuda_operator::flash_attention_cuda);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
