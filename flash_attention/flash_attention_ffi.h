#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Launches the FP32 Flash Attention forward kernel on device-resident tensors.
// q, k, v and output use the contiguous [B, H, N, D] layout.
cudaError_t cuda_operator_flash_attention_f32(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int32_t batch_size,
    int32_t head_num,
    int32_t seq_len,
    int32_t head_dim,
    int32_t causal,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif
