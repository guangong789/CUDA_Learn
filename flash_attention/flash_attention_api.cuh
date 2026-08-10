#pragma once

#include <cuda_runtime_api.h>

void launch_flash_attention_v0(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int head_num,
    int seq_len,
    int head_dim,
    bool causal = false,
    cudaStream_t stream = nullptr
);
