#include <flash_attention_api.cuh>
#include <flash_attention_ffi.h>

#include <limits>

extern "C" cudaError_t cuda_operator_flash_attention_f32(
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
) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (batch_size <= 0 || head_num <= 0 || seq_len <= 0 || head_dim != 64) {
        return cudaErrorInvalidValue;
    }

    const auto batch_heads = static_cast<int64_t>(batch_size) * head_num;
    if (batch_heads > 65535 || batch_heads > std::numeric_limits<int32_t>::max()) {
        return cudaErrorInvalidValue;
    }

    launch_flash_attention_v0(
        q,
        k,
        v,
        output,
        batch_size,
        head_num,
        seq_len,
        head_dim,
        causal != 0,
        stream
    );
    return cudaGetLastError();
}
