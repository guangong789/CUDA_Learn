#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <unistd.h>

#include <flash_attention_global.cuh>

constexpr int BM_BATCH_SIZE{4};
constexpr int BM_HEAD_NUM{8};
constexpr int BM_SEQ_LEN{1024};
constexpr int BM_HEAD_DIM{64};
constexpr int WARMUP{20};
constexpr int ITER{100};

double calc_tflops(float ms, bool causal) {
    double score_count = causal
        ? static_cast<double>(BM_SEQ_LEN) * (BM_SEQ_LEN + 1) / 2.0
        : static_cast<double>(BM_SEQ_LEN) * BM_SEQ_LEN;
    double flop = 4.0 * BM_BATCH_SIZE * BM_HEAD_NUM * score_count * BM_HEAD_DIM;
    return flop / (ms * 1e9);
}

float benchmark(
    const char* name,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_output,
    bool causal
) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP; ++i) {
        launch_flash_attention_v0(d_q, d_k, d_v, d_output,
            BM_BATCH_SIZE, BM_HEAD_NUM, BM_SEQ_LEN, BM_HEAD_DIM, causal);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITER; ++i) {
        launch_flash_attention_v0(d_q, d_k, d_v, d_output,
            BM_BATCH_SIZE, BM_HEAD_NUM, BM_SEQ_LEN, BM_HEAD_DIM, causal);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    float avg_ms = total_ms / ITER;

    printf("%-24s time=%8.4f ms  TFLOPS=%8.2f\n",
        name, avg_ms, calc_tflops(avg_ms, causal));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return avg_ms;
}

int main() {
    constexpr size_t tensor_size = static_cast<size_t>(BM_BATCH_SIZE)
        * BM_HEAD_NUM * BM_SEQ_LEN * BM_HEAD_DIM;
    constexpr size_t bytes = tensor_size * sizeof(float);

    float* h_q = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    float* h_k = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    float* h_v = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    random_tensor(h_q, tensor_size);
    random_tensor(h_k, tensor_size);
    random_tensor(h_v, tensor_size);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_output = nullptr;
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    printf("===== Flash Attention Benchmark =====\n");
    printf("B=%d  H=%d  N=%d  D=%d\n\n",
        BM_BATCH_SIZE, BM_HEAD_NUM, BM_SEQ_LEN, BM_HEAD_DIM);
    printf("Warming up GPU Power State (P-State Transition)...\n");
    for (int i = 0; i < 50; ++i) {
        launch_flash_attention_v0(d_q, d_k, d_v, d_output,
            BM_BATCH_SIZE, BM_HEAD_NUM, BM_SEQ_LEN, BM_HEAD_DIM, false);
    }
    cudaDeviceSynchronize();
    printf("Stabilizing thermal conditions...\n");
    sleep(1);

    benchmark("flash_attention_v0", d_q, d_k, d_v, d_output, false);
    benchmark("flash_attention_v0 causal", d_q, d_k, d_v, d_output, true);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);
    free(h_q);
    free(h_k);
    free(h_v);

    return 0;
}
