#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sgemm_global.cuh>

constexpr int WARMUP = 10;
constexpr int ITER = 100;

double calc_gflops(float ms) {
    double flop = 2.0 * M * N * K;
    return flop / (ms * 1e6);
}

void benchmark(const char* name, void (*kernel)(float*, float*, float*), float* dA, float* dB, float* dC) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP; ++i) {
        kernel(dA, dB, dC);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITER; ++i) {
        kernel(dA, dB, dC);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);

    float avg_ms = total_ms / ITER;
    double gflops = calc_gflops(avg_ms);

    printf("%-20s time=%8.4f ms  GFLOPS=%10.2f\n", name, avg_ms, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    constexpr size_t memA = M * K_PAD * sizeof(float);
    constexpr size_t memB = K * N_PAD * sizeof(float);
    constexpr size_t memC = M * N_PAD * sizeof(float);

    float* hA = (float*)calloc(M * K_PAD, sizeof(float));
    float* hB = (float*)calloc(K * N_PAD, sizeof(float));

    random_m(M, K, hA);
    random_m(K, N, hB);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    cudaMalloc(&dA, memA);
    cudaMalloc(&dB, memB);
    cudaMalloc(&dC, memC);

    cudaMemcpy(dA, hA, memA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, memB, cudaMemcpyHostToDevice);

    printf("===== SGEMM Benchmark =====\n");
    printf("M=%d  K=%d  N=%d\n\n", M, K, N);
    benchmark("sgemm_v5", launch_sgemm_v5, dA, dB, dC);
    benchmark("cuBLAS", launch_sgemm_cublas, dA, dB, dC);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);

    return 0;
}