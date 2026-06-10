#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <unistd.h>

#include <softmax_global.cuh>

constexpr int WARMUP = 50;
constexpr int ITER   = 1000;

double calc_bandwidth(float ms) {
    double bytes = 2.0 * M * N * sizeof(float);
    return bytes / (ms * 1e6);
}

float benchmark(const char* name, void (*kernel)(float*, float*), float* d_input, float* d_output, float baseline_ms = -1.f) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP; i++) {
        kernel(d_input, d_output);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < ITER; i++) {
        kernel(d_input, d_output);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.f;
    cudaEventElapsedTime(&total_ms, start, stop);

    float avg_ms = total_ms / ITER;
    double bw = calc_bandwidth(avg_ms);

    if (baseline_ms < 0.f) {
        printf("%-20s time=%8.4f ms  BW=%8.2f GB/s\n", name, avg_ms, bw);
    } else {
        printf( "%-20s time=%8.4f ms  BW=%8.2f GB/s  speedup=%6.2fx\n", name, avg_ms, bw, baseline_ms / avg_ms);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return avg_ms;
}

int main() {
    constexpr size_t bytes = M * N * sizeof(float);
    float* h_input = (float*)calloc(M * N, sizeof(float));
    random_m(M, N, h_input);

    float* d_input  = nullptr;
    float* d_output = nullptr;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    printf("===== Softmax Benchmark =====\n");
    printf("M=%d  N=%d\n\n", M, N);
    printf("Warming up GPU Power State ""(P-State Transition)...\n");
    for (int i = 0; i < 50; i++) {
        launch_softmax_v0(d_input, d_output);
    }
    cudaDeviceSynchronize();
    printf("Stabilizing thermal conditions...\n");
    sleep(1);
    float baseline_ms = benchmark("softmax_v0", launch_softmax_v0, d_input, d_output);
    benchmark("softmax_v1", launch_softmax_v1, d_input, d_output);
    benchmark("softmax_v2", launch_softmax_v2, d_input, d_output);
    benchmark("softmax_v3", launch_softmax_v3, d_input, d_output);
    benchmark("softmax_v4", launch_softmax_v4, d_input, d_output);
    benchmark("softmax_v5", launch_softmax_v5, d_input, d_output);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}