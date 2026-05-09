#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

// SHARED MEMORY

constexpr int THREAD_PER_BLOCK{256};

__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    shared[tid] = d_input[index];
    __syncthreads();
    for (int i = 1; i < blockDim.x; i <<= 1) {
        if (tid % (2 * i) == 0) {
            shared[tid] += shared[tid + i];
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_output[blockIdx.x] = shared[tid];
    }
}

bool check(float* output, float* res, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(output[i] - res[i]) > 0.005) return false;
    }
    return true;
}

int main() {
    constexpr int N = 32 * 1024 * 1024;
    float* input = (float*)malloc(N * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    constexpr int block_num = N / THREAD_PER_BLOCK;
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0 * (float)drand48() - 1.0;
    for (int i = 0; i < block_num; ++i) {
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; ++j) cur += input[i * THREAD_PER_BLOCK + j];
        res[i] = cur;
    }

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);

    reduce<<<Grid, Block>>>(d_input, d_output);
    cudaMemcpy(output, d_output, block_num *sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, res, block_num)) {
        printf("the ans is right\n");
    } else {
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; ++i) {
            std::cout << res[i] << ' ';
        }
    }

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}