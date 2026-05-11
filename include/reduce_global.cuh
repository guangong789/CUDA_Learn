#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr int N = 32 * 1024 * 1024;
constexpr int THREAD_PER_BLOCK{256};

void reduce_cpu(int block_num, int n_per_block, float * input, float *res) {
    for (int i = 0; i < block_num; ++i) {
        float cur = 0;
        for (int j = 0; j < n_per_block; ++j) cur += input[i * n_per_block + j];
        res[i] = cur;
    }
}

bool check(float* output, float* res, int n) {
    for (int i = 0; i < n; ++i) {
        if (abs(output[i] - res[i]) > 0.005) return false;
    }

    return true;
}