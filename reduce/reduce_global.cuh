#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr int N = 67108864;
constexpr int THREAD_PER_BLOCK = 256;
constexpr int NUM_PER_THREAD = 8;

constexpr int ALIGNMENT = 1024 * THREAD_PER_BLOCK;
constexpr int N_PADDED = ((N + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;

inline void reduce_cpu(int tpb, const float* input, float* res) {
    const int BLOCK_NUM = N_PADDED / tpb; 
    for (int i = 0; i < BLOCK_NUM; ++i) {
        float cur = 0.0f;
        for (int j = 0; j < tpb; ++j) {
            cur += input[i * tpb + j];
        }
        res[i] = cur;
    }
}

inline bool check(const float* output, const float* res, int n) {
    for (int i = 0; i < n; ++i) {
        if (std::abs(output[i] - res[i]) > 0.005f) {
            return false;
        }
    }
    return true;
}

void launch_reduce_v0(float* d_input, float* d_output, int tpb);
void launch_reduce_v1(float* d_input, float* d_output, int tpb);
void launch_reduce_v2(float* d_input, float* d_output, int tpb);
void launch_reduce_v3(float* d_input, float* d_output, int tpb);
void launch_reduce_v4(float* d_input, float* d_output, int tpb);
void launch_reduce_v5(float* d_input, float* d_output, int tpb);
void launch_reduce_v6(float* d_input, float* d_output, int tpb);
void launch_reduce_v7(float* d_input, float* d_output, int tpb);
void launch_reduce_v8(float* d_input, float* d_output, int tpb);