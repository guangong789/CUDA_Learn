#pragma once
#include <cstdio>
#include <float.h>
#include <algorithm>
#include <cuda_runtime.h>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

#define M(i, j) m[(i) * N + (j)]

constexpr int M{4096};
constexpr int N{1024};

inline void random_m(int rowNum, int colNum, float *m, bool ones = false) {
    for (int row = 0; row < rowNum; ++row) {
        for (int col = 0; col < colNum; ++col) {
            if (!ones) M(row, col) = 2.0f * (float)drand48() - 1.0f;
            else M(row, col) = 1.0f;
        }
    }
}

inline void softmax_cpu(const float* input, float* output) {
    for (int row = 0; row < M; ++row) {
        float max_val = -FLT_MAX;
        for (int col = 0; col < N; ++col) {
            max_val = fmaxf(max_val, input[row * N + col]);
        }

        float sum = 0.f;
        for (int col = 0; col < N; ++col) {
            float v = expf(input[row * N + col] - max_val);
            output[row * N + col] = v;
            sum += v;
        }

        for (int col = 0; col < N; ++col)  {
            output[row * N + col] /= sum;
        }
    }
}

inline bool softmax_cmp(const float* gpu, const float* ref, float atol = 1e-4f, float rtol = 1e-3f) {
    bool pass = true;
    float max_abs_err = 0.f;
    float max_rel_err = 0.f;
    int max_row = -1;
    int max_col = -1;

    int mismatch_cnt = 0;

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            int idx = row * N + col;

            float g = gpu[idx];
            float r = ref[idx];
            float abs_err = fabsf(g - r);
            float rel_err = abs_err / (fabsf(r) + 1e-12f);

            float tol = atol + rtol * fabsf(r);

            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                max_rel_err = rel_err;
                max_row = row;
                max_col = col;
            }
            
            if (abs_err > tol) {
                mismatch_cnt++;
                if (mismatch_cnt <= 10) {
                    printf("[%d,%d] ""gpu=%e ""ref=%e ""abs=%e ""rel=%e\n",
                        row, col, g, r, abs_err, rel_err);
                }
                pass = false;
            }
        }
    }

    printf("PASS            : %s\n", pass ? "TRUE" : "FALSE");
    printf("MISMATCH COUNT  : %d\n", mismatch_cnt);
    printf("MAX ABS ERROR   : %e\n", max_abs_err);
    printf("MAX REL ERROR   : %e\n", max_rel_err);
    if (max_row >= 0) {
        printf("WORST POSITION  : (%d,%d)\n", max_row, max_col);
    }

    return pass;
}

void launch_softmax_v0(float* input, float* output);
void launch_softmax_v1(float* input, float* output);
void launch_softmax_v2(float* input, float* output);
void launch_softmax_v3(float* input, float* output);

__device__ __forceinline__ float warpReduceMax(float val) {
    for (int stride = 16; stride > 0; stride >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, stride));
    }
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int stride = 16; stride > 0; stride >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, stride);
    }
    return val;
}