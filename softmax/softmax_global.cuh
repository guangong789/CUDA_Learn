#pragma once
#include <cstdio>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>

#define M(i, j) m[(i) * N + (j)]

constexpr int M{512};
constexpr int N{512};

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
            max_val = std::max(max_val, input[row * N + col]);
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

inline void softmax_cmp(const float* gpu, const float* ref, float atol = 1e-4f) {
    float max_err = 0.f;
    float avg_err = 0.f;
    bool pass = true;

    for (int row = 0; row < M; ++row) {
        float row_sum = 0.f;
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            float err = fabsf(gpu[idx] - ref[idx]);

            row_sum += gpu[idx];
            max_err = std::max(max_err, err);
            avg_err += err;

            if (err > atol) {
                printf("Mismatch [%d,%d]\nGPU=%f REF=%f ERR=%f\n", row, col, gpu[idx], ref[idx], err);
                pass = false;
                goto END;
            }
        }

        if (row < 5) {
            printf("row %d sum = %.6f\n", row, row_sum);
        }
    }
END:
    avg_err /= (M * N);
    printf("\nPASS=%s\nMAX_ERR=%e\nAVG_ERR=%e\n", pass ? "TRUE" : "FALSE", max_err, avg_err);
}

void launch_softmax_v0(float* input, float* output);