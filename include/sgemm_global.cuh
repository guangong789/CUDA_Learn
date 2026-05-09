#include <cstdio>
#include <cuda_runtime.h>
#include <cmath>

#define M(i, j) m[i * colNum + j]
#define H(i, j) h[i * N_PAD + j]
#define D(i, j) d[i * N_PAD + j]

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

// MATRIX
constexpr int M{1025};
constexpr int K{1026};
constexpr int N{1027};

constexpr int K_PAD = ((K + 3) / 4) * 4;
constexpr int N_PAD = ((N + 3) / 4) * 4;

#define A(i,j) a[(i)*K_PAD + (j)]
#define B(i,j) b[(i)*N_PAD + (j)]
#define C(i,j) c[(i)*N_PAD + (j)]

void random_m(int rowNum, int colNum, float *m, bool ones = false) {
    int row, col;
    for (row = 0; row < rowNum; ++row)
        for (col = 0; col < colNum; ++col)
        if (!ones) M(row, col) = 200.0f * (float)drand48() - 100.0f;
        else M(row, col) = 1.0f;
}

float cmp_m(const float *h, const float *d) {
    double diff = 0.0;
    double rel  = 0.0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float hv = H(i, j);
            float dv = D(i, j);

            float abs_err = fabsf(hv - dv);
            float denom = fmaxf(fabsf(hv), fabsf(dv));
            denom = fmaxf(denom, 1.0f);
            float rel_err = abs_err / denom;

            diff += abs_err;
            rel  += rel_err;
        }
    }

    diff /= (M * N);
    rel  /= (M * N);

    printf("mean_diff : %.6e\n", diff);
    printf("mean_rel  : %.6e\n", rel);

    return (float)rel;
}

void sgemm_cpu(float *a, float *b, float *c) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float value = 0.f;
            for (int k = 0; k < K; ++k) {
                value += A(row, k) * B(k, col);
            }
            C(row, col) = value;
        }
    }
}