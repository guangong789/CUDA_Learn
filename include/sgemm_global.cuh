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
        if (!ones) M(row, col) = 2.0f * (float)drand48() - 1.0f;
        else M(row, col) = 1.0f;
}

bool cmp_m(const float *h, const float *d) {
    const float atol = 1e-2f;
    const float rtol = 1e-2f;

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    int bad_count = 0;

    int worst_i = 0;
    int worst_j = 0;

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float rv = H(i, j);
            float tv = D(i, j);

            float abs_err = fabsf(rv - tv);
            float rel_err = abs_err / (fabsf(rv) + 1e-6f);

            if (abs_err > max_abs_err) {
                max_abs_err = abs_err;
                worst_i = i;
                worst_j = j;
            }
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (abs_err > atol + rtol * fabsf(rv)) bad_count++;
        }
    }

    printf("RESULT %s\n", bad_count == 0 ? "PASS" : "FAIL");
    printf("MAX_ABS %.8e\n", max_abs_err);
    printf("MAX_REL %.8e\n", max_rel_err);
    printf("BAD_COUNT %d\n", bad_count);

    if (bad_count > 0) {
        printf("WORST_POS %d %d\n", worst_i, worst_j);
        printf("REF %.8f\n", H(worst_i, worst_j));
        printf("TEST %.8f\n", D(worst_i, worst_j));
    }

    return bad_count == 0;
}

void sgemm_cpu(float *a, float *b, float *c) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            double value = 0.0;

            for (int k = 0; k < K; ++k) {
                value += (double)A(row, k) * (double)B(k, col);
            }
            C(row, col) = (float)value;
        }
    }
}