#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <float.h>
#include <vector>

#include <flash_attention_api.cuh>

#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<const float4*>(&(pointer))[0])

constexpr int FA_BATCH_SIZE{1};
constexpr int FA_HEAD_NUM{2};
constexpr int FA_SEQ_LEN{256};
constexpr int FA_HEAD_DIM{64};

inline void random_tensor(float* tensor, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        tensor[i] = 2.0f * static_cast<float>(drand48()) - 1.0f;
    }
}

inline void flash_attention_cpu(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int head_num,
    int seq_len,
    int head_dim,
    bool causal
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    std::vector<float> scores(seq_len);

    for (int bh = 0; bh < batch_size * head_num; ++bh) {
        for (int row = 0; row < seq_len; ++row) {
            float row_max = -FLT_MAX;

            for (int col = 0; col < seq_len; ++col) {
                if (causal && col > row) {
                    scores[col] = -FLT_MAX;
                    continue;
                }

                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    size_t q_idx = (static_cast<size_t>(bh) * seq_len + row) * head_dim + d;
                    size_t k_idx = (static_cast<size_t>(bh) * seq_len + col) * head_dim + d;
                    score += q[q_idx] * k[k_idx];
                }
                scores[col] = score * scale;
                row_max = fmaxf(row_max, scores[col]);
            }

            float row_sum = 0.0f;
            for (int col = 0; col < seq_len; ++col) {
                float probability = (causal && col > row) ? 0.0f : expf(scores[col] - row_max);
                scores[col] = probability;
                row_sum += probability;
            }

            for (int d = 0; d < head_dim; ++d) {
                float value = 0.0f;
                for (int col = 0; col < seq_len; ++col) {
                    size_t v_idx = (static_cast<size_t>(bh) * seq_len + col) * head_dim + d;
                    value += scores[col] * v[v_idx];
                }
                size_t o_idx = (static_cast<size_t>(bh) * seq_len + row) * head_dim + d;
                output[o_idx] = value / row_sum;
            }
        }
    }
}

inline bool flash_attention_cmp(
    const float* gpu,
    const float* ref,
    size_t size,
    float atol = 2e-4f,
    float rtol = 2e-3f
) {
    bool pass = true;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    size_t worst_idx = 0;
    int mismatch_cnt = 0;

    for (size_t i = 0; i < size; ++i) {
        float abs_err = fabsf(gpu[i] - ref[i]);
        float rel_err = abs_err / (fabsf(ref[i]) + 1e-12f);
        float tol = atol + rtol * fabsf(ref[i]);

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_rel_err = rel_err;
            worst_idx = i;
        }

        if (abs_err > tol) {
            mismatch_cnt++;
            if (mismatch_cnt <= 10) {
                printf("[%zu] gpu=%e ref=%e abs=%e rel=%e\n",
                    i, gpu[i], ref[i], abs_err, rel_err);
            }
            pass = false;
        }
    }

    printf("PASS            : %s\n", pass ? "TRUE" : "FALSE");
    printf("MISMATCH COUNT  : %d\n", mismatch_cnt);
    printf("MAX ABS ERROR   : %e\n", max_abs_err);
    printf("MAX REL ERROR   : %e\n", max_rel_err);
    printf("WORST INDEX     : %zu\n", worst_idx);

    return pass;
}
