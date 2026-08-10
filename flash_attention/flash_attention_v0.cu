#include <flash_attention_global.cuh>

// FP32 FLASH ATTENTION FORWARD

constexpr int WARP_SIZE{32};

template<int HEAD_DIM, int BLOCK_Q, int BLOCK_K, int WARPS_PER_BLOCK>
__global__ void flash_attention_v0(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int seq_len,
    bool causal
) {
    constexpr int ROWS_PER_WARP = BLOCK_Q / WARPS_PER_BLOCK;
    constexpr int VALUES_PER_THREAD = HEAD_DIM / WARP_SIZE;
    static_assert(BLOCK_K == WARP_SIZE);
    static_assert(BLOCK_Q % WARPS_PER_BLOCK == 0);
    static_assert(HEAD_DIM % WARP_SIZE == 0);

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bh = blockIdx.y;
    const int block_row = blockIdx.x * BLOCK_Q;
    const int warp_row = block_row + warp_id * ROWS_PER_WARP;
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));

    __shared__ float shared_q[BLOCK_Q][HEAD_DIM];
    __shared__ float shared_k[HEAD_DIM][BLOCK_K + 1];
    __shared__ float shared_v[BLOCK_K][HEAD_DIM];
    __shared__ float shared_p[BLOCK_Q][BLOCK_K];

    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP];
    float result[ROWS_PER_WARP][VALUES_PER_THREAD];

    #pragma unroll
    for (int row = 0; row < ROWS_PER_WARP; ++row) {
        row_max[row] = -FLT_MAX;
        row_sum[row] = 0.0f;
        #pragma unroll
        for (int item = 0; item < VALUES_PER_THREAD; ++item) {
            result[row][item] = 0.0f;
        }
    }

    // Q stays resident during the whole K/V loop.
    for (int idx = tid; idx < BLOCK_Q * HEAD_DIM / 4; idx += blockDim.x) {
        int row = idx / (HEAD_DIM / 4);
        int d = (idx % (HEAD_DIM / 4)) * 4;
        int global_row = block_row + row;
        size_t global_idx = (static_cast<size_t>(bh) * seq_len + global_row) * HEAD_DIM + d;
        float4 q_value = {0.0f, 0.0f, 0.0f, 0.0f};
        if (global_row < seq_len) {
            q_value = FETCH_FLOAT4_CONST(q[global_idx]);
        }
        FETCH_FLOAT4(shared_q[row][d]) = q_value;
    }
    __syncthreads();

    int last_query = min(block_row + BLOCK_Q, seq_len) - 1;

    for (int block_col = 0; block_col < seq_len; block_col += BLOCK_K) {
        if (causal && block_col > last_query) break;

        // Transposed K makes one key per lane conflict-free during QK^T.
        for (int idx = tid; idx < BLOCK_K * HEAD_DIM / 4; idx += blockDim.x) {
            int col = idx / (HEAD_DIM / 4);
            int d = (idx % (HEAD_DIM / 4)) * 4;
            int global_col = block_col + col;
            size_t global_idx = (static_cast<size_t>(bh) * seq_len + global_col) * HEAD_DIM + d;
            float4 k_value = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 v_value = {0.0f, 0.0f, 0.0f, 0.0f};
            if (global_col < seq_len) {
                k_value = FETCH_FLOAT4_CONST(k[global_idx]);
                v_value = FETCH_FLOAT4_CONST(v[global_idx]);
            }
            shared_k[d + 0][col] = k_value.x;
            shared_k[d + 1][col] = k_value.y;
            shared_k[d + 2][col] = k_value.z;
            shared_k[d + 3][col] = k_value.w;
            FETCH_FLOAT4(shared_v[col][d]) = v_value;
        }
        __syncthreads();

        float tile_max[ROWS_PER_WARP];
        float tile_sum[ROWS_PER_WARP];

        // Each lane owns one key and computes four score values.
        #pragma unroll
        for (int row = 0; row < ROWS_PER_WARP; ++row) {
            int query_row = warp_row + row;
            int key_col = block_col + lane_id;
            bool valid_row = query_row < seq_len;
            bool valid_key = key_col < seq_len && (!causal || key_col <= query_row);
            bool has_valid_key = valid_row && (!causal || block_col <= query_row);
            float score = 0.0f;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += shared_q[query_row - block_row][d] * shared_k[d][lane_id];
            }
            score = valid_row && valid_key ? score * scale : -FLT_MAX;

            float max_value = score;
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset));
            }

            float probability = has_valid_key ? expf(score - max_value) : 0.0f;
            shared_p[query_row - block_row][lane_id] = probability;

            float sum_value = probability;
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                sum_value += __shfl_xor_sync(0xffffffff, sum_value, offset);
            }

            tile_max[row] = max_value;
            tile_sum[row] = sum_value;
        }
        __syncwarp();

        // Compute the local P*V tile, then merge its online-softmax state.
        #pragma unroll
        for (int row = 0; row < ROWS_PER_WARP; ++row) {
            int query_row = warp_row + row;
            bool has_valid_key = query_row < seq_len && (!causal || block_col <= query_row);
            if (!has_valid_key) continue;

            float tile_result[VALUES_PER_THREAD] = {0.0f};
            #pragma unroll
            for (int col = 0; col < BLOCK_K; ++col) {
                float probability = shared_p[query_row - block_row][col];
                #pragma unroll
                for (int item = 0; item < VALUES_PER_THREAD; ++item) {
                    int d = item * WARP_SIZE + lane_id;
                    tile_result[item] += probability * shared_v[col][d];
                }
            }

            float new_max = fmaxf(row_max[row], tile_max[row]);
            float old_scale = expf(row_max[row] - new_max);
            float tile_scale = expf(tile_max[row] - new_max);

            #pragma unroll
            for (int item = 0; item < VALUES_PER_THREAD; ++item) {
                result[row][item] = result[row][item] * old_scale
                    + tile_result[item] * tile_scale;
            }
            row_sum[row] = row_sum[row] * old_scale + tile_sum[row] * tile_scale;
            row_max[row] = new_max;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < ROWS_PER_WARP; ++row) {
        int global_row = warp_row + row;
        if (global_row >= seq_len) continue;

        #pragma unroll
        for (int item = 0; item < VALUES_PER_THREAD; ++item) {
            int d = item * WARP_SIZE + lane_id;
            size_t global_idx = (static_cast<size_t>(bh) * seq_len + global_row) * HEAD_DIM + d;
            output[global_idx] = result[row][item] / row_sum[row];
        }
    }
}

void launch_flash_attention_v0(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int batch_size,
    int head_num,
    int seq_len,
    int head_dim,
    bool causal,
    cudaStream_t stream
) {
    constexpr int BLOCK_Q{32};
    constexpr int BLOCK_K{32};
    constexpr int WARPS_PER_BLOCK{8};
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    if (head_dim != FA_HEAD_DIM) {
        fprintf(stderr, "flash_attention_v0 only supports head_dim=%d, got %d\n",
            FA_HEAD_DIM, head_dim);
        return;
    }

    dim3 block{THREADS_PER_BLOCK};
    dim3 grid{static_cast<unsigned int>((seq_len + BLOCK_Q - 1) / BLOCK_Q),
              static_cast<unsigned int>(batch_size * head_num)};
    flash_attention_v0<FA_HEAD_DIM, BLOCK_Q, BLOCK_K, WARPS_PER_BLOCK>
        <<<grid, block, 0, stream>>>(q, k, v, output, seq_len, causal);
}

#ifdef FLASH_ATTENTION_STANDALONE

bool test_flash_attention(
    const float* h_q,
    const float* h_k,
    const float* h_v,
    float* h_output,
    float* h_ref,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    float* d_output,
    size_t tensor_size,
    bool causal
) {
    printf("===== Flash Attention %s =====\n", causal ? "Causal" : "Non-causal");
    flash_attention_cpu(h_q, h_k, h_v, h_ref,
        FA_BATCH_SIZE, FA_HEAD_NUM, FA_SEQ_LEN, FA_HEAD_DIM, causal);

    launch_flash_attention_v0(d_q, d_k, d_v, d_output,
        FA_BATCH_SIZE, FA_HEAD_NUM, FA_SEQ_LEN, FA_HEAD_DIM, causal);
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess) {
        printf("KERNEL LAUNCH   : %s\n", cudaGetErrorString(launch_error));
        return false;
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, tensor_size * sizeof(float), cudaMemcpyDeviceToHost);

    return flash_attention_cmp(h_output, h_ref, tensor_size);
}

int main() {
    constexpr size_t tensor_size = static_cast<size_t>(FA_BATCH_SIZE)
        * FA_HEAD_NUM * FA_SEQ_LEN * FA_HEAD_DIM;
    constexpr size_t bytes = tensor_size * sizeof(float);

    float* h_q = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    float* h_k = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    float* h_v = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    float* h_output = static_cast<float*>(calloc(tensor_size, sizeof(float)));
    float* h_ref = static_cast<float*>(calloc(tensor_size, sizeof(float)));

    random_tensor(h_q, tensor_size);
    random_tensor(h_k, tensor_size);
    random_tensor(h_v, tensor_size);

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_output = nullptr;
    cudaMalloc(&d_q, bytes);
    cudaMalloc(&d_k, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_q, h_q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    printf("B=%d  H=%d  N=%d  D=%d\n\n",
        FA_BATCH_SIZE, FA_HEAD_NUM, FA_SEQ_LEN, FA_HEAD_DIM);
    bool non_causal_pass = test_flash_attention(
        h_q, h_k, h_v, h_output, h_ref,
        d_q, d_k, d_v, d_output, tensor_size, false);
    bool causal_pass = test_flash_attention(
        h_q, h_k, h_v, h_output, h_ref,
        d_q, d_k, d_v, d_output, tensor_size, true);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);
    free(h_q);
    free(h_k);
    free(h_v);
    free(h_output);
    free(h_ref);

    return non_causal_pass && causal_pass ? 0 : 1;
}

#endif
