#include <sgemm_global.cuh>

// DOUBLE BUFFERING

template<unsigned int M_PER_BLOCK, unsigned int K_PER_BLOCK>
__device__ __inline__ void load_shared_a(float *a, float shared_a[2][K_PER_BLOCK][M_PER_BLOCK], int tid, int block_row, int step, int stage) {
    int s_row = tid / (K_PER_BLOCK / 4); 
    int s_col = (tid % (K_PER_BLOCK / 4)) * 4;
    int g_row = block_row + s_row;
    int g_col = step * K_PER_BLOCK + s_col;

    float trans_load[4];
    bool full_a = g_row < M && g_col + 3 < K;
    if (full_a) {
        FETCH_FLOAT4(trans_load[0]) = FETCH_FLOAT4(A(g_row, g_col));
        shared_a[stage][s_col + 0][s_row] = trans_load[0];
        shared_a[stage][s_col + 1][s_row] = trans_load[1];
        shared_a[stage][s_col + 2][s_row] = trans_load[2];
        shared_a[stage][s_col + 3][s_row] = trans_load[3];
    } else {
        for (int j = 0; j < 4; ++j) {
            shared_a[stage][s_col + j][s_row] = (g_row < M && g_col + j < K) ? A(g_row, g_col + j) : 0.0f;
        }
    }
}

template<unsigned int N_PER_BLOCK, unsigned int K_PER_BLOCK>
__device__ __inline__ void load_shared_b(float *b, float shared_b[2][K_PER_BLOCK][N_PER_BLOCK], int tid, int block_col, int step, int stage) {
    int s_row = tid / (N_PER_BLOCK / 4);
    int s_col = (tid % (N_PER_BLOCK / 4)) * 4;
    int g_row = step * K_PER_BLOCK + s_row;
    int g_col = block_col + s_col;

    bool full_b = g_row < K && g_col + 3 < N;
    if (full_b) {
        FETCH_FLOAT4(shared_b[stage][s_row][s_col]) = FETCH_FLOAT4(B(g_row, g_col));
    } else {
        for (int j = 0; j < 4; ++j) {
            shared_b[stage][s_row][s_col + j] = (g_row < K && g_col + j < N) ? B(g_row, g_col + j) : 0.0f;
        }
    }
}

template<unsigned int M_PER_BLOCK, unsigned int N_PER_BLOCK, unsigned int K_PER_BLOCK, unsigned int M_PER_THREAD, unsigned int N_PER_THREAD>
__global__ void sgemm_gpu(float *a, float *b, float *c) {
    constexpr unsigned int TILE_CNT = (K + K_PER_BLOCK - 1) / K_PER_BLOCK;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_row = blockIdx.y * M_PER_BLOCK;
    int block_col = blockIdx.x * N_PER_BLOCK;
    int thread_row = ty * M_PER_THREAD;
    int thread_col = tx * N_PER_THREAD;
    int tid = ty * blockDim.x + tx;

    __shared__ float shared_a[2][K_PER_BLOCK][M_PER_BLOCK];
    __shared__ float shared_b[2][K_PER_BLOCK][N_PER_BLOCK];
    float reg_a[M_PER_THREAD];
    float reg_b[N_PER_THREAD];
    float tmp[M_PER_THREAD][N_PER_THREAD] = {0.0f};

    int cur = 0;
    int next = 1;

    load_shared_a<M_PER_BLOCK, K_PER_BLOCK>(a, shared_a, tid, block_row, 0, cur);
    load_shared_b<N_PER_BLOCK, K_PER_BLOCK>(b, shared_b, tid, block_col, 0, cur);
    __syncthreads();

    for (int t = 0; t < (int)TILE_CNT - 1; ++t) {
        load_shared_a<M_PER_BLOCK, K_PER_BLOCK>(a, shared_a, tid, block_row, t + 1, next);
        load_shared_b<N_PER_BLOCK, K_PER_BLOCK>(b, shared_b, tid, block_col, t + 1, next);

        #pragma unroll
        for (int k = 0; k < K_PER_BLOCK; ++k) {
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(shared_a[cur][k][thread_row]);
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(shared_b[cur][k][thread_col]);
            #pragma unroll
            for (int i = 0; i < M_PER_THREAD; ++i) {
                #pragma unroll
                for (int j = 0; j < N_PER_THREAD; ++j) {
                    tmp[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();

        cur ^= 1;
        next ^= 1;
    }

    if (TILE_CNT > 0) {
        int valid_k = (K - (TILE_CNT - 1) * K_PER_BLOCK);
        #pragma unroll
        for (int k = 0; k < valid_k; ++k) {
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(shared_a[cur][k][thread_row]);
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(shared_b[cur][k][thread_col]);
            #pragma unroll
            for (int i = 0; i < M_PER_THREAD; ++i) {
                #pragma unroll
                for (int j = 0; j < N_PER_THREAD; ++j) {
                    tmp[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < M_PER_THREAD; ++i) {
        int global_row = block_row + thread_row + i;
        int global_col = block_col + thread_col;
        if (global_row < M && global_col + 3 < N) {
            FETCH_FLOAT4(C(global_row, global_col)) = FETCH_FLOAT4(tmp[i][0]);
        } else if (global_row < M) {
            for (int j = 0; j < N_PER_THREAD && global_col + j < N; ++j) {
                C(global_row, global_col + j) = tmp[i][j];
            }
        }
    }
}

void launch_sgemm_v5(float *a, float *b, float *c) {
    constexpr unsigned int M_PER_BLOCK{64};
    constexpr unsigned int N_PER_BLOCK{64};
    constexpr unsigned int K_PER_BLOCK{16};
    constexpr unsigned int M_PER_THREAD{4};
    constexpr unsigned int N_PER_THREAD{4};

    constexpr unsigned int M_THREAD_PER_BLOCK = M_PER_BLOCK / M_PER_THREAD;
    constexpr unsigned int N_THREAD_PER_BLOCK = N_PER_BLOCK / N_PER_THREAD; // 16

    dim3 block{N_THREAD_PER_BLOCK, M_THREAD_PER_BLOCK}; 
    dim3 grid{(N + N_PER_BLOCK - 1) / N_PER_BLOCK, (M + M_PER_BLOCK - 1) / M_PER_BLOCK};
    sgemm_gpu<M_PER_BLOCK, N_PER_BLOCK, K_PER_BLOCK, M_PER_THREAD, N_PER_THREAD><<<grid, block>>>(a, b, c);

    cudaDeviceSynchronize();
}

#ifdef SGEMM_STANDALONE

int main() {
    constexpr size_t mem_size_A = M * K_PAD * sizeof(float);
    constexpr size_t mem_size_B = K * N_PAD * sizeof(float);
    constexpr size_t mem_size_C = M * N_PAD * sizeof(float);

    float *mA_host = (float *)calloc(M * K_PAD, sizeof(float));
    float *mB_host = (float *)calloc(K * N_PAD, sizeof(float));
    float *mC_host_cpu = (float *)calloc(M * N_PAD, sizeof(float));
    float *mC_host_gpu = (float *)calloc(M * N_PAD, sizeof(float));

    random_m(M, K, mA_host);
    random_m(K, N, mB_host);

    float *mA_device, *mB_device, *mC_device;
    cudaMalloc((void **)&mA_device, mem_size_A);
    cudaMalloc((void **)&mB_device, mem_size_B);
    cudaMalloc((void **)&mC_device, mem_size_C);

    cudaMemcpy(mA_device, mA_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(mB_device, mB_host, mem_size_B, cudaMemcpyHostToDevice);

    // sgemm_cpu(mA_host, mB_host, mC_host_cpu);

    launch_sgemm_v5(mA_device, mB_device, mC_device);

    // cudaMemcpy(mC_host_gpu, mC_device, mem_size_C, cudaMemcpyDeviceToHost);
    // sgemm_cmp(mC_host_cpu, mC_host_gpu);

    cudaFree(mA_device);
    cudaFree(mB_device);
    cudaFree(mC_device);
    free(mA_host);
    free(mB_host);
    free(mC_host_cpu);
    free(mC_host_gpu);

    return 0;
}

#endif