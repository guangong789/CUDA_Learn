#include <sgemm_global.cuh>

// REGISTER TILING

template<unsigned int M_PER_BLOCK, unsigned int N_PER_BLOCK, unsigned int K_PER_BLOCK, 
    unsigned int M_PER_THREAD, unsigned int N_PER_THREAD>
__global__ void sgemm_gpu(float *a, float *b, float *c) {
    constexpr unsigned int TILE_CNT = (K + K_PER_BLOCK - 1) / K_PER_BLOCK;
    int tx = threadIdx.x, ty = threadIdx.y;  // [0, 15]
    int block_row = blockIdx.y * M_PER_BLOCK;
    int block_col = blockIdx.x * N_PER_BLOCK;
    // 块内每线程负责的起始点
    int thread_row = ty * M_PER_THREAD;
    int thread_col = tx * N_PER_THREAD;

    __shared__ float shared_a[M_PER_BLOCK][K_PER_BLOCK];  // {128, 8}
    __shared__ float shared_b[K_PER_BLOCK][N_PER_BLOCK];  // {8, 128}
    float reg_a[M_PER_THREAD] = {0.0f};
    float reg_b[N_PER_THREAD] = {0.0f}; 
    float tmp[M_PER_THREAD][N_PER_THREAD] = {0.0f};  // 8*8

    #pragma unroll
    for (int t = 0; t < TILE_CNT; ++t) {  // 分块加载
        // shared_a
        int tid = ty * blockDim.x + tx; 
        {
            int s_row = (tid * 4) / K_PER_BLOCK; 
            int s_col = (tid * 4) % K_PER_BLOCK;
            int g_row = block_row + s_row;
            int g_col = t * K_PER_BLOCK + s_col;

            bool full_a = g_row < M && g_col + 3 < K;
            if (full_a) {
                FETCH_FLOAT4(shared_a[s_row][s_col]) = FETCH_FLOAT4(A(g_row, g_col));
            } else {
                for (int j = 0; j < 4; ++j) {
                    shared_a[s_row][s_col + j] = (g_row < M && g_col + j < K) ? A(g_row, g_col + j) : 0.0f;
                }
            }
        }
        // shared_b
        {
            int s_row = (tid * 4) / N_PER_BLOCK;
            int s_col = (tid * 4) % N_PER_BLOCK;
            int g_row = t * K_PER_BLOCK + s_row;
            int g_col = block_col + s_col;

            bool full_b = g_row < K && g_col + 3 < N;
            if (full_b) {
                FETCH_FLOAT4(shared_b[s_row][s_col]) = FETCH_FLOAT4(B(g_row, g_col));
            } else {
                for (int j = 0; j < 4; ++j) {
                    shared_b[s_row][s_col + j] = (g_row < K && g_col + j < N) ? B(g_row, g_col + j) : 0.0f;
                }
            }
        }
        __syncthreads();

        // outer product
        int valid_k = (t == TILE_CNT - 1) ? (K - t * K_PER_BLOCK) : K_PER_BLOCK;
        #pragma unroll
        for (int k = 0; k < valid_k; ++k) {
            for (int i = 0; i < 8; ++i) {
                reg_a[i] = shared_a[thread_row + i][k];
            }
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(shared_b[k][thread_col]);
            FETCH_FLOAT4(reg_b[4]) = FETCH_FLOAT4(shared_b[k][thread_col + 4]);
            for (int i = 0; i < M_PER_THREAD; ++i) {
                for (int j = 0; j < N_PER_THREAD; ++j) {
                    tmp[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < M_PER_THREAD; ++i) {
        int global_row = block_row + thread_row + i;
        int global_col = block_col + thread_col;
        if (global_row < M && global_col + 7 < N) {
            FETCH_FLOAT4(C(global_row, global_col)) = FETCH_FLOAT4(tmp[i][0]);
            FETCH_FLOAT4(C(global_row, global_col + 4)) = FETCH_FLOAT4(tmp[i][4]);
        } else if (global_row < M) {
            for (int j = 0; global_col + j < N; ++j) {
                C(global_row, global_col + j) = tmp[i][j];
            }
        }
    }
}

void launch_sgemm_v3(float *a, float *b, float *c) {
    constexpr unsigned int M_PER_BLOCK{128};
    constexpr unsigned int N_PER_BLOCK{128};
    constexpr unsigned int K_PER_BLOCK{8};
    constexpr unsigned int M_PER_THREAD{8};
    constexpr unsigned int N_PER_THREAD{8};

    constexpr unsigned int M_THREAD_PER_BLOCK = M_PER_BLOCK / M_PER_THREAD;
    constexpr unsigned int N_THREAD_PER_BLOCK = N_PER_BLOCK / N_PER_THREAD;  // 16

    dim3 block{N_THREAD_PER_BLOCK, M_THREAD_PER_BLOCK};  // block(16, 16), float4 加载，每线程加载 1 float4, 计算 8*8
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

    launch_sgemm_v3(mA_device, mB_device, mC_device);

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