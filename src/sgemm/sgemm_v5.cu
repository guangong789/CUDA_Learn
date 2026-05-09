#include <sgemm_global.cuh>

// OUTER PRODUCT: TRANSPOSE

template<unsigned int M_PER_BLOCK, unsigned int N_PER_BLOCK, unsigned int K_PER_BLOCK, 
    unsigned int M_PER_THREAD, unsigned int N_PER_THREAD>
__global__ void sgemm_gpu(float *a, float *b, float *c) {
    constexpr unsigned int TILE_CNT = (K + K_PER_BLOCK - 1) / K_PER_BLOCK;
    int tx = threadIdx.x, ty = threadIdx.y;  // [0, 8)
    int block_row = blockIdx.y * M_PER_BLOCK;
    int block_col = blockIdx.x * N_PER_BLOCK;
    // 每线程负责的起始点
    int thread_row = ty * M_PER_THREAD;
    int thread_col = tx * N_PER_THREAD;

    __shared__ float shared_a[K_PER_BLOCK][M_PER_BLOCK];  // 32*32
    __shared__ float shared_b[K_PER_BLOCK][N_PER_BLOCK];
    float reg_for_trans[M_PER_THREAD] = {0.0f};
    float reg_a[M_PER_THREAD] = {0.0f};  // 4 scalars
    float reg_b[N_PER_THREAD] = {0.0f};  // 1 float4
    float tmp[M_PER_THREAD][N_PER_THREAD] = {0.0f};  // 4*4

    #pragma unroll
    for (int t = 0; t < (K + K_PER_BLOCK - 1) / K_PER_BLOCK; ++t) {  // 分块加载 
        int col_a = t * K_PER_BLOCK + tx * 4, col_b = block_col + tx * 4;
        for (int i = 0; i < M_PER_THREAD; ++i) {
            int row_offset = i * blockDim.y;
            int row_a = block_row + ty + row_offset, row_b = t * K_PER_BLOCK + ty + row_offset;
            int smem_row = ty + row_offset, smem_col = tx * 4;
            bool full_a = row_a < M && col_a + 3 < K;
            if (full_a) {
                 FETCH_FLOAT4(reg_for_trans[0]) = FETCH_FLOAT4(A(row_a, col_a));
            } else {
                for (int j = 0; j < 4; ++j) {
                    reg_for_trans[j] = (row_a < M && col_a + j < K) ? A(row_a, col_a + j) : 0.0f;
                }
            }
            // 不同行，相同列
            shared_a[tx*4][ty + row_offset] = reg_for_trans[0];
            shared_a[tx*4 + 1][ty + row_offset] = reg_for_trans[1];
            shared_a[tx*4 + 2][ty + row_offset] = reg_for_trans[2];
            shared_a[tx*4 + 3][ty + row_offset] = reg_for_trans[3];
            bool full_b = row_b < K && col_b < N;
            if (full_b) {
                FETCH_FLOAT4(shared_b[smem_row][smem_col]) = FETCH_FLOAT4(B(row_b, col_b));
            } else {
                for (int j = 0; j < 4; ++j) {
                    shared_b[smem_row][smem_col + j] = (row_b < K && col_b + j < N) ? B(row_b, col_b + j) : 0.0f;
                }
            }
        }
        __syncthreads();
        // outer product
        int valid_k = (t == TILE_CNT - 1) ? (K - t * K_PER_BLOCK) : K_PER_BLOCK;
        #pragma unroll
        for (int k = 0; k < valid_k; ++k) {
            FETCH_FLOAT4(reg_a[0]) = FETCH_FLOAT4(shared_a[k][thread_row]);
            FETCH_FLOAT4(reg_b[0]) = FETCH_FLOAT4(shared_b[k][thread_col]);
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
        if (global_row < M && global_col + 3 < N) {
            FETCH_FLOAT4(C(global_row, global_col)) = FETCH_FLOAT4(tmp[i][0]);
        } else if (global_row < M) {
            for (int j = 0; global_col + j < N; ++j) {
                C(global_row, global_col + j) = tmp[i][j];
            }
        }
    }
}

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

    sgemm_cpu(mA_host, mB_host, mC_host_cpu);

    // 处理数据大小
    constexpr unsigned int M_PER_BLOCK{32};
    constexpr unsigned int N_PER_BLOCK{32};
    constexpr unsigned int K_PER_BLOCK{32};
    constexpr unsigned int M_PER_THREAD{4};
    constexpr unsigned int N_PER_THREAD{4};

    constexpr unsigned int M_THREAD_PER_BLOCK = M_PER_BLOCK / M_PER_THREAD;
    constexpr unsigned int N_THREAD_PER_BLOCK = N_PER_BLOCK / N_PER_THREAD;  // 8

    dim3 block{N_THREAD_PER_BLOCK, M_THREAD_PER_BLOCK};  // float4 加载，每线程计算 4*4
    dim3 grid{(N + N_PER_BLOCK - 1) / N_PER_BLOCK, (M + M_PER_BLOCK - 1) / M_PER_BLOCK};
    sgemm_gpu<M_PER_BLOCK, N_PER_BLOCK, K_PER_BLOCK, M_PER_THREAD, N_PER_THREAD><<<grid, block>>>(mA_device, mB_device, mC_device);

    cudaDeviceSynchronize();
    cudaMemcpy(mC_host_gpu, mC_device, mem_size_C, cudaMemcpyDeviceToHost);

    cmp_m(mC_host_cpu, mC_host_gpu);

    cudaFree(mA_device);
    cudaFree(mB_device);
    cudaFree(mC_device);
    free(mA_host);
    free(mB_host);
    free(mC_host_cpu);
    free(mC_host_gpu);

    return 0;
}