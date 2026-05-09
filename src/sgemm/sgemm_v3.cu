#include <sgemm_global.cuh>

// FLOAT4
constexpr int LENGTH_FLOAT4{4};

template <unsigned int THREAD_CNT_M, unsigned int THREAD_CNT_N, unsigned int M_PER_BLOCK, unsigned int N_PER_BLOCK, unsigned int K_PER_BLOCK>
__global__ void sgemm_gpu(float *a, float *b, float *c) {
    const unsigned int TILE_CNT = (K + K_PER_BLOCK - 1) / K_PER_BLOCK;
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * M_PER_BLOCK, block_col = blockIdx.x * N_PER_BLOCK;  // 每个BLOCK的起始格点位置
    int thread_row = block_row + ty, thread_col = block_col + tx * LENGTH_FLOAT4;  // 每个线程的起始格点位置

    __shared__ float shared_a[M_PER_BLOCK][K_PER_BLOCK];
    __shared__ float shared_b[K_PER_BLOCK][N_PER_BLOCK];
    float tmp[LENGTH_FLOAT4] = {0.0f};

    int smem_row{ty}, smem_col{0};

    #pragma unroll
    for (int t = 0; t < TILE_CNT; ++t) {
        smem_col = tx * LENGTH_FLOAT4;
        int a_row = thread_row , a_col = t * K_PER_BLOCK + tx * LENGTH_FLOAT4;
        bool full_a = a_row < M && a_col + 3 < K;
        if (full_a) {
            FETCH_FLOAT4(shared_a[smem_row][smem_col]) = FETCH_FLOAT4(A(a_row, a_col));
        } else {
            for (int i = 0; i < 4; ++i) {
                shared_a[smem_row][smem_col + i] = (a_row < M && a_col + i < K) ? A(a_row, a_col + i) : 0.0f;;
            }
        }
        int b_row = t * K_PER_BLOCK + ty, b_col = thread_col;
        bool full_b = b_row < K && b_col + 3 < N;
        if (full_b) {
            FETCH_FLOAT4(shared_b[smem_row][smem_col]) = FETCH_FLOAT4(B(b_row, b_col));
        } else {
            for (int i = 0; i < 4; ++i) {
                shared_b[smem_row][smem_col + i] = (b_row < K && b_col + i < N) ? B(b_row, b_col + i) : 0.0f;
            }
        }
        __syncthreads();

        int valid_k = (t == TILE_CNT - 1) ? (K - t * K_PER_BLOCK) : K_PER_BLOCK;
        #pragma unroll
        for (int k = 0; k < valid_k; ++k) {
            float a_val = shared_a[smem_row][k];
            for (int j = 0; j < LENGTH_FLOAT4; ++j) {
                smem_col = tx * LENGTH_FLOAT4 + j;
                tmp[j] += a_val * shared_b[k][smem_col];
            }
        }
        __syncthreads();
    }

    if (thread_row < M && thread_col + 3 < N) {
        FETCH_FLOAT4(C(thread_row, thread_col)) = FETCH_FLOAT4(tmp[0]);
    } else if (thread_row < M) {
        for (int i = 0; thread_col + i < N; ++i) {
            C(thread_row, thread_col + i) = tmp[i];
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
   
    constexpr unsigned int M_PER_BLOCK{32};
    constexpr unsigned int N_PER_BLOCK{32};  // block 处理的数据
    constexpr unsigned int K_PER_BLOCK{32};
    constexpr unsigned int THREAD_CNT_M = M_PER_BLOCK;
    constexpr unsigned int THREAD_CNT_N = N_PER_BLOCK / LENGTH_FLOAT4;  // 线程数

    dim3 block{THREAD_CNT_N, THREAD_CNT_M};  // therad{8, 32}, 每线程处理 1*4
    dim3 grid{(N + N_PER_BLOCK - 1) / N_PER_BLOCK, (M + M_PER_BLOCK -1) / M_PER_BLOCK};
    sgemm_gpu<THREAD_CNT_M, THREAD_CNT_N, M_PER_BLOCK, N_PER_BLOCK, K_PER_BLOCK><<<grid, block>>>(mA_device, mB_device, mC_device);

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