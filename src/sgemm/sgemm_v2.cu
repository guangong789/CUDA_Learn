#include <sgemm_global.cuh>

// BLOCKING

// 通过i，j和BLOCK_SIZE来正确映射
template <unsigned int BLOCK_SIZE, unsigned int STRIDE, unsigned int TILE>  // BLOCK_SIZE：有多少线程，STRIDE：每个线程干多少活
__global__ void sgemm_gpu(float *a, float *b, float *c) {
    const unsigned int TILE_CNT = (K + TILE - 1) / TILE;  // TILE：每个BLOCK覆盖的格点
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * TILE, block_col = blockIdx.x * TILE;  // 每个BLOCK的起始格点位置
    int thread_row = block_row + ty, thread_col = block_col + tx;  // 每个线程的起始格点位置

    __shared__ float shared_a[TILE][TILE + 1];
    __shared__ float shared_b[TILE][TILE + 1];
    float tmp[STRIDE][STRIDE] = {0.0f};

    int smem_row{0}, smem_col{0};

    #pragma unroll
    for (int t = 0; t < TILE_CNT; ++t) {
        for (int i = 0; i < STRIDE; ++i) {
            for (int j = 0; j < STRIDE; ++j) {
                smem_row = ty + i * BLOCK_SIZE;
                smem_col = tx + j * BLOCK_SIZE;
                int a_row = block_row + (ty + i * BLOCK_SIZE), a_col = t * TILE + (tx + j * BLOCK_SIZE);  // 沿K维扫描
                shared_a[smem_row][smem_col] = (a_row < M && a_col < K) ? A(a_row, a_col) : 0.0f;
                int b_row = t * TILE + (ty + i * BLOCK_SIZE), b_col = block_col + (tx + j * BLOCK_SIZE);
                shared_b[smem_row][smem_col] = (b_row < K && b_col < N) ? B(b_row, b_col) : 0.0f;
            }
        }
        __syncthreads();

        int valid_k = (t == TILE_CNT - 1) ? (K - t * TILE) : TILE;
        #pragma unroll
        for (int k = 0; k < valid_k; ++k) {
            for (int j = 0; j < STRIDE; ++j) {
                for (int i = 0; i < STRIDE; ++i) {
                    smem_row = ty + i * BLOCK_SIZE;
                    smem_col = tx + j * BLOCK_SIZE;
                    tmp[i][j] += shared_a[smem_row][k] * shared_b[k][smem_col];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < STRIDE; ++i) {
        for (int j = 0; j < STRIDE; ++j) {
            int global_row = thread_row + i * BLOCK_SIZE, global_col = thread_col + j * BLOCK_SIZE;
            if (global_row < M && global_col < N) {
                C(global_row, global_col) = tmp[i][j];
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

    constexpr unsigned int STRIDE{2};  // 每线程处理的数据 2*2
    constexpr unsigned int THREAD_CNT_M{16};
    constexpr unsigned int THREAD_CNT_N{16};  // 线程数
    constexpr unsigned int M_PER_BLOCK = STRIDE * THREAD_CNT_M;
    constexpr unsigned int N_PER_BLOCK = STRIDE * THREAD_CNT_N;  // block 处理的数据

    dim3 block{THREAD_CNT_N, THREAD_CNT_M};
    dim3 grid{(N + N_PER_BLOCK - 1) / N_PER_BLOCK, (M + M_PER_BLOCK -1) / M_PER_BLOCK};  // 减少块的数量，增加每个线程的工作量
    sgemm_gpu<THREAD_CNT_N, STRIDE, N_PER_BLOCK><<<grid, block>>>(mA_device, mB_device, mC_device);

    cudaDeviceSynchronize();
    cudaMemcpy(mC_host_gpu, mC_device, mem_size_C, cudaMemcpyDeviceToHost);

    float err = cmp_m(mC_host_cpu, mC_host_gpu);
    if (err > 1e-3f) printf("FAILED\n");
    else printf("PASSED\n");

    cudaFree(mA_device);
    cudaFree(mB_device);
    cudaFree(mC_device);
    free(mA_host);
    free(mB_host);
    free(mC_host_cpu);
    free(mC_host_gpu);

    return 0;
}