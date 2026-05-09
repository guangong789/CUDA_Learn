#include <sgemm_global.cuh>

// SHARED MEMORY

template <unsigned int M_PER_BLOCK, unsigned int N_PER_BLOCK, unsigned int K_PER_BLOCK>
__global__ void sgemm_gpu(float *a, float *b, float *c) {
    constexpr unsigned int TILE_CNT = (K + K_PER_BLOCK - 1) / K_PER_BLOCK;
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * M_PER_BLOCK + ty;
    int col = blockIdx.x * N_PER_BLOCK + tx;

    __shared__ float shared_a[M_PER_BLOCK][K_PER_BLOCK];
    __shared__ float shared_b[K_PER_BLOCK][N_PER_BLOCK];

    float value = 0.0f;

    #pragma unroll
    for (int t = 0; t < TILE_CNT; ++t) {  // 每一轮 K_PER_BLOCK 长度
        int col_a = t * K_PER_BLOCK + tx;
        shared_a[ty][tx] = (row < M && col_a < K) ? A(row, col_a) : 0.0f;
        int row_b = t * K_PER_BLOCK + ty;
        shared_b[ty][tx] = (row_b < K && col < N) ? B(row_b, col) : 0.0f;
        __syncthreads();

        for (int k = 0; k < K_PER_BLOCK; ++k) {
            value += shared_a[ty][k] * shared_b[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C(row, col) = value;
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
    constexpr unsigned int N_PER_BLOCK{32};
    constexpr unsigned int K_PER_BLOCK{32};

    dim3 block{N_PER_BLOCK, M_PER_BLOCK};
    dim3 grid{(N + N_PER_BLOCK - 1) / N_PER_BLOCK, (M + M_PER_BLOCK - 1) / M_PER_BLOCK};
    sgemm_gpu<M_PER_BLOCK, N_PER_BLOCK, K_PER_BLOCK><<<grid, block>>>(mA_device, mB_device, mC_device);

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