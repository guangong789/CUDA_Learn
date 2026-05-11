#include <reduce_global.cuh>

// SHUFFLE

constexpr int WARP_SIZE{32};

template <unsigned int NUM_PER_BLOCK, unsigned int NUM_PER_THREAD>
__global__ void reduce(float* d_input, float* d_output) {
    float sum = 0.0f;

    int tid = threadIdx.x;
    int offset = blockIdx.x * NUM_PER_BLOCK;
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        sum += d_input[offset + i * THREAD_PER_BLOCK + tid];
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);  // warp 的一半
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // register -> shared memory
    __shared__ float warpLevelSums[32];
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;
    if (laneId == 0) {
        warpLevelSums[warpId] = sum;
    }
    __syncthreads();
    // shared memory -> register
    if (warpId == 0) {
        sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.0f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }
    if (tid == 0) {
        d_output[blockIdx.x] = sum;
    }
}

int main() {
    constexpr int N = 32 * 1024 * 1024;
    float* input = (float*)malloc(N * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    constexpr int block_num = 1024;
    constexpr int num_per_block =  N / 1024;
    constexpr int num_per_thread = num_per_block / THREAD_PER_BLOCK;  //  每个 thread 处理的数
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0 * (float)drand48() - 1.0;
    
    reduce_cpu(block_num, num_per_block, input, res);

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);

    for (int i = 0; i < 10; ++i) {
        reduce<num_per_block, num_per_thread><<<Grid, Block>>>(d_input, d_output);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, block_num *sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, res, block_num)) printf("the ans is right\n");
    else printf("the ans is wrong\n");

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}