#include <reduce_global.cuh>

// COMPLETELY UNROLL

__device__ void warpReduce(volatile float* cache, int tid) {
    cache[tid] = cache[tid] + cache[tid + 32];
    cache[tid] = cache[tid] + cache[tid + 16];
    cache[tid] = cache[tid] + cache[tid + 8];
    cache[tid] = cache[tid] + cache[tid + 4];
    cache[tid] = cache[tid] + cache[tid + 2];
    cache[tid] = cache[tid] + cache[tid + 1];
}

__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[THREAD_PER_BLOCK];

    int tid = threadIdx.x;
    int index = blockIdx.x * 2 * blockDim.x + tid;
    shared[tid] = d_input[index] + d_input[index + blockDim.x];
    __syncthreads();
    if (THREAD_PER_BLOCK >= 512) {
        if (tid < 256) shared[tid] += shared[tid + 256];
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 256) {
        if (tid < 128) shared[tid] += shared[tid + 128];
        __syncthreads();
    }
    if (THREAD_PER_BLOCK >= 128) {
        if (tid < 64) shared[tid] += shared[tid + 64];
        __syncthreads();
    }
    if (tid < 32) {
        warpReduce(shared, tid);
    }
    if (tid == 0) {
        d_output[blockIdx.x] = shared[tid];
    }
}

int main() {
    float* input = (float*)malloc(N * sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    constexpr int block_num = N / (2 * THREAD_PER_BLOCK);
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0 * (float)drand48() - 1.0;
    
    reduce_cpu(block_num, 2 * THREAD_PER_BLOCK, input, res);

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 Grid(block_num);
    dim3 Block(THREAD_PER_BLOCK);
    reduce<<<Grid, Block>>>(d_input, d_output);

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