#include <reduce_global.cuh>

// MULTI BLOCK SIZE

__device__ void warpReduce(volatile float* cache, int tid) {
    cache[tid] = cache[tid] + cache[tid + 32];
    cache[tid] = cache[tid] + cache[tid + 16];
    cache[tid] = cache[tid] + cache[tid + 8];
    cache[tid] = cache[tid] + cache[tid + 4];
    cache[tid] = cache[tid] + cache[tid + 2];
    cache[tid] = cache[tid] + cache[tid + 1];
}

template <int NUM_PER_BLOCK, int NUM_PER_THREAD, int TPB>
__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[TPB];

    int tid = threadIdx.x;
    int offset = blockIdx.x * NUM_PER_BLOCK;
    shared[tid] = 0;
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; ++i) {
        shared[tid] += d_input[offset + i * THREAD_PER_BLOCK + tid];
    }
    __syncthreads();
    if (TPB >= 512) {
        if (tid < 256) shared[tid] += shared[tid + 256];
        __syncthreads();
    }
    if (TPB >= 256) {
        if (tid < 128) shared[tid] += shared[tid + 128];
        __syncthreads();
    }
    if (TPB >= 128) {
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

void launch_reduce_v7(float* d_input, float* d_output, int tpb) {
    constexpr int thread_num = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num =  thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block = N_PADDED / block_num;

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<num_per_block, NUM_PER_THREAD, THREAD_PER_BLOCK><<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();
}

int main() {
    float* input = (float*)calloc(N_PADDED, sizeof(float));
    float* d_input; 
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));

    constexpr int thread_num = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num =  thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block = N_PADDED / block_num;
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0 * (float)drand48() - 1.0;

    // reduce_cpu(num_per_block, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v7(d_input, d_output, THREAD_PER_BLOCK);

    // cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output, res, block_num)) printf("The ans is right\n");
    // else printf("The ans is wrong\n");

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}