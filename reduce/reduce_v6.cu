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

template <int TPB>
__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[TPB];

    int tid = threadIdx.x;
    int index = blockIdx.x * 2 * blockDim.x + tid;
    shared[tid] = d_input[index] + d_input[index + blockDim.x];
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

void launch_reduce_v6(float* d_input, float* d_output, int tpb) {
    int block_num = N_PADDED / (2 * tpb);

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<THREAD_PER_BLOCK><<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();
}

int main() {
    float* input = (float*)calloc(N_PADDED, sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));

    constexpr int block_num = N_PADDED / THREAD_PER_BLOCK;
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0 * (float)drand48() - 1.0;

    // reduce_cpu(2 * THREAD_PER_BLOCK, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v6(d_input, d_output, THREAD_PER_BLOCK);

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