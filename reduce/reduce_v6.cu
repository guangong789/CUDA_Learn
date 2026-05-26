#include <reduce_global.cuh>

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
    int index = blockIdx.x * NUM_PER_THREAD * blockDim.x + 4 * tid;

    float4 reg0 = FETCH_FLOAT4(d_input + index);
    float4 reg1 = FETCH_FLOAT4(d_input + index + 4 * blockDim.x);

    float sum = 0.0f;
    sum += reg0.x + reg0.y + reg0.z + reg0.w;
    sum += reg1.x + reg1.y + reg1.z + reg1.w;

    shared[tid] = sum;
    __syncthreads();

    if (TPB >= 1024) {
        if (tid < 512) { shared[tid] += shared[tid + 512]; }
        __syncthreads();
    }
    if (TPB >= 512) {
        if (tid < 256) { shared[tid] += shared[tid + 256]; }
        __syncthreads();
    }
    if (TPB >= 256) {
        if (tid < 128) { shared[tid] += shared[tid + 128]; }
        __syncthreads();
    }
    if (TPB >= 128) {
        if (tid < 64) { shared[tid] += shared[tid + 64]; }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(shared, tid);
    }

    if (tid == 0) {
        d_output[blockIdx.x] = shared[0];
    }
}

void launch_reduce_v6(float* d_input, float* d_output, int tpb)  {
    int block_num = N_PADDED / (NUM_PER_THREAD * tpb);

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<THREAD_PER_BLOCK><<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();
}

int main() {
    constexpr int block_num = N_PADDED / (NUM_PER_THREAD * THREAD_PER_BLOCK);

    float* input  = (float*)calloc(N_PADDED, sizeof(float));
    float* output = (float*)malloc(block_num * sizeof(float));
    float* res    = (float*)malloc(block_num * sizeof(float));

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0f * (float)drand48() - 1.0f;

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v6(d_input, d_output, THREAD_PER_BLOCK);

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}