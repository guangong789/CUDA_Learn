#include <reduce_global.cuh>

// IDLE THREAD: ADD DURING LOAD

__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[THREAD_PER_BLOCK];

    int tx = threadIdx.x;
    int tid = blockIdx.x * 2 * blockDim.x + tx;
    shared[tx] = d_input[tid] + d_input[tid + blockDim.x];
    __syncthreads();
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tx < i) {
            shared[tx] += shared[tx + i];
        }
        __syncthreads();
    }
    if (tx == 0) {
        d_output[blockIdx.x] = shared[tx];
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