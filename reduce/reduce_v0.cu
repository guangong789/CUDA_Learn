#include <reduce_global.cuh>
#include <random>

// BASELINE

__global__ void reduce(float* d_input, float* d_output) {
    int tx  = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;

    #pragma unroll
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (tx % (2 * stride) == 0) {
            d_input[tid] += d_input[tid + stride];
        }
        __syncthreads();
    }

    if (tx == 0) {
        d_output[blockIdx.x] = d_input[tid];
    }
}

void launch_reduce_v0(float* d_input, float* d_output, int tpb) {
    int block_num = N_PADDED / tpb;

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<<<grid, block>>>(d_input, d_output);
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
 
    // reduce_cpu(THREAD_PER_BLOCK, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v0(d_input, d_output, THREAD_PER_BLOCK);

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