#include <reduce_global.cuh>

// DIVERGENCE BRANCH

template <int TPB>
__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[TPB];

    int tx = threadIdx.x;
    int tid = blockIdx.x * blockDim.x + tx;
    shared[tx] = d_input[tid];
    __syncthreads();

    #pragma unroll
    for (int i = 1; i < blockDim.x; i <<= 1) {
        if (tx < blockDim.x / (2 * i)) {
            int source = tx * 2 * i;
            shared[source] += shared[source + i];
        }
        __syncthreads();
    }

    if (tx == 0) {
        d_output[blockIdx.x] = shared[tx];
    }
}

void launch_reduce_v2(float* d_input, float* d_output, int tpb) {
    int block_num = N_PADDED / tpb;

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

    // reduce_cpu(THREAD_PER_BLOCK, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v2(d_input, d_output, THREAD_PER_BLOCK);

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