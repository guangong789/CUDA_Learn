#include <reduce_global.cuh>
#include <random>

// BASELINE

__global__ void reduce(float* d_input, float* d_output, int tpb) {
    int tx  = threadIdx.x;
    int tid = blockIdx.x * tpb + tx;

    for (int stride = 1; stride < tpb; stride <<= 1) {
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

    reduce<<<grid, block>>>(d_input, d_output, tpb);
    cudaDeviceSynchronize();
}

int main() {
    constexpr int block_num = N / THREAD_PER_BLOCK;

    float* input  = (float*)calloc(N_PADDED, sizeof(float));
    float* output = (float*)malloc(block_num * sizeof(float));
    float* res    = (float*)malloc(block_num * sizeof(float));

    float* d_input  = nullptr;
    float* d_output = nullptr;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < N; ++i) {
        input[i] = dis(gen);
    }

    reduce_cpu(THREAD_PER_BLOCK, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v0(d_input, d_output, THREAD_PER_BLOCK);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    if (check(output, res, block_num)) printf("The ans is right\n");
    else printf("The ans is wrong\n");

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}