#include <reduce_global.cuh>
#include <cub/cub.cuh>

// CUB REFERENCE

template <int NUM_PER_BLOCK, int NUM_PER_THREAD, int TPB>
__global__ void reduce(float* d_input, float* d_output) {
    using BlockLoad   = cub::BlockLoad<float, TPB, NUM_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockReduce = cub::BlockReduce<float, TPB>;

    __shared__ union TempStorage {
        typename BlockLoad::TempStorage   load;
        typename BlockReduce::TempStorage reduce;
    } temp;

    int offset = blockIdx.x * NUM_PER_BLOCK;
    float thread_data[NUM_PER_THREAD];

    BlockLoad(temp.load).Load(d_input + offset, thread_data);

    __syncthreads(); 

    float thread_sum = 0.f;
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i++) {
        thread_sum += thread_data[i];
    }

    float block_sum = BlockReduce(temp.reduce).Sum(thread_sum);

    if (threadIdx.x == 0) {
        d_output[blockIdx.x] = block_sum;
    }
}

void launch_reduce_cub(float* d_input, float* d_output, int tpb) {
    constexpr int thread_num = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num  = thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block = N_PADDED / block_num;

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<num_per_block, NUM_PER_THREAD, THREAD_PER_BLOCK><<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();
}

int main() {
    constexpr int thread_num = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num  = thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block = N_PADDED / block_num;

    float* input = (float*)calloc(N_PADDED, sizeof(float));
    float* output = (float*)malloc(block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    for (int i = 0; i < N; i++) input[i] = 2.f * (float)drand48() - 1.f;

    reduce_cpu(num_per_block, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_cub(d_input, d_output, THREAD_PER_BLOCK);

    cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    if (check(output, res, block_num)) printf("The ans is right\n");
    else printf("The ans is wrong\n");

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(res);

    return 0;
}