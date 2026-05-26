#include <reduce_global.cuh>
#include <cub/cub.cuh>

template<int NUM_PER_BLOCK, int TPB>
__global__ void reduce(float* d_input, float* d_output)
{
    using Vec = float4;

    constexpr int VEC_PER_THREAD = 2;
    constexpr int FLOAT_PER_VEC  = 4;

    using BlockLoad =
        cub::BlockLoad<
            Vec,
            TPB,
            VEC_PER_THREAD,
            cub::BLOCK_LOAD_DIRECT>;

    using BlockReduce =
        cub::BlockReduce<
            float,
            TPB,
            cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockReduce::TempStorage reduce;
    } temp;

    int offset =
        blockIdx.x * NUM_PER_BLOCK;

    Vec thread_data[VEC_PER_THREAD];

    BlockLoad(temp.load).Load(
        reinterpret_cast<Vec*>(d_input)
        + offset / FLOAT_PER_VEC,
        thread_data);

    __syncthreads();

    float thread_sum = 0.f;

    #pragma unroll
    for(int i=0;i<VEC_PER_THREAD;++i)
    {
        thread_sum += thread_data[i].x;
        thread_sum += thread_data[i].y;
        thread_sum += thread_data[i].z;
        thread_sum += thread_data[i].w;
    }

    float block_sum =
        BlockReduce(temp.reduce)
            .Sum(thread_sum);

    if(threadIdx.x==0)
    {
        d_output[blockIdx.x]
            = block_sum;
    }
}

void launch_reduce_cub(
    float* d_input,
    float* d_output,
    int tpb)
{
    constexpr int NUM_PER_BLOCK =
        NUM_PER_THREAD
        * THREAD_PER_BLOCK;

    constexpr int BLOCK_NUM =
        N_PADDED
        / NUM_PER_BLOCK;

    dim3 grid(BLOCK_NUM);
    dim3 block(tpb);

    reduce<
        NUM_PER_BLOCK,
        THREAD_PER_BLOCK>
    <<<grid,block>>>(
        d_input,
        d_output);

    cudaDeviceSynchronize();
}

int main() {
    constexpr int thread_num     = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num      = thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block  = N_PADDED / block_num;

    float* input  = (float*)calloc(N_PADDED, sizeof(float));
    float* output = (float*)malloc(block_num * sizeof(float));
    float* res    = (float*)malloc(block_num * sizeof(float));

    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));
    cudaMalloc((void**)&d_output, block_num * sizeof(float));

    for (int i = 0; i < N; i++) input[i] = 2.0f * (float)drand48() - 1.0f;

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_cub(d_input, d_output, THREAD_PER_BLOCK);

    cudaFree(d_input);
    cudaFree(d_output);
    free(input);
    free(output);
    free(res);

    return 0;
}