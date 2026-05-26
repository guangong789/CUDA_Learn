#include <reduce_global.cuh>

// SHUFFLE

constexpr int WARP_SIZE{32};

template <int NUM_PER_BLOCK, int NUM_PER_THREAD>
__global__ void reduce(float* d_input, float* d_output) {
    float sum = 0.0f;
    int tid = threadIdx.x;
    int offset = blockIdx.x * NUM_PER_BLOCK;

    // 以 float4 向量化跨步加载
    #pragma unroll
    for (int i = 0; i < NUM_PER_THREAD; i += 4) {
        int index = offset + i * THREAD_PER_BLOCK + tid * 4;
        float4 reg = FETCH_FLOAT4(d_input + index);
        sum += reg.x + reg.y + reg.z + reg.w;
    }

    // 第一阶段：Warp 内部寄存器级洗牌规约（完全摆脱 Shared Memory 依赖）
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    // 跨 Warp 通信：仅用极小的临时共享内存暂存每个 Warp 的局部和
    __shared__ float warpLevelSums[32];
    const int laneId = tid % WARP_SIZE;
    const int warpId = tid / WARP_SIZE;

    if (laneId == 0) {
        warpLevelSums[warpId] = sum;
    }
    __syncthreads();

    // 第二阶段：Warp0 处理整个 Block 的残余和
    if (warpId == 0) {
        sum = (laneId < blockDim.x / 32) ? warpLevelSums[laneId] : 0.0f;
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
    }

    if (tid == 0) {
        d_output[blockIdx.x] = sum;
    }
}

void launch_reduce_v7(float* d_input, float* d_output, int tpb) {
    constexpr int thread_num     = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num      = thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block  = N_PADDED / block_num;

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<num_per_block, NUM_PER_THREAD><<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();
}

int main() {
    float* input = (float*)calloc(N_PADDED, sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));

    constexpr int thread_num     = N_PADDED / NUM_PER_THREAD;
    constexpr int block_num      = thread_num / THREAD_PER_BLOCK;
    constexpr int num_per_block  = N_PADDED / block_num;

    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res    = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) {
        input[i] = 2.0 * (float)drand48() - 1.0;
    }

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v7(d_input, d_output, THREAD_PER_BLOCK);

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}