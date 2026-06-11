#include <softmax_global.cuh>

// warp-level block reduction

template<int NUM_PER_THREAD>
__global__ void softmax_v3(float* input, float* output) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int col = tid * NUM_PER_THREAD;
    const int idx = row * N + col;

    float4 vals = FETCH_FLOAT4(input[idx]);
    float local_max = fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w));

    const int NUM_WARPS = N / (NUM_PER_THREAD * 32);  // 8
    __shared__ float smem[NUM_WARPS];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    // warp reduce max
    float warp_max = warpReduceMax(local_max);
    if (lane_id == 0) smem[warp_id] = warp_max;
    __syncthreads();
    // block reduce max
    if (warp_id == 0) {
        float block_max = (lane_id < NUM_WARPS) ? smem[lane_id] : -FLT_MAX;
        block_max = warpReduceMax(block_max);
        if (lane_id == 0) smem[0] = block_max;
    }
    __syncthreads();
    float max_val = smem[0];
    // exp
    float4 exps = {expf(vals.x - max_val), expf(vals.y - max_val), expf(vals.z - max_val), expf(vals.w - max_val)};
    float local_sum = exps.x + exps.y + exps.z + exps.w;
    // warp reduce sum
    float warp_sum = warpReduceSum(local_sum);
    if (lane_id == 0) smem[warp_id] = warp_sum;
    __syncthreads();
    // block reduce sum
    if (warp_id == 0) {
        float block_sum = (lane_id < NUM_WARPS) ? smem[lane_id] : 0.0f;
        block_sum = warpReduceSum(block_sum);
        if (lane_id == 0) smem[0] = block_sum;
    }
    __syncthreads();
    float sum = smem[0];
    // normalize
    float4 out = {exps.x / sum, exps.y / sum, exps.z / sum, exps.w / sum};
    FETCH_FLOAT4(output[idx]) = out;
}

void launch_softmax_v3(float* input, float* output) {
    constexpr int NUM_PER_THREAD = 4;
    dim3 block{N / NUM_PER_THREAD};
    dim3 grid{M};

    softmax_v3<NUM_PER_THREAD><<<grid, block>>>(input, output);
}

#ifdef SOFTMAX_STANDALONE

int main() {
    constexpr size_t m_size = M * N * sizeof(float);
    float *mH_input = (float *)calloc(M * N, sizeof(float));
    float *mH_output = (float *)calloc(M * N, sizeof(float));
    float *mH_ref = (float *)calloc(M * N, sizeof(float));

    random_m(M, N, mH_input);

    float *mD_input, *mD_output;
    cudaMalloc((void **)&mD_input, m_size);
    cudaMalloc((void **)&mD_output, m_size);
    cudaMemcpy(mD_input, mH_input, m_size, cudaMemcpyHostToDevice);

    // softmax_cpu(mH_input, mH_ref);

    launch_softmax_v3(mD_input, mD_output);
    cudaDeviceSynchronize();

    // cudaMemcpy(mH_output, mD_output, m_size, cudaMemcpyDeviceToHost);
    // softmax_cmp(mH_output, mH_ref);

    cudaFree(mD_input);
    cudaFree(mD_output);
    free(mH_input);
    free(mH_output);
    free(mH_ref);

    return 0;
}

#endif