#include <softmax_global.cuh>

// warp per row

template<int WARPS_PER_BLOCK>
__global__ void softmax_v4(float* input, float* output) {
    const int tid = threadIdx.x;  // [0, 127]
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    // thread reduce max
    float thread_max = -FLT_MAX;
    #pragma unroll
    for (int col = lane_id * 4; col < N; col += 32 * 4) {
        float4 f4 = FETCH_FLOAT4(input[row * N + col]);
        float f4_max = fmaxf(fmaxf(f4.x, f4.y), fmaxf(f4.z, f4.w));
        thread_max = fmaxf(thread_max, f4_max);
    }
    // warp reduce max
    float max_val = warpReduceMax(thread_max);
    max_val = __shfl_sync(0xffffffff, max_val, 0);
    // exp & reduce sum
    float thread_sum = 0.f;
    #pragma unroll
    for (int col = lane_id * 4; col < N; col += 32 * 4) {
        float4 f4 = FETCH_FLOAT4(input[row * N + col]);
        float exp = expf(f4.x - max_val) + expf(f4.y - max_val) + expf(f4.z - max_val) + expf(f4.w - max_val);
        thread_sum += exp;
    }
    float sum_val = warpReduceSum(thread_sum);
    sum_val = __shfl_sync(0xffffffff, sum_val, 0);
    // normalize
    #pragma unroll
    for (int col = lane_id * 4; col < N; col += 32 * 4) {
        float inv_sum = 1.f / sum_val;
        float4 f4 = FETCH_FLOAT4(input[row * N + col]);
        float4 exp4 = {expf(f4.x - max_val), expf(f4.y - max_val), expf(f4.z - max_val), expf(f4.w - max_val)};
        FETCH_FLOAT4(output[row * N + col]) = {exp4.x * inv_sum, exp4.y * inv_sum, exp4.z * inv_sum, exp4.w * inv_sum};
    }
}

void launch_softmax_v4(float* input, float* output) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREAD_PER_BLOCK =  WARPS_PER_BLOCK * 32;
    dim3 block{THREAD_PER_BLOCK};
    dim3 grid{M / WARPS_PER_BLOCK};

    softmax_v4<WARPS_PER_BLOCK><<<grid, block>>>(input, output);
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

    launch_softmax_v4(mD_input, mD_output);
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