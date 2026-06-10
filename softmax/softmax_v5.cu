#include <softmax_global.cuh>

// online softmax

template<int WARPS_PER_BLOCK>
__global__ void softmax_v5(float* input, float* output) {
    const int tid = threadIdx.x;  // [0, 127]
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    int row = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    
    // thread online
    float thread_max = -FLT_MAX;
    float thread_sum = 0.0f;
    #pragma unroll
    for (int col = lane_id; col < N; col += 32) {
        float cur = input[row * N + col];
        float new_max = fmaxf(cur, thread_max);
        thread_sum = thread_sum * expf(thread_max - new_max) + expf(cur - new_max);
        thread_max = new_max;
    }
    // block online
    Pair out = warpReduceOnline(thread_max, thread_sum);
    float row_max = __shfl_sync(0xffffffff, out.m, 0);
    float row_sum = __shfl_sync(0xffffffff, out.s, 0);
    // normalize
    #pragma unroll
    for (int col = lane_id; col < N; col += 32) {
        float exp = expf(input[row * N + col] - row_max);
        output[row * N + col] = exp / row_sum;
    }
}

void launch_softmax_v5(float* input, float* output) {
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREAD_PER_BLOCK =  WARPS_PER_BLOCK * 32;
    dim3 block{THREAD_PER_BLOCK};
    dim3 grid{M / WARPS_PER_BLOCK};

    softmax_v5<WARPS_PER_BLOCK><<<grid, block>>>(input, output);
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

    launch_softmax_v5(mD_input, mD_output);
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