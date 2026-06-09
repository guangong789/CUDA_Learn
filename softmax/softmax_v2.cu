#include <softmax_global.cuh>

// float4 vectorized

template<int NUM_PER_THREAD>
__global__ void softmax_v2(float* input, float* output) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int col = tid * NUM_PER_THREAD;
    const int idx = row * N + col;

    __shared__ float smem[N / NUM_PER_THREAD];
    float4 vals = FETCH_FLOAT4(input[idx]);
    float local_max = fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w));
    smem[tid] = local_max;
    __syncthreads();
    // Block Reduce Max
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        }
        __syncthreads();
    }
    if (tid < 32) {
        float vmax = warpReduceMax(smem[tid]);
        if (tid == 0) smem[0] = vmax;
    }
    __syncthreads();
    const float max_val = smem[0];
    // Exp
    float4 exps = {expf(vals.x - max_val), expf(vals.y - max_val), expf(vals.z - max_val), expf(vals.w - max_val)};
    float local_sum = exps.x + exps.y + exps.z + exps.w;
    smem[tid] = local_sum;
    __syncthreads();
    // Block Reduce Sum
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        float vsum = warpReduceSum(smem[tid]);
        if (tid == 0) smem[0] = vsum;
    }
    __syncthreads();
    const float sum = smem[0];
    // Normalize
    float4 out = {exps.x / sum, exps.y / sum, exps.z / sum, exps.w / sum};
    FETCH_FLOAT4(output[idx]) = out;
}

void launch_softmax_v2(float* input, float* output) {
    constexpr int NUM_PER_THREAD = 4;
    dim3 block{N / NUM_PER_THREAD};
    dim3 grid{M};

    softmax_v2<NUM_PER_THREAD><<<grid, block>>>(input, output);
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

    launch_softmax_v2(mD_input, mD_output);
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