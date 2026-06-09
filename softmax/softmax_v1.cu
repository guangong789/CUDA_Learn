#include <softmax_global.cuh>

// shuffle

__global__ void softmax_v1(float *input, float *output) {
    int row = blockIdx.x, col = threadIdx.x;
    int index = row * N + col;

    __shared__ float shared_m[N];
    float val = input[index];
    shared_m[col] = val;
    __syncthreads();
    
    // reduce max
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (col < stride) {
            shared_m[col] = fmaxf(shared_m[col], shared_m[col + stride]);
        }
        __syncthreads();
    }
    if (col < 32) {
        float vmax = warpReduceMax(shared_m[col]);
        if (col == 0) shared_m[0] = vmax;
    }
    __syncthreads();
    float max_val = shared_m[0];
    // exp
    float exp_val = expf(val - max_val);
    shared_m[col] = exp_val;
    __syncthreads();
    // reduce sum
    for(int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if(col < stride) shared_m[col] += shared_m[col + stride];
        __syncthreads();
    }
    if (col < 32) {
        float vsum = warpReduceSum(shared_m[col]);
        if (col == 0) shared_m[0] = vsum;
    }
    __syncthreads();
    float sum = shared_m[0];
    // normalize
    output[index] = exp_val / sum;
}

void launch_softmax_v1(float* input, float* output) {
    dim3 block{N};
    dim3 grid{M};

    softmax_v1<<<grid, block>>>(input, output);
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

    launch_softmax_v1(mD_input, mD_output);
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