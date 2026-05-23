#include <softmax_global.cuh>

__global__ void softmax_v0(float *input, float *output) {
    int row = blockIdx.x, col = threadIdx.x;
    int index = row * N + col;

    if (row >= M) return;

    __shared__ float shared_m[N];
    
    float val = -FLT_MAX;
    if (col < N) val = input[index];
    shared_m[col] = val;
    __syncthreads();
    // reduce max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            shared_m[col] = max(shared_m[col], shared_m[col + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_m[0];
    // exp
    float exp_val = 0.0f;
    if (col < N) exp_val = expf(val - max_val);
    shared_m[col] = exp_val;
    __syncthreads();
    // reduce sum
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if(col < stride) shared_m[col] += shared_m[col + stride];
        __syncthreads();
    }
    // normalize
    float sum = shared_m[0];
    if (col < N) output[index] = exp_val / sum;
}

void launch_softmax_v0(float* input, float* output) {
    dim3 block{N};
    dim3 grid{M};

    softmax_v0<<<grid, block>>>(input, output);
    cudaDeviceSynchronize();
}

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

    softmax_cpu(mH_input, mH_ref);

    launch_softmax_v0(mD_input, mD_output);
    cudaMemcpy(mH_output, mD_output, m_size, cudaMemcpyDeviceToHost);

    softmax_cmp(mH_ref, mH_output);

    cudaFree(mD_input);
    cudaFree(mD_output);
    free(mH_input);
    free(mH_output);
    free(mH_ref);

    return 0;
}