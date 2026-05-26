#include <reduce_global.cuh>

// FLOAT4 VECTOR LOAD + ADD DURING LOAD
 
template <int TPB>
__global__ void reduce(float* d_input, float* d_output) {
    __shared__ float shared[TPB];

    int tx = threadIdx.x; 
    int tid = blockIdx.x * NUM_PER_THREAD * blockDim.x + 4 * tx;

    float4 reg0 = FETCH_FLOAT4(d_input + tid);
    float4 reg1 = FETCH_FLOAT4(d_input + tid + 4 * blockDim.x);
   
    float sum = 0.f;
    sum += reg0.x + reg0.y + reg0.z + reg0.w;
    sum += reg1.x + reg1.y + reg1.z + reg1.w;
    shared[tx] = sum;
    __syncthreads();
 
    #pragma unroll
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (tx < i) {
            shared[tx] += shared[tx + i];
        }
        __syncthreads();
    }

    if (tx == 0) {
        d_output[blockIdx.x] = shared[0];
    }
}

void launch_reduce_v4(float* d_input, float* d_output, int tpb){
    int block_num = N_PADDED / (NUM_PER_THREAD * tpb);

    dim3 grid(block_num);
    dim3 block(tpb);

    reduce<THREAD_PER_BLOCK><<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();
}

int main() {
    float* input = (float*)calloc(N_PADDED, sizeof(float));
    float* d_input;
    cudaMalloc((void**)&d_input, N_PADDED * sizeof(float));

    constexpr int block_num = N_PADDED / (NUM_PER_THREAD * THREAD_PER_BLOCK);
    float* output = (float*)malloc(block_num * sizeof(float));
    float* d_output;
    cudaMalloc((void**)&d_output, block_num * sizeof(float));
    float* res = (float*)malloc(block_num * sizeof(float));

    for (int i = 0; i < N; ++i) input[i] = 2.0 * (float)drand48() - 1.0;

    // reduce_cpu(NUM_PER_THREAD*THREAD_PER_BLOCK, input, res);

    cudaMemcpy(d_input, input, N_PADDED * sizeof(float), cudaMemcpyHostToDevice);
    launch_reduce_v4(d_input, d_output, THREAD_PER_BLOCK);

    // cudaMemcpy(output, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    // if (check(output, res, block_num)) printf("The ans is right\n");
    // else printf("The ans is wrong\n");

    free(input);
    free(output);
    free(res);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}