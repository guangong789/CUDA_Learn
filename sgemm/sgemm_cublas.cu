#include <cublas_v2.h>
#include <sgemm_global.cuh>

void launch_cublas_sgemm(float *a, float *b, float *c) {
    cublasHandle_t handle;

    cublasCreate(&handle);

    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b, N_PAD, a, K_PAD, &beta, c, N_PAD);

    cudaDeviceSynchronize();

    cublasDestroy(handle);
}

#ifdef SGEMM_STANDALONE

int main() {
    constexpr size_t mem_size_A = M * K_PAD * sizeof(float);
    constexpr size_t mem_size_B = K * N_PAD * sizeof(float);
    constexpr size_t mem_size_C = M * N_PAD * sizeof(float);

    float *mA_host = (float *)calloc(M * K_PAD, sizeof(float));
    float *mB_host = (float *)calloc(K * N_PAD, sizeof(float));
    float *mC_host_cpu = (float *)calloc(M * N_PAD, sizeof(float));
    float *mC_host_gpu = (float *)calloc(M * N_PAD, sizeof(float));

    random_m(M, K, mA_host);
    random_m(K, N, mB_host);

    float *mA_device, *mB_device, *mC_device;
    cudaMalloc((void **)&mA_device, mem_size_A);
    cudaMalloc((void **)&mB_device, mem_size_B);
    cudaMalloc((void **)&mC_device, mem_size_C);

    cudaMemcpy(mA_device, mA_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(mB_device, mB_host, mem_size_B, cudaMemcpyHostToDevice);

    // sgemm_cpu(mA_host, mB_host, mC_host_cpu);

    launch_cublas_sgemm(mA_device, mB_device, mC_device);

    // cudaMemcpy(mC_host_gpu, mC_device, mem_size_C, cudaMemcpyDeviceToHost);
    // cmp_m(mC_host_cpu, mC_host_gpu);

    cudaFree(mA_device);
    cudaFree(mB_device);
    cudaFree(mC_device);
    free(mA_host);
    free(mB_host);
    free(mC_host_cpu);
    free(mC_host_gpu);

    return 0;
}

#endif