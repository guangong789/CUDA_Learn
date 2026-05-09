#pragma once
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
}

inline int _getSPcores(const cudaDeviceProp& prop) {
    int mp = prop.multiProcessorCount;
    switch (prop.major) {
        case 2: return (prop.minor == 1) ? mp * 48 : mp * 32; // Fermi
        case 3: return mp * 192;                             // Kepler
        case 5: return mp * 128;                             // Maxwell
        case 6: return (prop.minor == 0) ? mp * 64 : mp * 128; // Pascal
        case 7: return mp * 64;                              // Volta & Turing
        case 8: return (prop.minor == 0) ? mp * 64 : mp * 128; // Ampere
        case 9: return mp * 128;                             // Hopper & Blackwell
        default: return mp * 128; // 默认值
    }
}

void setGPU() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        printf("No CUDA compatible GPU found\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    int cores = _getSPcores(prop);

    printf("==================== GPU Device Info ====================\n");
    printf("Device Name           : %s\n", prop.name);
    printf("Compute Capability    : %d.%d\n", prop.major, prop.minor);
    printf("SM Count              : %d\n", prop.multiProcessorCount);
    printf("CUDA Cores (Total)    : %d\n", cores);
    printf("CUDA Cores / SM       : %d\n", cores / prop.multiProcessorCount);
    
    printf("\n-------------------- Memory Hierarchy --------------------\n");
    printf("Global Memory         : %.2f GB\n", (float)prop.totalGlobalMem / (1024.f * 1024.f * 1024.f));
    printf("L2 Cache Size         : %.2f MB\n", (float)prop.l2CacheSize / (1024.f * 1024.f));
    printf("Shared Mem / SM       : %.2f KB\n", (float)prop.sharedMemPerMultiprocessor / 1024.f);
    printf("Max Shared Mem / Block: %.2f KB\n", (float)prop.sharedMemPerBlock / 1024.f);
    printf("Registers / SM        : %d\n", prop.regsPerMultiprocessor);
    
    printf("\n----------------- Memory Bound Analysis -----------------\n");
    printf("Memory Bus Width      : %d bits\n", prop.memoryBusWidth);
    printf("Memory Clock Rate     : %.2f GHz\n", (float)prop.memoryClockRate / 1e6);
    
    double memBW = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6;
    printf("Theoretical Bandwidth : %.2f GB/s\n", memBW);
    
    printf("\n---------------- Compute Bound Analysis -----------------\n");
    double peakFlops = (double)cores * prop.clockRate * 2.0 / 1.0e6;
    printf("GPU Max Clock Rate    : %.2f GHz\n", (float)prop.clockRate / 1e6);
    printf("Peak FP32 Performance : %.2f GFLOPS\n", peakFlops);

    printf("\n---------------- Execution Configuration ----------------\n");
    printf("Warp Size             : %d\n", prop.warpSize);
    printf("Max Threads / Block   : %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads / SM      : %d\n", prop.maxThreadsPerMultiProcessor);
    printf("=========================================================\n");
}