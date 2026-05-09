#include <cstdio>
#include <cuda_runtime.h>
#include <stbHelper/help_stb.cuh>

constexpr int CHANNELS{3};

__global__ void ColorToGray(unsigned char* input, unsigned char* output, int Pwidth, int Pheight) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < Pheight && col < Pwidth) {
        int target = row * Pwidth + col;
        int source = target * CHANNELS;

        unsigned char r = input[source];
        unsigned char g = input[source + 1];
        unsigned char b = input[source + 2];

        output[target] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
    }
}

int main() {
    int width, height, cpp;
    
    unsigned char* h_input = stbi_load("/home/yan/MAIN/src/main/input/CtoG.jpg", &width, &height, &cpp, 3);
    if (!h_input) {
        printf("Not Found\n");
        return -1;
    }

    int img_size = width * height;
    unsigned char* h_output = (unsigned char*)malloc(img_size);

    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, img_size * 3);
    cudaMalloc(&d_output, img_size);

    cudaMemcpy(d_input, h_input, img_size * 3, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    ColorToGray<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(h_output, d_output, img_size, cudaMemcpyDeviceToHost);

    stbi_write_jpg("/home/yan/MAIN/src/main/output/gray.jpg", width, height, 1, h_output, 100);

    stbi_image_free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}