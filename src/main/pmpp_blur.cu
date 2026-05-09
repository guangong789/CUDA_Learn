#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <stbHelper/help_stb.cuh>

constexpr int CHANNELS{3};
constexpr int BLUR_SIZE{5};

__global__ void blur_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        for (int c = 0; c < CHANNELS; ++c) {
            int pixelSum = 0;
            int pixelCnt = 0;

            for (int rowOffset = -BLUR_SIZE; rowOffset <= BLUR_SIZE; ++rowOffset) {
                for (int colOffset = -BLUR_SIZE; colOffset <= BLUR_SIZE; ++colOffset) {
                    int curRow = row + rowOffset;
                    int curCol = col + colOffset;
                    if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                        int index = (curRow * width + curCol) * CHANNELS + c;

                        pixelSum += input[index];
                        ++pixelCnt;
                    }
                }
            }

            int outIndex = (row * width + col) * CHANNELS + c;
            output[outIndex] = static_cast<unsigned char>(pixelSum / pixelCnt);
        }
    }
}

int main() {
    int width;
    int height;
    int cpp;

    unsigned char* input_h = stbi_load("/home/yan/MAIN/src/main/input/CtoG.jpg", &width, &height, &cpp, CHANNELS);
    if (!input_h) {
        printf("Image load failed\n");
        return -1;
    }

    size_t img_size = width * height * CHANNELS * sizeof(unsigned char);

    unsigned char* input_d;
    unsigned char* output_d;

    cudaMalloc((void**)&input_d, img_size);
    cudaMalloc((void**)&output_d, img_size);
    cudaMemcpy(input_d, input_h, img_size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    blur_kernel<<<grid, block>>>(input_d, output_d, width, height);
    cudaDeviceSynchronize();

    unsigned char* output_h = (unsigned char*)malloc(img_size);

    cudaMemcpy(output_h, output_d, img_size, cudaMemcpyDeviceToHost);

    stbi_write_jpg("/home/yan/MAIN/src/main/output/blur.jpg", width, height, CHANNELS, output_h, 100);

    stbi_image_free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);

    return 0;
}