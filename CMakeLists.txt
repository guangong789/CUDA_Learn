cmake_minimum_required(VERSION 4.0)
project(CUDA_Learning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(sgemm)
add_subdirectory(reduce)
add_subdirectory(softmax)
add_subdirectory(flash_attention)
add_subdirectory(src/main)
add_subdirectory(src/ncu_learn)