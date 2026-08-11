set(FLASH_ATTENTION_KERNELS
    flash_attention_v0.cu
)

# Reusable C ABI library for non-C++ callers such as the Rust binding.
add_library(cuda_operator_flash_attention STATIC
    ${FLASH_ATTENTION_KERNELS}
    flash_attention_ffi.cu
)

target_include_directories(cuda_operator_flash_attention
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(cuda_operator_flash_attention PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_compile_options(cuda_operator_flash_attention
    PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
)

foreach(KERNEL_FILE ${FLASH_ATTENTION_KERNELS})
    get_filename_component(TARGET_NAME
        ${KERNEL_FILE}
        NAME_WE
    )

    add_executable(${TARGET_NAME} ${KERNEL_FILE})

    target_include_directories(${TARGET_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_definitions(${TARGET_NAME}
        PRIVATE
        FLASH_ATTENTION_STANDALONE
    )

    target_compile_options(${TARGET_NAME}
        PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo -Xptxas=-v>
    )
endforeach()

# Benchmark
add_executable(bm_flash_attention
    bm_flash_attention.cu
    ${FLASH_ATTENTION_KERNELS}
)

target_include_directories(bm_flash_attention
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(bm_flash_attention
    PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
)
