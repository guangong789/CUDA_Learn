set(SOFTMAX_KERNELS
    softmax_v0.cu
    softmax_v1.cu
    softmax_v2.cu
    softmax_v3.cu
    softmax_v4.cu
    softmax_v5.cu
)

foreach(KERNEL_FILE ${SOFTMAX_KERNELS})
    get_filename_component(TARGET_NAME
        ${KERNEL_FILE}
        NAME_WE
    )

    add_executable(
        ${TARGET_NAME}
        ${KERNEL_FILE}
    )

    target_include_directories(${TARGET_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_definitions(${TARGET_NAME}
        PRIVATE
        SOFTMAX_STANDALONE
    )

    target_compile_options(${TARGET_NAME}
        PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo -Xptxas=-v>
    )

endforeach()

# Benchmark
add_executable(
    bm_softmax
    bm_softmax.cu
    ${SOFTMAX_KERNELS}
)

target_include_directories(
    bm_softmax
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(
    bm_softmax
    PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
)