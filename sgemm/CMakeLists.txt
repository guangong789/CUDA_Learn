set(SGEMM_KERNELS
    sgemm_v0.cu
    sgemm_v1.cu
    sgemm_v2.cu
    sgemm_v3.cu
    sgemm_v4.cu
    sgemm_v5.cu
    sgemm_v6.cu
    sgemm_cublas.cu

)

foreach(KERNEL_FILE ${SGEMM_KERNELS})
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
        SGEMM_STANDALONE
    )

    target_link_libraries(${TARGET_NAME}
        PRIVATE
        cublas
    )

    target_compile_options(${TARGET_NAME}
        PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
    )

endforeach()

add_executable(bm_sgemm
    bm_sgemm.cu
    ${SGEMM_KERNELS}
)

target_include_directories(bm_sgemm
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(bm_sgemm
    PRIVATE
    cublas
)

target_compile_options(bm_sgemm
    PRIVATE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
)