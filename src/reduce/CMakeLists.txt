file(GLOB REDUCE_SOURCES *.cu)

foreach(source_file ${REDUCE_SOURCES})
    get_filename_component(exe_name ${source_file} NAME_WE)
    
    add_executable(${exe_name} ${source_file})
    
    target_compile_options(${exe_name} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-g -G>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo>
    )

    # target_link_libraries(${exe_name} PRIVATE cublas)
endforeach()