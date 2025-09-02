set(CUTLASS_KERNEL_FLAGS ${CUTLASS_KERNEL_FLAGS}
  -fsycl
  -fsycl-device-code-split=per_kernel
  -fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21
  "SHELL:-Xspirv-translator=intel_gpu_pvc --spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  "SHELL:-Xspirv-translator=intel_gpu_bmg_g21 --spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
  -Xs "-options \"-igc_opts 'VISAOptions=-perfmodel,VectorAliasBBThreshold=1000,ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" -options -ze-opt-large-register-file"
)

# Path to the configuration tool
set(CONFIG_TOOL ${CMAKE_CURRENT_SOURCE_DIR}/config-tool.py)

# Input and output files
# The name of this file must be kept in sync with the best known CUTLASS config.
# TODO: Re-enable gemm config input to come from `CUTLASSLibrary_BENCHMARK_CONFIG_DIR`
# set(GEMM_CONFIG_INPUT  ${CUTLASSLibrary_BENCHMARK_CONFIG_DIR}/input_gemm.in)
set(GEMM_CONFIG_INPUT ${CMAKE_CURRENT_SOURCE_DIR}/gemm/input_gemm.in)
set(GEMM_CONFIG_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gemm_table.hpp)
set(GEMM_CONFIG_NAME "gemm_config")

# Use a custom command to generate a C++ header with the configuration table
# from the CUTLASS benchmark configuration.
add_custom_command(
    OUTPUT ${GEMM_CONFIG_OUTPUT}
    COMMAND ${CMAKE_COMMAND} -E echo "Generating GEMM config header..."
    COMMAND ${Python3_EXECUTABLE} ${CONFIG_TOOL} ${GEMM_CONFIG_INPUT} -o ${GEMM_CONFIG_OUTPUT} --name ${GEMM_CONFIG_NAME}
    DEPENDS ${GEMM_CONFIG_INPUT} ${CONFIG_TOOL}
    COMMENT "Generate GEMM configuration"
    VERBATIM
)

# Create a target that other targets can depend on
add_custom_target(generate_gemm_config DEPENDS ${GEMM_CONFIG_OUTPUT})

Python3_add_library(cutlass_kernel MODULE WITH_SOABI python_main.cpp)

target_compile_options(cutlass_kernel PRIVATE "-fsycl" "-fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21" "-fpreview-breaking-changes")
target_compile_options(cutlass_kernel PRIVATE "-DCUTLASS_ENABLE_SYCL")
target_compile_options(cutlass_kernel PRIVATE "-DSYCL_INTEL_TARGET")
target_compile_definitions(cutlass_kernel PRIVATE GEMM_CONFIG_HEADER=\"${GEMM_CONFIG_OUTPUT}\")
target_compile_definitions(cutlass_kernel PRIVATE GEMM_CONFIG_NAME=\"${GEMM_CONFIG_NAME}\")

target_link_options(cutlass_kernel PRIVATE ${CUTLASS_KERNEL_FLAGS})
target_link_libraries(cutlass_kernel PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_include_directories(cutlass_kernel PUBLIC "${CUTLASSLibrary_INCLUDE_DIR}" "${CUTLASSLibrary_INCLUDE_TOOL_DIR}" "${CUTLASSLibrary_INCLUDE_APPLICATION_DIR}" "${CUTLASSLibrary_INCLUDE_BENCHMARK_DIR}")

add_dependencies(cutlass_kernel generate_gemm_config)

add_subdirectory(gemm)
add_subdirectory(attention)

install(TARGETS cutlass_kernel LIBRARY DESTINATION .)
