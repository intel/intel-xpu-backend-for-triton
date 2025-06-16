set(CUTLASS_KERNEL_FLAGS ${CUTLASS_KERNEL_FLAGS}
  -fsycl
  -fsycl-device-code-split=per_kernel
  -fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21
  "SHELL:-Xspirv-translator=intel_gpu_pvc --spirv-ext=+SPV_INTEL_split_barrier"
  "SHELL:-Xspirv-translator=intel_gpu_bmg_g21 --spirv-ext=+SPV_INTEL_split_barrier"
  -Xs "-options \"-igc_opts 'VISAOptions=-perfmodel,VectorAliasBBThreshold=1000,ExtraOCLOptions=-cl-intel-256-GRF-per-thread'\" -options -ze-opt-large-register-file"
)

Python3_add_library(cutlass_kernel MODULE WITH_SOABI python_main.cpp)

target_compile_options(cutlass_kernel PRIVATE "-fsycl" "-fsycl-targets=intel_gpu_pvc,intel_gpu_bmg_g21" "-fpreview-breaking-changes")
target_compile_options(cutlass_kernel PRIVATE "-DCUTLASS_ENABLE_SYCL")
target_compile_options(cutlass_kernel PRIVATE "-DSYCL_INTEL_TARGET")

target_link_options(cutlass_kernel PRIVATE ${CUTLASS_KERNEL_FLAGS})
target_link_libraries(cutlass_kernel PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_include_directories(cutlass_kernel PUBLIC "${CUTLASSLibrary_INCLUDE_DIR}" "${CUTLASSLibrary_INCLUDE_TOOL_DIR}" "${CUTLASSLibrary_INCLUDE_APPLICATION_DIR}")

add_subdirectory(gemm)
add_subdirectory(attention)

install(TARGETS cutlass_kernel LIBRARY DESTINATION .)
