set(CUTLASS_KERNEL_FLAGS ${CUTLASS_KERNEL_FLAGS}
  -fsycl
  -fsycl-device-code-split=per_kernel
  -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier
)

Python3_add_library(cutlass_kernel MODULE WITH_SOABI python_main.cpp)

target_compile_options(cutlass_kernel PRIVATE "-fsycl" "-fpreview-breaking-changes")
target_compile_options(cutlass_kernel PRIVATE "-DCUTLASS_ENABLE_SYCL")
target_compile_options(cutlass_kernel PRIVATE "-DSYCL_INTEL_TARGET")

target_link_options(cutlass_kernel PRIVATE ${CUTLASS_KERNEL_FLAGS})
target_link_libraries(cutlass_kernel PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_include_directories(cutlass_kernel PUBLIC "${CUTLASSLibrary_INCLUDE_DIR}" "${CUTLASSLibrary_INCLUDE_TOOL_DIR}")

install(TARGETS cutlass_kernel LIBRARY DESTINATION .)
