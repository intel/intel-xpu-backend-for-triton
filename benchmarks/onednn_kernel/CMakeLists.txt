set(oneDNN_KERNEL_FLAGS ${oneDNN_KERNEL_FLAGS}
  -fsycl
  -fsycl-device-code-split=per_kernel
)

Python3_add_library(onednn_kernel MODULE WITH_SOABI python_main.cpp)

target_compile_options(onednn_kernel PRIVATE "-fsycl")
target_compile_options(onednn_kernel PRIVATE "-DDNNL_CPU_RUNTIME=SYCL")
target_compile_options(onednn_kernel PRIVATE "-DDNNL_GPU_RUNTIME=SYCL")

target_link_options(onednn_kernel PRIVATE ${oneDNN_KERNEL_FLAGS} -ldnnl)
target_link_libraries(onednn_kernel PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

target_include_directories(onednn_kernel PUBLIC "${oneDNNLibrary_INCLUDE_DIR}")

add_subdirectory(softmax)

install(TARGETS onednn_kernel LIBRARY DESTINATION .)
