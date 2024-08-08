# XeTLA library is required.
find_package(XeTLALibrary REQUIRED)

add_library(xetla_kernel SHARED python_main.cpp)
set_target_properties(xetla_kernel PROPERTIES PREFIX "")
target_compile_options(xetla_kernel PRIVATE "-fPIC")
target_compile_options(xetla_kernel PRIVATE "-fsycl")
target_link_options(xetla_kernel PRIVATE "-fsycl")
target_link_libraries(xetla_kernel PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(xetla_kernel PUBLIC "${TORCH_IPEX_LIBRARIES}")
target_include_directories(xetla_kernel PUBLIC "${PYTHON_INCLUDE_DIRS}")
target_include_directories(xetla_kernel PUBLIC "${TORCH_IPEX_INCLUDE_DIRS}")
target_include_directories(xetla_kernel PUBLIC "${XeTLALibrary_INCLUDE_DIR}")

add_subdirectory(softmax)
add_subdirectory(gemm)
add_subdirectory(stream_k_gemm)
add_subdirectory(flash_attention)
