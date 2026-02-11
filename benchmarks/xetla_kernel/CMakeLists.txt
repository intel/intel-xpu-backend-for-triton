set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS}
  -fsycl
  -fsycl-device-code-split=per_kernel
)

if (USE_AOT_DEVLIST)
  set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -fsycl-targets=spir64_gen)
endif()

# For large registers mode, enable 256 registers for kernels
set(XETLA_OFFLINE_OPTIONS "-doubleGRF")
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-disable-indvars-opt")
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-codegen")
# For registers usage verbose at AOT
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -printregusage")
# Enable bank conflict reduction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -enableBCR")
# Optimization to reduce the tokens used for DPAS instruction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -DPASTokenReduction")

set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -Xs )
if (USE_AOT_DEVLIST)
  set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-device pvc -options '${XETLA_OFFLINE_OPTIONS}'")
else()
  set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "${XETLA_OFFLINE_OPTIONS}")
endif()

Python3_add_library(xetla_kernel MODULE WITH_SOABI python_main.cpp)
target_compile_options(xetla_kernel PRIVATE "-fPIC")
target_compile_options(xetla_kernel PRIVATE "-fsycl")
target_compile_options(xetla_kernel PUBLIC "-DXETPP_NEW_XMAIN")
target_link_options(xetla_kernel PRIVATE ${XETLA_KERNEL_FLAGS})
target_link_libraries(xetla_kernel PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_include_directories(xetla_kernel PUBLIC "${XeTLALibrary_INCLUDE_DIR}")

add_subdirectory(softmax)
add_subdirectory(gemm)
add_subdirectory(stream_k_gemm)
add_subdirectory(split_k_gemm)
add_subdirectory(flash_attention)

install(TARGETS xetla_kernel LIBRARY DESTINATION .)
