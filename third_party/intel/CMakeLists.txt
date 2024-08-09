include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_subdirectory(include)
add_subdirectory(lib)

set(CMPLR_ROOT "$ENV{CMPLR_ROOT}")
set(SYCL_INCLUDE_DIR "${CMPLR_ROOT}/include")
set(SYCL_LINK_DIR "${CMPLR_ROOT}/lib")

# Disable sycl deprecation warnings as errors 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

add_triton_plugin(TritonXPU
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc

  LINK_LIBS
  PostProcessLLVMIR
  TritonGENToLLVM
  TritonGENToLLVMIRTranslation
  TritonIntelGPUToLLVM
  TritonIntelGPUTransforms
  TritonToTritonGPUWarp
)
target_compile_features(TritonXPU PRIVATE cxx_std_17)
target_include_directories(TritonXPU PRIVATE ${SYCL_INCLUDE_DIR} "${SYCL_INCLUDE_DIR}/sycl")
target_link_libraries(TritonXPU PRIVATE sycl ze_loader)