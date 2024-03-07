include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)

add_triton_plugin(TritonXPU
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc

  LINK_LIBS
  TritonIntelGPUToLLVM
  TritonGENToLLVM  
  NVGPUIntelToLLVM
)
