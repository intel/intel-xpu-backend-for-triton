include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

add_triton_plugin(TritonXPU
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc

  LINK_LIBS
  MLIRGPUToLLVMSPV
  TritonGENToLLVM
  TritonGENToLLVMIRTranslation
  TritonIntelGPUToLLVM
  TritonIntelGPUTransforms
)

target_link_libraries(TritonXPU PRIVATE Python3::Module pybind11::headers)

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
