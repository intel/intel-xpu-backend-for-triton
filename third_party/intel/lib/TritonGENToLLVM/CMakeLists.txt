add_library(TritonGenISAIntrinsics STATIC IMPORTED)
set_target_properties(TritonGenISAIntrinsics
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libGenISAIntrinsics.a
)

add_triton_library(TritonGENToLLVM
  TritonGENToLLVMPass.cpp

  DEPENDS
  TritonGENToLLVMConversionPassIncGen
  TritonGenISAIntrinsics

  LINK_LIBS PUBLIC
  TritonGenISAIntrinsics
)
