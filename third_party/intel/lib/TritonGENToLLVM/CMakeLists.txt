add_library(GenISAIntrinsics STATIC IMPORTED GLOBAL)
set_target_properties(GenISAIntrinsics
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libGenISAIntrinsics.a
)

add_triton_library(TritonGENToLLVM
  Attributes.cpp
  TritonGENToLLVMPass.cpp
  GenIntrinsicHelper.cpp

  DEPENDS
  TritonGENToLLVMConversionPassIncGen
  GenISAIntrinsics

  LINK_LIBS PUBLIC
  GenISAIntrinsics
  MLIRLLVMDialect
  MLIRSPIRVDialect
  TritonIntelUtils
  MLIRTargetLLVMIRImport
)
