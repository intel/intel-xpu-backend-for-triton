add_library(GenISAIntrinsics STATIC IMPORTED)
set_target_properties(GenISAIntrinsics
  PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libGenISAIntrinsics.a
)

add_triton_library(GENToLLVM
  GENToLLVMPass.cpp

  DEPENDS
  GENConversionPassIncGen
  GenISAIntrinsics

  LINK_LIBS PUBLIC    
  GenISAIntrinsics
)
