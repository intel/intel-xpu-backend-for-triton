set(GenISALibPath "${CMAKE_CURRENT_BINARY_DIR}/libGenISAIntrinsics.a")
file(GLOB GenISAChunks "${CMAKE_CURRENT_SOURCE_DIR}/libGenISA_chunk_*")

add_custom_command(
        OUTPUT ${GenISALibPath}
        COMMAND cat ${CMAKE_CURRENT_SOURCE_DIR}/libGenISA_chunk_?? > ${GenISALibPath}
        DEPENDS ${GenISAChunks}
)

add_custom_target(GenISALib ALL DEPENDS ${GenISALibPath})

add_library(GenISAIntrinsics STATIC IMPORTED GLOBAL)
set_target_properties(GenISAIntrinsics
        PROPERTIES IMPORTED_LOCATION ${GenISALibPath}
)
add_dependencies(GenISAIntrinsics GenISALib)

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
