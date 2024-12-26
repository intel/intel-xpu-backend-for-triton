get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(triton-translate
  triton-translate.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
)
llvm_update_compile_flags(triton-translate)
target_link_libraries(triton-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRTranslateLib
  MLIRSupport
  TritonGENToLLVMIRTranslation
)
mlir_check_link_libraries(triton-translate)

# Move `triton-translate` to the folder where `lit` expects to find it
add_custom_command(TARGET triton-translate  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy
  "$<TARGET_FILE:triton-translate>"
  "${triton_BINARY_DIR}/bin"
)
