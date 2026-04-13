get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# Disable triton-opt due to MLIR RTTI linker issues with test libraries
# This is a testing tool not required for runtime kernel execution
# The Python module libtriton.so builds successfully without it
# add_executable(triton-opt triton-opt.cpp)
#
# target_compile_options(triton-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
# target_link_libraries(triton-opt PRIVATE
#   TritonIntelLLVMIR
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialect
#   # AMD and Proton test libraries are optional
#   $<$<BOOL:${TRITON_ENABLE_AMD}>:TritonAMDGPUTestAnalysis>
#   $<$<BOOL:${TRITON_ENABLE_PROTON}>:TritonTestProton>
#   # MLIR core
#   MLIROptLib
#   MLIRPass
#   MLIRIR
#   MLIRSupport
#   MLIRRegisterAllDialects
#   MLIRRegisterAllPasses
#   MLIRTransforms
# )
#
# mlir_check_all_link_libraries(triton-opt)

# Disable debugging tools due to LLVM linker issues
# These are not required for runtime kernel execution
# add_executable(triton-reduce triton-reduce.cpp)
# mlir_check_all_link_libraries(triton-reduce)
# target_compile_options(triton-reduce PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
#
# target_link_libraries(triton-reduce PRIVATE
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialect
#   # AMD and Proton test libraries are optional
#   $<$<BOOL:${TRITON_ENABLE_AMD}>:TritonAMDGPUTestAnalysis>
#   $<$<BOOL:${TRITON_ENABLE_PROTON}>:TritonTestProton>
#   # MLIR core
#   MLIRReduceLib
#   MLIRPass
#   MLIRPassManager
#   MLIRIR
#   MLIRSupport
#   MLIRRegisterAllDialects
#   MLIRRegisterAllPasses
#   MLIRTransforms
# )
#
# mlir_check_all_link_libraries(triton-reduce)
#
# add_executable(triton-lsp triton-lsp.cpp)
#
# target_compile_options(triton-lsp PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
# target_link_libraries(triton-lsp PRIVATE
#   ${triton_libs}
#   # tests
#   TritonTestAnalysis
#   TritonTestDialect
#   # AMD and Proton test libraries are optional
#   $<$<BOOL:${TRITON_ENABLE_AMD}>:TritonAMDGPUTestAnalysis>
#   $<$<BOOL:${TRITON_ENABLE_PROTON}>:TritonTestProton>
#   # MLIR core
#   MLIRLspServerLib
#   MLIRPass
#   MLIRPassManager
#   MLIRIR
#   MLIRSupport
#   MLIRRegisterAllDialects
#   MLIRRegisterAllPasses
#   MLIRTransforms
# )
#
# mlir_check_all_link_libraries(triton-lsp)

# Disable triton-llvm-opt to avoid LLVM linker issues
# This is a testing tool not required for runtime
# add_executable(triton-llvm-opt triton-llvm-opt.cpp)
# add_dependencies(triton-llvm-opt intrinsics_gen)
# target_compile_options(triton-llvm-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
# target_link_libraries(triton-llvm-opt PRIVATE
#   TritonLLVMIR
#
#   LLVMAnalysis
#   LLVMCore
#   LLVMSupport
#   LLVMOption
#   LLVMCodeGen
#   TritonIntelLLVMIR
#   TritonIntelGPUIR
#   )
# export_executable_symbols_for_plugins(triton-llvm-opt)

# Disable triton-tensor-layout (debugging tool) due to LLVM linker issues
# add_executable(triton-tensor-layout triton-tensor-layout.cpp)
# target_compile_options(triton-tensor-layout PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
# target_link_libraries(triton-tensor-layout PRIVATE
#   ${triton_libs}
#   TritonTestAnalysis
#   TritonTestDialect
#   # AMD and Proton test libraries are optional
#   $<$<BOOL:${TRITON_ENABLE_PROTON}>:TritonTestProton>
#   $<$<BOOL:${TRITON_ENABLE_AMD}>:TritonAMDGPUTestAnalysis>
#   MLIRRegisterAllDialects
#   MLIRRegisterAllPasses
#   MLIRTransforms
#   )
