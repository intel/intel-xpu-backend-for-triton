add_triton_ut(
  NAME DPASAnalysis
  SRCS DPASAnalysisTest.cpp
  LIBS
    TritonIntelGPUIR
    TritonGPUIR
    TritonIR
    TritonGPUTransforms
    TritonIntelAnalysis
    TritonIntelGPUTransforms
    TritonNvidiaGPUTransforms
    MLIRArithDialect
    MLIRFuncDialect
  DEFS
    TRITON_DPAS_UNIT_TESTING
)

# Suppress deprecation warnings for OpBuilder::create<> API
# TODO: Migrate to modern MLIR operation creation patterns
target_compile_options(DPASAnalysis PRIVATE -Wno-deprecated-declarations)
