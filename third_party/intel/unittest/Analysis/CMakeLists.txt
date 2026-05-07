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
if(NOT WIN32)
  target_compile_options(DPASAnalysis PRIVATE -Wno-deprecated-declarations)
endif()

add_triton_ut(
  NAME AliasReuseAnalysis
  SRCS AliasReuseAnalysisTest.cpp
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
    MLIRSCFDialect
)

if(NOT WIN32)
  target_compile_options(AliasReuseAnalysis PRIVATE -Wno-deprecated-declarations)
endif()

add_triton_ut(
  NAME SpatialReuseAnalysis
  SRCS SpatialReuseAnalysisTest.cpp
  LIBS
    TritonAnalysis
    TritonGPUIR
    TritonGPUTransforms
    TritonIR
    TritonIntelAnalysis
    TritonIntelGPUIR
    TritonIntelGPUTransforms
    TritonNvidiaGPUIR
    TritonNvidiaGPUTransforms
    MLIRArithDialect
    MLIRFuncDialect
)
