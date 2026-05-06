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
  NAME StrideInfo
  SRCS StrideInfoTest.cpp
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
    MLIRSCFDialect
)

# Suppress deprecation warnings for OpBuilder::create<> API
if(NOT WIN32)
  target_compile_options(StrideInfo PRIVATE -Wno-deprecated-declarations)
endif()

add_triton_ut(
  NAME TemporalReuseAnalysis
  SRCS TemporalReuseAnalysisTest.cpp
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
    MLIRSCFDialect
)
