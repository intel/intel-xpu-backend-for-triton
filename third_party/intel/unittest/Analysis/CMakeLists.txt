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

add_triton_ut(
  NAME RegisterPressure
  SRCS RegisterPressureTest.cpp
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
)

add_triton_ut(
  NAME AliasAnalysis
  SRCS AliasAnalysisTest.cpp
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

add_triton_ut(
  NAME ReuseAnalysis
  SRCS ReuseAnalysisTest.cpp
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

add_triton_ut(
  NAME SignedDivRemDeduction
  SRCS SignedDivRemDeductionTest.cpp
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

add_triton_ut(
  NAME SignednessProver
  SRCS SignednessProverTest.cpp
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
    MLIRParser
    MLIRSCFDialect
)

add_triton_ut(
  NAME SymbolicBounds
  SRCS SymbolicBoundsTest.cpp
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
    MLIRParser
    MLIRSCFDialect
)

add_triton_ut(
  NAME SymbolicBoundsOracle
  SRCS SymbolicBoundsOracleTest.cpp
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
    MLIRParser
    MLIRSCFDialect
)
