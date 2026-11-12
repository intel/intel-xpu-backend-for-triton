# All Intel unit tests share one executable: every extra test executable costs a
# full link against the Triton/MLIR/LLVM static libraries. Add new tests (and any
# libraries they need) to the lists below instead of creating a new executable.
add_triton_ut(
  NAME TritonIntelUnitTests
  SRCS
    Analysis/AliasAnalysisTest.cpp
    Analysis/DPASAnalysisTest.cpp
    Analysis/RegisterPressureTest.cpp
    Analysis/ReuseAnalysisTest.cpp
    Analysis/SignednessProverTest.cpp
    Analysis/SpatialReuseAnalysisTest.cpp
    Analysis/StrideInfoTest.cpp
    Analysis/SymbolicBoundsOracleTest.cpp
    Analysis/SymbolicBoundsTest.cpp
    Analysis/TemporalReuseAnalysisTest.cpp
    Dialect/TritonIntelGPU/BlockIOUtilsTest.cpp
    Dialect/TritonIntelGPU/DPAStoLinearLayoutTest.cpp
    Dialect/TritonIntelGPU/LinearLayoutConversionsTest.cpp
    Dialect/TritonIntelGPU/UtilityTest.cpp
  LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRParser
    MLIRSCFDialect
    TritonAnalysis
    TritonGPUIR
    TritonGPUTransforms
    TritonIR
    TritonIntelAnalysis
    TritonIntelGPUIR
    TritonIntelGPUTransforms
    TritonNvidiaGPUIR
    TritonNvidiaGPUTransforms
)
set_source_files_properties(Analysis/DPASAnalysisTest.cpp
  PROPERTIES COMPILE_DEFINITIONS TRITON_DPAS_UNIT_TESTING)

add_subdirectory(Conversion)
add_subdirectory(Transforms)
