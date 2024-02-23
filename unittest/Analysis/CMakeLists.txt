message(STATUS "Dialect libs: ${dialect_libs}")
message(STATUS "Conversion libs: ${conversion_libs}")
message(STATUS "Triton libs: ${triton_libs}")
# message(FATAL_ERROR "end")

add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    ${dialect_libs}
    ${triton_libs}
)
