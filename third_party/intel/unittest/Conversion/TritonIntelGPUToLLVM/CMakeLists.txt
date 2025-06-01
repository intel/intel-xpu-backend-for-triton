add_triton_ut(
  NAME TestXeAsmFormat
  SRCS XeAsmFormatTest.cpp
  LIBS
    TritonGPUToLLVM
    TritonIntelGPUToLLVM
    MLIRUBToLLVM
)
