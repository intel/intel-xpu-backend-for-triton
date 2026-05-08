add_triton_ut(
  NAME TestXeAsmFormat
  SRCS XeAsmFormatTest.cpp
  LIBS
    TritonIntelXeAsm
    MLIRArithDialect
)
