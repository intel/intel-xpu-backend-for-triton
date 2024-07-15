# These tests should not be called for Intel.
return()

add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PTXAsmFormatTest.cpp
	LIBS TritonGPUToLLVM TritonNVIDIAGPUToLLVM
)
