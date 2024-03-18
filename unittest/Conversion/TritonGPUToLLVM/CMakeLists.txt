add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PTXAsmFormatTest.cpp
	LIBS TritonGPUToLLVM TritonIntelGPUToLLVM
)

add_triton_ut(
	NAME TestEmitIndices
	SRCS EmitIndicesTest.cpp DumpLayout.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR TritonNVIDIAGPUToLLVM
)
