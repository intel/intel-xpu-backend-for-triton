add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR TritonTransforms ${dialect_libs} ${conversion_libs} ${triton_libs}
)
