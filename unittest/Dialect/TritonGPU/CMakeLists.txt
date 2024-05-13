add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR
)
add_triton_ut(
	NAME Dialect
	SRCS DialectTest.cpp
	LIBS TritonGPUIR
)
# FIXME: Fail on XPU.
return()
add_triton_ut(
	NAME LinearLayoutConversions
	SRCS LinearLayoutConversionsTest.cpp
	LIBS TritonGPUIR
)
