add_triton_ut(
  NAME DPAStoLinearLayout
  SRCS DPAStoLinearLayoutTest.cpp
  LIBS
    TritonIntelGPUIR
    TritonGPUTransforms
    TritonIntelAnalysis
    TritonIntelGPUTransforms
    TritonNvidiaGPUTransforms
)
add_triton_ut(
	NAME LinearLayoutConversionsIntel
	SRCS LinearLayoutConversionsTest.cpp
	LIBS
		TritonGPUIR
		TritonGPUTransforms
		TritonIntelAnalysis
		TritonIntelGPUTransforms
		TritonNvidiaGPUTransforms
)
