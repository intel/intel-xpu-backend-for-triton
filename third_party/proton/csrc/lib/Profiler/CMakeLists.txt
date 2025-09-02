set(SOURCE_FILES
    Cupti/CuptiPCSampling.cpp
    Cupti/CuptiProfiler.cpp
    RocTracer/RoctracerProfiler.cpp
	  Instrumentation/InstrumentationProfiler.cpp
	  Instrumentation/CudaRuntime.cpp
	  Instrumentation/HipRuntime.cpp
	  Instrumentation/Metadata.cpp
)

if(TRITON_BUILD_PROTON_XPU)
    list(APPEND SOURCE_FILES Xpupti/XpuptiProfiler.cpp)
endif()

add_proton_library(ProtonProfiler ${SOURCE_FILES})
