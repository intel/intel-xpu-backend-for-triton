set(SOURCE_FILES
	Device.cpp
	GPU/CudaApi.cpp
	GPU/CuptiApi.cpp
	GPU/HipApi.cpp
	GPU/HsaApi.cpp
	GPU/RoctracerApi.cpp
)

if(TRITON_BUILD_PROTON_XPU)
    list(APPEND SOURCE_FILES GPU/XpuApi.cpp)
	list(APPEND SOURCE_FILES GPU/XpuptiApi.cpp)
endif()

add_proton_library(ProtonDriver ${SOURCE_FILES})
