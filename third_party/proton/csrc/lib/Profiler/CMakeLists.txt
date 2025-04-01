set(SOURCE_FILES
    Cupti/CuptiPCSampling.cpp
    Cupti/CuptiProfiler.cpp
    RocTracer/RoctracerProfiler.cpp
)
message(STATUS "ENABLE_XPUTPI_PROFILER: ${ENABLE_XPUTPI_PROFILER}")
if(ENABLE_XPUTPI_PROFILER)
    list(APPEND SOURCE_FILES Xpupti/XpuptiProfiler.cpp)
endif()

add_proton_library(ProtonProfiler ${SOURCE_FILES})
