add_triton_library(TritonIntelAnalysis
    AliasAnalysis.cpp
    Allocation.cpp
    AxisInfoExt.cpp
    Liveness.cpp
    RegisterPressure.cpp
    SpatialReuseAnalysis.cpp
    StrideInfo.cpp
    TemporalReuseAnalysis.cpp
    Membar.cpp
    Utility.cpp
    Range.cpp

    DEPENDS
    TritonTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    TritonIR
)
