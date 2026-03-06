add_triton_library(TritonIntelAnalysis
    Allocation.cpp
    AxisInfo.cpp
    Liveness.cpp
    StrideInfo.cpp
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
