add_triton_library(TritonIntelAnalysis
    Allocation.cpp
    AxisInfoExt.cpp
    StrideInfo.cpp
    Liveness.cpp
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
