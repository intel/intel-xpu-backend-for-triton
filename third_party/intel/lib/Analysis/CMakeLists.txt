add_triton_library(TritonIntelAnalysis
    Allocation.cpp
    AxisInfo.cpp
    DPAS.cpp
    Liveness.cpp
    Membar.cpp
    Utility.cpp

    DEPENDS
    TritonTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    TritonIR
)
