add_triton_library(TritonIntelAnalysis
    AxisInfo.cpp
    DPAS.cpp
    Liveness.cpp
    Utility.cpp

    DEPENDS
    TritonTableGen
    TritonGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonIR
)
