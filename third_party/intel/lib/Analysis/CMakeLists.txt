add_triton_library(TritonIntelAnalysis
    DPAS.cpp

    DEPENDS
    TritonTableGen
    TritonGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonIR
)
