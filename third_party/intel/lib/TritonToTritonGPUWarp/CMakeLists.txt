add_triton_library(TritonToTritonGPUWarp
    TritonToTritonGPUWarpPass.cpp

    DEPENDS
    TritonGPUWarpConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
