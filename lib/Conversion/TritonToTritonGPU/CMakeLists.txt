add_triton_library(TritonToTritonGPU
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp
    TritonToTritonGPUWarpPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
