add_triton_library(TritonGPUToLLVM
    TypeConverter.cpp
    Utility.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGENXDialect
    MLIRGPUDialect
    MLIRGPUToGENXTransforms
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonIntelGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    NVGPUIR
)
