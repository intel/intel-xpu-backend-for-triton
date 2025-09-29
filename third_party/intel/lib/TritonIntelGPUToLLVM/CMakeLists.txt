add_triton_library(TritonIntelGPUToLLVM
    AllocateSharedMemory.cpp
    BF16Casts.cpp
    ControlFlowOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/DPAS.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    Fp4ToFpOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    PrintOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    SPMDOpToLLVM.cpp
    SPIRVTargetInfo.cpp
    TargetInfo.cpp
    TensorPtrOpsToLLVM.cpp
    TritonGPUToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    XeAsmFormat.cpp

    DEPENDS
    TritonIntelGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    GPUToTritonGEN
    MLIRGPUToLLVMSPV
    MLIRSPIRVToLLVM
    TritonGENIR
    TritonGENToLLVM
    TritonGENToSPIRV
    TritonIntelGPUIR
    TritonIntelUtils
)
