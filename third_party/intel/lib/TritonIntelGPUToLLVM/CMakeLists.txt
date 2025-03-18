add_triton_library(TritonIntelGPUToLLVM
    ArithOpsToLLVM.cpp
    BF16Casts.cpp
    ControlFlowOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    DotOpToLLVM/DPAS.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
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
    TritonOpsToLLVM.cpp
    TypeConverter.cpp
    UpcastMXFPToLLVM.cpp
    Utility.cpp

    DEPENDS
    TritonIntelGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    GPUToTritonGEN
    MLIRGPUToLLVMSPV
    MLIRSPIRVToLLVM
    TritonGENIR
    TritonGENToLLVM
    TritonIntelGPUIR
    TritonIntelUtils
)
