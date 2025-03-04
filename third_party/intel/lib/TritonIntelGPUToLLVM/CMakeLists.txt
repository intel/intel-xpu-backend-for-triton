add_triton_library(TritonIntelGPUToLLVM
    AllocateSharedMemory.cpp
    ArithOpsToLLVM.cpp
    BF16Casts.cpp
    ControlFlowOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    DotOpToLLVM/DPAS.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    Fp4ToFpOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    PrintOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    SPMDOpToLLVM.cpp
    TargetInfo.cpp
    TensorPtrOpsToLLVM.cpp
    TritonGPUToLLVM.cpp
    TritonOpsToLLVM.cpp
    TypeConverter.cpp
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
