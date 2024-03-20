add_triton_library(TritonIntelGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandDPAS.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/DPAS.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    AssertOpToLLVM.cpp
    PrintOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    ControlFlowOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    Utility.cpp
    TensorPtrOpsToLLVM.cpp
    ClusterOpsToLLVM.cpp
    PTXAsmFormat.cpp
    AllocateSharedMemory.cpp
    TargetInfo.cpp

    DEPENDS
    TritonIntelGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonNvidiaGPUTransforms
    NVGPUIR
    TritonGENIR
    TritonGENToLLVM
    TritonIntelGPUIR
    GPUToTritonGEN
)
