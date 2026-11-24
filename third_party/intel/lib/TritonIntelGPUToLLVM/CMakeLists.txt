add_triton_library(TritonIntelXeAsm
    XeAsmFormat.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRLLVMDialect
    MLIRTransforms
    TritonIR
)

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
    MemoryOpToLLVM.cpp
    PrintOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    SPIRVTargetInfo.cpp
    TargetInfo.cpp
    TensorDescOpsToLLVM.cpp
    TritonGPUToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    WarpIdOpToLLVM.cpp

    DEPENDS
    TritonIntelGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    GPUToTritonGEN
    MLIRGPUToLLVMSPV
    MLIRSPIRVToLLVM
    TritonGENIR
    TritonGENToLLVM
    TritonGENToSPIRV
    TritonInstrumentToLLVM
    TritonIntelGPUIR
    TritonIntelUtils
    TritonIntelXeAsm
)
