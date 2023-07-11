add_mlir_conversion_library(TritonGPUToSPIRV
    TritonGPUToSPIRV.cpp
    ViewOpToSPIRV.cpp
    ElementwiseOpToSPIRV.cpp
    TritonGPUToSPIRVPass.cpp
    ConvertLayoutOpToSPIRV/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToSPIRV/SharedToDotOperandMMAv1.cpp
    ConvertLayoutOpToSPIRV/SharedToDotOperandMMAv2.cpp
    ConvertLayoutOpToSPIRV.cpp
    DotOpToSPIRV/FMA.cpp
    DotOpToSPIRV/MMAv1.cpp
    DotOpToSPIRV/MMAv2.cpp
    LoadStoreOpToSPIRV.cpp
    ReduceOpToSPIRV.cpp
    TypeConverter.cpp
    Utility.cpp
    DotOpToSPIRV.cpp


    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToSPIRV

    DEPENDS
    TritonGPUToSPIRVConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUOps
    MLIRGPUToNVVMTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
