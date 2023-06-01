
MESSAGE(STATUS "johnlu add TritonGPUToSPIRV")
add_mlir_conversion_library(TritonGPUToSPIRV
    TritonGPUToSPIRV.cpp
    ViewOpToSPIRV.cpp
    ElementwiseOpToSPIRV.cpp
    TritonGPUToSPIRVPass.cpp
    LoadStoreOpToSPIRV.cpp
    ReduceOpToSPIRV.cpp
    ConvertLayoutOpToSPIRV.cpp
    TypeConverter.cpp
    DotOpHelpers.cpp
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

MESSAGE(STATUS "johnlu conversion_libs ${conversion_libs}")