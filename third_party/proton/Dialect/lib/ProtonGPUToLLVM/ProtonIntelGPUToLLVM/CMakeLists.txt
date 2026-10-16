include_directories(${PROJECT_SOURCE_DIR}/third_party/intel/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/intel/lib)

add_triton_library(ProtonIntelGPUToLLVM
    TargetInfo.cpp
    IntelPatternProtonGPUOpToLLVM.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonIntelGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonIntelGPUToLLVM
    TritonIntelUtils
    MLIRGPUToLLVMSPV
)
