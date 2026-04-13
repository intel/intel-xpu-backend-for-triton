add_triton_library(TritonToTritonGPU
    RelayoutTritonGPU.cpp
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
)

# ProtonIR is optional; only link it when Proton is enabled. When
# TRITON_ENABLE_PROTON=OFF the Proton dialect libraries are not built, so we
# must not add ProtonIR here.
if (TRITON_ENABLE_PROTON)
  target_link_libraries(TritonToTritonGPU PUBLIC ProtonIR)
endif()
