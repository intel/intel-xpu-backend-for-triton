add_triton_library(TritonAnnotateModule
    TritonAnnotateModule.cpp

    DEPENDS
    TritonAnnotateModulePassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
)
