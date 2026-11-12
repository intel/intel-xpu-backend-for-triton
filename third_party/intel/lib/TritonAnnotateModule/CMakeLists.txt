add_triton_library(TritonAnnotateModule
    TritonAnnotateModule.cpp

    DEPENDS
    TritonAnnotateModulePassIncGen
    TritonTableGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
)
