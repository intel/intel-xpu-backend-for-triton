add_triton_library(TritonRaiseBlockPointer
    TritonRaiseBlockPointer.cpp

    DEPENDS
    TritonRaiseBlockPointerPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
)
