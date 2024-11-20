add_triton_library(TritonIntelLLVMIR
        LLVMIRFreezeMaskedDivRem.cpp

        DEPENDS
        LLVMIRIncGen
        )