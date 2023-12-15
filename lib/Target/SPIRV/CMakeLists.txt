add_mlir_translation_library(TritonSPIRV
        SPIRVTranslation.cpp

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        TritonLLVMIR
        )
