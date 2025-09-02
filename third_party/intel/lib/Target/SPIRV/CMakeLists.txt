# SPIRV-LLVM-Translator is required.
find_package(SPIRVToLLVMTranslator)

# Check if there is the LLVM SPIR-V backend.
is_llvm_target_library("SPIRV" spirv_present_result INCLUDED_TARGETS)

if(spirv_present_result)
  message(STATUS "Found SPIR-V Backend")
  add_compile_definitions(LLVM_SPIRV_BACKEND_TARGET_PRESENT)
  add_mlir_translation_library(TritonSPIRV
    SPIRVTranslation.cpp

    LINK_COMPONENTS
    Core
    # spirv backend
    SPIRVCodeGen

    LINK_LIBS PUBLIC
    TritonLLVMIR
    # spirv tools
    LLVMSPIRVLib
  )
else()
  add_mlir_translation_library(TritonSPIRV
    SPIRVTranslation.cpp

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    TritonLLVMIR
    # spirv tools
    LLVMSPIRVLib
  )
endif()

# Add SPIRV-LLVM-Translator include directory.
target_include_directories(TritonSPIRV PRIVATE ${SPIRVToLLVMTranslator_INCLUDE_DIR})

set_property(GLOBAL APPEND PROPERTY TRITON_LIBS TritonSPIRV)
llvm_update_compile_flags(TritonSPIRV)

# unused target
set_target_properties(llvm-spirv PROPERTIES EXCLUDE_FROM_ALL TRUE)
