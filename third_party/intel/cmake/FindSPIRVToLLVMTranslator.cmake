# Try to find SPIRV-LLVM-Translator.
#
include(FetchContent)

if (NOT SPIRVToLLVMTranslator_FOUND)

    set(SPIRVToLLVMTranslator_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/SPIRVToLLVMTranslator")
    message(STATUS "SPIRV-LLVM location is not specified. Will try to download
                  SPIRVToLLVMTranslator from https://github.com/KhronosGroup/SPIRV-LLVM-Translator into
                  ${SPIRVToLLVMTranslator_SOURCE_DIR}")
    file(READ spirv-llvm-translator.conf SPIRVToLLVMTranslator_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${SPIRVToLLVMTranslator_TAG}" SPIRVToLLVMTranslator_TAG)
    FetchContent_Declare(spirv-llvm-translator
            GIT_REPOSITORY    https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
            GIT_TAG           ${SPIRVToLLVMTranslator_TAG}
            SOURCE_DIR ${SPIRVToLLVMTranslator_SOURCE_DIR}
            )

    FetchContent_GetProperties(spirv-llvm-translator)
    if(NOT spirv-llvm-translator_POPULATED)

            set(LLVM_CONFIG ${LLVM_LIBRARY_DIR}/../bin/llvm-config)
            set(LLVM_DIR "${LLVM_LIBRARY_DIR}/cmake/llvm" CACHE PATH "Path to LLVM build dir " FORCE)
            set(LLVM_SPIRV_BUILD_EXTERNAL YES CACHE BOOL "Build SPIRV-LLVM Translator as external" FORCE)

            FetchContent_MakeAvailable(spirv-llvm-translator)

            # FIXME: Don't apply patch when Agama driver is updated.
            execute_process(
                COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                RESULT_VARIABLE PATCH_RESULT
            )
            if(NOT PATCH_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to apply 3122.patch to SPIRV-LLVM-Translator")
            endif()

    endif()

    set(SPIRVToLLVMTranslator_INCLUDE_DIR "${SPIRVToLLVMTranslator_SOURCE_DIR}/include"
            CACHE INTERNAL "SPIRVToLLVMTranslator_INCLUDE_DIR")

    find_package_handle_standard_args(
            SPIRVToLLVMTranslator
            FOUND_VAR SPIRVToLLVMTranslator_FOUND
            REQUIRED_VARS
                SPIRVToLLVMTranslator_SOURCE_DIR)

endif (NOT SPIRVToLLVMTranslator_FOUND)
