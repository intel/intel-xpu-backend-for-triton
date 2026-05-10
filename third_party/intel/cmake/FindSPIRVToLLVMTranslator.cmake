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

    set(LLVM_CONFIG ${LLVM_LIBRARY_DIR}/../bin/llvm-config)
    set(LLVM_DIR "${LLVM_LIBRARY_DIR}/cmake/llvm" CACHE PATH "Path to LLVM build dir " FORCE)
    set(LLVM_SPIRV_BUILD_EXTERNAL YES CACHE BOOL "Build SPIRV-LLVM Translator as external" FORCE)
    set(SPIRVToLLVMTranslator_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/spirv-llvm-translator-build")

    # indirect check that translator build succeeded
    if(DEFINED SPIRVToLLVMTranslator_LAST_TAG AND SPIRVToLLVMTranslator_LAST_TAG STREQUAL SPIRVToLLVMTranslator_TAG)
            message(STATUS "Using existing SPIRV-LLVM-Translator sources at ${SPIRVToLLVMTranslator_SOURCE_DIR}")

            # Import targets directly without FetchContent_MakeAvailable()
            add_subdirectory(${SPIRVToLLVMTranslator_SOURCE_DIR} ${SPIRVToLLVMTranslator_BINARY_DIR})
    else()
            # Sources don't exist, download via FetchContent
            message(STATUS "Downloading SPIRV-LLVM-Translator...")

            FetchContent_MakeAvailable(spirv-llvm-translator)

            # FIXME: Don't apply patch when LTS driver is updated.
            execute_process(
                    COMMAND git apply --check ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                    WORKING_DIRECTORY ${SPIRVToLLVMTranslator_SOURCE_DIR}
                    ERROR_QUIET
                    RESULT_VARIABLE PATCH_RESULT
            )
            if(PATCH_RESULT EQUAL 0)
            execute_process(
                    COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                    WORKING_DIRECTORY ${SPIRVToLLVMTranslator_SOURCE_DIR}
                    RESULT_VARIABLE PATCH_RESULT
            )
            else()
            execute_process( # Check if the patch is already applied
                    COMMAND git apply --reverse --check ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                    WORKING_DIRECTORY ${SPIRVToLLVMTranslator_SOURCE_DIR}
                    RESULT_VARIABLE PATCH_RESULT
            )
            endif()
            if(NOT PATCH_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to apply 3122.patch to SPIRV-LLVM-Translator")
            endif()

            # FIXME: Don't apply patch when driver is updated to support SPV_INTEL_subgroup_matrix_multiply_accumulate_float[4|8]
            execute_process(
                COMMAND git apply --check ${CMAKE_CURRENT_LIST_DIR}/revert_3609.patch
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                ERROR_QUIET
                RESULT_VARIABLE PATCH_RESULT
            )
            if(PATCH_RESULT EQUAL 0)
                execute_process(
                        COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/revert_3609.patch
                        WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                        RESULT_VARIABLE PATCH_RESULT
                )
            else()
                execute_process( # Check if the patch is already applied
                        COMMAND git apply --reverse --check ${CMAKE_CURRENT_LIST_DIR}/revert_3609.patch
                        WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                        RESULT_VARIABLE PATCH_RESULT
                )
            endif()
            if(NOT PATCH_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to apply revert_3609.patch to SPIRV-LLVM-Translator")
            endif()
    endif()

    set(SPIRVToLLVMTranslator_INCLUDE_DIR "${SPIRVToLLVMTranslator_SOURCE_DIR}/include"
            CACHE INTERNAL "SPIRVToLLVMTranslator_INCLUDE_DIR")
    # helps to not rebuild translator
    set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR
            "${CMAKE_BINARY_DIR}/_deps/spirv-llvm-translator-build/SPIRV-Headers"
            CACHE STRING "Path to SPIRV-Headers" FORCE)
    set(SPIRVToLLVMTranslator_LAST_TAG
            "${SPIRVToLLVMTranslator_TAG}"
            CACHE STRING "Last built SPIRV-LLVM-Translator tag" FORCE)

    find_package_handle_standard_args(
            SPIRVToLLVMTranslator
            FOUND_VAR SPIRVToLLVMTranslator_FOUND
            REQUIRED_VARS
                SPIRVToLLVMTranslator_SOURCE_DIR)

endif (NOT SPIRVToLLVMTranslator_FOUND)
