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
            GIT_CONFIG        core.autocrlf=false
            SOURCE_DIR ${SPIRVToLLVMTranslator_SOURCE_DIR}
            )

    FetchContent_GetProperties(spirv-llvm-translator)
    if(NOT spirv-llvm-translator_POPULATED)
            set(LLVM_CONFIG ${LLVM_LIBRARY_DIR}/../bin/llvm-config)
            set(LLVM_DIR "${LLVM_LIBRARY_DIR}/cmake/llvm" CACHE PATH "Path to LLVM build dir " FORCE)
            set(LLVM_SPIRV_BUILD_EXTERNAL YES CACHE BOOL "Build SPIRV-LLVM Translator as external" FORCE)

            FetchContent_MakeAvailable(spirv-llvm-translator)

            # FIXME: Don't apply patch when LTS driver is updated.
            set(PATCH_STATUS "failed")
            set(PATCH_COMMAND "git apply --check")
            execute_process(
                COMMAND git apply --check ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                OUTPUT_VARIABLE PATCH_CHECK_STDOUT
                ERROR_VARIABLE PATCH_CHECK_STDERR
                RESULT_VARIABLE PATCH_RESULT
            )
            if(PATCH_CHECK_STDOUT)
                message(STATUS "git apply --check stdout: ${PATCH_CHECK_STDOUT}")
            endif()
            if(PATCH_CHECK_STDERR)
                message(STATUS "git apply --check stderr: ${PATCH_CHECK_STDERR}")
            endif()
            set(PATCH_STDOUT "${PATCH_CHECK_STDOUT}")
            set(PATCH_STDERR "${PATCH_CHECK_STDERR}")
            if(PATCH_RESULT EQUAL 0)
                set(PATCH_COMMAND "git apply")
                execute_process(
                        COMMAND git apply ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                        WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                        OUTPUT_VARIABLE PATCH_APPLY_STDOUT
                        ERROR_VARIABLE PATCH_APPLY_STDERR
                        RESULT_VARIABLE PATCH_RESULT
                )
                if(PATCH_APPLY_STDOUT)
                    message(STATUS "git apply stdout: ${PATCH_APPLY_STDOUT}")
                endif()
                if(PATCH_APPLY_STDERR)
                    message(STATUS "git apply stderr: ${PATCH_APPLY_STDERR}")
                endif()
                set(PATCH_STDOUT "${PATCH_APPLY_STDOUT}")
                set(PATCH_STDERR "${PATCH_APPLY_STDERR}")
                if(PATCH_RESULT EQUAL 0)
                    set(PATCH_STATUS "applied")
                endif()
            else()
                set(PATCH_COMMAND "git apply --reverse --check")
                execute_process( # Check if the patch is already applied
                        COMMAND git apply --reverse --check ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                        WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                        OUTPUT_VARIABLE PATCH_REVERSE_CHECK_STDOUT
                        ERROR_VARIABLE PATCH_REVERSE_CHECK_STDERR
                        RESULT_VARIABLE PATCH_RESULT
                )
                if(PATCH_REVERSE_CHECK_STDOUT)
                    message(STATUS "git apply --reverse --check stdout: ${PATCH_REVERSE_CHECK_STDOUT}")
                endif()
                if(PATCH_REVERSE_CHECK_STDERR)
                    message(STATUS "git apply --reverse --check stderr: ${PATCH_REVERSE_CHECK_STDERR}")
                endif()
                set(PATCH_STDOUT "${PATCH_REVERSE_CHECK_STDOUT}")
                set(PATCH_STDERR "${PATCH_REVERSE_CHECK_STDERR}")
                if(PATCH_RESULT EQUAL 0)
                    set(PATCH_STATUS "already_applied")
                endif()
            endif()
            if(PATCH_STATUS STREQUAL "failed")
                if(PATCH_STDOUT)
                    message(STATUS "failed command stdout: ${PATCH_STDOUT}")
                endif()
                if(PATCH_STDERR)
                    message(STATUS "failed command stderr: ${PATCH_STDERR}")
                endif()
                message(FATAL_ERROR "Failed during '${PATCH_COMMAND}' for 3122.patch in SPIRV-LLVM-Translator (exit code: ${PATCH_RESULT})")
            elseif(PATCH_STATUS STREQUAL "already_applied")
                message(STATUS "3122.patch is already applied to SPIRV-LLVM-Translator")
            else()
                message(STATUS "3122.patch applied to SPIRV-LLVM-Translator")
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
