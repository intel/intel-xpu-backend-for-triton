# Try to find SPIRV-LLVM-Translator.
#
include(FetchContent)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

            # Diagnose patching problems early: repository status, EOL config, and
            # raw newline style of the patch file often explain why "git apply"
            # fails even when the file is otherwise present.
            execute_process(
                COMMAND git config --show-origin --get core.autocrlf
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                OUTPUT_VARIABLE SPIRV_LLVM_AUTOCONFIG
                ERROR_VARIABLE SPIRV_LLVM_AUTOCONFIG_ERR
                RESULT_VARIABLE SPIRV_LLVM_AUTOCONFIG_RESULT
            )
            if(SPIRV_LLVM_AUTOCONFIG)
                string(STRIP "${SPIRV_LLVM_AUTOCONFIG}" SPIRV_LLVM_AUTOCONFIG)
                message(STATUS "SPIRV-LLVM-Translator git core.autocrlf: ${SPIRV_LLVM_AUTOCONFIG}")
            endif()
            if(SPIRV_LLVM_AUTOCONFIG_ERR)
                message(STATUS "SPIRV-LLVM-Translator git core.autocrlf stderr: ${SPIRV_LLVM_AUTOCONFIG_ERR}")
            endif()

            execute_process(
                COMMAND git config --show-origin --get core.eol
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                OUTPUT_VARIABLE SPIRV_LLVM_CORE_EOL
                ERROR_VARIABLE SPIRV_LLVM_CORE_EOL_ERR
                RESULT_VARIABLE SPIRV_LLVM_CORE_EOL_RESULT
            )
            if(SPIRV_LLVM_CORE_EOL)
                string(STRIP "${SPIRV_LLVM_CORE_EOL}" SPIRV_LLVM_CORE_EOL)
                message(STATUS "SPIRV-LLVM-Translator git core.eol: ${SPIRV_LLVM_CORE_EOL}")
            endif()
            if(SPIRV_LLVM_CORE_EOL_ERR)
                message(STATUS "SPIRV-LLVM-Translator git core.eol stderr: ${SPIRV_LLVM_CORE_EOL_ERR}")
            endif()

            execute_process(
                COMMAND git status --short --untracked-files=normal
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                OUTPUT_VARIABLE SPIRV_LLVM_STATUS
                ERROR_VARIABLE SPIRV_LLVM_STATUS_ERR
                RESULT_VARIABLE SPIRV_LLVM_STATUS_RESULT
            )
            if(SPIRV_LLVM_STATUS)
                message(STATUS "SPIRV-LLVM-Translator git status before patch:\n${SPIRV_LLVM_STATUS}")
            else()
                message(STATUS "SPIRV-LLVM-Translator git status before patch: clean")
            endif()
            if(SPIRV_LLVM_STATUS_ERR)
                message(STATUS "SPIRV-LLVM-Translator git status stderr: ${SPIRV_LLVM_STATUS_ERR}")
            endif()

            execute_process(
                COMMAND git diff --check
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                OUTPUT_VARIABLE SPIRV_LLVM_DIFF_CHECK
                ERROR_VARIABLE SPIRV_LLVM_DIFF_CHECK_ERR
                RESULT_VARIABLE SPIRV_LLVM_DIFF_CHECK_RESULT
            )
            if(SPIRV_LLVM_DIFF_CHECK)
                message(STATUS "SPIRV-LLVM-Translator git diff --check before patch:\n${SPIRV_LLVM_DIFF_CHECK}")
            endif()
            if(SPIRV_LLVM_DIFF_CHECK_ERR)
                message(STATUS "SPIRV-LLVM-Translator git diff --check stderr: ${SPIRV_LLVM_DIFF_CHECK_ERR}")
            endif()

            execute_process(
                COMMAND git ls-files --eol lib/SPIRV/SPIRVWriter.cpp
                WORKING_DIRECTORY ${spirv-llvm-translator_SOURCE_DIR}
                OUTPUT_VARIABLE SPIRV_LLVM_WRITER_EOL
                ERROR_VARIABLE SPIRV_LLVM_WRITER_EOL_ERR
                RESULT_VARIABLE SPIRV_LLVM_WRITER_EOL_RESULT
            )
            if(SPIRV_LLVM_WRITER_EOL)
                message(STATUS "SPIRV-LLVM-Translator lib/SPIRV/SPIRVWriter.cpp eol info: ${SPIRV_LLVM_WRITER_EOL}")
            endif()
            if(SPIRV_LLVM_WRITER_EOL_ERR)
                message(STATUS "SPIRV-LLVM-Translator lib/SPIRV/SPIRVWriter.cpp eol stderr: ${SPIRV_LLVM_WRITER_EOL_ERR}")
            endif()

            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "from pathlib import Path; import sys; p = Path(sys.argv[1]); b = p.read_bytes(); crlf = b.count(b'\\r\\n'); lf = b.count(b'\\n'); cr = b.count(b'\\r'); print(f'{p}: CRLF={crlf} LF={lf} CR={cr}')" ${CMAKE_CURRENT_LIST_DIR}/3122.patch
                OUTPUT_VARIABLE SPIRV_LLVM_PATCH_EOL
                ERROR_VARIABLE SPIRV_LLVM_PATCH_EOL_ERR
                RESULT_VARIABLE SPIRV_LLVM_PATCH_EOL_RESULT
            )
            if(SPIRV_LLVM_PATCH_EOL)
                message(STATUS "SPIRV-LLVM-Translator patch file eol info: ${SPIRV_LLVM_PATCH_EOL}")
            endif()
            if(SPIRV_LLVM_PATCH_EOL_ERR)
                message(STATUS "SPIRV-LLVM-Translator patch file eol stderr: ${SPIRV_LLVM_PATCH_EOL_ERR}")
            endif()

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
