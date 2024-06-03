# Try to find SPIRV-LLVM-Translator.
#
include(FetchContent)

if (NOT SPIRVToLLVMTranslator_FOUND)

    set(SPIRVToLLVMTranslator_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/SPIRVToLLVMTranslator")
    set(SPIRVToLLVMTranslator_BINARY_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/SPIRVToLLVMTranslator_binary")
    message(STATUS "SPIRV-LLVM location is not specified. Will try to download
                  SPIRVToLLVMTranslator from https://github.com/KhronosGroup/SPIRV-LLVM-Translator into
                  ${SPIRVToLLVMTranslator_SOURCE_DIR}")
    file(READ ${CMAKE_SOURCE_DIR}/cmake/spirv-llvm-translator.conf SPIRVToLLVMTranslator_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${SPIRVToLLVMTranslator_TAG}" SPIRVToLLVMTranslator_TAG)
    FetchContent_Declare(spirv-llvm-translator
            GIT_REPOSITORY    https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
            GIT_TAG           ${SPIRVToLLVMTranslator_TAG}
            SOURCE_DIR ${SPIRVToLLVMTranslator_SOURCE_DIR}
            )

    find_package(Git)
    set(GIT_HASH "Unknown")
    if(Git_FOUND)
        if (EXISTS ${SPIRVToLLVMTranslator_SOURCE_DIR})
            # Get the latest abbreviated commit hash of the working branch
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
                    WORKING_DIRECTORY ${SPIRVToLLVMTranslator_SOURCE_DIR}
                    OUTPUT_VARIABLE GIT_HASH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()
    endif()

    if(NOT "${GIT_HASH}" STREQUAL "${SPIRVToLLVMTranslator_TAG}")
        FetchContent_GetProperties(spirv-llvm-translator)
        if(NOT spirv-llvm-translator_POPULATED)
            FetchContent_Populate(spirv-llvm-translator)
        endif()
    endif()

    find_package(SPIRVHeaders)
    # set the SPIRV Headers director in advance
    set(LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR
            "${SPIRV_HEADERS_SOURCE_DIR}"
            CACHE INTERNAL "LLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR")
    # disable the SPIRV Tools in llvm-spriv translator.
    set(CMAKE_DISABLE_FIND_PACKAGE_SPIRV-Tools TRUE)
    set(CMAKE_DISABLE_FIND_PACKAGE_SPIRV-Tools-tools TRUE)
    add_subdirectory(${SPIRVToLLVMTranslator_SOURCE_DIR} ${SPIRVToLLVMTranslator_BINARY_DIR})

    set(SPIRVToLLVMTranslator_INCLUDE_DIR "${SPIRVToLLVMTranslator_SOURCE_DIR}/include"
            CACHE INTERNAL "SPIRVToLLVMTranslator_INCLUDE_DIR")

    find_package_handle_standard_args(
            SPIRVToLLVMTranslator
            FOUND_VAR SPIRVToLLVMTranslator_FOUND
            REQUIRED_VARS
                SPIRVToLLVMTranslator_SOURCE_DIR)

endif (NOT SPIRVToLLVMTranslator_FOUND)
