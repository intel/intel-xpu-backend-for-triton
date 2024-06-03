# Download SPIRV-Header in advance.
#
include(FetchContent)

if (NOT SPIRVHeaders_FOUND)

    set(SPIRV_HEADERS_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/SPIRV-Headers"
            CACHE INTERNAL "SPIRV_HEADERS_SOURCE_DIR")
    message(STATUS "SPIR-V Headers location is not specified. Will try to download
          spirv.hpp from https://github.com/KhronosGroup/SPIRV-Headers into
          ${SPIRV_HEADERS_SOURCE_DIR}")
    file(READ ${CMAKE_SOURCE_DIR}/cmake/spirv-headers-tag.conf SPIRV_HEADERS_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${SPIRV_HEADERS_TAG}" SPIRV_HEADERS_TAG)
    FetchContent_Declare(spirv-headers
            GIT_REPOSITORY    https://github.com/KhronosGroup/SPIRV-Headers.git
            GIT_TAG           ${SPIRV_HEADERS_TAG}
            SOURCE_DIR ${SPIRV_HEADERS_SOURCE_DIR}
    )

    find_package(Git)
    set(GIT_HASH "Unknown")
    if(Git_FOUND)
        if (EXISTS ${SPIRV_HEADERS_SOURCE_DIR})
            # Get the latest abbreviated commit hash of the working branch
            execute_process(
                    COMMAND ${GIT_EXECUTABLE} log -1 --format=%H
                    WORKING_DIRECTORY ${SPIRV_HEADERS_SOURCE_DIR}
                    OUTPUT_VARIABLE GIT_HASH
                    OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()
    endif()

    if(NOT "${GIT_HASH}" STREQUAL "${SPIRV_HEADERS_TAG}")
        FetchContent_GetProperties(spirv-headers)
        if(NOT spirv-headers_POPULATED)
            FetchContent_Populate(spirv-headers)
        endif()
    endif()

    find_package_handle_standard_args(
            SPIRVHeaders
            FOUND_VAR SPIRVHeaders_FOUND
            REQUIRED_VARS
            SPIRV_HEADERS_SOURCE_DIR)

endif (NOT SPIRVHeaders_FOUND)
