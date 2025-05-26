# Try to find CUTLASS library.

include(FetchContent)

if (NOT CUTLASSLibrary_FOUND)
    # TODO: switch ot FetchContent_MakeAvailable once CUTLASS supports it
    cmake_policy(SET CMP0169 OLD)

    set(CUTLASSLibrary_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/CUTLASSLibrary")
    message(STATUS "CUTLASSLibrary is not specified. Will try to download
                  CUTLASS library from https://github.com/codeplaysoftware/cutlass-sycl.git into
                  ${CUTLASSLibrary_SOURCE_DIR}")
    file(READ cutlass_kernel/cutlass-library.conf CUTLASSLibrary_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${CUTLASSLibrary_TAG}" CUTLASSLibrary_TAG)
    FetchContent_Declare(cutlass-library
            GIT_REPOSITORY    https://github.com/codeplaysoftware/cutlass-sycl.git
            GIT_TAG           ${CUTLASSLibrary_TAG}
            SOURCE_DIR ${CUTLASSLibrary_SOURCE_DIR}
            )

    FetchContent_GetProperties(cutlass-library)
    if(NOT cutlass-library_POPULATED)
       FetchContent_Populate(cutlass-library)
    endif()

    set(CUTLASSLibrary_INCLUDE_DIR "${CUTLASSLibrary_SOURCE_DIR}/include" CACHE INTERNAL "CUTLASSLibrary_SOURCE_DIR")
    set(CUTLASSLibrary_INCLUDE_TOOL_DIR "${CUTLASSLibrary_SOURCE_DIR}/tools/util/include" CACHE INTERNAL "CUTLASSLibrary_SOURCE_DIR")

    find_package_handle_standard_args(
            CUTLASSLibrary
            FOUND_VAR CUTLASSLibrary_FOUND
            REQUIRED_VARS
            CUTLASSLibrary_SOURCE_DIR)

endif (NOT CUTLASSLibrary_FOUND)
