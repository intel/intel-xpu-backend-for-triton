# Try to find XeTLA library.

include(FetchContent)

if (NOT XeTLALibrary_FOUND)
    # TODO: switch ot FetchContent_MakeAvailable once XeTLA supports it
    cmake_policy(SET CMP0169 OLD)

    set(XeTLALibrary_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/XeTLALibrary")
    message(STATUS "XeTLALibrary is not specified. Will try to download
                  XeTLA library from https://github.com/intel/xetla into
                  ${XeTLALibrary_SOURCE_DIR}")
    file(READ xetla_kernel/xetla-library.conf XeTLALibrary_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${XeTLALibrary_TAG}" XeTLALibrary_TAG)
    FetchContent_Declare(xetla-library
            GIT_REPOSITORY    https://github.com/intel/xetla.git
            GIT_TAG           ${XeTLALibrary_TAG}
            SOURCE_DIR ${XeTLALibrary_SOURCE_DIR}
            )

    FetchContent_GetProperties(xetla-library)
    if(NOT xetla-library_POPULATED)
       FetchContent_MakeAvailable(xetla-library)
    endif()

    set(XeTLALibrary_INCLUDE_DIR "${XeTLALibrary_SOURCE_DIR}/include"
            CACHE INTERNAL "XeTLALibrary_SOURCE_DIR")

    find_package_handle_standard_args(
            XeTLALibrary
            FOUND_VAR XeTLALibrary_FOUND
            REQUIRED_VARS
            XeTLALibrary_SOURCE_DIR)

endif (NOT XeTLALibrary_FOUND)
