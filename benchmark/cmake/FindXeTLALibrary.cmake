# Try to find XeTLA library.
#
include(FetchContent)

if (NOT XeTLALibrary_FOUND)

    set(XeTLALibrary_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/XeTLALibrary")
    message(STATUS "XeTLALibrary is not specified. Will try to download
                  XeTLA library from https://github.com/intel/xetla into
                  ${XeTLALibrary_SOURCE_DIR}")
    file(READ xetla-library.conf XeTLALibrary_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${XeTLALibrary_TAG}" XeTLALibrary_TAG)
    FetchContent_Declare(xetla-library
            GIT_REPOSITORY    https://github.com/intel/xetla.git
            GIT_TAG           ${XeTLALibrary_TAG}
            SOURCE_DIR ${XeTLALibrary_SOURCE_DIR}
            )

    FetchContent_Declare(xetla-library)

    # add the XeTLA library.
    #set(XETLA_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xetla/include)

    set(XeTLALibrary_INCLUDE_DIR "${XeTLALibrary_SOURCE_DIR}/include"
            CACHE INTERNAL "XeTLALibrary_SOURCE_DIR")

    find_package_handle_standard_args(
            XeTLALibrary
            FOUND_VAR XeTLALibrary_FOUND
            REQUIRED_VARS
            XeTLALibrary_SOURCE_DIR)

endif (NOT XeTLALibrary_FOUND)
