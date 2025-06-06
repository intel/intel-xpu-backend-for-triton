# Try to find ONEDNN library.

include(FetchContent)

if (NOT ONEDNNLibrary_FOUND)
    # TODO: switch ot FetchContent_MakeAvailable once ONEDNN supports it
    cmake_policy(SET CMP0169 OLD)

    set(ONEDNNLibrary_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/ONEDNNLibrary")
    message(STATUS "ONEDNNLibrary is not specified. Will try to download
                  ONEDNN library from https://github.com/uxlfoundation/oneDNN.git into
                  ${ONEDNNLibrary_SOURCE_DIR}")
    file(READ onednn_kernel/onednn-library.conf ONEDNNLibrary_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${ONEDNNLibrary_TAG}" ONEDNNLibrary_TAG)
    FetchContent_Declare(onednn-library
            GIT_REPOSITORY    https://github.com/uxlfoundation/oneDNN.git
            GIT_TAG           ${ONEDNNLibrary_TAG}
            SOURCE_DIR ${ONEDNNLibrary_SOURCE_DIR}
            )

    FetchContent_GetProperties(onednn-library)
    if(NOT onednn-library_POPULATED)
       FetchContent_Populate(onednn-library)
    endif()

    set(ONEDNNLibrary_INCLUDE_DIR "${ONEDNNLibrary_SOURCE_DIR}/include" CACHE INTERNAL "ONEDNNLibrary_SOURCE_DIR")

    find_package_handle_standard_args(
            ONEDNNLibrary
            FOUND_VAR ONEDNNLibrary_FOUND
            REQUIRED_VARS
            ONEDNNLibrary_SOURCE_DIR)

endif (NOT ONEDNNLibrary_FOUND)
