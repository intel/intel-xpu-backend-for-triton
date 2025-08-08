# Try to find oneDNN library.

include(FetchContent)

if (NOT oneDNNLibrary_FOUND)
    # TODO: switch to FetchContent_MakeAvailable once oneDNN supports it
    cmake_policy(SET CMP0169 OLD)

    set(oneDNNLibrary_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/oneDNNLibrary")
    message(STATUS "oneDNNLibrary is not specified. Will try to download
                  oneDNN library from https://github.com/uxlfoundation/oneDNN.git into
                  ${oneDNNLibrary_SOURCE_DIR}")
    file(READ onednn_kernel/onednn-library.conf oneDNNLibrary_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${oneDNNLibrary_TAG}" oneDNNLibrary_TAG)
    FetchContent_Declare(onednn-library
            GIT_REPOSITORY    https://github.com/uxlfoundation/oneDNN.git
            GIT_TAG           ${oneDNNLibrary_TAG}
            SOURCE_DIR ${oneDNNLibrary_SOURCE_DIR}
            )

    FetchContent_GetProperties(onednn-library)
    if(NOT onednn-library_POPULATED)
       FetchContent_Populate(onednn-library)
    endif()

    set(oneDNNLibrary_INCLUDE_DIR "${oneDNNLibrary_SOURCE_DIR}/include" CACHE INTERNAL "oneDNNLibrary_SOURCE_DIR")

    find_package_handle_standard_args(
            oneDNNLibrary
            FOUND_VAR oneDNNLibrary_FOUND
            REQUIRED_VARS
            oneDNNLibrary_SOURCE_DIR)

endif (NOT oneDNNLibrary_FOUND)
