# Try to find SYCL-TLA library.

include(FetchContent)

if (NOT SyclTlaLibrary_FOUND)
    # TODO: switch to FetchContent_MakeAvailable once SYCL-TLA supports it
    cmake_policy(SET CMP0169 OLD)

    set(SyclTlaLibrary_SOURCE_DIR
            "${CMAKE_CURRENT_BINARY_DIR}/SyclTlaLibrary")
    message(STATUS "SyclTlaLibrary is not specified. Will try to download
                  SYCL-TLA library from https://github.com/intel/sycl-tla.git into
                  ${SyclTlaLibrary_SOURCE_DIR}")
    file(READ sycl_tla_kernel/sycl-tla-library.conf SyclTlaLibrary_TAG)
    # Strip the potential trailing newline from tag
    string(STRIP "${SyclTlaLibrary_TAG}" SyclTlaLibrary_TAG)
    FetchContent_Declare(sycl-tla-library
            GIT_REPOSITORY    https://github.com/intel/sycl-tla.git
            GIT_TAG           ${SyclTlaLibrary_TAG}
            SOURCE_DIR ${SyclTlaLibrary_SOURCE_DIR}
            )

    FetchContent_GetProperties(sycl-tla-library)
    if(NOT sycl-tla-library_POPULATED)
       FetchContent_Populate(sycl-tla-library)
    endif()

    set(SyclTlaLibrary_INCLUDE_DIR "${SyclTlaLibrary_SOURCE_DIR}/include" CACHE INTERNAL "SyclTlaLibrary_SOURCE_DIR")
    set(SyclTlaLibrary_INCLUDE_TOOL_DIR "${SyclTlaLibrary_SOURCE_DIR}/tools/util/include" CACHE INTERNAL "SyclTlaLibrary_SOURCE_DIR")
    set(SyclTlaLibrary_INCLUDE_APPLICATION_DIR "${SyclTlaLibrary_SOURCE_DIR}/applications" CACHE INTERNAL "SyclTlaLibrary_SOURCE_DIR")
    set(SyclTlaLibrary_INCLUDE_BENCHMARK_DIR "${SyclTlaLibrary_SOURCE_DIR}/benchmarks" CACHE INTERNAL "SyclTlaLibrary_SOURCE_DIR")
    set(SyclTlaLibrary_BENCHMARK_CONFIG_DIR "${SyclTlaLibrary_SOURCE_DIR}/benchmarks/device/pvc/input_files" CACHE INTERNAL "SyclTlaLibrary_SOURCE_DIR")

    find_package_handle_standard_args(
            SyclTlaLibrary
            FOUND_VAR SyclTlaLibrary_FOUND
            REQUIRED_VARS
            SyclTlaLibrary_SOURCE_DIR)

endif (NOT SyclTlaLibrary_FOUND)
