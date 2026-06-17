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

    # Apply local patches to the fetched SYCL-TLA source.
    # 0001: causal FMHA forward hangs on Xe because the new mainloop uses a
    # workgroup-scope split barrier inside a K-loop whose trip count is
    # per-subgroup under causal masking (divergent -> deadlock). The legacy
    # kernel already guards this with subgroup scope; this patch applies the
    # same fix to applications/flash_attention_v2/collective/xe_fmha_fwd_mainloop.hpp.
    # Remove once https://github.com/intel/sycl-tla/pull/830 fix is present
    file(GLOB SyclTlaLibrary_PATCHES
         "${CMAKE_CURRENT_SOURCE_DIR}/sycl_tla_kernel/patches/*.patch")
    foreach(_patch ${SyclTlaLibrary_PATCHES})
        # Idempotent: skip if already applied (git apply --reverse --check succeeds).
        execute_process(
            COMMAND git apply --reverse --check "${_patch}"
            WORKING_DIRECTORY "${SyclTlaLibrary_SOURCE_DIR}"
            RESULT_VARIABLE _already_applied
            OUTPUT_QUIET ERROR_QUIET)
        if(_already_applied EQUAL 0)
            message(STATUS "SYCL-TLA patch already applied: ${_patch}")
        else()
            execute_process(
                COMMAND git apply "${_patch}"
                WORKING_DIRECTORY "${SyclTlaLibrary_SOURCE_DIR}"
                RESULT_VARIABLE _patch_result
                OUTPUT_VARIABLE _patch_output
                ERROR_VARIABLE _patch_output)
            if(NOT _patch_result EQUAL 0)
                message(FATAL_ERROR
                    "Failed to apply SYCL-TLA patch ${_patch}:\n${_patch_output}")
            endif()
            message(STATUS "Applied SYCL-TLA patch: ${_patch}")
        endif()
    endforeach()

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
