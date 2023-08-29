list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(DEFINED ENV{TRITON_INTEL_LLVM_DIR})
  message(STATUS "[intel_xpu_backend]: using user provide llvm path")
else()
  if(DEFINED ENV{TRITON_CODEGEN_INTEL_XPU_BACKEND})
    message(STATUS "[intel_xpu_backend]: using intel_llvm")

    set(TRITON_INTEL_LLVM_DIR $ENV{HOME}/.triton/intel_llvm)
    message(
      STATUS "[intel_xpu_backend]: set intel llvm path ${TRITON_INTEL_LLVM_DIR}"
    )
    if(EXISTS ${TRITON_INTEL_LLVM_DIR}/lib/cmake)
      message(STATUS "[intel_xpu_backend]: using previous exists llvm")
    else()
      include(FetchContent)
      function(download_file url)
        cmake_policy(SET CMP0135 NEW)

        FetchContent_Declare(intel-llvm URL ${url} SOURCE_DIR
                                            ${TRITON_INTEL_LLVM_DIR})

        FetchContent_Populate(intel-llvm)
      endfunction(download_file)

      set(LLVM_DOWNLOAD_URL
          https://github.com/intel/intel-xpu-backend-for-triton/releases/download/llvm_pkg_v0.1/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-18.04-release.tar.xz
      )
      set(OS_INFO_FILE "/etc/os-release")

      message(
        STATUS
          "[intel_xpu_backend]: downloading intel llvm for intel_xpu_backend, path ${TRITON_INTEL_LLVM_DIR}"
      )
      if(EXISTS ${OS_INFO_FILE})
        file(STRINGS ${OS_INFO_FILE} OS_INFO_KEY_LIST REGEX "^(ID|VERSION_ID)")
        set(OS_ID "")
        set(OS_VERSION_ID "")
        foreach(_os_info ${OS_INFO_KEY_LIST})
          if("${_os_info}" MATCHES "^(ID)=(.*)$")
            string(TOLOWER "${CMAKE_MATCH_2}" OS_ID)
          elseif("${_os_info}" MATCHES "(VERSION_ID)=(.*)$")
            string(TOLOWER "${CMAKE_MATCH_2}" OS_VERSION_ID)
            string(REGEX REPLACE "^\"(.*)\"$" "\\1" OS_VERSION_ID
                                 "${OS_VERSION_ID}")
          endif()
        endforeach()

        if("${OS_ID}" STREQUAL "ubuntu")
          if("${OS_VERSION_ID}" STREQUAL "22.04")
            set(LLVM_DOWNLOAD_URL
                https://github.com/intel/intel-xpu-backend-for-triton/releases/download/llvm_pkg_v0.1/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-22.04-release.tar.xz
            )
            message(
              STATUS
                "[intel_xpu_backend]: downloading intel llvm for ${OS_ID} ${OS_VERSION_ID}"
            )
          endif()
        elseif("${OS_ID}" STREQUAL "centos")
          if("${OS_VERSION_ID}" STREQUAL "7")
            set(LLVM_DOWNLOAD_URL
                https://github.com/intel/intel-xpu-backend-for-triton/releases/download/llvm_pkg_v0.1/llvm+mlir-17.0.0-x86_64-linux-gnu-centos-7-release.tar.xz
            )
            message(
              STATUS
                "[intel_xpu_backend]: downloading intel llvm for ${OS_ID} ${OS_VERSION_ID}"
            )
          endif()
        endif()
      endif()

      message(
        STATUS
          "[intel_xpu_backend]: downloading intel llvm from ${LLVM_DOWNLOAD_URL}"
      )
      download_file(${LLVM_DOWNLOAD_URL})
    endif()

    message(
      STATUS
        "[intel_xpu_backend]: Triton uses LLVM version ${LLVM_PACKAGE_VERSION} with revision ${LLVM_GIT_REVISION}"
    )
    message(
      STATUS
        "[intel_xpu_backend]: Use the LLVM version with revision to re-config for the intel_xpu_backend"
    )
    # hints from
    # https://stackoverflow.com/questions/10205986/how-to-capture-cmake-command-line-arguments
    # Propagate the CMake vars to the intel xpu backend

    set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_DIR=${TRITON_INTEL_LLVM_DIR}")
    set(CMAKE_ARGS
        "${CMAKE_ARGS};-DLLVM_INCLUDE_DIRS=${TRITON_INTEL_LLVM_DIR}/include")
    set(CMAKE_ARGS
        "${CMAKE_ARGS};-DLLVM_LIBRARY_DIR=${TRITON_INTEL_LLVM_DIR}/lib")

  else()
    message(STATUS "[intel_xpu_backend]: using original llvm ")

    set(XPU_BACKEND_RECURSIVE ON)
  endif()
endif()

if(XPU_BACKEND_RECURSIVE)
  set(TRITON_BUILD_PYTHON_MODULE ON)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

  add_subdirectory(include)
  add_subdirectory(lib)
  add_subdirectory(python)
else()
  get_cmake_property(CACHE_VARS CACHE_VARIABLES)
  foreach(CACHE_VAR ${CACHE_VARS})
    get_property(
      CACHE_VAR_HELPSTRING
      CACHE ${CACHE_VAR}
      PROPERTY HELPSTRING)
    get_property(
      CACHE_VAR_TYPE
      CACHE ${CACHE_VAR}
      PROPERTY TYPE)
    if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
      set(CACHE_VAR_TYPE)
    else()
      set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
    endif()

    if(CACHE_VAR STREQUAL "TRITON_BUILD_TUTORIALS")
      set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=OFF")
    elseif(CACHE_VAR STREQUAL "TRITON_BUILD_PYTHON_MODULE")
      # We still need the Python module, since there are cross-dependencies.
      set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=ON")
    elseif(CACHE_VAR STREQUAL "LLVM_LIBRARY_DIR")
      continue()
    elseif(CACHE_VAR STREQUAL "LLVM_DIR")
      continue()
    elseif(CACHE_VAR STREQUAL "LLVM_INCLUDE_DIRS")
      continue()
    else()
      set(CMAKE_ARGS
          "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=${${CACHE_VAR}}")
    endif()
  endforeach()

  execute_process(
    COMMAND ${CMAKE_COMMAND} ${CMAKE_ARGS} -DXPU_BACKEND_RECURSIVE=ON -S
            ${PROJECT_SOURCE_DIR} -B ${CMAKE_CURRENT_BINARY_DIR}/triton)

  add_custom_target(
    intel_xpu_backend_for_triton ALL
    COMMAND ${CMAKE_COMMAND} --build . -j --config ${CMAKE_BUILD_TYPE} --target
            intel_xpu_backend_for_triton
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/triton
    COMMENT "Building intel_xpu_backend_for_triton"
    VERBATIM USES_TERMINAL)

endif()
