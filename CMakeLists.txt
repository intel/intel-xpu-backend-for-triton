
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if (DEFINED ENV{TRITON_INTEL_LLVM_DIR})
MESSAGE(STATUS "[intel_xpu_backend]: using TRITON_INTEL_LLVM_DIR")

# set_target_properties(SPIRV-Tools-shared PROPERTIES EXCLUDE_FROM_ALL ON)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/llvm)
MESSAGE(STATUS "[intel_xpu_backend]: using previous exists llvm")
else()
include(FetchContent)
function(download_file url hash)

FetchContent_Declare(private-llvm
URL ${url}
SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/llvm
)

FetchContent_Populate(private-llvm)
endfunction(download_file)

MESSAGE(STATUS "[intel_xpu_backend]: downloading private llvm for intel_xpu_backend")
download_file(
  https://github.com/chuanqi129/inductor-tools/releases/download/0.1/llvm+mlir-17.0.0-x86_64-linux-gnu-ubuntu-22.04-release.tar.xz
  40ea513050b3689201c7877854909662
)

MESSAGE(STATUS "[intel_xpu_backend]: unziping private-llvm")
add_custom_target(
    unpack-llvm
    COMMAND
        ${CMAKE_COMMAND} -E tar xf ${CMAKE_CURRENT_SOURCE_DIR}/llvm/private-llvm.tar.xz
)

set(ENV{TRITON_INTEL_LLVM_DIR}  ${CMAKE_CURRENT_SOURCE_DIR}/llvm)
MESSAGE(STATUS "[intel_xpu_backend]: set llvm path ENV{TRITON_INTEL_LLVM_DIR}")
endif()

else ()
MESSAGE(STATUS "[intel_xpu_backend]: using original llvm ")

set(XPU_BACKEND_RECURSIVE ON)
endif()

if(XPU_BACKEND_RECURSIVE)
set(TRITON_BUILD_PYTHON_MODULE ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(python)
else()

MESSAGE(STATUS "[intel_xpu_backend]:Triton uses LLVM version ${LLVM_PACKAGE_VERSION} with revision ${LLVM_GIT_REVISION}")
MESSAGE(STATUS "[intel_xpu_backend]:Use the LLVM version with revision to re-config for the intel_xpu_backend")

# hints from https://stackoverflow.com/questions/10205986/how-to-capture-cmake-command-line-arguments
# Propagate the CMake vars to the intel xpu backend

set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_DIR=$ENV{TRITON_INTEL_LLVM_DIR}")
set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_INCLUDE_DIRS=$ENV{TRITON_INTEL_LLVM_DIR}/include")
set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_LIBRARY_DIR=$ENV{TRITON_INTEL_LLVM_DIR}/lib")

get_cmake_property(CACHE_VARS CACHE_VARIABLES)
foreach(CACHE_VAR ${CACHE_VARS})
    get_property(CACHE_VAR_HELPSTRING CACHE ${CACHE_VAR} PROPERTY HELPSTRING)
    get_property(CACHE_VAR_TYPE CACHE ${CACHE_VAR} PROPERTY TYPE)
    if(CACHE_VAR_TYPE STREQUAL "UNINITIALIZED")
        set(CACHE_VAR_TYPE)
    else()
        set(CACHE_VAR_TYPE :${CACHE_VAR_TYPE})
    endif()

    if(CACHE_VAR STREQUAL "TRITON_BUILD_TUTORIALS")
        set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=OFF")
    elseif(CACHE_VAR STREQUAL "TRITON_BUILD_PYTHON_MODULE")
        set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=OFF")
    elseif(CACHE_VAR STREQUAL "LLVM_LIBRARY_DIR")
        continue()
    elseif(CACHE_VAR STREQUAL "LLVM_DIR")
        continue()
    elseif(CACHE_VAR STREQUAL "LLVM_INCLUDE_DIRS")
        continue()
    else()
        set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=${${CACHE_VAR}}")
    endif()
endforeach()

execute_process(COMMAND ${CMAKE_COMMAND}
        ${CMAKE_ARGS}
        -DXPU_BACKEND_RECURSIVE=ON
        -S ${PROJECT_SOURCE_DIR}
        -B ${CMAKE_CURRENT_BINARY_DIR}/triton)

add_custom_target(intel_xpu_backend_for_triton ALL
        COMMAND
        ${CMAKE_COMMAND} --build . --config ${CMAKE_BUILD_TYPE} --target intel_xpu_backend_for_triton
        WORKING_DIRECTORY  ${CMAKE_CURRENT_BINARY_DIR}/triton
        COMMENT "Building intel_xpu_backend_for_triton"
        VERBATIM
        USES_TERMINAL)

endif()
