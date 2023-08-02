
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


set(XPU_BACKEND_RECURSIVE ON)

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

set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_DIR=$ENV{TRITON_INTEL_LLVM_DIR}/build")
set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_INCLUDE_DIRS=$ENV{TRITON_INTEL_LLVM_DIR}/build/include")
set(CMAKE_ARGS "${CMAKE_ARGS};-DLLVM_LIBRARY_DIR=$ENV{TRITON_INTEL_LLVM_DIR}/build/lib")

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
        # Disable the tutorial component
        set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=OFF")
    elseif(CACHE_VAR STREQUAL "TRITON_BUILD_PYTHON_MODULE")
        # Disable the tutorial component
        set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=OFF")
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
