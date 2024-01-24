cmake_minimum_required(VERSION 3.18)

if(POLICY CMP0116)
# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
  cmake_policy(SET CMP0116 OLD)
endif()

# Use the PYTHON_THIRD_PARTY_PATH var to detect whether XPU backend is built as a plug-in
if(PYTHON_THIRD_PARTY_PATH AND NOT XPU_BACKEND_RECURSIVE_CONFIG)

  # hints from
  # https://stackoverflow.com/questions/10205986/how-to-capture-cmake-command-line-arguments
  # Propagate the CMake vars to the intel xpu backend
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
      set(CMAKE_ARGS "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=ON")
    elseif(CACHE_VAR STREQUAL "TRITON_INTEL_LLVM_DIR")
      string(REPLACE ";" "\\;" CACHE_VAR_VAL "${${CACHE_VAR}}")
      set(TRITON_INTEL_LLVM_DIR ${CACHE_VAR_VAL})
    elseif(CACHE_VAR STREQUAL "CMAKE_LIBRARY_OUTPUT_DIRECTORY" OR
            CACHE_VAR STREQUAL "CMAKE_BUILD_TYPE" OR
            CACHE_VAR STREQUAL "CMAKE_MAKE_PROGRAM" OR
            CACHE_VAR STREQUAL "CMAKE_EXPORT_COMPILE_COMMANDS" OR
            CACHE_VAR STREQUAL "CMAKE_VERBOSE_MAKEFILE" OR
            CACHE_VAR STREQUAL "PYTHON_INCLUDE_DIRS" OR
            CACHE_VAR STREQUAL "PYBIND11_INCLUDE_DIR" OR
            CACHE_VAR STREQUAL "PYTHON_LIBRARY" OR
            CACHE_VAR STREQUAL "GenISAIntrinsics_LIB_DIR"
    )
      # Keep ; semantic by replacing it with \\;
      string(REPLACE ";" "\\;" CACHE_VAR_VAL "${${CACHE_VAR}}")
      set(CMAKE_ARGS
      "${CMAKE_ARGS};-D${CACHE_VAR}${CACHE_VAR_TYPE}=${CACHE_VAR_VAL}")
    endif()
  endforeach()

  if(DEFINED ENV{TRITON_INTEL_LLVM_DIR})
    message(STATUS "[intel_xpu_backend]: using user provide llvm path")
    set(TRITON_INTEL_LLVM_DIR  $ENV{TRITON_INTEL_LLVM_DIR})
  endif()

  if(DEFINED TRITON_INTEL_LLVM_DIR)
    message(
            STATUS "[intel_xpu_backend]: set intel llvm path ${TRITON_INTEL_LLVM_DIR}"
    )
    set(CMAKE_ARGS
            "${CMAKE_ARGS};-DLLVM_INCLUDE_DIRS=${TRITON_INTEL_LLVM_DIR}/include")
    set(CMAKE_ARGS
            "${CMAKE_ARGS};-DLLVM_LIBRARY_DIR=${TRITON_INTEL_LLVM_DIR}/lib")
  else()
    message(FATAL_ERROR "Can't find intel llvm repo for Intel XPU backend. Please install it and provide explicit path with environment var TRITON_INTEL_LLVM_DIR= or cmake var -DTRITON_INTEL_LLVM_DIR=")
  endif()
  # re-config the CMake as a root cmake file
  execute_process(COMMAND ${CMAKE_COMMAND} ${CMAKE_ARGS}
                  "-DTRITON_CODEGEN_BACKENDS=intel"
                  -DXPU_BACKEND_RECURSIVE_CONFIG=ON
                  -DPYTHON_THIRD_PARTY_PATH=${PYTHON_THIRD_PARTY_PATH}
                  -G Ninja -S
                  ${CMAKE_CURRENT_SOURCE_DIR} -B ${CMAKE_CURRENT_BINARY_DIR}/triton
                  ECHO_OUTPUT_VARIABLE)

  add_custom_target(intel_xpu_backend_for_triton ALL
                    COMMAND ${CMAKE_COMMAND} --build . -j --config ${CMAKE_BUILD_TYPE} --target
                    triton
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/triton
                    COMMENT "Building intel_xpu_backend_for_triton"
                    VERBATIM USES_TERMINAL)

  # clean up the intel xpu backend in the TRITON_CODEGEN_BACKENDS var.
  # Because the intel xpu backend build a stand along share library instead of link everything into a mono libtriton.so.
  set(NEW_CODEGEN_BACKENDS)
  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    if(CODEGEN_BACKEND STREQUAL "intel")
      continue()
    endif()
    set(NEW_CODEGEN_BACKENDS "${NEW_CODEGEN_BACKENDS};${CODEGEN_BACKEND}")
  endforeach()

  set(TRITON_CODEGEN_BACKENDS ${NEW_CODEGEN_BACKENDS} PARENT_SCOPE)
  return()
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(triton)
include(CTest)

if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()



# Options
option(TRITON_BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)
set(TRITON_CODEGEN_BACKENDS "" CACHE STRING "Enable different codegen backends")

# Ensure Python3 vars are set correctly
# used conditionally in this file and by lit tests

# Customized release build type with assertions: TritonRelBuildWithAsserts
set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT WIN32)
  find_library(TERMINFO_LIBRARY tinfo)
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17")

# Third-party
include_directories(${PYBIND11_INCLUDE_DIR})



# #########
# LLVM
# #########
if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

# GenISAIntrinsics
if (NOT TARGET GenISAIntrinsics)
  if (EXISTS "${GenISAIntrinsics_LIB_DIR}/libGenISAIntrinsics.a")
    set(GenISAIntrinsics_LDFLAGS "-L${GenISAIntrinsics_LIB_DIR}")
  elseif (EXISTS "${LLVM_LIBRARY_DIR}/libGenISAIntrinsics.a")
    set(GenISAIntrinsics_LDFLAGS "-L${LLVM_LIBRARY_DIR}")
  else()
    message(FATAL_ERROR "Can't find libGenISAIntrinsics.a at ${LLVM_LIBRARY_DIR}. Please install it into that directory, or provide explicit path with -DGenISAIntrinsics_LIB_DIR=")
  endif()
endif()
set(GenISAIntrinsics_LIBRARY
  libGenISAIntrinsics.a
)
message(STATUS "GenISAIntrinsics_LDFLAGS: ${GenISAIntrinsics_LDFLAGS}")

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Utilities
function(add_triton_object name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT)
  target_sources(${name}
    PRIVATE ${ARG_UNPARSED_ARGUMENTS}
    INTERFACE $<TARGET_OBJECTS:${name}>
  )


  # add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} PUBLIC ${ARG_LINK_LIBS})
  endif()
endfunction(add_triton_object)

set_property(GLOBAL PROPERTY TRITON_LIBS "")
function(add_triton_library name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  add_triton_object(${name} ${ARGN})
  llvm_update_compile_flags(${name})
endfunction()

set_property(GLOBAL PROPERTY TRITON_PLUGINS "")
function(add_triton_plugin name)
  set_property(GLOBAL APPEND PROPERTY TRITON_PLUGINS ${name})
  add_triton_object(${name} ${ARGN})
endfunction()


# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default -fvisibility=hidden")

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files

# link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)

# find_package(PythonLibs REQUIRED)
set(TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# TODO: Figure out which target is sufficient to fix errors; triton is
# apparently not enough. Currently set linking libstdc++fs for all targets
# to support some old version GCC compilers like 8.3.0.
if (NOT WIN32 AND NOT APPLE)
  link_libraries(stdc++fs)
endif()


# -----

# ------
if(TRITON_BUILD_PYTHON_MODULE)

  if (XPU_BACKEND_RECURSIVE_CONFIG)
    set(TRITON_LIB_NAME "intel_xpu_backend_for_triton")
  else()
    set(TRITON_LIB_NAME "triton")
  endif()

  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  include_directories(${PYTHON_SRC_PATH})

  if(PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
  else()
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    include_directories(${Python3_INCLUDE_DIRS})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()

  foreach(CODEGEN_BACKEND ${TRITON_CODEGEN_BACKENDS})
    if(NOT PYTHON_THIRD_PARTY_PATH)
      set(PYTHON_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/triton/backends)
    endif()
    add_subdirectory(third_party/${CODEGEN_BACKEND})
  endforeach()


  get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
  get_property(triton_plugins GLOBAL PROPERTY TRITON_PLUGINS)
  set(TRITON_LIBRARIES
    ${triton_libs}
    ${triton_plugins}
    TritonSPIRV

    # mlir
    MLIRAMDGPUDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    MLIRGPUToNVVMTransforms
    MLIRGPUToGPURuntimeTransforms
    MLIRGPUTransforms
    MLIRIR
    MLIRControlFlowToLLVM
    MLIRBytecodeWriter
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRMathToLLVM
    MLIRROCDLToLLVMIRTranslation
    MLIRGENXToLLVMIRTranslation
    MLIRGPUDialect
    MLIRSCFToControlFlow
    MLIRIndexToLLVM
    MLIRGPUToROCDLTransforms

    # LLVM
    LLVMPasses
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMNVPTXCodeGen
    # LLVMNVPTXAsmPrinter
    LLVMAMDGPUCodeGen
    LLVMAMDGPUAsmParser

  )

  # Define triton library
  string(JOIN "," TRITON_BACKENDS_TUPLE ${TRITON_CODEGEN_BACKENDS})
  set(TRITON_BACKENDS_TUPLE "(${TRITON_BACKENDS_TUPLE})")
  add_compile_definitions(TRITON_BACKENDS_TUPLE=${TRITON_BACKENDS_TUPLE})
  add_compile_definitions(TRITON_LIB_NAME=${TRITON_LIB_NAME})
  add_library(triton SHARED ${PYTHON_SRC_PATH}/main.cc
                  ${PYTHON_SRC_PATH}/ir.cc
                  ${PYTHON_SRC_PATH}/passes.cc
                  ${PYTHON_SRC_PATH}/interpreter.cc
                  ${PYTHON_SRC_PATH}/llvm.cc)

  set_target_properties(triton PROPERTIES OUTPUT_NAME ${TRITON_LIB_NAME})

  # Link triton with its dependencies
  target_link_libraries(triton PUBLIC ${TRITON_LIBRARIES})
  if(WIN32)
    target_link_libraries(triton PRIVATE ${CMAKE_DL_LIBS})
  else()
    target_link_libraries(triton PRIVATE z)
  endif()
  target_link_options(triton PRIVATE ${LLVM_LDFLAGS} ${GenISAIntrinsics_LDFLAGS})
endif()

if(UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

if(TRITON_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

  # Check if the platform is MacOS
  if(APPLE)
    set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
  endif()

  target_link_libraries(triton PRIVATE ${PYTHON_LDFLAGS})
endif()

add_subdirectory(bin)
add_subdirectory(test)
add_subdirectory(unittest)
