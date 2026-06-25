# arch_parser.c depends on <sycl/sycl.hpp> and is normally JIT-compiled at
# runtime by the Python driver using `icpx`. To unit-test it we need the SYCL
# toolchain at configure time. When it is not available (the common case for a
# plain build), skip this test so the rest of the build is unaffected.
#
# The surrounding Triton build uses a non-SYCL host compiler (e.g. gcc), so we
# cannot build this test as an ordinary CMake target: add_sycl_to_target() adds
# `-fsycl`, which only icpx understands. Instead we compile and link the test
# with icpx directly via a custom command, and register the resulting binary as
# a CTest test. This also avoids find_package(IntelSYCL), whose config script
# raises a FATAL_ERROR (not suppressed by QUIET) on hosts with a partial oneAPI
# install -- e.g. the Windows CI runner.
find_program(ICPX_EXECUTABLE NAMES icpx)

if(NOT ICPX_EXECUTABLE)
  message(STATUS "SYCL compiler (icpx) not found: skipping ArchParser unit test")
  return()
endif()

set(_ARCH_PARSER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../../backend/arch_parser.c)
set(_ARCH_PARSER_TEST ${CMAKE_CURRENT_SOURCE_DIR}/ArchParserTest.cpp)
set(_ARCH_PARSER_BIN ${CMAKE_CURRENT_BINARY_DIR}/ArchParserTest)

# GoogleTest is built by the host compiler, but its static archives are ELF and
# ABI-compatible with icpx (both use libstdc++ on Linux), so icpx can link them.
# Pull include dirs and library paths from the existing gtest/gmock targets so
# we track however the top-level unittest build provisions GoogleTest.
# arch_parser.c is C++ despite its extension (hence `-x c++`); SYCL requires
# RTTI, so unlike add_triton_ut we do not pass -fno-rtti.
add_custom_command(
  OUTPUT ${_ARCH_PARSER_BIN}
  COMMAND ${ICPX_EXECUTABLE}
          -fsycl
          -std=c++17
          "-I$<JOIN:$<TARGET_PROPERTY:gtest,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
          "-I$<JOIN:$<TARGET_PROPERTY:gmock,INTERFACE_INCLUDE_DIRECTORIES>,;-I>"
          ${_ARCH_PARSER_TEST}
          -x c++ ${_ARCH_PARSER_SRC} -x none
          "$<TARGET_FILE:gtest_main>"
          "$<TARGET_FILE:gtest>"
          "$<TARGET_FILE:gmock>"
          -lpthread
          -o ${_ARCH_PARSER_BIN}
  DEPENDS ${_ARCH_PARSER_TEST} ${_ARCH_PARSER_SRC} gtest_main gtest gmock
  COMMENT "Building ArchParser unit test with icpx"
  VERBATIM
  COMMAND_EXPAND_LISTS)

add_custom_target(ArchParser ALL DEPENDS ${_ARCH_PARSER_BIN})

# Custom-command binaries cannot be auto-discovered with gtest_discover_tests,
# so register the executable as a single CTest test (it runs all gtest cases).
add_test(NAME ArchParser COMMAND ${_ARCH_PARSER_BIN})

add_dependencies(TritonUnitTests ArchParser)
