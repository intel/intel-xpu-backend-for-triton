# arch_parser.c depends on <sycl/sycl.hpp> and is normally JIT-compiled at
# runtime by the Python driver using `icpx`. To unit-test it we need the SYCL
# toolchain at configure time. When it is not available (the common case for a
# plain build), skip this test so the rest of the build is unaffected.
find_package(IntelSYCL QUIET)

if(NOT IntelSYCL_FOUND)
  message(STATUS "IntelSYCL not found: skipping ArchParser unit test")
  return()
endif()

set(_ARCH_PARSER_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../backend/arch_parser.c)

add_executable(ArchParser ArchParserTest.cpp ${_ARCH_PARSER_SRC})

target_link_libraries(ArchParser PRIVATE GTest::gtest_main gmock)

# arch_parser.c is C++ despite its extension; SYCL requires RTTI, so do not
# apply the -fno-rtti used by add_triton_ut.
set_source_files_properties(${_ARCH_PARSER_SRC} PROPERTIES LANGUAGE CXX)

# Apply -fsycl and SYCL include/library paths to both the parser and the test.
add_sycl_to_target(TARGET ArchParser SOURCES ArchParserTest.cpp ${_ARCH_PARSER_SRC})

add_test(NAME ArchParser COMMAND ArchParser)
gtest_discover_tests(ArchParser DISCOVERY_TIMEOUT 60)

add_dependencies(TritonUnitTests ArchParser)
