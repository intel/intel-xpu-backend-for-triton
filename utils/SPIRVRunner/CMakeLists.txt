cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(reproducer)
set(CMAKE_CXX_COMPILER icpx)
set(BUILD_SHARED_LIBS OFF)

if(DEFINED ENV{ONEAPI_ROOT})
  set(ONEAPI_ROOT "$ENV{ONEAPI_ROOT}")
endif()
if(NOT ONEAPI_ROOT)
  set(ONEAPI_ROOT "/opt/intel/oneapi")
endif()

message(STATUS "ONEAPI_ROOT: ${ONEAPI_ROOT}")

list(APPEND CMAKE_PREFIX_PATH "${ONEAPI_ROOT}/tbb/latest/lib/cmake/tbb/")

find_package(Torch REQUIRED)

# Include LLVM Support Library for CLI parsing
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_library(llvm_parser OBJECT llvm_parser.cpp)
target_compile_options(llvm_parser PRIVATE -fno-rtti)

include(ExternalProject)
ExternalProject_Add(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    PREFIX ${CMAKE_BINARY_DIR}/nlohmann_json
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
set(JSON_INCLUDE_DIR ${CMAKE_BINARY_DIR}/nlohmann_json/src/json/include/)

set(COMPILE_FLAGS "-fsycl -Wall")
set(LINK_FLAGS "-fsycl -lze_loader")

set(SYCL_FUNCTIONS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/intel/backend/include")
set(TARGET_NAME SPIRVRunner)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp $<TARGET_OBJECTS:llvm_parser>)
target_include_directories(${TARGET_NAME} PRIVATE
	"${ONEAPI_ROOT}/compiler/latest/include" ${SYCL_FUNCTIONS_INCLUDE_DIR} ${JSON_INCLUDE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_dependencies(${TARGET_NAME} json)
llvm_map_components_to_libnames(LLVM_LIBS Support)
target_link_libraries(${TARGET_NAME} PRIVATE "${TORCH_LIBRARIES}" "${LLVM_LIBS}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
