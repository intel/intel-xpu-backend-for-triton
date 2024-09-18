cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(reproducer)
set(CMAKE_CXX_COMPILER icpx)
set(BUILD_SHARED_LIBS OFF)

list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb/")

find_package(Torch REQUIRED)

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

# Add preview-breaking-changes for ABI compatibility with SYCL library linked by PyTorch: https://github.com/pytorch/pytorch/commit/92bebb46fa9fd60523d8aeb7b5f1a3f488c4cd93
set(COMPILE_FLAGS "-fsycl -Wall -fpreview-breaking-changes")
set(LINK_FLAGS "-fsycl -lze_loader")

set(SYCL_FUNCTIONS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/intel/backend/include")

set(TARGET_NAME SPIRVRunner)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
target_include_directories(${TARGET_NAME} PRIVATE "/opt/intel/oneapi/compiler/latest/include" ${SYCL_FUNCTIONS_INCLUDE_DIR} ${JSON_INCLUDE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
add_dependencies(${TARGET_NAME} json)

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
