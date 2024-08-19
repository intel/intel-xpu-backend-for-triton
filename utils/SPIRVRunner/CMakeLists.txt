cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(reproducer)

set(CMAKE_CXX_COMPILER icpx)
set(BUILD_SHARED_LIBS OFF)

list(APPEND CMAKE_PREFIX_PATH "/opt/intel/oneapi/tbb/latest/lib/cmake/tbb/")

find_package(Torch REQUIRED)

set(COMPILE_FLAGS "-fsycl -Wall -fpreview-breaking-changes")
set(LINK_FLAGS "-fsycl -lze_loader")

set(SYCL_FUNCTIONS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/intel/backend/include")

set(TARGET_NAME SPIRVRunner)
add_executable(${TARGET_NAME} ${TARGET_NAME}.cpp)
target_include_directories(${TARGET_NAME} PRIVATE "/opt/intel/oneapi/compiler/latest/include" ${SYCL_FUNCTIONS_INCLUDE_DIR})
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")

target_link_libraries(${TARGET_NAME} "${TORCH_LIBRARIES}")
set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)
