cmake_minimum_required(VERSION 3.20)

project(
  pti-python-binaries
  VERSION 0.0.1
  LANGUAGES C CXX)

include(FetchContent)

if(DEFINED ENV{PTI_PINNED_COMMIT})
  set(PTI_PINNED_COMMIT $ENV{PTI_PINNED_COMMIT})
else()
  message(FATAL_ERROR "PTI_PINNED_COMMIT env var not defined")
endif()

if(DEFINED ENV{LEVELZERO_INCLUDE_DIR})
  set(LevelZero_INCLUDE_DIR "$ENV{LEVELZERO_INCLUDE_DIR}" CACHE PATH "Path to Level Zero includes")
  message(STATUS "Using Level Zero from ENV: ${LevelZero_INCLUDE_DIR}")
else()
  message(STATUS "LEVELZERO_INCLUDE_DIR env var not defined; try to use system version")
endif()

if(DEFINED ENV{LEVELZERO_LIBRARY})
  set(LevelZero_LIBRARY "$ENV{LEVELZERO_LIBRARY}" CACHE PATH "Path to Level Zero library")
  message(STATUS "Using Level Zero from ENV: ${LevelZero_LIBRARY}")
else()
  message(STATUS "LEVELZERO_LIBRARY env var not defined; try to use system version")
endif()

FetchContent_Declare(
  pti-lib
  GIT_REPOSITORY https://github.com/intel/pti-gpu.git
  GIT_TAG        ${PTI_PINNED_COMMIT}
  SOURCE_SUBDIR sdk
)

if(NOT APPLE)
  list(APPEND CMAKE_INSTALL_RPATH $ORIGIN)
endif()

# Sets the installation directories to be inside the root of the virtual
# environment, .e.g., .venv/lib/libpti_view.so (note: this is non-standard).
# However, this is what other oneAPI components and PyTorch XPU (Intel backend)
# expects.
set(CMAKE_INSTALL_LIBDIR "${SKBUILD_DATA_DIR}/lib")
set(CMAKE_INSTALL_INCLUDEDIR "${SKBUILD_DATA_DIR}/include")
set(CMAKE_INSTALL_BINDIR "${SKBUILD_DATA_DIR}/bin")
set(CMAKE_INSTALL_DOCDIR "${SKBUILD_DATA_DIR}/share")


FetchContent_MakeAvailable(pti-lib)
