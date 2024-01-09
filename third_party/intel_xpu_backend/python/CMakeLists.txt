
# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding Intel XPU Backend Python module")
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/intel_xpu_backend_for_triton.cc)
    include_directories("." ${PYTHON_SRC_PATH})
    include_directories(../include)

    if(PYTHON_INCLUDE_DIRS)
        include_directories(${PYTHON_INCLUDE_DIRS})
    else()
        find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
        include_directories(${Python3_INCLUDE_DIRS})
        link_directories(${Python3_LIBRARY_DIRS})
        link_libraries(${Python3_LIBRARIES})
        add_link_options(${Python3_LINK_OPTIONS})
    endif()

    add_library(intel_xpu_backend_for_triton SHARED ${PYTHON_SRC})

    get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
    get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

    set(INTEL_EXTENSION_LIBRARIES
            TritonAnalysis
            TritonTransforms
            TritonGPUTransforms
            TritonSPIRV
            TritonGPUToSPIRV
            ${dialect_libs}
            ${conversion_libs}

            # optimizations
            MLIRBytecodeWriter
            MLIRPass
            MLIRTransforms
            MLIRSupport
            # things go crazy in the library loops in Analysis and GPUTransforms and
            # CMake supporting in --starg-group and --end-group.
            # here we repeat the link libraries to work around it.
            TritonAnalysis
            TritonTransforms
            TritonGPUTransforms
            ${dialect_libs}
            ${conversion_libs}
            )
    target_link_libraries(intel_xpu_backend_for_triton PRIVATE ${LLVM_LIBRARIES})
    target_link_libraries(intel_xpu_backend_for_triton PRIVATE ${INTEL_EXTENSION_LIBRARIES})
    # TODO: Figure out which target is sufficient to fix errors; intel_xpu_backend_for_triton is
    # apparently not enough
    link_libraries(stdc++fs)

    target_link_options(intel_xpu_backend_for_triton PRIVATE ${LLVM_LDFLAGS})

    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/lib/libsycl-fallback-imf.spv
         ${CMAKE_CURRENT_SOURCE_DIR}/lib/libsycl-fallback-imf-fp64.spv
         ${CMAKE_CURRENT_SOURCE_DIR}/lib/libsycl-fallback-cassert.spv
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/xpu/lib)

    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/utils/sycl.cpp
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/xpu/utils)

    file(INSTALL
         ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
         ${CMAKE_CURRENT_SOURCE_DIR}/extensions.py
         DESTINATION ${PYTHON_THIRD_PARTY_PATH}/xpu/)
endif()
