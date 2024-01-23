add_triton_plugin(TritonXPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc)

# Python module
if(TRITON_BUILD_PYTHON_MODULE)
    message(STATUS "Adding Intel XPU Backend Python module")

    file(INSTALL
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/lib/libsycl-spir64-unknown-unknown.bc
            DESTINATION ${PYTHON_THIRD_PARTY_PATH}/intel/lib)

    file(INSTALL
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/bin/spirv-dis
            DESTINATION ${PYTHON_THIRD_PARTY_PATH}/intel/bin
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                             GROUP_READ GROUP_WRITE GROUP_EXECUTE
                             WORLD_READ             WORLD_EXECUTE)

    file(INSTALL
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/driver.c
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/__init__.py
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/build.py
            ${CMAKE_CURRENT_SOURCE_DIR}/backend/driver.py
            DESTINATION ${PYTHON_THIRD_PARTY_PATH}/intel/)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/backend/compiler.py.in
                   ${CMAKE_CURRENT_BINARY_DIR}/backend/compiler.py
                   @ONLY)
    file(INSTALL
            ${CMAKE_CURRENT_BINARY_DIR}/backend/compiler.py
            DESTINATION
            ${PYTHON_THIRD_PARTY_PATH}/intel/)

endif()
