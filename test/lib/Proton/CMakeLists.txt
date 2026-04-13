# Proton unit tests currently assume a generated Proton dialect header
# (Dialect.h.inc) that is not available in the Intel XPU fork layout.
# When Proton itself is not being built (TRITON_BUILD_PROTON=OFF), or when
# the dialect headers are missing, skip building TritonTestProton entirely.
if(NOT TRITON_BUILD_PROTON)
  message(STATUS "Skipping TritonTestProton (TRITON_BUILD_PROTON=OFF)")
  return()
endif()

add_library(TritonTestProton TestScopeIdAllocation.cpp)
target_link_libraries(TritonTestProton PUBLIC MLIRPass ProtonAnalysis)
target_compile_options(TritonTestProton PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
