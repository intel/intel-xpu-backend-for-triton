# A plain library, like third_party/amd/test/lib/Analysis: test passes must not enter the
# global TRITON_LIBS property, which the production Python module links in full.
add_library(TritonIntelTestAnalysis
  TestAxisInfo.cpp
  TestLivenessAnalysis.cpp
  TestRangeAnalysis.cpp
  TestRegisterPressureAnalysis.cpp
  TestStrideInfo.cpp
)
add_dependencies(TritonIntelTestAnalysis
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
)
target_link_libraries(TritonIntelTestAnalysis MLIRPass)
target_compile_options(TritonIntelTestAnalysis PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
