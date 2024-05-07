set(TARGET bgemm_performance)

#The SYCL kernel should be compiled with "-fsycl"
set_source_files_properties(bgemm.cpp PROPERTIES COMPILE_FLAGS "-fsycl")

add_library(${TARGET} STATIC bgemm.cpp)
target_compile_options(${TARGET} PUBLIC "-fPIC")
target_link_libraries(${TARGET} PUBLIC "${TORCH_LIBRARIES}")
target_link_libraries(${TARGET} PUBLIC "${TORCH_IPEX_LIBRARIES}")
target_link_libraries(${TARGET} PUBLIC OpenCL)
target_link_libraries(${TARGET} PUBLIC gtest gtest_main)
target_include_directories(${TARGET} PUBLIC "${Python_INCLUDE_DIRS}")
target_include_directories(${TARGET} PUBLIC "${TORCH_IPEX_INCLUDE_DIRS}")
target_include_directories(${TARGET} PUBLIC "${XeTLALibrary_INCLUDE_DIR}")
