

#The SYCL kernel should be compiled with "-fsycl"
set_source_files_properties(softmax.cpp PROPERTIES COMPILE_FLAGS "-fsycl")

add_library(xetla_kernel STATIC softmax.cpp)
target_compile_options(xetla_kernel PUBLIC "-fPIC")
target_link_libraries(xetla_kernel "${TORCH_LIBRARIES}")
target_link_libraries(xetla_kernel "${TORCH_IPEX_LIBRARIES}")
target_include_directories(xetla_kernel PUBLIC "${Python_INCLUDE_DIRS}")
target_include_directories(xetla_kernel PUBLIC "${TORCH_IPEX_INCLUDE_DIRS}")
target_include_directories(xetla_kernel PUBLIC "${XeTLALibrary_INCLUDE_DIR}")
