add_library(GenISAIntrinsics STATIC IMPORTED GLOBAL)
set_target_properties(GenISAIntrinsics PROPERTIES IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/lib/libGenISAIntrinsics.a")
set_target_properties(GenISAIntrinsics PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/include")
