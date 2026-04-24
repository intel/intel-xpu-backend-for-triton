// RUN: env TRITON_INTEL_2DBLOCK_ASSERT=1 triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s --check-prefix=ASSERT
// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s --check-prefix=NOASSERT

module attributes {"ttg.threads-per-warp" = 16 : i32} {
llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // ASSERT: llvm.call spir_funccc @__assert_fail
  // NOASSERT-NOT: __assert_fail
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<2xi16>
  llvm.return
}
}

// -----

module attributes {"ttg.threads-per-warp" = 16 : i32} {
llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // ASSERT: llvm.call spir_funccc @__assert_fail
  // NOASSERT-NOT: __assert_fail
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8, tile_width=32, tile_height=16, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
  llvm.return
}
}

// -----

module attributes {"ttg.threads-per-warp" = 16 : i32} {
llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr<1>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xi16>) {
  // ASSERT: llvm.call spir_funccc @__assert_fail
  // NOASSERT-NOT: __assert_fail
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8, tile_width=32, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi16>)
  llvm.return
}
}
