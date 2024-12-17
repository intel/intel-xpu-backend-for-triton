// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<8xi16>, %b : vector<8xi32>) {
  // CHECK:      llvm.func @triton_gen.dpas(%arg0: vector<8xi32>, %arg1: vector<8xi16>, %arg2: vector<8xi32>) {
  // CHECK-NEXT:   %0 = triton_gen.dpas %arg0, %arg1, %arg2 {pa = i8, pb = i8, rc = 8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi32>, vector<8xi16>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

llvm.func @foo(%arg0: !llvm.ptr, %arg1: !llvm.ptr)

llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr) {
  // CHECK: llvm.func @triton_gen.cache_controls(%arg0: !llvm.ptr)
  // CHECK-NEXT: %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>} : !llvm.ptr -> i32
  %0 = llvm.load %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.load_cache_control<0, Cached, 0>, #triton_gen.load_cache_control<1, Uncached, 0>>} : !llvm.ptr -> i32
  // CHECK-NEXT: llvm.store %0, %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, WriteBack, 1>, #triton_gen.store_cache_control<1, Streaming, 1>>} : i32, !llvm.ptr
  llvm.store %0, %arg0 {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, WriteBack, 1>, #triton_gen.store_cache_control<1, Streaming, 1>>} : i32, !llvm.ptr
  // CHECK-NEXT: llvm.call @foo(%arg0, %arg0) {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<0, Cached, 1>>} : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @foo(%arg0, %arg0) {triton_gen.DecorationCacheControlINTEL = #triton_gen.decoration_cache_control<#triton_gen.store_cache_control<0, Uncached, 0>, #triton_gen.load_cache_control<0, Cached, 1>>} : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.return
}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT:   %0 = triton_gen.2Dblockload %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  llvm.return
}

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<16xf32>) {
  // CHECK:      llvm.func @triton_gen.2Dblockstore(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: vector<16xf32>) {
  // CHECK-NEXT:   triton_gen.2Dblockstore %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<16xf32>)
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=16, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<16xf32>)
  llvm.return
}

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.func @triton_gen.2Dblockprefetch(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT:    triton_gen.2Dblockprefetch %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 1, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}

llvm.func @triton_gen.sub_group_block_read(%ptr : !llvm.ptr<1>) {
  // CHECK:      llvm.func @triton_gen.sub_group_block_read(%arg0: !llvm.ptr<1>) {
  // CHECK-NEXT:   triton_gen.sub_group_block_read %arg0 : !llvm.ptr<1> -> vector<2xi16>
  triton_gen.sub_group_block_read %ptr : !llvm.ptr<1> -> vector<2xi16>
  llvm.return
}

llvm.func @triton_gen.sub_group_block_write(%ptr : !llvm.ptr<3>, %val : i32) {
  // CHECK:      llvm.func @triton_gen.sub_group_block_write(%arg0: !llvm.ptr<3>, %arg1: i32) {
  // CHECK-NEXT:    triton_gen.sub_group_block_write %arg0, %arg1 : !llvm.ptr<3>, i32
  triton_gen.sub_group_block_write %ptr, %val : !llvm.ptr<3>, i32
  llvm.return
}
