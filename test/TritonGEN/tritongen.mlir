// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @triton_gen_special_regs() -> i32 {
  // CHECK-LABEL: triton_gen_special_regs
  // CHECK: triton_gen.workitem.id.x : i32
  %0 = triton_gen.workitem.id.x : i32
  // CHECK: triton_gen.workitem.id.y : i32
  %1 = triton_gen.workitem.id.y : i32
  // CHECK: triton_gen.workitem.id.z : i32
  %2 = triton_gen.workitem.id.z : i32
  // CHECK: triton_gen.workgroup.id.x : i32
  %3 = triton_gen.workgroup.id.x : i32
  // CHECK: triton_gen.workgroup.id.y : i32
  %4 = triton_gen.workgroup.id.y : i32
  // CHECK: triton_gen.workgroup.id.z : i32
  %5 = triton_gen.workgroup.id.z : i32
  // CHECK: triton_gen.workgroup.dim.x : i32
  %6 = triton_gen.workgroup.dim.x : i32
  // CHECK: triton_gen.workgroup.dim.y : i32
  %7 = triton_gen.workgroup.dim.y : i32
  // CHECK: triton_gen.workgroup.dim.z : i32
  %8 = triton_gen.workgroup.dim.z : i32
  // CHECK: triton_gen.grid.dim.x : i32
  %9 = triton_gen.grid.dim.x : i32
  // CHECK: triton_gen.grid.dim.y : i32
  %10 = triton_gen.grid.dim.y : i32
  // CHECK: triton_gen.grid.dim.z : i32
  %11 = triton_gen.grid.dim.z : i32
  // CHECK: triton_gen.subgroup.id : i32
  %12 = triton_gen.subgroup.id : i32
  llvm.return %0 : i32
}

llvm.func @triton_gen.barrier() {
  // CHECK-LABEL: triton_gen.barrier
  // CHECK: triton_gen.barrier
  triton_gen.barrier
  llvm.return
}

llvm.func @triton_gen.sub_group_shuffle() {
  // CHECK-LABEL: triton_gen.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = triton_gen.sub_group_shuffle xor %0, %0 : i32 -> i32
  %1 = triton_gen.sub_group_shuffle xor %0, %0 : i32 -> i32
  // CHECK: %2 = triton_gen.sub_group_shuffle up %0, %0 : i32 -> i32
  %2 = triton_gen.sub_group_shuffle up %0, %0 : i32 -> i32
  // CHECK: %3 = triton_gen.sub_group_shuffle down %0, %0 : i32 -> i32
  %3 = triton_gen.sub_group_shuffle down %0, %0 : i32 -> i32
  // CHECK: %4 = triton_gen.sub_group_shuffle idx %0, %0 : i32 -> i32
  %4 = triton_gen.sub_group_shuffle idx %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = triton_gen.sub_group_shuffle xor %5, %0 : i8 -> i8
  %6 = triton_gen.sub_group_shuffle xor %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = triton_gen.sub_group_shuffle xor %7, %0 : i16 -> i16
  %8 = triton_gen.sub_group_shuffle xor %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = triton_gen.sub_group_shuffle xor %9, %0 : i64 -> i64
  %10 = triton_gen.sub_group_shuffle xor %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = triton_gen.sub_group_shuffle xor %11, %0 : f16 -> f16
  %12 = triton_gen.sub_group_shuffle xor %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = triton_gen.sub_group_shuffle xor %13, %0 : f32 -> f32
  %14 = triton_gen.sub_group_shuffle xor %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = triton_gen.sub_group_shuffle xor %15, %0 : f64 -> f64
  %16 = triton_gen.sub_group_shuffle xor %15, %0 : f64 -> f64
  llvm.return
}

llvm.func @triton_gen.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // CHECK:      llvm.func @triton_gen.dpas(%arg0: vector<8xi32>, %arg1: vector<16xi8>, %arg2: vector<32xi8>) {
  // CHECK-NEXT:   %0 = triton_gen.dpas %arg0, %arg1, %arg2 {pa = i8, pb = i8, rc = 8} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  %0 = triton_gen.dpas %c, %a, %b {pa=i8, pb=i8, rc=8} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.func @triton_gen.2Dblockload(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT:   %0 = triton_gen.2Dblockload %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  llvm.return
}

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xf32>) {
  // CHECK:      llvm.func @triton_gen.2Dblockstore(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: vector<8xf32>) {
  // CHECK-NEXT:   triton_gen.2Dblockstore %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  llvm.return
}

llvm.func @triton_gen.2Dblockprefetch(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK:      llvm.func @triton_gen.2Dblockprefetch(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // CHECK-NEXT:    triton_gen.2Dblockprefetch %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false, cache_control = Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  triton_gen.2Dblockprefetch %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false, cache_control=Default} : (!llvm.ptr, i32, i32, i32, i32, i32)
  llvm.return
}
