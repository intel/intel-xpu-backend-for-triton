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

llvm.func @triton_gen.fptofp(%a: f32, %b: f16) {
  // CHECK-LABEL: triton_gen.fptofp
  // CHECK:      %0 = triton_gen.fptofp %arg0 {roundingMode = rte} : f32 to f16
  // CHECK-NEXT: %1 = triton_gen.fptofp %arg0 {roundingMode = rtn} : f32 to f16
  // CHECK-NEXT: %2 = triton_gen.fptofp %arg0 {roundingMode = rtp} : f32 to f16
  // CHECK-NEXT: %3 = triton_gen.fptofp %arg0 {roundingMode = rtz} : f32 to f16
  // CHECK-NEXT: %4 = triton_gen.fptofp %arg1 {roundingMode = rte} : f16 to f32
  // CHECK-NEXT: %5 = triton_gen.fptofp %arg1 {roundingMode = rtn} : f16 to f32
  // CHECK-NEXT: %6 = triton_gen.fptofp %arg1 {roundingMode = rtp} : f16 to f32
  // CHECK-NEXT: %7 = triton_gen.fptofp %arg1 {roundingMode = rtz} : f16 to f32
  // CHECK-NEXT: %8 = triton_gen.fptofp %arg1 : f16 to f32
  %0 = triton_gen.fptofp %a {roundingMode = rte} : f32 to f16
  %1 = triton_gen.fptofp %a {roundingMode = rtn} : f32 to f16
  %2 = triton_gen.fptofp %a {roundingMode = rtp} : f32 to f16
  %3 = triton_gen.fptofp %a {roundingMode = rtz} : f32 to f16
  %4 = triton_gen.fptofp %b {roundingMode = rte} : f16 to f32
  %5 = triton_gen.fptofp %b {roundingMode = rtn} : f16 to f32
  %6 = triton_gen.fptofp %b {roundingMode = rtp} : f16 to f32
  %7 = triton_gen.fptofp %b {roundingMode = rtz} : f16 to f32
  %8 = triton_gen.fptofp %b : f16 to f32
  llvm.return
}

llvm.func @triton_gen.2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: %0 = triton_gen.2Dblockload %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 16, tile_width = 16, tile_height = 16, v_blocks = 1, transpose = false, vnni_transform = false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  %0 = triton_gen.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16, tile_width=16, tile_height=16, v_blocks=1, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  llvm.return
}

llvm.func @triton_gen.2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xf32>) {
  // CHECK: triton_gen.2Dblockstore %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 1, transpose = false, vnni_transform = false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  triton_gen.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32, tile_width=8, tile_height=8, v_blocks=1, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  llvm.return
}
