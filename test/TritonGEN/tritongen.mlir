// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @gen_special_regs() -> i32 {
  // CHECK-LABEL: gen_special_regs
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
