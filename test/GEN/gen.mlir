// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

llvm.func @gen_special_regs() -> i32 {
  // CHECK-LABEL: gen_special_regs
  // CHECK: gen.workitem.id.x : i32
  %0 = gen.workitem.id.x : i32
  // CHECK: gen.workitem.id.y : i32
  %1 = gen.workitem.id.y : i32
  // CHECK: gen.workitem.id.z : i32
  %2 = gen.workitem.id.z : i32
  // CHECK: gen.workgroup.id.x : i32
  %3 = gen.workgroup.id.x : i32
  // CHECK: gen.workgroup.id.y : i32
  %4 = gen.workgroup.id.y : i32
  // CHECK: gen.workgroup.id.z : i32
  %5 = gen.workgroup.id.z : i32
  // CHECK: gen.workgroup.dim.x : i32
  %6 = gen.workgroup.dim.x : i32
  // CHECK: gen.workgroup.dim.y : i32
  %7 = gen.workgroup.dim.y : i32
  // CHECK: gen.workgroup.dim.z : i32
  %8 = gen.workgroup.dim.z : i32
  // CHECK: gen.grid.dim.x : i32
  %9 = gen.grid.dim.x : i32
  // CHECK: gen.grid.dim.y : i32
  %10 = gen.grid.dim.y : i32
  // CHECK: gen.grid.dim.z : i32
  %11 = gen.grid.dim.z : i32
  llvm.return %0 : i32
}

llvm.func @gen.barrier() {
  // CHECK-LABEL: gen.barrier
  // CHECK: gen.barrier
  gen.barrier
  llvm.return
}

llvm.func @gen.sub_group_shuffle() {
  // CHECK-LABEL: gen.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = gen.sub_group_shuffle xor %0, %0 : i32 -> i32
  %1 = gen.sub_group_shuffle xor %0, %0 : i32 -> i32
  // CHECK: %2 = gen.sub_group_shuffle up %0, %0 : i32 -> i32
  %2 = gen.sub_group_shuffle up %0, %0 : i32 -> i32
  // CHECK: %3 = gen.sub_group_shuffle down %0, %0 : i32 -> i32
  %3 = gen.sub_group_shuffle down %0, %0 : i32 -> i32
  // CHECK: %4 = gen.sub_group_shuffle idx %0, %0 : i32 -> i32
  %4 = gen.sub_group_shuffle idx %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = gen.sub_group_shuffle xor %5, %0 : i8 -> i8
  %6 = gen.sub_group_shuffle xor %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = gen.sub_group_shuffle xor %7, %0 : i16 -> i16
  %8 = gen.sub_group_shuffle xor %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = gen.sub_group_shuffle xor %9, %0 : i64 -> i64
  %10 = gen.sub_group_shuffle xor %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = gen.sub_group_shuffle xor %11, %0 : f16 -> f16
  %12 = gen.sub_group_shuffle xor %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = gen.sub_group_shuffle xor %13, %0 : f32 -> f32
  %14 = gen.sub_group_shuffle xor %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = gen.sub_group_shuffle xor %15, %0 : f64 -> f64
  %16 = gen.sub_group_shuffle xor %15, %0 : f64 -> f64
  llvm.return
}

llvm.func @gen.fptofp(%a: f32, %b: f16) {
  // CHECK-LABEL: gen.fptofp
  // CHECK:      %0 = gen.conv.fptofp %arg0 {roundingMode = rte} : f32 to f16
  // CHECK-NEXT: %1 = gen.conv.fptofp %arg0 {roundingMode = rtn} : f32 to f16
  // CHECK-NEXT: %2 = gen.conv.fptofp %arg0 {roundingMode = rtp} : f32 to f16
  // CHECK-NEXT: %3 = gen.conv.fptofp %arg0 {roundingMode = rtz} : f32 to f16
  // CHECK-NEXT: %4 = gen.conv.fptofp %arg1 {roundingMode = rte} : f16 to f32
  // CHECK-NEXT: %5 = gen.conv.fptofp %arg1 {roundingMode = rtn} : f16 to f32
  // CHECK-NEXT: %6 = gen.conv.fptofp %arg1 {roundingMode = rtp} : f16 to f32
  // CHECK-NEXT: %7 = gen.conv.fptofp %arg1 {roundingMode = rtz} : f16 to f32
  %0 = gen.conv.fptofp %a {roundingMode = rte} : f32 to f16
  %1 = gen.conv.fptofp %a {roundingMode = rtn} : f32 to f16
  %2 = gen.conv.fptofp %a {roundingMode = rtp} : f32 to f16
  %3 = gen.conv.fptofp %a {roundingMode = rtz} : f32 to f16
  %4 = gen.conv.fptofp %b {roundingMode = rte} : f16 to f32
  %5 = gen.conv.fptofp %b {roundingMode = rtn} : f16 to f32
  %6 = gen.conv.fptofp %b {roundingMode = rtp} : f16 to f32
  %7 = gen.conv.fptofp %b {roundingMode = rtz} : f16 to f32
  llvm.return
}
