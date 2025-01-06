// RUN: triton-opt -convert-tritongen-to-llvm -split-input-file %s | FileCheck %s

// CHECK-DAG: llvm.func spir_funccc @_Z31intel_work_group_barrier_arriveii(i32, i32) attributes {convergent, no_unwind, will_return}
// CHECK-DAG: llvm.func spir_funccc @_Z29intel_work_group_barrier_waitii(i32, i32) attributes {convergent, no_unwind, will_return}

llvm.func @triton_gen.split_barrier() {
  // CHECK-LABEL: triton_gen.split_barrier() {
  // CHECK-DAG: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:     llvm.call spir_funccc @_Z31intel_work_group_barrier_arriveii([[ZERO]], [[ONE]]) {{.*}} : (i32, i32) -> ()
  // CHECK-DAG: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK:     llvm.call spir_funccc @_Z29intel_work_group_barrier_waitii([[ZERO]], [[ONE]]) {{.*}} : (i32, i32) -> ()
  %0 = triton_gen.split_barrier_init
  triton_gen.split_barrier_signal %0
  triton_gen.split_barrier_wait %0
  llvm.return
}
