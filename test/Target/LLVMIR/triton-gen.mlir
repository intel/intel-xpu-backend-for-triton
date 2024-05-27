// RUN: triton-translate -triton-to-llvmir -split-input-file %s | FileCheck %s

// CHECK: define spir_kernel void @test_intel_reqd_sub_group_size() !intel_reqd_sub_group_size ![[REQD_SUB_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_intel_reqd_sub_group_size() attributes {triton_gen.intel_reqd_sub_group_size = [32 : i32]} {
  llvm.return
}
// CHECK: define spir_kernel void @test_max_work_group_size() !max_work_group_size ![[MAX_WORK_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_max_work_group_size() attributes {triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
  llvm.return
}
// CHECK: define spir_kernel void @test_reqd_work_group_size() !reqd_work_group_size ![[REQD_WORK_GROUP_SIZE:.*]] {
llvm.func spir_kernelcc @test_reqd_work_group_size() attributes {triton_gen.reqd_work_group_size = [128 : i32, 1 : i32, 2 : i32]} {
  llvm.return
}

// CHECK-DAG: ![[REQD_SUB_GROUP_SIZE]] = !{i64 32}
// CHECK-DAG: ![[MAX_WORK_GROUP_SIZE]] = !{i64 128, i64 1, i64 1}
// CHECK-DAG: ![[REQD_WORK_GROUP_SIZE]] = !{i64 128, i64 1, i64 2}
