// RUN: triton-opt -convert-gpu-to-tritongen %s | FileCheck %s

module attributes {
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  llvm.func @triton_gen.sub_group_reduce() {
    %0 = llvm.mlir.constant(0.0 : f32) : f32
    %1 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: triton_gen.sub_group_reduce add %0 {size = 16} : f32
    %2 = gpu.subgroup_reduce add %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce mul %0 {size = 16} : f32
    %3 = gpu.subgroup_reduce mul %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce min %1 {size = 16} : i32
    %4 = gpu.subgroup_reduce minui %1 : (i32) -> (i32)
    // CHECK: triton_gen.sub_group_reduce min %1 {size = 16} : i32
    %5 = gpu.subgroup_reduce minsi %1 : (i32) -> (i32)
    // CHECK: triton_gen.sub_group_reduce min %0 {size = 16} : f32
    %6 = gpu.subgroup_reduce minimumf %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce min %0 {size = 16} : f32
    %7 = gpu.subgroup_reduce minnumf %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce max %1 {size = 16} : i32
    %8 = gpu.subgroup_reduce maxui %1 : (i32) -> (i32)
    // CHECK: triton_gen.sub_group_reduce max %1 {size = 16} : i32
    %9 = gpu.subgroup_reduce maxsi %1 : (i32) -> (i32)
    // CHECK: triton_gen.sub_group_reduce max %0 {size = 16} : f32
    %10 = gpu.subgroup_reduce maximumf %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce max %0 {size = 16} : f32
    %11 = gpu.subgroup_reduce maxnumf %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce and %1 {size = 16} : i32
    %12 = gpu.subgroup_reduce and %1 : (i32) -> (i32)
    // CHECK: triton_gen.sub_group_reduce or %1 {size = 16} : i32
    %13 = gpu.subgroup_reduce or %1 : (i32) -> (i32)
    // CHECK: triton_gen.sub_group_reduce xor %1 {size = 16} : i32
    %14 = gpu.subgroup_reduce xor %1 : (i32) -> (i32)
    llvm.return
  }
}
}
