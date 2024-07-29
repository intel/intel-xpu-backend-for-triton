// RUN: triton-opt -convert-gpu-to-tritongen %s | FileCheck %s

module attributes {
  "triton_gpu.threads-per-warp" = 16 : i32,
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Kernel, Addresses, GroupNonUniformShuffle, Int64], []>, #spirv.resource_limits<subgroup_size = 16>>
} {

gpu.module @kernels {
  llvm.func @triton_gen.sub_group_reduce() {
    %0 = llvm.mlir.constant(0.0 : f32) : f32
    // CHECK: triton_gen.sub_group_reduce add %0 {size = 16} : f32
    %1 = gpu.subgroup_reduce add %0 : (f32) -> (f32)
    // CHECK: triton_gen.sub_group_reduce max %0 {size = 16} : f32
    %2 = gpu.subgroup_reduce maxnumf %0 : (f32) -> (f32)
    llvm.return
  }
}
}
