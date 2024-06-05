// RUN: triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: check that the spirv target env is inserted
// CHECK: spirv.target_env{{.*}}#spirv.resource_limits<subgroup_size = 16>
module attributes { "triton_gpu.threads-per-warp" = 16 : i32, "triton_gpu.num-warps" = 4 : i32 } { }
