// RUN: triton-opt %s -split-input-file --triton-annotate-module='support-f4-conversion' --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s --check-prefixes=CHECK-SPIRV
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm  -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [32], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_intel_gpu.support_bfloat16_conversion, triton_intel_gpu.support_subgroup_matrix_multiply_accumulate, triton_intel_gpu.support_2d_block_io, triton_intel_gpu.target_arch = "spir16", ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fp4_to_bf16_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_tensor_descriptor %arg0, [%c16_i32], [%c1_i64] : <i8>, !tt.tensordesc<16xi8, #blocked>
    %3 = tt.descriptor_load %2[%1] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<16xi8, #blocked> -> tensor<16xi8, #blocked>
    %4 = ttg.fp4_to_fp %3 {axis = 0 : i32} : tensor<16xi8, #blocked> -> tensor<32xbf16, #blocked1>
    %5 = arith.muli %0, %c32_i32 : i32
    %6 = tt.make_tensor_descriptor %arg1, [%c32_i32], [%c1_i64] : <bf16>, !tt.tensordesc<32xbf16, #blocked1>
    tt.descriptor_store %6[%5], %4 : !tt.tensordesc<32xbf16, #blocked1>, tensor<32xbf16, #blocked1>
    tt.return
  }
}

// CHECK-SPIRV-COUNT-2: llvm.call spir_funccc @_Z36__builtin_spirv_ConvertE2M1ToBF16EXTDv16_i
// CHECK-LABEL: llvm.func spir_kernelcc @fp4_to_bf16_kernel
// CHECK-DAG: [[C6:%.+]] = llvm.mlir.constant(6 : i32) : i32
// CHECK-DAG: [[C16:%.+]] = llvm.mlir.constant(16 : i32) : i32
// CHECK-DAG: [[C24:%.+]] = llvm.mlir.constant(24 : i32) : i32
// CHECK-DAG: [[C28:%.+]] = llvm.mlir.constant(28 : i32) : i32
// CHECK-DAG: [[MASK:%.+]] = llvm.mlir.constant(-2118123520 : i32) : i32
// CHECK-DAG: [[SCALE:%.+]] = llvm.mlir.constant(8.50705917E+37 : f32) : f32
// CHECK-NOT: vector<16xbf16>
// CHECK: [[X:%.+]] = llvm.zext %{{.+}} : i8 to i32
// CHECK: [[LO:%.+]] = llvm.shl [[X]], [[C28]] : i32
// CHECK: [[LO1:%.+]] = llvm.ashr [[LO]], [[C6]] : i32
// CHECK: [[LO2:%.+]] = llvm.and [[LO1]], [[MASK]] : i32
// CHECK: [[LO3:%.+]] = llvm.bitcast [[LO2]] : i32 to f32
// CHECK: [[LO4:%.+]] = llvm.fmul [[LO3]], [[SCALE]] : f32
// CHECK: [[LO5:%.+]] = llvm.bitcast [[LO4]] : f32 to i32
// CHECK: [[LO6:%.+]] = llvm.lshr [[LO5]], [[C16]] : i32
// CHECK: [[LO7:%.+]] = llvm.trunc [[LO6]] : i32 to i16
// CHECK: llvm.bitcast [[LO7]] : i16 to bf16
// CHECK: [[HI:%.+]] = llvm.shl [[X]], [[C24]] : i32
// CHECK: [[HI1:%.+]] = llvm.ashr [[HI]], [[C6]] : i32
// CHECK: [[HI2:%.+]] = llvm.and [[HI1]], [[MASK]] : i32
// CHECK: [[HI3:%.+]] = llvm.bitcast [[HI2]] : i32 to f32
// CHECK: [[HI4:%.+]] = llvm.fmul [[HI3]], [[SCALE]] : f32
// CHECK: [[HI5:%.+]] = llvm.bitcast [[HI4]] : f32 to i32
// CHECK: [[HI6:%.+]] = llvm.lshr [[HI5]], [[C16]] : i32
// CHECK: [[HI7:%.+]] = llvm.trunc [[HI6]] : i32 to i16
// CHECK: llvm.bitcast [[HI7]] : i16 to bf16
// CHECK-COUNT-30: llvm.fmul %{{.+}}, [[SCALE]] : f32
// CHECK-NOT: llvm.fmul
// CHECK: llvm.return

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<4xi8, #blocked>) -> tensor<8xf16, #blocked1> {
    %dst = ttg.fp4_to_fp %src {axis = 0 : i32} : tensor<4xi8, #blocked> -> tensor<8xf16, #blocked1>
    // CHECK-SPIRV: llvm.call spir_funccc @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTDv8_i
    // CHECK-LABEL: llvm.func spir_kernelcc @convert
    // CHECK-NOT: vector<8xf16>
    // CHECK-COUNT-8: llvm.fptrunc %{{.+}} : f32 to f16
    // CHECK-NOT: llvm.fptrunc
    tt.return %dst : tensor<8xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<2xi8, #blocked>) -> tensor<4xf16, #blocked1> {
    %dst = ttg.fp4_to_fp %src {axis = 0 : i32} : tensor<2xi8, #blocked> -> tensor<4xf16, #blocked1>
    // CHECK-SPIRV: llvm.call spir_funccc @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTDv4_i
    // CHECK-LABEL: llvm.func spir_kernelcc @convert
    // CHECK-NOT: vector<8xf16>
    // CHECK-COUNT-4: llvm.fptrunc %{{.+}} : f32 to f16
    // CHECK-NOT: llvm.fptrunc
    tt.return %dst : tensor<4xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.target_arch = "spir64" } {
  tt.func public @convert(%src: tensor<1xi8, #blocked>) -> tensor<2xf16, #blocked1> {
    %dst = ttg.fp4_to_fp %src {axis = 0 : i32} : tensor<1xi8, #blocked> -> tensor<2xf16, #blocked1>
    // CHECK-SPIRV: llvm.call spir_funccc @_Z36__builtin_spirv_ConvertE2M1ToFP16EXTDv2_i
    // CHECK-LABEL: llvm.func spir_kernelcc @convert
    // CHECK-NOT: vector<8xf16>
    // CHECK-COUNT-2: llvm.fptrunc %{{.+}} : f32 to f16
    // CHECK-NOT: llvm.fptrunc
    tt.return %dst : tensor<2xf16, #blocked1>
  }
}
