// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm -canonicalize | FileCheck %s

// Regression test: fp4_to_fp fallback path when the input tensor is loaded
// as packed i32 words (sizePerThread=8 forces a vector<2xi32> descriptor load).
// Each i32 holds 4 packed i8 values; the lowering must extract all 4 nibbles
// per element (not 2), producing 16 bf16 outputs for 8 input bytes.

#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {triton_intel_gpu.support_bfloat16_conversion, triton_intel_gpu.support_subgroup_matrix_multiply_accumulate, triton_intel_gpu.support_2d_block_io, triton_intel_gpu.target_arch = "spir64", ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fp4_vec2xi32_path(%src_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c8_i32 : i32
    %2 = tt.make_tensor_descriptor %src_ptr, [%c1024_i32], [%c1_i64] : <i8>, !tt.tensordesc<8xi8, #blocked>
    %3 = tt.descriptor_load %2[%1] : !tt.tensordesc<8xi8, #blocked> -> tensor<8xi8, #blocked>
    %4 = ttg.fp4_to_fp %3 {axis = 0 : i32} : tensor<8xi8, #blocked> -> tensor<16xbf16, #blocked1>
    %5 = arith.muli %0, %c16_i32 : i32
    %6 = tt.make_tensor_descriptor %dst_ptr, [%c1024_i32], [%c1_i64] : <bf16>, !tt.tensordesc<16xbf16, #blocked1>
    tt.descriptor_store %6[%5], %4 : !tt.tensordesc<16xbf16, #blocked1>, tensor<16xbf16, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: llvm.func {{.*}}@fp4_vec2xi32_path
// COM: Table-lookup fallback used (hardware builtin requires explicit capability).
// CHECK-NOT: __builtin_spirv_ConvertE2M1ToBF16INTEL
// COM: 16 bf16 results written (8 i8 inputs x 2 nibbles each):
// CHECK-COUNT-16: llvm.extractelement {{.*}} : vector<16xbf16>
