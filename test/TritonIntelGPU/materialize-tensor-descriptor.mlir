// RUN: triton-opt %s -split-input-file --tritonintelgpu-materialize-block-pointer | FileCheck %s

// COM: Tensor descriptor tests - descriptors are always row-major (last stride = 1)
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @materialize_tensor_descriptor(
  tt.func public @materialize_tensor_descriptor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 15 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %pitch_odd: i64 {tt.divisibility = 15 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c15_i32 = arith.constant 15 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64

    // COM: Row-major tensor descriptors with proper alignment
    // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32}
    // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
    %0 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x32xf16, #dot_a>>
    %1 = tt.make_tensor_descriptor %arg0, [%c32_i32, %c64_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<32x64xf16, #dot_b>>
    %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf16, #dot_a>> -> tensor<64x32xf16, #dot_a>
    %3 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x64xf16, #dot_b>> -> tensor<32x64xf16, #dot_b>
    // CHECK: tt.descriptor_store {{.*}} {ttig.block_io = "row_major"{{.*}}}
    tt.descriptor_store %0[%c0_i32, %c0_i32], %2 : !tt.tensordesc<tensor<64x32xf16, #dot_a>>, tensor<64x32xf16, #dot_a>
    // CHECK: tt.descriptor_store {{.*}} {ttig.block_io = "row_major"{{.*}}}
    tt.descriptor_store %1[%c0_i32, %c0_i32], %3 : !tt.tensordesc<tensor<32x64xf16, #dot_b>>, tensor<32x64xf16, #dot_b>

    // COM: Non-64 divisible pitch - should not get block_io attribute
    // CHECK-NOT: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
    %4 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch_odd, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x32xf16, #dot_a>>
    %5 = tt.descriptor_load %4[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf16, #dot_a>> -> tensor<64x32xf16, #dot_a>

    // COM: Base pointer not aligned to 4 bytes - should not get block_io attribute
    // CHECK-NOT: tt.descriptor_store {{.*}} {ttig.block_io = "row_major"{{.*}}}
    %6 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<64x32xf16, #dot_a>>
    %7 = tt.descriptor_load %6[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xf16, #dot_a>> -> tensor<64x32xf16, #dot_a>
    tt.descriptor_store %6[%c0_i32, %c0_i32], %7 : !tt.tensordesc<tensor<64x32xf16, #dot_a>>, tensor<64x32xf16, #dot_a>

    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @_attn_fwd(
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2097152_i64 = arith.constant 2097152 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %0 = tt.get_program_id z : i32
    %1 = tt.get_program_id x : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.extui %1 : i32 to i64
    %4 = arith.muli %3, %c2097152_i64 : i64
    %5 = arith.extui %2 : i32 to i64
    %6 = arith.muli %5, %c65536_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
    %9 = arith.muli %0, %c128_i32 : i32
    %10 = tt.make_tensor_descriptor %8, [%c1024_i32, %c64_i32], [%c64_i64, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    // COM: 4 bytes aligned base (value got from addptr, addi, muli), baseWidth.
    // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
    %11 = tt.descriptor_load %10[%9, %c0_i32] : !tt.tensordesc<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    // CHECK: tt.descriptor_store {{.*}} {ttig.block_io = "row_major"{{.*}}}
    tt.descriptor_store %10[%9, %c0_i32], %11 : !tt.tensordesc<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: Ensure i64 element type is supported in materialize block pointer.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @materialize_tensor_descriptor_i64(
  tt.func public @materialize_tensor_descriptor_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64

    // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
    %0 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c32_i32], [%pitch, %c1_i64] : !tt.ptr<i64>, !tt.tensordesc<tensor<64x32xi64, #dot_a>>
    %1 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<64x32xi64, #dot_a>> -> tensor<64x32xi64, #dot_a>
    // CHECK: tt.descriptor_store {{.*}} {ttig.block_io = "row_major"{{.*}}}
    tt.descriptor_store %0[%c0_i32, %c0_i32], %1 : !tt.tensordesc<tensor<64x32xi64, #dot_a>>, tensor<64x32xi64, #dot_a>
    tt.return
  }
}

// -----

// COM: 3D tensor descriptor
#dpas_3d = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 2], repCluster = [1, 1, 1]}>
#dot_a_3d = #ttg.dot_op<{opIdx = 0, parent = #dpas_3d, kWidth = 1}>
#dot_b_3d = #ttg.dot_op<{opIdx = 1, parent = #dpas_3d, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "xpu", "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @materialize_tensor_descriptor(
  tt.func public @materialize_tensor_descriptor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // CHECK: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"{{.*}}}
    %31 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32, %c0_i32], [%pitch, %pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<tensor<4x64x32xf16>>
    %34 = tt.descriptor_load %31[%c0_i32, %c0_i32, %c0_i32] : !tt.tensordesc<tensor<4x64x32xf16>> -> tensor<4x64x32xf16, #dot_a_3d>
    // CHECK: tt.descriptor_store {{.*}} {ttig.block_io = "row_major"{{.*}}}
    tt.descriptor_store %31[%c0_i32, %c0_i32, %c0_i32], %34 : !tt.tensordesc<tensor<4x64x32xf16>> , tensor<4x64x32xf16, #dot_a_3d>
    tt.return
  }
}
