// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// ttig.descriptor_gather lowers to one load per `vec` registers that are adjacent
// columns of one row, predicated on the chunk's row and first column (vec divides
// shape[1] and is aligned with y). With ttig.support_predicated_io the loads are
// triton_gen.predicated_load; otherwise a branch with a zero default.

// Test 1: per element (y alignment unprovable), no predicated IO, i16 offsets.
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @per_element(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<8xi16, #slice_x>, %arg2: i32) -> tensor<8x16xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c16_i64 = arith.constant 16 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c16_i64, %c1_i64] : <f16>, <1x16xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %arg2] : (!tt.tensordesc<1x16xf16>, tensor<8xi16, #slice_x>, i32) -> tensor<8x16xf16, #blocked>
    tt.return %result : tensor<8x16xf16, #blocked>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @per_element
// CHECK:     %[[ZERO:.*]] = llvm.mlir.zero : i16
// CHECK:     %[[X64:.*]] = llvm.sext %{{.*}} : i16 to i64
// CHECK:     %[[ROW_OK:.*]] = llvm.icmp "ult" %[[X64]], %{{.*}} : i64
// CHECK:     %[[COL_OK:.*]] = llvm.icmp "ult" %{{.*}}, %{{.*}} : i64
// CHECK:     %[[PRED:.*]] = llvm.and %[[ROW_OK]], %[[COL_OK]] : i1
// CHECK:     llvm.cond_br %[[PRED]], ^{{.*}}, ^{{.*}}(%[[ZERO]] : i16)
// CHECK:     llvm.load %{{.*}} : !llvm.ptr<1> -> i16
// CHECK-COUNT-3: llvm.load %{{.*}} : !llvm.ptr<1> -> i16
// CHECK-NOT: triton_gen.predicated_load

// -----

// Test 2: vectorized + predicated IO (coalesced vLLM layout).
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  tt.func public @vectorized_predicated(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<64xi32, #slice_x>) -> tensor<64x128xbf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %result = ttig.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<1x128xbf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x128xbf16, #blocked>
    tt.return %result : tensor<64x128xbf16, #blocked>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @vectorized_predicated
// CHECK-NOT: llvm.cond_br
// CHECK:     %[[ZERO:.*]] = llvm.mlir.zero : vector<4xi32>
// CHECK-COUNT-8: triton_gen.predicated_load %{{.*}}, %{{.*}}, %[[ZERO]] {cache_control = Default} : (!llvm.ptr<1>, i1, vector<4xi32>) -> vector<4xi32>
// CHECK-NOT: triton_gen.predicated_load
// CHECK-NOT: llvm.cond_br
// CHECK:     llvm.return

// -----

// Test 3: vectorized, no predicated IO -> predicated block with a zero vector.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @vectorized_branch(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<64xi32, #slice_x>) -> tensor<64x128xbf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %result = ttig.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<1x128xbf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x128xbf16, #blocked>
    tt.return %result : tensor<64x128xbf16, #blocked>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @vectorized_branch
// CHECK:     %[[ZERO:.*]] = llvm.mlir.zero : vector<4xi32>
// CHECK:     llvm.cond_br %{{.*}}, ^{{.*}}, ^{{.*}}(%[[ZERO]] : vector<4xi32>)
// CHECK:     llvm.load %{{.*}} : !llvm.ptr<1> -> vector<4xi32>
// CHECK-NOT: triton_gen.predicated_load

// -----

// Test 4: base divisibility 4 bytes caps vec at 2 bf16.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  tt.func public @base_alignment_caps_vec(%arg0: !tt.ptr<bf16> {tt.divisibility = 4 : i32}, %arg1: tensor<64xi32, #slice_x>) -> tensor<64x128xbf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c128_i32 = arith.constant 128 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c128_i32], [%c128_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %result = ttig.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<1x128xbf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x128xbf16, #blocked>
    tt.return %result : tensor<64x128xbf16, #blocked>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @base_alignment_caps_vec
// CHECK-COUNT-32: triton_gen.predicated_load {{.*}} -> i32
// CHECK-NOT: triton_gen.predicated_load

// -----

// Test 5: shape[1] = 12 (divisibility 4) caps vec at 4 bf16.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice_x = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_predicated_io} {
  tt.func public @shape_caps_vec(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: tensor<64xi32, #slice_x>) -> tensor<64x128xbf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4096_i32 = arith.constant 4096 : i32
    %c12_i32 = arith.constant 12 : i32
    %c128_i64 = arith.constant 128 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c4096_i32, %c12_i32], [%c128_i64, %c1_i64] : <bf16>, <1x128xbf16>
    %result = ttig.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<1x128xbf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x128xbf16, #blocked>
    tt.return %result : tensor<64x128xbf16, #blocked>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @shape_caps_vec
// CHECK-COUNT-16: triton_gen.predicated_load {{.*}} -> vector<2xi32>
// CHECK-NOT: triton_gen.predicated_load

// -----

// Test 6: broadcast registers. A zero register basis gives 16 logical registers
// with 8 unique values: each canonical register is loaded once, and the result
// struct holds the unique values only.
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8]], warp = [], block = []}>
#slice_x = #ttg.slice<{dim = 1, parent = #linear}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_predicated_io} {
  tt.func public @broadcast_registers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<8xi32, #slice_x>) -> tensor<8x16xf16, #linear> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c16_i64 = arith.constant 16 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c16_i64, %c1_i64] : <f16>, <1x16xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<1x16xf16>, tensor<8xi32, #slice_x>, i32) -> tensor<8x16xf16, #linear>
    tt.return %result : tensor<8x16xf16, #linear>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @broadcast_registers
// CHECK-COUNT-8: triton_gen.predicated_load {{.*}} -> i16
// CHECK-NOT: triton_gen.predicated_load
// CHECK:     llvm.return %{{.*}} : !llvm.struct<(f16, f16, f16, f16, f16, f16, f16, f16)>

// -----

// Test 7: DPAS dot-operand result layout lowers through RV too.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#slice_x = #ttg.slice<{dim = 1, parent = #dot0}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_predicated_io} {
  tt.func public @dot_operand(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: tensor<64xi32, #slice_x>) -> tensor<64x32xf16, #dot0> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i32 = arith.constant 256 : i32
    %c32_i64 = arith.constant 32 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c32_i64, %c1_i64] : <f16>, <1x32xf16>
    %result = ttig.descriptor_gather %desc[%arg1, %c0_i32] : (!tt.tensordesc<1x32xf16>, tensor<64xi32, #slice_x>, i32) -> tensor<64x32xf16, #dot0>
    tt.return %result : tensor<64x32xf16, #dot0>
  }
}

// CHECK-LABEL: llvm.func spir_kernelcc @dot_operand
// CHECK-NOT: triton_gen.sub_group_gather_load
// CHECK-COUNT-32: triton_gen.predicated_load {{.*}} -> i16
// CHECK-NOT: triton_gen.predicated_load
