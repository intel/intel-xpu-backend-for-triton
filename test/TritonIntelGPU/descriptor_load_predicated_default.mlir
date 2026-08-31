// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,PREDICATED
// RUN: env TRITON_INTEL_PREDICATED_LOAD=0 triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,NO-PREDICATED
// RUN: env TRITON_INTEL_PREDICATED_LOAD=1 triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefixes=CHECK,PREDICATED

// Pin the *default* lowering of tt.descriptor_load on the non-2D-block path.
//
// RUN 1 deliberately leaves TRITON_INTEL_PREDICATED_LOAD unset: it is the only
// line here that pins the default, and the only one whose expected output
// changes if the default is flipped back. RUN 2 and RUN 3 pin the explicit
// `=0` and `=1` overrides.
//
// The pass order mirrors production (compiler.py: lower_to_2d_block_load ->
// allocate_shared_memory -> to_llvmir). No chunk here puts a descriptor load
// inside an scf.for, so --convert-scf-to-cf is not needed: createPredicatedBlock
// splits the enclosing block, which an unlowered scf.for region rejects.
//
// TRITON_INTEL_PREDICATED_STORE is deliberately never set: descriptor stores are
// already predicated by default, and pinning the var here would hide that.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // shape=[5,5] is not divisible by block_shape=[4,4], so the fallback emits
  // PER-ELEMENT boundary masks.
  // CHECK-LABEL: llvm.func spir_kernelcc @desc_load_default_per_element(
  tt.func public @desc_load_default_per_element(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // The mask IR is bit-for-bit identical in both arms, so the predicate
    // operand is intentionally left as a wildcard; the load form is what
    // discriminates them.
    // CHECK:              %[[PTR0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK:              %[[OTHER0:.*]] = llvm.bitcast %{{.*}} : vector<1xf32> to i32
    // PREDICATED-NEXT:    %[[V0:.*]] = triton_gen.predicated_load %[[PTR0]], %{{.*}}, %[[OTHER0]] {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32
    // NO-PREDICATED-NEXT: llvm.cond_br %{{.*}}, ^[[BB_LOAD:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]](%[[OTHER0]] : i32)
    // NO-PREDICATED-NEXT: ^[[BB_LOAD]]:
    // NO-PREDICATED-NEXT: %[[LOADED0:.*]] = llvm.load %[[PTR0]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // NO-PREDICATED-NEXT: llvm.br ^[[BB_MERGE]](%[[LOADED0]] : i32)
    // NO-PREDICATED-NEXT: ^[[BB_MERGE]](%[[V0:.*]]: i32):
    // CHECK-NEXT:         llvm.bitcast %[[V0]] : i32 to f32
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // shape=[8,8] and offset=4 are both divisible by block_shape=[4,4], so the
  // fallback emits BLOCK-LEVEL masks whose predicate is uniformly true. This is
  // the regime the old "IGC folds control flow better" rationale was about: it
  // still emits a mask, and after the flip it still becomes a predicated load.
  // CHECK-LABEL: llvm.func spir_kernelcc @desc_load_default_block_level(
  tt.func public @desc_load_default_block_level(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c8_i32, %c8_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // CHECK:              %[[PTR0:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK:              %[[OTHER0:.*]] = llvm.bitcast %{{.*}} : vector<1xf32> to i32
    // PREDICATED-NEXT:    %[[V0:.*]] = triton_gen.predicated_load %[[PTR0]], %{{.*}}, %[[OTHER0]] {cache_control = Default} : (!llvm.ptr<1>, i1, i32) -> i32
    // NO-PREDICATED-NEXT: llvm.cond_br %{{.*}}, ^[[BB_LOAD:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]](%[[OTHER0]] : i32)
    // NO-PREDICATED-NEXT: ^[[BB_LOAD]]:
    // NO-PREDICATED-NEXT: %[[LOADED0:.*]] = llvm.load %[[PTR0]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // NO-PREDICATED-NEXT: llvm.br ^[[BB_MERGE]](%[[LOADED0]] : i32)
    // NO-PREDICATED-NEXT: ^[[BB_MERGE]](%[[V0:.*]]: i32):
    // CHECK-NEXT:         llvm.bitcast %[[V0]] : i32 to f32
    %3 = tt.descriptor_load %0[%c4_i32, %c4_i32] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

// Same shape as @desc_load_default_per_element, but the module does NOT carry
// ttig.support_predicated_io. The capability gate short-circuits first, so every
// arm - including the ones that request predication - must take the branch path.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @desc_load_no_capability(
  tt.func public @desc_load_no_capability(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32) -> (tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // CHECK-NOT:  triton_gen.predicated_load
    // CHECK:      %[[PTR:.*]] = llvm.bitcast %{{.*}} : !llvm.ptr<1> to !llvm.ptr<1>
    // CHECK:      %[[OTHER:.*]] = llvm.bitcast %{{.*}} : vector<1xf32> to i32
    // CHECK-NEXT: llvm.cond_br %{{.*}}, ^[[BB_LOAD:bb[0-9]+]], ^[[BB_MERGE:bb[0-9]+]](%[[OTHER]] : i32)
    // CHECK-NEXT: ^[[BB_LOAD]]:
    // CHECK-NEXT: %[[LOADED:.*]] = llvm.load %[[PTR]] {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK-NEXT: llvm.br ^[[BB_MERGE]](%[[LOADED]] : i32)
    // CHECK-NOT:  triton_gen.predicated_load
    %3 = tt.descriptor_load %0[%arg1, %arg2] : !tt.tensordesc<4x4xf32> -> tensor<4x4xf32, #blocked>
    tt.return %3 : tensor<4x4xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [1, 0]}>

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_predicated_io"} {
  // The DescriptorStoreOp default is unconditional in every arm: it is already
  // on, and TRITON_INTEL_PREDICATED_LOAD must not perturb it.
  // CHECK-LABEL: llvm.func spir_kernelcc @desc_store_default(
  tt.func public @desc_store_default(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: tensor<4x4xf32, #blocked>) {
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c5_i32 = arith.constant 5 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c5_i32, %c5_i32], [%c1_i64, %c4_i64] {order = array<i32: 0>} : <f32>, <4x4xf32>

    // CHECK: triton_gen.predicated_store %{{.*}}, %{{.*}}, %{{.*}} {cache_control = Default} : (!llvm.ptr<1>, i32, i1) -> ()
    tt.descriptor_store %0[%arg1, %arg2], %arg3 : !tt.tensordesc<4x4xf32>, tensor<4x4xf32, #blocked>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// Only the predicated arm forwards the eviction-policy hint to an LSC
// cache-control decoration; the branch arm drops it silently. Defaulting
// descriptor loads to predication therefore starts emitting L1IAR_L3C where
// nothing was emitted before - a second, independent behavioural change that
// must not regress unnoticed.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttig.support_predicated_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @desc_load_default_evict_first(
  tt.func @desc_load_default_evict_first(%desc: !tt.tensordesc<128xf32>) {
    %c0_i32 = arith.constant 0 : i32
    // PREDICATED:        triton_gen.predicated_load %{{.*}}, %{{.*}}, %{{.*}} {cache_control = L1IAR_L3C} : (!llvm.ptr<1>, i1, i32) -> i32
    // NO-PREDICATED-NOT: L1IAR_L3C
    // NO-PREDICATED:     llvm.cond_br
    // NO-PREDICATED-NOT: L1IAR_L3C
    %val = tt.descriptor_load %desc[%c0_i32] evictionPolicy = evict_first : !tt.tensordesc<128xf32> -> tensor<128xf32, #blocked1>
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>

// A descriptor load that IS 2D-block-eligible never reaches the predication
// gate, even with ttig.support_predicated_io present. If 2D lowering ever
// declined this tile, it would fall through to the masked path and the
// CHECK-NOT lines would fire.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.support_predicated_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @desc_load_2d_block_immune(
  tt.func public @desc_load_2d_block_immune(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i32, %arg5: i32) {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x32xf16>
    // CHECK-NOT:     triton_gen.predicated_load
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 2, transpose = false, vnni_transform = false, cache_control = Default}
    // CHECK-NOT:     triton_gen.predicated_load
    %load = tt.descriptor_load %desc[%arg4, %arg5] {ttig.block_io = "row_major"} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #dot0>
    tt.return
  }
}
