// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

// With the module attribute ttig.store_l1_write_back (set where the driver's
// default L1 store policy bypasses L1), a 2D block store whose rows are narrower
// than a 64 B cache line is written back through L1, so the two messages that
// write one line merge in L1 instead of reaching the next level as partial
// writes.

// A 16-bit DPAS tile stored as 8x16 blocks (32 B rows): write-back.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_narrow_rows
  tt.func public @store_f16_narrow_rows(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = L1WB_L3WB}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.return
  }
}

// -----
// Without the module attribute the driver default stays.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_write_back_default
  tt.func public @store_f16_write_back_default(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.return
  }
}

// -----
// 32-bit rows are a whole line already: default.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 8], B = [8, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f32_whole_line_rows
  tt.func public @store_f32_whole_line_rows(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f32>, <64x64xf32, #dpas>
    // CHECK-COUNT-1: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf32, #dpas>, tensor<64x64xf32, #dpas>
    tt.return
  }
}

// -----
// A work-group tile whose rows are narrower than a line (64x16 f16 = 32 B) leaves
// the rest of each line to another work-group, which L1 cannot merge: default.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_narrow_tile
  tt.func public @store_f16_narrow_tile(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x16xf16, #dpas>
    // CHECK-COUNT-1: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf16, #dpas>, tensor<64x16xf16, #dpas>
    tt.return
  }
}

// -----
// A kernel with an atomic may hand the tile to another work-group, and a release
// does not flush L1 on these parts: default.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_then_release
  tt.func public @store_f16_then_release(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %flag: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.atomic_store release, gpu, %flag, %c1_i32 : !tt.ptr<i32>
    tt.return
  }
}

// -----
// The same when the atomic is in a noinline callee.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_then_call
  tt.func public @store_f16_then_call(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %flag: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.call @publish(%flag) : (!tt.ptr<i32>) -> ()
    tt.return
  }
  tt.func private @publish(%flag: !tt.ptr<i32>) attributes {noinline = true} {
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %0 = tt.atomic_rmw exch, release, gpu, %flag, %c1_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// -----
// The same with an atomic poll (not an AtomicOpInterface op),
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_then_poll
  tt.func public @store_f16_then_poll(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %flag: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    %0 = tt.atomic_poll acquire, gpu, %flag, %c0_i32 : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----
// and with inline asm.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_then_inline_asm
  tt.func public @store_f16_then_inline_asm(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %flag: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    // CHECK: llvm.inline_asm
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    %0 = tt.elementwise_inline_asm "mov (M1, 1) $0(0, 0)<1> $1(0, 0)<0;1,0>" {constraints = "=rw,rw", packed_element = 1 : i32, pure = false} %c1_i32 : i32 -> i32
    tt.return
  }
}

// -----
// Pointer stores: an 8-bit tile stored as 16-bit 8x16 blocks (32 B rows) is
// written back, also with work-group tile rows of exactly one line (64 B); a
// store with an explicit cache modifier keeps the default.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttig.store_l1_write_back", "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @pointer_store_i8_narrow_rows
  tt.func public @pointer_store_i8_narrow_rows(%arg0: !tt.ptr<i8>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_a}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_a}>> -> tensor<256x1xi32, #dot_a>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #dot_a>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #dot_a>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_a}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_a}>> -> tensor<1x64xi32, #dot_a>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #dot_a> -> tensor<256x64xi32, #dot_a>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dot_a> -> tensor<256x64xi32, #dot_a>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #dot_a>
    %9 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<256x64x!tt.ptr<i8>, #dot_a>
    %addr = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<i8>, #dot_a>, tensor<256x64xi32, #dot_a>
    %cst = arith.constant dense<0> : tensor<256x64xi8, #dot_a>
    // CHECK-COUNT-16: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = L1WB_L3WB}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<i8>, #dot_a>
    // CHECK-COUNT-16: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.store %addr, %cst {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>, ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<i8>, #dot_a>
    tt.return
  }
}

// -----
// Volatile and `.cv` loads keep the default too. LowerTo2DBlockLoad leaves them
// to the regular lowering on every target: a 2D block load keeps neither the
// volatile flag nor the cache policy, which would also hide them from the check.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.store_l1_write_back"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @store_f16_then_volatile_and_cv_loads
  tt.func public @store_f16_then_volatile_and_cv_loads(%src: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dst: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i64) {
    // CHECK-NOT: triton_gen.2Dblockload
    // CHECK: llvm.load volatile
    // CHECK-NOT: triton_gen.2Dblockload
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-NOT: triton_gen.2Dblockstore
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot_a}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot_a}>> -> tensor<64x1xi32, #dot_a>
    %2 = arith.constant dense<32> : tensor<64x1xi32, #dot_a>
    %3 = arith.muli %1, %2 : tensor<64x1xi32, #dot_a>
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot_a}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot_a}>> -> tensor<1x32xi32, #dot_a>
    %6 = tt.broadcast %3 : tensor<64x1xi32, #dot_a> -> tensor<64x32xi32, #dot_a>
    %7 = tt.broadcast %5 : tensor<1x32xi32, #dot_a> -> tensor<64x32xi32, #dot_a>
    %8 = arith.addi %6, %7 : tensor<64x32xi32, #dot_a>
    %9 = tt.splat %src : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot_a>
    %10 = tt.addptr %9, %8 : tensor<64x32x!tt.ptr<f16>, #dot_a>, tensor<64x32xi32, #dot_a>
    %11 = tt.load %10 {isVolatile = true, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot_a>
    %12 = tt.load %10 {cachePolicy = #tt.cache_policy<cache_modifier = cv, eviction_policy = evict_normal>, ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot_a>
    %13 = arith.addf %11, %12 : tensor<64x32xf16, #dot_a>
    %14 = tt.splat %dst : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot_a>
    %15 = tt.addptr %14, %8 : tensor<64x32x!tt.ptr<f16>, #dot_a>, tensor<64x32xi32, #dot_a>
    tt.store %15, %13 : tensor<64x32x!tt.ptr<f16>, #dot_a>
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %dst, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <64x64xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %cst {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xf16, #dpas>, tensor<64x64xf16, #dpas>
    tt.return
  }
}
