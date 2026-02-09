// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm  --check-prefixes=ALL-LAYOUT

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 2], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 8, 2], order = [2, 1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<i8>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #slice}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #slice}>> -> tensor<256x1xi32, #slice>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #slice>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #slice>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #slice}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #slice}>> -> tensor<1x64xi32, #slice>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #slice> -> tensor<256x64xi32, #slice>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #slice> -> tensor<256x64xi32, #slice>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #slice>
    %9 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<256x64x!tt.ptr<i8>, #slice>
    %addr = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<i8>, #slice>, tensor<256x64xi32, #slice>
    %cst = arith.constant dense<0> : tensor<256x64xi8, #slice>
    // ALL-LAYOUT-COUNT-32: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<i8>, #slice>

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 2], order = [1, 0]}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<i8>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #blocked>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #blocked>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #blocked>
    %9 = tt.splat %arg0 : !tt.ptr<i8> -> tensor<256x64x!tt.ptr<i8>, #blocked>
    %addr = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<i8>, #blocked>, tensor<256x64xi32, #blocked>
    %cst = arith.constant dense<0> : tensor<256x64xi8, #blocked>
    // ALL-LAYOUT-COUNT-8: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 4, v_blocks = 1, cache_control = Default}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<i8>, #blocked>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 2}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<i8>) {
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
    // CHECK-COUNT-16: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<i8>, #dot_a>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_a}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_a}>> -> tensor<256x1xi32, #dot_a>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #dot_a>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #dot_a>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_a}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_a}>> -> tensor<1x64xi32, #dot_a>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #dot_a> -> tensor<256x64xi32, #dot_a>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dot_a> -> tensor<256x64xi32, #dot_a>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #dot_a>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>, #dot_a>
    %addr = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<f32>, #dot_a>, tensor<256x64xi32, #dot_a>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #dot_a>
    // CHECK-COUNT-64: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 8, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f32>, #dot_a>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 1}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_b}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dot_b}>> -> tensor<256x1xi32, #dot_b>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #dot_b>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #dot_b>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_b}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot_b}>> -> tensor<1x64xi32, #dot_b>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #dot_b> -> tensor<256x64xi32, #dot_b>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dot_b> -> tensor<256x64xi32, #dot_b>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #dot_b>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>, #dot_b>
    %addr = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<f32>, #dot_b>, tensor<256x64xi32, #dot_b>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #dot_b>
    // CHECK-COUNT-64: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f32>, #dot_b>

    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2]}>
module attributes {ttig.support_2d_block_io, "ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @regular_pointer_block_io
  tt.func public @regular_pointer_block_io(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dpas}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #dpas}>> -> tensor<256x1xi32, #dpas>
    %2 = arith.constant dense<64> : tensor<256x1xi32, #dpas>
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #dpas>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dpas}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dpas}>> -> tensor<1x64xi32, #dpas>
    %6 = tt.broadcast %3 : tensor<256x1xi32, #dpas> -> tensor<256x64xi32, #dpas>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #dpas> -> tensor<256x64xi32, #dpas>
    %8 = arith.addi %6, %7 : tensor<256x64xi32, #dpas>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>, #dpas>
    %addr = tt.addptr %9, %8 : tensor<256x64x!tt.ptr<f32>, #dpas>, tensor<256x64xi32, #dpas>
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #dpas>
    // CHECK-COUNT-16: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default}
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<256x64x!tt.ptr<f32>, #dpas>

    tt.return
  }
}

// -----

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  tt.func public @triton_tem_fused_mm_0(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst_0 = arith.constant dense<92> : tensor<1x64xi32, #mma>
    %cst_1 = arith.constant dense<92> : tensor<32x1xi32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %2 = arith.muli %1, %c8_i32 : i32
    %3 = arith.subi %c3_i32, %2 : i32
    %4 = arith.minsi %3, %c8_i32 : i32
    %5 = arith.remsi %0, %4 : i32
    %6 = arith.addi %2, %5 : i32
    %7 = arith.remsi %0, %c16_i32 : i32
    %8 = arith.divsi %7, %4 : i32
    %9 = arith.cmpi sge, %6, %c0_i32 : i32
    %10 = arith.cmpi sge, %8, %c0_i32 : i32
    %11 = arith.muli %6, %c32_i32 : i32
    %12 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %13 = tt.splat %11 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %14 = arith.addi %13, %12 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %15 = arith.muli %8, %c64_i32 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %17 = tt.splat %15 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %18 = arith.addi %17, %16 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %19 = tt.expand_dims %14 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<32x1xi32, #mma>
    %20 = tt.expand_dims %18 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xi32, #mma>
    %21 = arith.cmpi slt, %19, %cst_1 : tensor<32x1xi32, #mma>
    %22 = arith.cmpi slt, %20, %cst_0 : tensor<1x64xi32, #mma>
    %23 = tt.broadcast %21 : tensor<32x1xi1, #mma> -> tensor<32x64xi1, #mma>
    %24 = tt.broadcast %22 : tensor<1x64xi1, #mma> -> tensor<32x64xi1, #mma>
    %25 = arith.andi %23, %24 : tensor<32x64xi1, #mma>
    %26 = arith.muli %19, %cst_1 : tensor<32x1xi32, #mma>
    %27 = tt.broadcast %20 : tensor<1x64xi32, #mma> -> tensor<32x64xi32, #mma>
    %28 = tt.broadcast %26 : tensor<32x1xi32, #mma> -> tensor<32x64xi32, #mma>
    %29 = arith.addi %27, %28 : tensor<32x64xi32, #mma>
    %30 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x64x!tt.ptr<f32>, #mma>
    %31 = tt.addptr %30, %29 : tensor<32x64x!tt.ptr<f32>, #mma>, tensor<32x64xi32, #mma>
    // COM: mask constancy is 4, which is smaller than the tile shape.
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.store %31, %cst, %25 {ttig.block_io = "row_major"} : tensor<32x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
