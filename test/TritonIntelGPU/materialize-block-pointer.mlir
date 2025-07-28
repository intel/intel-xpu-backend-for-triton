// RUN: triton-opt %s -split-input-file --tritonintelgpu-materialize-block-pointer | FileCheck %s

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  // CHECK-LABEL: tt.func public @materialize_block_pointer(
  tt.func public @materialize_block_pointer(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 15 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %pitch_odd: i64 {tt.divisibility = 15 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c15_i64 = arith.constant 15 : i64
    %c15_i32 = arith.constant 15 : i32

    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32, ttig.block_io = "row_major"}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"}
    %3 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot_a>>
    %4 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot_b>>
    %5 = tt.load %3 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %6 = tt.load %4 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>, ttig.block_io = "row_major"}
    tt.store %3, %5 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>, ttig.block_io = "row_major"}
    tt.store %4, %6 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "column_major"}
    %7 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #dot_a>>
    %8 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x64xf16, #dot_b>>
    %9 = tt.load %7 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %10 = tt.load %8 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %7, %9 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>, ttig.block_io = "column_major"}
    tt.store %8, %10 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non-constant stride on fast changing dim.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %11 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #dot_a>>
    %12 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %pitch], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x64xf16, #dot_b>>
    %13 = tt.load %11 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %14 = tt.load %12 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %11, %13 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %12, %14 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non-64 divisible pitch.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %15 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch_odd], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x32xf16, #dot_a>>
    %16 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c1_i64, %pitch_odd], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<32x64xf16, #dot_b>>
    %17 = tt.load %15 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %18 = tt.load %16 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %15, %17 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %16, %18 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non 4 bytes aligned base.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %19 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot_a>>
    %20 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot_b>>
    %21 = tt.load %19 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %22 = tt.load %20 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %19, %21 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %20, %22 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non 4 bytes aligned baseWidth.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %23 = tt.make_tensor_ptr %arg0, [%c0_i64, %c15_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot_a>>
    %24 = tt.make_tensor_ptr %arg0, [%c0_i64, %c15_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot_b>>
    %25 = tt.load %23 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %26 = tt.load %24 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %23, %25 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %24, %26 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    // COM: Non 4 bytes aligned offsetX.
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 1>, padding = 1 : i32}
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32}
    %27 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c15_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf16, #dot_a>>
    %28 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c15_i32] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot_b>>
    %29 = tt.load %27 {boundaryCheck = array<i32: 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    %30 = tt.load %28 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<32x64xf16, #dot_b>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %27, %29 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<64x32xf16, #dot_a>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>}
    tt.store %28, %30 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<32x64xf16, #dot_b>>

    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [16, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 16], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: f32, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2097152_i64 = arith.constant 2097152 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = tt.get_program_id z : i32
    %1 = tt.get_program_id x : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.extsi %1 : i32 to i64
    %4 = arith.muli %3, %c2097152_i64 : i64
    %5 = arith.extsi %2 : i32 to i64
    %6 = arith.muli %5, %c65536_i64 : i64
    %7 = arith.addi %4, %6 : i64
    %8 = tt.addptr %arg0, %7 : !tt.ptr<f16>, i64
    %9 = arith.muli %0, %c128_i32 : i32
    %10 = tt.make_tensor_ptr %8, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%9, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    // COM: 4 bytes aligned base (value got from addptr, addi, muli), baseWidth and offsetX (value got from muli).
    // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0>, padding = 1 : i32, ttig.block_io = "row_major"}
    %11 = tt.load %10 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>, ttig.block_io = "row_major"}
    tt.store %10, %11 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>
    tt.return
  }
}

// -----

// COM: Ensure load is annotated when its base ptr is the result of a `tt.advance` operation fed by a select operation.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_sg_2d_block} {
  tt.func public @load_fed_by_advance_op_in_loop(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c384_i32 = arith.constant 384 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c8_i32 : i32
    %2 = arith.extsi %arg2 : i32 to i64
    %3 = arith.extsi %arg1 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%3, %2], [%2, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x128xf32, #blocked>>
    %5 = scf.for %arg3 = %c0_i32 to %arg2 step %c128_i32 iter_args(%arg4 = %4) -> (!tt.ptr<tensor<8x128xf32, #blocked>>) : i32 {
      %7 = arith.remsi %arg3, %c384_i32 : i32
      %8 = arith.cmpi eq, %7, %c0_i32 : i32
      %12 = arith.select %8, %4, %arg4 : !tt.ptr<tensor<8x128xf32, #blocked>>
      %14 = tt.advance %12, [%1, %arg3] : <tensor<8x128xf32, #blocked>>
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"}
      %15 = tt.load %14 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x128xf32, #blocked>>
      // CHECK: tt.store {{.*}} {boundaryCheck = array<i32: 1>, ttig.block_io = "row_major"}
      tt.store %14, %15 {boundaryCheck = array<i32: 1>} : !tt.ptr<tensor<8x128xf32, #blocked>>
      scf.yield %12 : !tt.ptr<tensor<8x128xf32, #blocked>>
    }
    tt.return
  }
}

// -----

// COM: Ensure pointer with stride [0, 1] is considered as row major.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  tt.func public @tensor_of_ptr(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %18 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.expand_dims %18 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %20 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x32x!tt.ptr<bf16>, #blocked>
    %21 = tt.addptr %20, %19 : tensor<1x32x!tt.ptr<bf16>, #blocked>, tensor<1x32xi32, #blocked>
    %22 = tt.broadcast %21 : tensor<1x32x!tt.ptr<bf16>, #blocked> -> tensor<256x32x!tt.ptr<bf16>, #blocked>
    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    %50 = tt.load %22 : tensor<256x32x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// COM: Ensure i64 element type is supported in materialize block pointer.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.support_sg_2d_block} {
  // CHECK-LABEL: tt.func public @materialize_block_pointer(
  tt.func public @materialize_block_pointer(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64

    // CHECK: tt.load {{.*}} {ttig.block_io = "row_major"}
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%pitch, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xi64, #dot_a>>
    %1 = tt.load %0 : !tt.ptr<tensor<64x32xi64, #dot_a>>
    // CHECK: tt.store {{.*}} {ttig.block_io = "row_major"}
    tt.store %0, %1 : !tt.ptr<tensor<64x32xi64, #dot_a>>

    tt.return
  }
}
