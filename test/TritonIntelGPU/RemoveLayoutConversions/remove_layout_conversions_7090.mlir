// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/7090
// COM: Kernel triton_poi_fused__to_copy_permute_7 (pointwise, 3-D tensor
// COM: descriptor): descriptor_load -> tt.trans -> store, with no dot.
// COM:
// COM: The descriptor_load carries ttig.block_io = "row_major" but its coalesced
// COM: layout does not validate as a 2D block load. Before the fix it was treated
// COM: as cheap, so RemoveLayoutConversions rematerialized it and back-propagated
// COM: the store layout across tt.trans, de-coalescing the load (vec4 -> vec1
// COM: gather). isExpensiveLoadOrStore now anchors such a load, so it keeps its
// COM: coalesced sizePerThread = [1, 1, 4] layout (vec4 on the contiguous inner
// COM: dim) and the store-side convert_layout is preserved instead of being
// COM: folded into the load.

// CHECK-DAG: #[[$LOAD:.+]] = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 8, 2], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>

#blocked = #ttg.blocked<{sizePerThread = [8, 1, 1], threadsPerWarp = [8, 1, 2], warpsPerCTA = [1, 2, 4], order = [0, 2, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 8], warpsPerCTA = [2, 4, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 8, 2], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4, 1], threadsPerWarp = [8, 2, 1], warpsPerCTA = [8, 1, 1], order = [1, 0, 2]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.2d_block_io_base_alignment = 64 : i32, ttig.min_sg_size = 16 : i32, ttig.support_256b_load_store, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "pvc"} {
  // CHECK-LABEL: @triton_poi_fused__to_copy_permute_7
  // COM: The block_io load stays anchored in its coalesced (vec4) layout.
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %{{.*}}[{{.*}}] {ttig.block_io = "row_major"{{.*}}} : !tt.tensordesc<8x64x8xf32> -> tensor<8x64x8xf32, #[[$LOAD]]>
  // CHECK: %[[TRANS:.*]] = tt.trans %[[LOAD]] {order = array<i32: 1, 2, 0>}
  // COM: The store-side convert_layout is preserved (not folded into the load).
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x8x8xf16, #{{.+}}> -> tensor<64x8x8xf16, #[[$STORE:.+]]>
  // CHECK: tt.store %{{.*}}, %[[CVT]], %{{.*}} : tensor<64x8x8x!tt.ptr<f16>, #[[$STORE]]>
  tt.func public @triton_poi_fused__to_copy_permute_7(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c28_i32 = arith.constant 28 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1792_i64 = arith.constant 1792 : i64
    %c28_i64 = arith.constant 28 : i64
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<1792> : tensor<1x8x1xi32, #blocked>
    %cst_0 = arith.constant dense<64> : tensor<1x1x8xi32, #blocked>
    %cst_1 = arith.constant dense<28> : tensor<1x1x8xi32, #blocked>
    %cst_2 = arith.constant dense<28> : tensor<1x8x1xi32, #blocked>
    %cst_3 = arith.constant dense<64> : tensor<64x1x1xi32, #blocked1>
    %0 = tt.get_program_id z : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<64x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<64x1xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %6 = tt.expand_dims %4 {axis = 2 : i32} : tensor<64x1xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<64x1x1xi32, #blocked>
    %7 = tt.expand_dims %5 {axis = 2 : i32} : tensor<64x1xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<64x1x1xi32, #blocked1>
    %8 = tt.splat %1 : i32 -> tensor<64x1x1xi32, #blocked>
    %9 = tt.splat %1 : i32 -> tensor<64x1x1xi32, #blocked1>
    %10 = arith.addi %8, %6 : tensor<64x1x1xi32, #blocked>
    %11 = arith.addi %9, %7 : tensor<64x1x1xi32, #blocked1>
    %12 = arith.cmpi slt, %11, %cst_3 : tensor<64x1x1xi32, #blocked1>
    %13 = tt.get_program_id y : i32
    %14 = arith.muli %13, %c8_i32 : i32
    %15 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>>
    %16 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>>
    %17 = tt.expand_dims %15 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>>
    %18 = tt.expand_dims %16 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = tt.expand_dims %17 {axis = 2 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 2, parent = #blocked}>> -> tensor<1x8x1xi32, #blocked>
    %20 = tt.splat %14 : i32 -> tensor<1x8x1xi32, #blocked>
    %21 = arith.addi %20, %19 : tensor<1x8x1xi32, #blocked>
    %22 = arith.cmpi slt, %21, %cst_2 : tensor<1x8x1xi32, #blocked>
    %23 = tt.get_program_id x : i32
    %24 = arith.muli %23, %c8_i32 : i32
    %25 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1x1x8xi32, #blocked>
    %26 = tt.splat %24 : i32 -> tensor<1x1x8xi32, #blocked>
    %27 = arith.addi %26, %25 : tensor<1x1x8xi32, #blocked>
    %28 = arith.cmpi slt, %27, %cst_1 : tensor<1x1x8xi32, #blocked>
    %29 = tt.make_tensor_descriptor %arg0, [%c28_i32, %c64_i32, %c28_i32], [%c1792_i64, %c28_i64, %c1_i64] : <f32>, <8x64x8xf32>
    %30 = tt.descriptor_load %29[%24, %1, %14] {ttig.block_io = "row_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<8x64x8xf32> -> tensor<8x64x8xf32, #blocked2>
    %31 = tt.trans %30 {order = array<i32: 1, 2, 0>} : tensor<8x64x8xf32, #blocked2> -> tensor<64x8x8xf32, #blocked3>
    %32 = arith.muli %27, %cst_0 : tensor<1x1x8xi32, #blocked>
    %33 = tt.broadcast %10 : tensor<64x1x1xi32, #blocked> -> tensor<64x1x8xi32, #blocked>
    %34 = tt.broadcast %32 : tensor<1x1x8xi32, #blocked> -> tensor<64x1x8xi32, #blocked>
    %35 = arith.addi %33, %34 : tensor<64x1x8xi32, #blocked>
    %36 = arith.muli %21, %cst : tensor<1x8x1xi32, #blocked>
    %37 = tt.broadcast %35 : tensor<64x1x8xi32, #blocked> -> tensor<64x8x8xi32, #blocked>
    %38 = tt.broadcast %36 : tensor<1x8x1xi32, #blocked> -> tensor<64x8x8xi32, #blocked>
    %39 = arith.addi %37, %38 : tensor<64x8x8xi32, #blocked>
    %40 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x8x8x!tt.ptr<f16>, #blocked>
    %41 = tt.addptr %40, %39 : tensor<64x8x8x!tt.ptr<f16>, #blocked>, tensor<64x8x8xi32, #blocked>
    %42 = tt.broadcast %28 : tensor<1x1x8xi1, #blocked> -> tensor<1x8x8xi1, #blocked>
    %43 = tt.broadcast %22 : tensor<1x8x1xi1, #blocked> -> tensor<1x8x8xi1, #blocked>
    %44 = arith.andi %42, %43 : tensor<1x8x8xi1, #blocked>
    %45 = tt.broadcast %44 : tensor<1x8x8xi1, #blocked> -> tensor<64x8x8xi1, #blocked>
    %46 = tt.broadcast %12 : tensor<64x1x1xi1, #blocked1> -> tensor<64x8x8xi1, #blocked1>
    %47 = ttg.convert_layout %46 : tensor<64x8x8xi1, #blocked1> -> tensor<64x8x8xi1, #blocked>
    %48 = arith.andi %45, %47 : tensor<64x8x8xi1, #blocked>
    %49 = arith.truncf %31 : tensor<64x8x8xf32, #blocked3> to tensor<64x8x8xf16, #blocked3>
    %50 = ttg.convert_layout %49 : tensor<64x8x8xf16, #blocked3> -> tensor<64x8x8xf16, #blocked>
    tt.store %41, %50, %48 : tensor<64x8x8x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
