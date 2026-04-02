[tritonintelgpu-materialize-block-pointer]: Considering op: %10 = tt.load %9 : tensor<8x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonintelgpu-materialize-block-pointer]: Considering tensor of pointer of memory accessing op: %10 = tt.load %9 : tensor<8x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonintelgpu-materialize-block-pointer]: Found non-contiguous row: 1
[tritonintelgpu-materialize-block-pointer]: Considering op: tt.store %14, %10 : tensor<8x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonintelgpu-materialize-block-pointer]: Considering tensor of pointer of memory accessing op: tt.store %14, %10 : tensor<8x16x!tt.ptr<f16>, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
[tritonintelgpu-materialize-block-pointer]: Found non-contiguous row: 1
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_bfloat16_conversion, ttig.support_predicated_io, ttig.support_subgroup_matrix_multiply_accumulate, ttig.target_arch = "spir64"} {
  tt.func public @minimal_2d_load_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<8x1xi32, #blocked>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8x1xi32, #blocked>
    %2 = arith.muli %1, %cst : tensor<8x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<8x1x!tt.ptr<f16>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<8x1x!tt.ptr<f16>, #blocked>, tensor<8x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<8x1x!tt.ptr<f16>, #blocked> -> tensor<8x16x!tt.ptr<f16>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<8x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<8x16x!tt.ptr<f16>, #blocked>, tensor<8x16xi32, #blocked>
    %10 = tt.load %9 {ttig.block_io = "row_major"} : tensor<8x16x!tt.ptr<f16>, #blocked>
    %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<8x1x!tt.ptr<f16>, #blocked>
    %12 = tt.addptr %11, %2 : tensor<8x1x!tt.ptr<f16>, #blocked>, tensor<8x1xi32, #blocked>
    %13 = tt.broadcast %12 : tensor<8x1x!tt.ptr<f16>, #blocked> -> tensor<8x16x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %8 : tensor<8x16x!tt.ptr<f16>, #blocked>, tensor<8x16xi32, #blocked>
    tt.store %14, %10 {ttig.block_io = "row_major"} : tensor<8x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
