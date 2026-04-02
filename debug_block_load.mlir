// Minimal tensor-of-pointers kernel for debugging 2D block load lowering path.
// Extracted from minimal_kernel.py TTIR dump (post-canonicalize/CSE, pre-MaterializeBlockPointer).
//
// Pattern: ptr + offs_m[:, None] * 64 + offs_n[None, :]
//   → tensor<8x16x!tt.ptr<f16>> with stride=[64,1], contiguity=[1,16]
//   → eligible for row_major 2D block IO
//
// Usage with triton-opt (from build dir):
//   triton-opt debug_block_load.mlir --tritonintelgpu-materialize-block-pointer --mlir-print-ir-after-all
//
// Comprehensive pipeline from TTIR to LLVM+GenISA:
//   triton-opt debug_block_load.mlir \
//     --tritonintelgpu-materialize-block-pointer \
//     --convert-triton-intel-gpu-to-llvm \
//     --convert-tritongen-to-llvm \
//     --mlir-print-ir-after-all 2>&1 | less

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "xpu",
  "ttg.threads-per-warp" = 32 : i32,
  ttig.min_sg_size = 16 : i32,
  ttig.support_2d_block_io,
  ttig.support_bfloat16_conversion,
  ttig.support_predicated_io,
  ttig.support_subgroup_matrix_multiply_accumulate,
  ttig.target_arch = "spir64"
} {
  tt.func public @minimal_2d_load_kernel(
      %ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    // stride_row = 64 (constexpr)
    %cst = arith.constant dense<64> : tensor<8x1xi32, #blocked>

    // offs_m = [0,1,2,...,7]
    %offs_m = tt.make_range {end = 8 : i32, start = 0 : i32}
              : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_2d = tt.expand_dims %offs_m {axis = 1 : i32}
              : tensor<8xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
              -> tensor<8x1xi32, #blocked>

    // row_offsets = offs_m * 64
    %row_offsets = arith.muli %offs_m_2d, %cst : tensor<8x1xi32, #blocked>

    // base + row_offsets → tensor<8x1x!tt.ptr<f16>>
    %base_splat = tt.splat %ptr
              : !tt.ptr<f16> -> tensor<8x1x!tt.ptr<f16>, #blocked>
    %row_ptrs = tt.addptr %base_splat, %row_offsets
              : tensor<8x1x!tt.ptr<f16>, #blocked>, tensor<8x1xi32, #blocked>

    // offs_n = [0,1,2,...,15]
    %offs_n = tt.make_range {end = 16 : i32, start = 0 : i32}
              : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_2d = tt.expand_dims %offs_n {axis = 0 : i32}
              : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
              -> tensor<1x16xi32, #blocked>

    // broadcast to 8x16
    %row_ptrs_bc = tt.broadcast %row_ptrs
              : tensor<8x1x!tt.ptr<f16>, #blocked>
              -> tensor<8x16x!tt.ptr<f16>, #blocked>
    %col_offs_bc = tt.broadcast %offs_n_2d
              : tensor<1x16xi32, #blocked>
              -> tensor<8x16xi32, #blocked>

    // final tensor of pointers: ptr + m*64 + n
    %ptrs = tt.addptr %row_ptrs_bc, %col_offs_bc
              : tensor<8x16x!tt.ptr<f16>, #blocked>, tensor<8x16xi32, #blocked>

    // ===== THIS IS THE LOAD WE'RE TRACING =====
    %data = tt.load %ptrs : tensor<8x16x!tt.ptr<f16>, #blocked>

    // Store (same pattern)
    %out_splat = tt.splat %out_ptr
              : !tt.ptr<f16> -> tensor<8x1x!tt.ptr<f16>, #blocked>
    %out_row_ptrs = tt.addptr %out_splat, %row_offsets
              : tensor<8x1x!tt.ptr<f16>, #blocked>, tensor<8x1xi32, #blocked>
    %out_bc = tt.broadcast %out_row_ptrs
              : tensor<8x1x!tt.ptr<f16>, #blocked>
              -> tensor<8x16x!tt.ptr<f16>, #blocked>
    %out_ptrs = tt.addptr %out_bc, %col_offs_bc
              : tensor<8x16x!tt.ptr<f16>, #blocked>, tensor<8x16xi32, #blocked>
    tt.store %out_ptrs, %data : tensor<8x16x!tt.ptr<f16>, #blocked>

    tt.return
  }
}
