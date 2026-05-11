// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=tt.load

// COM: Row-major pointer-based load with broadcast (stride=0). The pass
// COM: converts this to ttig.2d_block_load_from_ptr, retaining the full
// COM: pointer tensor.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @tensor_of_pointers_load
  tt.func @tensor_of_pointers_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: ttig.2d_block_load_from_ptr %4 {row_major} {base_height = 1 : i32, base_pitch = 64 : i32, base_width = 64 : i32}
    %5 = tt.load %4 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Column-major pointer load. The memory_layout is set to column_major,
// COM: which tells the LLVM lowering to use transposed block reads.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @pointer_load_column_major
  tt.func @pointer_load_column_major(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<32x64xf16, #dot1> {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #dot1}>> -> tensor<1x64xi32, #dot1>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #dot1>
    %3 = tt.addptr %2, %1 : tensor<1x64x!tt.ptr<f16>, #dot1>, tensor<1x64xi32, #dot1>
    %4 = tt.broadcast %3 : tensor<1x64x!tt.ptr<f16>, #dot1> -> tensor<32x64x!tt.ptr<f16>, #dot1>
    // CHECK: ttig.2d_block_load_from_ptr %4 {column_major} {base_height = 1 : i32, base_pitch = 128 : i32, base_width = 128 : i32}
    %5 = tt.load %4 {ttig.block_io = "column_major"} : tensor<32x64x!tt.ptr<f16>, #dot1>
    tt.return %5 : tensor<32x64xf16, #dot1>
  }
}

// -----

// COM: Masked pointer load with both mask and other. Both are forwarded
// COM: to the ttig.2d_block_load_from_ptr op. The mask is a constant true
// COM: (uniform constancy) which satisfies the 2D block I/O tile constraints.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_pointer_load_with_other
  tt.func @masked_pointer_load_with_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i1) -> tensor<64x32xf16, #dot0> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %mask = tt.splat %arg1 : i1 -> tensor<64x32xi1, #dot0>
    // CHECK: ttig.2d_block_load_from_ptr
    %7 = tt.load %4, %mask, %cst {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %7 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Masked pointer load WITHOUT an explicit 'other' value. The pass must
// COM: synthesize a zero splat so the verifier constraint (other required when
// COM: mask is present) is satisfied. The mask is a uniform splat which has
// COM: sufficient constancy for 2D block I/O.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @masked_pointer_load_no_other
  tt.func @masked_pointer_load_no_other(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i1) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %mask = tt.splat %arg1 : i1 -> tensor<64x32xi1, #dot0>
    // CHECK: %[[ZERO_CST:.*]] = arith.constant 0.000000e+00 : f16
    // CHECK: %[[ZERO_SPLAT:.*]] = tt.splat %[[ZERO_CST]]
    // CHECK: ttig.2d_block_load_from_ptr
    %7 = tt.load %4, %mask {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %7 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Pointer load with 1D->2D reshape stride attribute. The pass reads the
// COM: stride from ttig.block_io_stride to compute pitch, skips tile validation,
// COM: and propagates the attribute to the output op.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @pointer_load_1d_reshape_stride
  tt.func @pointer_load_1d_reshape_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: ttig.2d_block_load_from_ptr %4 {row_major} {base_height = 64 : i32, base_pitch = 512 : i32, base_width = 64 : i32, ttig.block_io_stride = 256 : i64}
    %5 = tt.load %4 {ttig.block_io = "row_major", ttig.block_io_stride = 256 : i64} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Pointer load with ttig.one_matrix_per_load attribute. The pass must
// COM: propagate this attribute to the resulting ttig.2d_block_load_from_ptr.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @pointer_load_one_matrix_per_load
  tt.func @pointer_load_one_matrix_per_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: ttig.2d_block_load_from_ptr
    // CHECK-SAME: ttig.one_matrix_per_load
    %5 = tt.load %4 {ttig.block_io = "row_major", ttig.one_matrix_per_load} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}

// -----

// COM: Env var TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=1 forces the attribute on
// COM: all loads, even those without it originally.
// RUN: env TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT=1 triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --check-prefix=ENV-CHECK
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // ENV-CHECK-LABEL: tt.func @pointer_load_env_override
  tt.func @pointer_load_env_override(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf16, #dot0> {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x32x!tt.ptr<f16>, #dot0>
    %3 = tt.addptr %2, %1 : tensor<1x32x!tt.ptr<f16>, #dot0>, tensor<1x32xi32, #dot0>
    %4 = tt.broadcast %3 : tensor<1x32x!tt.ptr<f16>, #dot0> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // ENV-CHECK: ttig.2d_block_load_from_ptr
    // ENV-CHECK-SAME: ttig.one_matrix_per_load
    %5 = tt.load %4 {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
    tt.return %5 : tensor<64x32xf16, #dot0>
  }
}
