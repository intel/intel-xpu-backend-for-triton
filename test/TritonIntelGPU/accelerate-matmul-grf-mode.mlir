// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul="grf-mode=128" | FileCheck %s --check-prefix=GRF128
// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul="split-chained-dots-along-n=true" | FileCheck %s --check-prefix=SPLIT

// COM: A 10 KB consumer copy fits the 256-entry GRF the default mode can escalate to, but not an explicit 128-entry GRF.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// CHECK-NOT: #ttig.dpas
// GRF128-DAG: #[[$QK:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// GRF128-DAG: #[[$PV:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 4]{{.*}}>
// GRF128-NOT: #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_decode_grf_mode
  // GRF128-LABEL: chained_dot_decode_grf_mode
  tt.func public @chained_dot_decode_grf_mode(
    %arg0: tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<4x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<4x32xf32, #[[$DPAS]]>
    // GRF128: tt.dot {{.*}} -> tensor<4x32xf32, #[[$QK]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x32xf32, #blocked>
    %t = arith.truncf %d : tensor<4x32xf32, #blocked> to tensor<4x32xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<4x32xbf16, #blocked> -> tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<4x128xf32, #[[$DPAS]]>
    // GRF128: tt.dot {{.*}} -> tensor<4x128xf32, #[[$PV]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<4x128xf32, #blocked1>
    tt.return %r : tensor<4x128xf32, #blocked1>
  }
}

// -----

// COM: More than 32 warps cannot escalate to a larger GRF mode, so the same 10 KB consumer copy goes along N.
// CHECK-DAG: #[[$QK:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [64, 1]{{.*}}>
// CHECK-DAG: #[[$PV:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 64]{{.*}}>
// CHECK-NOT: #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 64], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 64], order = [1, 0]}>
module attributes {"ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_decode_many_warps
  tt.func public @chained_dot_decode_many_warps(
    %arg0: tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<4x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x32xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<4x32xf32, #[[$QK]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x32xf32, #blocked>
    %t = arith.truncf %d : tensor<4x32xf32, #blocked> to tensor<4x32xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<4x32xbf16, #blocked> -> tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<4x128xf32, #[[$PV]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<4x128xf32, #blocked1>
    tt.return %r : tensor<4x128xf32, #blocked1>
  }
}

// -----

// COM: A target that escalates to 512 GRF entries holds a 24 KB consumer copy, so both dots stay on M.
// CHECK-DAG: #[[$QK:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// CHECK-DAG: #[[$PV:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 1]{{.*}}>
// CHECK-NOT: #ttig.dpas
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.max_grf_mode" = "512", "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_decode_max_grf_512
  tt.func public @chained_dot_decode_max_grf_512(
    %arg0: tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<16x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<4x512xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x16xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<4x512xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<4x16xf32, #[[$QK]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x16xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<4x16xf32, #blocked>
    %t = arith.truncf %d : tensor<4x16xf32, #blocked> to tensor<4x16xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<4x16xbf16, #blocked> -> tensor<4x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<4x512xf32, #[[$PV]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<4x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<16x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<4x512xf32, #blocked1>
    tt.return %r : tensor<4x512xf32, #blocked1>
  }
}

// -----

// COM: With every warp on M, each warp would hold the full 16 KB B of the QK dot, 20 KB in total, more than even the
// COM: 256-entry GRF. By default the warps stay on M and the module is marked as a candidate for the backend; on
// COM: request one factor of two of the warps moves to N, for both dots of the chain.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// CHECK-NOT: #ttig.dpas
// CHECK: module attributes {{.*}}ttig.chained_dot_split_candidate = 1 : i32
// GRF128: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// GRF128: module attributes {{.*}}ttig.chained_dot_split_candidate = 1 : i32
// SPLIT: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 2]{{.*}}>
// SPLIT-NOT: #ttig.dpas
// SPLIT-NOT: chained_dot_split_candidate
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_head_dim_128
  // GRF128-LABEL: chained_dot_head_dim_128
  // SPLIT-LABEL: chained_dot_head_dim_128
  tt.func public @chained_dot_head_dim_128(
    %arg0: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    // SPLIT: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x128xf32, #[[$DPAS]]>
    // SPLIT: tt.dot {{.*}} -> tensor<64x128xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// COM: At head dim 64 each warp holds 11 KB with every warp on M. That fits the 256-entry GRF the default mode can
// COM: escalate to, so the module is neither marked nor split, but not an explicit 128-entry GRF.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// CHECK-NOT: chained_dot_split_candidate
// GRF128: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// GRF128: module attributes {{.*}}ttig.chained_dot_split_candidate = 1 : i32
// SPLIT: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_head_dim_64
  // GRF128-LABEL: chained_dot_head_dim_64
  // SPLIT-LABEL: chained_dot_head_dim_64
  tt.func public @chained_dot_head_dim_64(
    %arg0: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>) -> tensor<64x64xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.return %r : tensor<64x64xf32, #blocked>
  }
}

// -----

// COM: Only the consumer of the chain is too large: its 32 KB B makes 41 KB per warp with every warp on M, while the
// COM: producer holds 6.5 KB. The decision covers the whole chain, so on request the producer moves to N together
// COM: with the consumer and the chain keeps one layout.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// CHECK-NOT: #ttig.dpas
// CHECK: module attributes {{.*}}ttig.chained_dot_split_candidate = 1 : i32
// GRF128: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// GRF128: module attributes {{.*}}ttig.chained_dot_split_candidate = 1 : i32
// SPLIT: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [4, 2]{{.*}}>
// SPLIT-NOT: #ttig.dpas
// SPLIT-NOT: chained_dot_split_candidate
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_only_consumer_exceeds
  // GRF128-LABEL: chained_dot_only_consumer_exceeds
  // SPLIT-LABEL: chained_dot_only_consumer_exceeds
  tt.func public @chained_dot_only_consumer_exceeds(
    %arg0: tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x256xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked1>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    // SPLIT: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // CHECK: tt.dot {{.*}} -> tensor<64x256xf32, #[[$DPAS]]>
    // SPLIT: tt.dot {{.*}} -> tensor<64x256xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x256xf32, #blocked1>
    tt.return %r : tensor<64x256xf32, #blocked1>
  }
}

// -----

// COM: With one warp there is no warp to move. The chain is far over the GRF, but it is neither marked nor split.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 1]{{.*}}>
// CHECK-NOT: chained_dot_split_candidate
// GRF128: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 1]{{.*}}>
// GRF128-NOT: chained_dot_split_candidate
// SPLIT: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}warpsPerCTA = [1, 1]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_one_warp
  // GRF128-LABEL: chained_dot_one_warp
  // SPLIT-LABEL: chained_dot_one_warp
  tt.func public @chained_dot_one_warp(
    %arg0: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // SPLIT: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // SPLIT: tt.dot {{.*}} -> tensor<64x128xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}

// -----

// COM: The head dim 128 chain on 8-wide DPAS (min_sg_size 8). It is over the 8 KB a 256-entry GRF holds at 8 lanes,
// COM: but the split was only measured with 16-wide DPAS, so it is neither marked nor split.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}executionSize = 8{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// CHECK-NOT: chained_dot_split_candidate
// GRF128: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}executionSize = 8{{.*}}warpsPerCTA = [8, 1]{{.*}}>
// GRF128-NOT: chained_dot_split_candidate
// SPLIT: #[[$DPAS:.+]] = #ttig.dpas<{{.*}}executionSize = 8{{.*}}warpsPerCTA = [8, 1]{{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 8 : i32, "ttig.min_sg_size" = 8 : i32, "ttig.support_subgroup_matrix_multiply_accumulate"} {
  // CHECK-LABEL: chained_dot_execution_size_8
  // GRF128-LABEL: chained_dot_execution_size_8
  // SPLIT-LABEL: chained_dot_execution_size_8
  tt.func public @chained_dot_execution_size_8(
    %arg0: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
    %arg1: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>) -> tensor<64x128xf32, #blocked1> {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked1>
    // SPLIT: tt.dot {{.*}} -> tensor<64x64xf32, #[[$DPAS]]>
    %d = tt.dot %arg0, %arg1, %cst_0 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    %t = arith.truncf %d : tensor<64x64xf32, #blocked> to tensor<64x64xbf16, #blocked>
    %c = ttg.convert_layout %t : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    // SPLIT: tt.dot {{.*}} -> tensor<64x128xf32, #[[$DPAS]]>
    %r = tt.dot %c, %arg2, %cst_1 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} :
      tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x128xf32, #blocked1>
    tt.return %r : tensor<64x128xf32, #blocked1>
  }
}
