// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication | FileCheck %s --check-prefix=RDD
// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication --tritongpu-reorder-instructions | FileCheck %s --check-prefix=PIPE

// COM: Reduced from the _attn_bwd dq loop (flash_attention_benchmark.py, BLOCK_M2=64,
// COM: BLOCK_N2=64, num_warps=16, D_HEAD=128). An earlier pass keeps the conversion
// COM: of the loop-invariant `do` tile in the loop. The shared-memory staging store
// COM: must still happen once, before the loop and the prefetches; only the
// COM: local_load stays in the loop.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttig.descriptor_prefetch
  // RDD:       scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       tt.dot
  // RDD:       scf.yield
  // PIPE-LABEL: @loop_invariant_cvt_src
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  ttig.descriptor_prefetch
  // PIPE:       scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC]] :
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       tt.dot
  // PIPE:       scf.yield
  tt.func public @loop_invariant_cvt_src(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    ttig.descriptor_prefetch %desc[%c0, %c0] : !tt.tensordesc<64x128xf16>
    %res:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst, %off = %c0) -> (tensor<64x64xf32, #mma>, i32) : i32 {
      %next = arith.addi %off, %c64 : i32
      ttig.descriptor_prefetch %desc[%next, %c0] : !tt.tensordesc<64x128xf16>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp, %next : tensor<64x64xf32, #mma>, i32
    }
    tt.return %res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: A source defined inside the loop keeps its allocation inside the loop.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_variant_cvt_src
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       %[[DO:.*]] = tt.load
  // RDD:       %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  // PIPE-LABEL: @loop_variant_cvt_src
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NEXT:  ttg.local_alloc %[[DO]] :
  tt.func public @loop_variant_cvt_src(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: Source defined outside BOTH loops; the conversion sits in the inner
// COM: loop, which has a descriptor_prefetch before it. The source is
// COM: invariant w.r.t. both loops, so the allocation anchors right after it,
// COM: outside both loops.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @nested_loop_loop_invariant_cvt_src
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD:       scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC]] :
  // PIPE-LABEL: @nested_loop_loop_invariant_cvt_src
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE:       scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC]] :
  tt.func public @nested_loop_loop_invariant_cvt_src(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n_outer: i32, %n_inner: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %outer_res = scf.for %i = %c0 to %n_outer step %c1 iter_args(%acc_outer = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %inner_res = scf.for %j = %c0 to %n_inner step %c1 iter_args(%acc = %acc_outer) -> (tensor<64x64xf32, #mma>) : i32 {
        ttig.descriptor_prefetch %desc[%j, %c0] : !tt.tensordesc<64x128xf16>
        %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
        scf.yield %dp : tensor<64x64xf32, #mma>
      }
      scf.yield %inner_res : tensor<64x64xf32, #mma>
    }
    tt.return %outer_res : tensor<64x64xf32, #mma>
  }
}
