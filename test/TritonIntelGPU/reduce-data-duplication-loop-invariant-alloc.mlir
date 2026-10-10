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

// -----

// COM: Control: a loop-invariant source, but the loop writes shared memory
// COM: (a local_store into a staging buffer) before the conversion. Moving the
// COM: allocation's store above that write could break the ordering that
// COM: TritonGPUReorderInstructions preserves, so the allocation must stay
// COM: at the conversion, after the local_store.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_after_local_store
  // RDD:       %[[BUF:.*]] = ttg.local_alloc : ()
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       ttg.local_store %{{.*}}, %[[BUF]] :
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  // PIPE-LABEL: @loop_invariant_cvt_src_after_local_store
  // PIPE:       %[[BUF:.*]] = ttg.local_alloc : ()
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for
  // PIPE:       ttg.local_store %{{.*}}, %[[BUF]] :
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_invariant_cvt_src_after_local_store(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %x_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %x = tt.load %x_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      ttg.local_store %x, %buf : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: Control: a loop-invariant source, but the loop has a CTA barrier before
// COM: the conversion. `ttg.barrier` has no memory-effect interface, so its
// COM: effects are unknown and the allocation must stay after it. RDD only:
// COM: TritonGPUReorderInstructions treats unknown effects as non-writes and
// COM: hoists the allocation itself, which is upstream behavior.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_after_barrier
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       ttg.barrier local
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_invariant_cvt_src_after_barrier(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      ttg.barrier local
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: The loop-invariant source is a function argument, which has no defining
// COM: op. The allocation is placed right before the loop, after the ops that
// COM: precede it, so the store happens once.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_func_arg
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %arg0 :
  // RDD-NEXT:  scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       tt.dot
  // RDD:       scf.yield
  // PIPE-LABEL: @loop_invariant_cvt_src_func_arg
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttig.descriptor_prefetch
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %arg0 :
  // PIPE-NEXT:  scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC]] :
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       tt.dot
  // PIPE:       scf.yield
  tt.func public @loop_invariant_cvt_src_func_arg(%do: tensor<64x128xf16, #blocked>, %desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
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

// COM: The source is an outer loop's iter_arg and the conversion sits in the
// COM: inner loop. The source is invariant w.r.t. the inner loop only, so the
// COM: allocation leaves the inner loop but stays in the outer loop body,
// COM: right before the inner loop. Ops earlier in the outer body (here a
// COM: barrier) are not crossed, so they do not block it.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @nested_loop_outer_iter_arg_cvt_src
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %[[DO:arg[0-9]+]] = %{{.*}})
  // RDD-NEXT:  ttg.barrier local
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC]] :
  // PIPE-LABEL: @nested_loop_outer_iter_arg_cvt_src
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %[[DO:arg[0-9]+]] = %{{.*}})
  // PIPE-NEXT:  ttg.barrier local
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC]] :
  tt.func public @nested_loop_outer_iter_arg_cvt_src(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n_outer: i32, %n_inner: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do0 = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %outer_res:2 = scf.for %i = %c0 to %n_outer step %c1 iter_args(%acc_outer = %cst, %do = %do0) -> (tensor<64x64xf32, #mma>, tensor<64x128xf16, #blocked>) : i32 {
      ttg.barrier local
      %inner_res = scf.for %j = %c0 to %n_inner step %c1 iter_args(%acc = %acc_outer) -> (tensor<64x64xf32, #mma>) : i32 {
        ttig.descriptor_prefetch %desc[%j, %c0] : !tt.tensordesc<64x128xf16>
        %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
        scf.yield %dp : tensor<64x64xf32, #mma>
      }
      %do_next = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      scf.yield %inner_res, %do_next : tensor<64x64xf32, #mma>, tensor<64x128xf16, #blocked>
    }
    tt.return %outer_res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: Control: the source is an iter_arg of the conversion's own loop, so it
// COM: changes every iteration and the allocation stays at the conversion.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_carried_cvt_src
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %[[DO:arg[0-9]+]] = %{{.*}})
  // RDD-NEXT:  tt.load
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  // PIPE-LABEL: @loop_carried_cvt_src
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %[[DO:arg[0-9]+]] = %{{.*}})
  // PIPE-NEXT:  tt.load
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_carried_cvt_src(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do0 = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %res:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst, %do = %do0) -> (tensor<64x64xf32, #mma>, tensor<64x128xf16, #blocked>) : i32 {
      %do_next = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp, %do_next : tensor<64x64xf32, #mma>, tensor<64x128xf16, #blocked>
    }
    tt.return %res#0 : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: Control: a function-argument source, but the loop writes shared memory
// COM: before the conversion, so the allocation must stay after the
// COM: local_store, as for an op-defined source.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_func_arg_after_local_store
  // RDD:       %[[BUF:.*]] = ttg.local_alloc : ()
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       ttg.local_store %{{.*}}, %[[BUF]] :
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %arg0 :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  // PIPE-LABEL: @loop_invariant_cvt_src_func_arg_after_local_store
  // PIPE:       %[[BUF:.*]] = ttg.local_alloc : ()
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for
  // PIPE:       ttg.local_store %{{.*}}, %[[BUF]] :
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %arg0 :
  // PIPE-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_invariant_cvt_src_func_arg_after_local_store(%do: tensor<64x128xf16, #blocked>, %x_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %x = tt.load %x_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      ttg.local_store %x, %buf : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: The _attn_bwd hit-case shape (flash_attention_benchmark.py, BLOCK_M2=64,
// COM: BLOCK_N2=64, num_warps=16, D_HEAD=128): one loop-invariant `do` tile is
// COM: converted in two sibling loops (the masked and unmasked dq loops), to a
// COM: different dot-operand parent in each, and both loops prefetch. Each
// COM: conversion gets its own allocation right after the load, outside both
// COM: loops; the second one is anchored across the whole first loop, its
// COM: prefetches and the first allocation.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#mma1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_sibling_loops
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NEXT:  %[[ALLOC1:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  %[[ALLOC0:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttig.descriptor_prefetch
  // RDD:       scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC0]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.yield
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC1]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.yield
  // COM: TritonGPUReorderInstructions re-anchors each allocation right after
  // COM: the load, in program order, which swaps the two.
  // PIPE-LABEL: @loop_invariant_cvt_src_sibling_loops
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NEXT:  %[[ALLOC0:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  %[[ALLOC1:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  ttig.descriptor_prefetch
  // PIPE:       scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC0]] :
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.yield
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC1]] :
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.yield
  tt.func public @loop_invariant_cvt_src_sibling_loops(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %desc0: !tt.tensordesc<32x128xf16>, %desc1: !tt.tensordesc<64x128xf16>, %vT0: tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>>, %vT1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n0: i32, %n1: i32) -> (tensor<64x32xf32, #mma1>, tensor<64x64xf32, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma1>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    ttig.descriptor_prefetch %desc0[%c0, %c0] : !tt.tensordesc<32x128xf16>
    %res0 = scf.for %i = %c0 to %n0 step %c1 iter_args(%acc = %cst0) -> (tensor<64x32xf32, #mma1>) : i32 {
      ttig.descriptor_prefetch %desc0[%i, %c0] : !tt.tensordesc<32x128xf16>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT0, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 1}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 2}>> -> tensor<64x32xf32, #mma1>
      scf.yield %dp : tensor<64x32xf32, #mma1>
    }
    ttig.descriptor_prefetch %desc1[%c0, %c0] : !tt.tensordesc<64x128xf16>
    %res1 = scf.for %i = %c0 to %n1 step %c1 iter_args(%acc = %cst1) -> (tensor<64x64xf32, #mma>) : i32 {
      ttig.descriptor_prefetch %desc1[%i, %c0] : !tt.tensordesc<64x128xf16>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT1, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res0, %res1 : tensor<64x32xf32, #mma1>, tensor<64x64xf32, #mma>
  }
}
