// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication | FileCheck %s --check-prefix=RDD
// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication --tritongpu-reorder-instructions | FileCheck %s --check-prefix=PIPE

// COM: Reduced from the _attn_bwd dq loop (#8283): the loop-invariant `do` is
// COM: stored once, before the loop; only the local_load stays in the loop.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NEXT:  ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       tt.dot
  // RDD:       scf.yield
  // PIPE-LABEL: @loop_invariant_cvt_src
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NEXT:  ttig.descriptor_prefetch
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  scf.for
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

// COM: Control: the source is defined in the loop, so the allocation stays at
// COM: the conversion, not before the scf.if.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_variant_cvt_src_in_if
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NEXT:  scf.if
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_variant_cvt_src_in_if(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32, %cond: i1) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      %r = scf.if %cond -> (tensor<64x64xf32, #mma>) {
        %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
        scf.yield %dp : tensor<64x64xf32, #mma>
      } else {
        scf.yield %acc : tensor<64x64xf32, #mma>
      }
      scf.yield %r : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: The source is defined outside both loops: the allocation goes before the
// COM: outer loop.
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

// COM: Control: the loop writes shared memory before the conversion, so the
// COM: allocation stays at the conversion.
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

// COM: Control: as above with the write after the conversion, which it still
// COM: precedes on the next iteration.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_before_local_store
  // RDD:       %[[BUF:.*]] = ttg.local_alloc : ()
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  // RDD:       ttg.local_store %{{.*}}, %[[BUF]] :
  // PIPE-LABEL: @loop_invariant_cvt_src_before_local_store
  // PIPE:       %[[BUF:.*]] = ttg.local_alloc : ()
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.for
  // PIPE-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  ttg.local_load %[[ALLOC]] :
  // PIPE:       ttg.local_store %{{.*}}, %[[BUF]] :
  tt.func public @loop_invariant_cvt_src_before_local_store(%do_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %x_ptr: tensor<64x128x!tt.ptr<f16>, #blocked>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %do = tt.load %do_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      %x = tt.load %x_ptr : tensor<64x128x!tt.ptr<f16>, #blocked>
      ttg.local_store %x, %buf : tensor<64x128xf16, #blocked> -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: Control: `ttg.barrier` has unknown effects, so the allocation stays after
// COM: it. RDD only: TritonGPUReorderInstructions treats them as no write.
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

// COM: Control: the source is an iter_arg of the loop, so the allocation stays
// COM: at the conversion.
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

// COM: The _attn_bwd shape: `do` is converted in two sibling dq loops; each
// COM: conversion gets an allocation before its own loop.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#mma1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_sibling_loops
  // RDD:       %[[DO:.*]] = tt.load
  // RDD-NEXT:  ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC0:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC0]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.yield
  // RDD-NEXT:  }
  // RDD-NEXT:  ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC1:.*]] = ttg.local_alloc %[[DO]] :
  // RDD-NEXT:  scf.for
  // RDD-NOT:   ttg.local_alloc
  // RDD:       ttg.local_load %[[ALLOC1]] :
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.yield
  // PIPE-LABEL: @loop_invariant_cvt_src_sibling_loops
  // PIPE:       %[[DO:.*]] = tt.load
  // PIPE-NEXT:  ttig.descriptor_prefetch
  // PIPE-NEXT:  %[[ALLOC0:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  scf.for
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       ttg.local_load %[[ALLOC0]] :
  // PIPE-NOT:   ttg.local_alloc
  // PIPE:       scf.yield
  // PIPE-NEXT:  }
  // PIPE-NEXT:  ttig.descriptor_prefetch
  // PIPE-NEXT:  %[[ALLOC1:.*]] = ttg.local_alloc %[[DO]] :
  // PIPE-NEXT:  scf.for
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

// -----

// COM: Control: a function-argument source stays at the conversion, as in
// COM: TritonGPUReorderInstructions.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_func_arg
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %arg0 :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_invariant_cvt_src_func_arg(%do: tensor<64x128xf16, #blocked>, %desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      ttig.descriptor_prefetch %desc[%i, %c0] : !tt.tensordesc<64x128xf16>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// COM: Control: splat and constant sources stay at the conversion.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 2], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // RDD-LABEL: @loop_invariant_cvt_src_splat
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %{{.*}} :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_invariant_cvt_src_splat(%x: f16, %desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do = tt.splat %x : f16 -> tensor<64x128xf16, #blocked>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      ttig.descriptor_prefetch %desc[%i, %c0] : !tt.tensordesc<64x128xf16>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }

  // RDD-LABEL: @loop_invariant_cvt_src_constant
  // RDD-NOT:   ttg.local_alloc
  // RDD:       scf.for
  // RDD:       ttig.descriptor_prefetch
  // RDD-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %{{.*}} :
  // RDD-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_invariant_cvt_src_constant(%desc: !tt.tensordesc<64x128xf16>, %vT: tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %do = arith.constant dense<1.000000e+00> : tensor<64x128xf16, #blocked>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      ttig.descriptor_prefetch %desc[%i, %c0] : !tt.tensordesc<64x128xf16>
      %do_cvt = ttg.convert_layout %do : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %dp = tt.dot %do_cvt, %vT, %acc, inputPrecision = tf32 : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x64xf32, #mma>
      scf.yield %dp : tensor<64x64xf32, #mma>
    }
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
