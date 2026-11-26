// RUN: triton-opt %s -split-input-file --tritonintelgpu-reduce-data-duplication | FileCheck %s

#dpas1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 32], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: no_duplication
  tt.func public @no_duplication() {
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas2>
    // CHECK-NOT: ttg.local_alloc
    // CHECK-NOT: ttg.local_load
    %108 = ttg.convert_layout %cst1 : tensor<64x32xf16, #dpas1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas1, kWidth = 1}>>
    %109 = ttg.convert_layout %cst2 : tensor<64x32xf16, #dpas2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas2, kWidth = 2}>>
    tt.return
  }
}

// -----

#dpas1 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [32, 1], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
#dpas2 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 32], repCluster = [1, 2], A = [8, 16], B = [16, 32], C = [8, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: reduce_duplication
  tt.func public @reduce_duplication() {
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas1>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dpas2>
    // CHECK: ttg.local_alloc
    // CHECK: ttg.local_load
    %108 = ttg.convert_layout %cst1 : tensor<64x32xf16, #dpas1> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas1, kWidth = 2}>>
    // CHECK: ttg.convert_layout
    %109 = ttg.convert_layout %cst2 : tensor<64x32xf16, #dpas2> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas2, kWidth = 1}>>
    tt.return
  }
}

// -----

// COM: #8890: a K-contiguous A tile loaded in the loop. Staging it would store it
// COM: on every iteration anyway, in an order the dot operand reads one element at
// COM: a time, so the conversion is left to the generic lowering.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: loop_variant_blocked_source
  // CHECK-NOT:   ttg.local_alloc
  // CHECK:       scf.for
  // CHECK:       %[[A:.*]] = tt.load
  // CHECK-NEXT:  ttg.convert_layout %[[A]] :
  // CHECK-NOT:   ttg.local_alloc
  // CHECK:       tt.return
  tt.func public @loop_variant_blocked_source(%a_ptr: tensor<8x256x!tt.ptr<bf16>, #blocked>, %b: tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<8x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #mma>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<8x64xf32, #mma>) : i32 {
      %a = tt.load %a_ptr : tensor<8x256x!tt.ptr<bf16>, #blocked>
      %a_cvt = ttg.convert_layout %a : tensor<8x256xbf16, #blocked> -> tensor<8x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %d = tt.dot %a_cvt, %b, %acc, inputPrecision = tf32 : tensor<8x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<8x64xf32, #mma>
      scf.yield %d : tensor<8x64xf32, #mma>
    }
    tt.return %res : tensor<8x64xf32, #mma>
  }
}

// -----

// COM: A tile computed in the loop from a loaded one is still staged through a
// COM: buffer: only a tile loaded in the loop is left to the generic lowering.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: loop_variant_computed_source
  // CHECK:       scf.for
  // CHECK:       %[[X:.*]] = arith.mulf
  // CHECK-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[X]] :
  // CHECK-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_variant_computed_source(%a_ptr: tensor<8x256x!tt.ptr<bf16>, #blocked>, %b: tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<8x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #mma>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<8x64xf32, #mma>) : i32 {
      %a = tt.load %a_ptr : tensor<8x256x!tt.ptr<bf16>, #blocked>
      %x = arith.mulf %a, %a : tensor<8x256xbf16, #blocked>
      %x_cvt = ttg.convert_layout %x : tensor<8x256xbf16, #blocked> -> tensor<8x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %d = tt.dot %x_cvt, %b, %acc, inputPrecision = tf32 : tensor<8x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<8x64xf32, #mma>
      scf.yield %d : tensor<8x64xf32, #mma>
    }
    tt.return %res : tensor<8x64xf32, #mma>
  }
}

// -----

// COM: The same tile loaded and converted before the loop, as K and V in
// COM: _attn_bwd: it is stored once, so it is still staged through a buffer.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: function_level_blocked_source
  // CHECK:       %[[A:.*]] = tt.load
  // CHECK-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[A]] :
  // CHECK-NEXT:  ttg.local_load %[[ALLOC]] :
  // CHECK:       scf.for
  tt.func public @function_level_blocked_source(%a_ptr: tensor<8x256x!tt.ptr<bf16>, #blocked>, %b: tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<8x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #mma>
    %a = tt.load %a_ptr : tensor<8x256x!tt.ptr<bf16>, #blocked>
    %a_cvt = ttg.convert_layout %a : tensor<8x256xbf16, #blocked> -> tensor<8x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<8x64xf32, #mma>) : i32 {
      %d = tt.dot %a_cvt, %b, %acc, inputPrecision = tf32 : tensor<8x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<256x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<8x64xf32, #mma>
      scf.yield %d : tensor<8x64xf32, #mma>
    }
    tt.return %res : tensor<8x64xf32, #mma>
  }
}

// -----

// COM: Control: a DPAS result computed in the loop and duplicated across the
// COM: warps along N of its consumer is still staged through a buffer.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: loop_variant_dpas_source
  // CHECK:       scf.for
  // CHECK:       %[[P:.*]] = arith.truncf
  // CHECK-NEXT:  %[[ALLOC:.*]] = ttg.local_alloc %[[P]] :
  // CHECK-NEXT:  ttg.local_load %[[ALLOC]] :
  tt.func public @loop_variant_dpas_source(%q: tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>, %k: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %v: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %n: i32) -> tensor<8x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x64xf32, #mma>
    %res = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<8x64xf32, #mma>) : i32 {
      %qk = tt.dot %q, %k, %cst, inputPrecision = tf32 : tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<8x64xf32, #mma>
      %p = arith.truncf %qk : tensor<8x64xf32, #mma> to tensor<8x64xbf16, #mma>
      %p_cvt = ttg.convert_layout %p : tensor<8x64xbf16, #mma> -> tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %d = tt.dot %p_cvt, %v, %acc, inputPrecision = tf32 : tensor<8x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<8x64xf32, #mma>
      scf.yield %d : tensor<8x64xf32, #mma>
    }
    tt.return %res : tensor<8x64xf32, #mma>
  }
}
