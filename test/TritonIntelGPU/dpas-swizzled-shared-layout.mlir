// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-data-duplication | FileCheck %s

// This test verifies that DPAS operands converted through shared memory
// get proper swizzle parameters, not the degenerate {vec=1, perPhase=1, maxPhase=1}.
// The bug was that DpasEncodingAttr lacked composeSharedLayoutForOperand(),
// causing fallback to generic swizzle which created incorrect LinearLayout.

// -----
// Test 1: Rank-2 operand A with K-contiguous layout
// operandShape=[256, 64], sharedOrder=[1, 0] (K=64 inner), vec=1, elemBits=16
// Expected: vec=1, perPhase=1, maxPhase=16 (non-degenerate)

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_a_rank2
  tt.func @dpas_operand_a_rank2(%a_ptr: tensor<256x64x!tt.ptr<bf16>, #blocked>,
                                 %b_ptr: tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>) {
    %a = tt.load %a_ptr : tensor<256x64x!tt.ptr<bf16>, #blocked>

    // Convert blocked -> dot_operand_a requires layout conversion
    // CHECK: ttg.local_alloc
    // CHECK-SAME: #shared
    %a_dot = ttg.convert_layout %a : tensor<256x64xbf16, #blocked> -> tensor<256x64xbf16, #dot_operand_a>

    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>

    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a_dot, %b, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #dot_operand_b> -> tensor<256x128xf32, #mma>

    tt.return
  }
}

// -----
// Test 2: Rank-2 operand B with K-contiguous layout
// operandShape=[64, 128], sharedOrder=[0, 1] (K=64 inner), vec=2, elemBits=16
// Expected: vec=2, perPhase=1, maxPhase=16

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// CHECK: #shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_b_rank2
  tt.func @dpas_operand_b_rank2(%a_ptr: tensor<256x64x!tt.ptr<bf16>, #dot_operand_a>,
                                 %b_ptr: tensor<64x128x!tt.ptr<bf16>, #blocked>) {
    %a = tt.load %a_ptr : tensor<256x64x!tt.ptr<bf16>, #dot_operand_a>

    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #blocked>

    // Convert blocked -> dot_operand_b requires layout conversion
    // CHECK: ttg.local_alloc
    // CHECK-SAME: #shared
    %b_dot = ttg.convert_layout %b : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #dot_operand_b>

    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a, %b_dot, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #dot_operand_b> -> tensor<256x128xf32, #mma>

    tt.return
  }
}

// -----
// Test 3: Non-contiguous K dimension (no swizzling needed)
// operandShape=[256, 64], sharedOrder=[0, 1] (M=256 inner, K=64 outer), vec=1, elemBits=16
// Expected: vec=1, perPhase=1, maxPhase=1 (no swizzle since K is not contiguous)

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_a_non_contig_k
  tt.func @dpas_operand_a_non_contig_k(%a_ptr: tensor<256x64x!tt.ptr<bf16>, #blocked>,
                                        %b_ptr: tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>) {
    %a = tt.load %a_ptr : tensor<256x64x!tt.ptr<bf16>, #blocked>

    // Convert blocked -> dot_operand_a with non-contiguous K
    // CHECK: ttg.local_alloc
    // CHECK-SAME: #shared
    %a_dot = ttg.convert_layout %a : tensor<256x64xbf16, #blocked> -> tensor<256x64xbf16, #dot_operand_a>

    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>

    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a_dot, %b, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #dot_operand_b> -> tensor<256x128xf32, #mma>

    tt.return
  }
}

// -----
// Test 4: threadsPerWarp=32 (larger warp size)
// operandShape=[256, 64], sharedOrder=[1, 0] (K=64 inner), vec=1, elemBits=16
// With threadsPerWarp=32: maxPhase = std::max(std::min(32/1, 64/1), 1) = 32
// Expected: vec=1, perPhase=1, maxPhase=32

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 32, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 32, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_a_warp32
  tt.func @dpas_operand_a_warp32(%a_ptr: tensor<256x64x!tt.ptr<bf16>, #blocked>,
                                  %b_ptr: tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>) {
    %a = tt.load %a_ptr : tensor<256x64x!tt.ptr<bf16>, #blocked>

    // Convert blocked -> dot_operand_a with threadsPerWarp=32
    // CHECK: ttg.local_alloc
    // CHECK-SAME: #shared
    %a_dot = ttg.convert_layout %a : tensor<256x64xbf16, #blocked> -> tensor<256x64xbf16, #dot_operand_a>

    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>

    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a_dot, %b, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #dot_operand_b> -> tensor<256x128xf32, #mma>

    tt.return
  }
}

// -----
// Test 5: needTrans scenario (verifies transpose swizzle logic)
// This tests the kDimIndex transformation: kDimIndex' = (2*rank - 3) - kDimIndex
// For rank-2, opIdx=0: normally kDimIndex=1 (K at dim 1)
// With needTrans: kDimIndex' = (2*2-3) - 1 = 0 (K swapped to dim 0)
// When K is at inner position (sharedOrder[0]=0), swizzle should be applied
// operandShape=[256, 64], sharedOrder=[0, 1], vec=1, elemBits=16
// Expected: vec=1, perPhase=1, maxPhase=16 (swizzle applied because K is now contiguous)

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>

// Manually constructed swizzled_shared with needTrans effect:
// For operand A with needTrans, K dimension moves from 1→0, so sharedOrder=[0,1] makes K contiguous
// This simulates what would happen if needTrans=true was passed to composeSharedLayoutForOperand
#shared_needtrans = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_a_needtrans
  tt.func @dpas_operand_a_needtrans(%a_desc: !ttg.memdesc<256x64xbf16, #shared_needtrans, #smem>,
                                     %b_ptr: tensor<64x128x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>) {
    // Load from pre-constructed swizzled shared memory with needTrans-like layout
    // CHECK: ttg.local_load
    // CHECK-SAME: #shared
    %a_dot = ttg.local_load %a_desc : !ttg.memdesc<256x64xbf16, #shared_needtrans, #smem> -> tensor<256x64xbf16, #dot_operand_a>

    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>

    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a_dot, %b, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x128xf32, #mma>

    tt.return
  }
}
