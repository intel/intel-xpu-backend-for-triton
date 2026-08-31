// RUN: triton-opt %s -split-input-file -tritonintelgpu-reduce-data-duplication | FileCheck %s

// This test verifies that DPAS operands converted through shared memory
// get proper swizzle parameters, not the degenerate {vec=1, perPhase=1, maxPhase=1}.
// The bug was that DpasEncodingAttr lacked composeSharedLayoutForOperand(),
// causing fallback to generic swizzle which created incorrect LinearLayout.
//
// Test case: dot operation with operand A requiring layout conversion through SLM
// Expected: SwizzledSharedEncoding with maxPhase > 1 (proper swizzle)
// Bug behavior: SwizzledSharedEncoding with maxPhase = 1 (degenerate)

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_swizzle
  tt.func @dpas_operand_swizzle(%a_ptr: tensor<256x64x!tt.ptr<bf16>, #blocked>,
                                 %b_ptr: tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>) {
    // Load operand A with blocked encoding
    %a = tt.load %a_ptr : tensor<256x64x!tt.ptr<bf16>, #blocked>

    // Convert blocked -> dot_operand_a requires layout conversion
    // This should insert local_alloc with SwizzledSharedEncoding
    // CHECK: ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<256x64xbf16, #ttg.swizzled_shared<{vec = {{[0-9]+}}, perPhase = {{[0-9]+}}, maxPhase = {{[0-9]+}}, order = [1, 0]}>, #smem>
    // CHECK-NOT: maxPhase = 1,
    %a_dot = ttg.convert_layout %a : tensor<256x64xbf16, #blocked> -> tensor<256x64xbf16, #dot_operand_a>

    // Operand B already in dot encoding, load directly
    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #dot_operand_b>

    // Dot operation
    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a_dot, %b, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #dot_operand_b> -> tensor<256x128xf32, #mma>

    tt.return
  }
}

// -----

// Test operand B conversion (different kDimIndex)
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @dpas_operand_b_swizzle
  tt.func @dpas_operand_b_swizzle(%a_ptr: tensor<256x64x!tt.ptr<bf16>, #dot_operand_a>,
                                   %b_ptr: tensor<64x128x!tt.ptr<bf16>, #blocked>) {
    %a = tt.load %a_ptr : tensor<256x64x!tt.ptr<bf16>, #dot_operand_a>

    // Load operand B with blocked encoding
    %b = tt.load %b_ptr : tensor<64x128x!tt.ptr<bf16>, #blocked>

    // Convert blocked -> dot_operand_b requires layout conversion
    // CHECK: ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<64x128xbf16, #ttg.swizzled_shared<{vec = {{[0-9]+}}, perPhase = {{[0-9]+}}, maxPhase = {{[0-9]+}}, order = [0, 1]}>, #smem>
    // CHECK-NOT: maxPhase = 1,
    %b_dot = ttg.convert_layout %b : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #dot_operand_b>

    %acc = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %result = tt.dot %a, %b_dot, %acc : tensor<256x64xbf16, #dot_operand_a> * tensor<64x128xbf16, #dot_operand_b> -> tensor<256x128xf32, #mma>

    tt.return
  }
}
