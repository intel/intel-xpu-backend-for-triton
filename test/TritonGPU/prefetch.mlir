// RUN: triton-opt %s -split-input-file -tritongpu-prefetch -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.nvidia_mma<{version = 2, warpsPerCTA = [4, 1]}>
#A_OP = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>


// CHECK: tt.func @matmul_loop_mixed
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = triton_gpu.extract_slice %[[A0:.*]][0, 0] [128, 16]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = triton_gpu.convert_layout %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = triton_gpu.extract_slice %[[B0:.*]][0, 0] [16, 128]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = triton_gpu.convert_layout %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = triton_gpu.extract_slice %[[arg_a0]][0, 16] [128, 16]
// CHECK-DAG:   %[[A_REM:.*]] = triton_gpu.convert_layout %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = triton_gpu.extract_slice %[[arg_b0]][16, 0] [16, 128]
// CHECK-DAG:   %[[B_REM:.*]] = triton_gpu.convert_layout %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = triton_gpu.extract_slice {{.*}}[0, 0] [128, 16]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = triton_gpu.convert_layout %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = triton_gpu.extract_slice {{.*}}[0, 0] [16, 128]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = triton_gpu.convert_layout %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "triton_gpu.num-warps" = 4 : i32 } {
tt.func @matmul_loop_mixed(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.broadcast %A : (!tt.ptr<f8E5M2>) -> tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.broadcast %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf8E5M2, #AL>
  %a_init = triton_gpu.convert_layout %a_ : (tensor<128x32xf8E5M2, #AL>) -> tensor<128x32xf8E5M2, #A>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
  %b_init = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x32xf8E5M2, #A>, tensor<32x128xf16, #B>, tensor<128x128xf32, #C>) {
    %a_op_ = triton_gpu.convert_layout %a : (tensor<128x32xf8E5M2, #A>) -> tensor<128x32xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x32xf8E5M2, #A_OP> -> tensor<128x32xf16, #A_OP>
    %b_op = triton_gpu.convert_layout %b : (tensor<32x128xf16, #B>) -> tensor<32x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf8E5M2, #AL>
    %next_a = triton_gpu.convert_layout %next_a_ : (tensor<128x32xf8E5M2, #AL>) -> tensor<128x32xf8E5M2, #A>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %next_b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x32xf8E5M2, #A>, tensor<32x128xf16, #B>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module
