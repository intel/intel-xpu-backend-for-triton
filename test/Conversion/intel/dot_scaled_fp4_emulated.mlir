// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s --check-prefix=NEG

// COM: mxfp4 x mxfp4 tt.dot_scaled on a target without block-scaled DPAS, with the operand and scale layouts that
// COM: AccelerateMatmul gives it, lowered to bf16 DPAS that decode and scale the packed operands themselves.
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 2]], lane = [[0, 0], [0, 0], [0, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [16, 0], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled(
  // NEG-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled(
  // NEG-NOT: triton_gen.bdpas
  // COM: Without block-scaled DPAS, one int8-shaped tile is 4 bf16 DPAS: 4 M x 2 N x 2 K tiles per warp here.
  // CHECK-DAG: llvm.mlir.constant(8.50705917E+37 : f32) : f32
  // COM: A nibble moved to bits [31:28], shifted right by 6 and masked, is 2^-126 times its e2m1 value.
  // CHECK: [[MASK:%.*]] = llvm.mlir.constant(-2118123520 : i32) : i32
  // CHECK-NEXT: [[SIX:%.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK-NEXT: [[SHR:%.*]] = llvm.ashr {{%.*}}, [[SIX]] : i32
  // CHECK-NEXT: [[BITS:%.*]] = llvm.and [[SHR]], [[MASK]] : i32
  // CHECK-NEXT: [[TINY:%.*]] = llvm.bitcast [[BITS]] : i32 to f32
  // COM: Both multiplies are kept out of TRITON_INTEL_FAST_MATH (ttig.fast_math or not): with `fast`, LLVM could
  // COM: reassociate them into one by 2^126 times the scale, which overflows.
  // CHECK-NEXT: [[VAL:%.*]] = llvm.fmul [[TINY]], {{%.*}} {triton_gen.keep_fp_math} : f32
  // COM: The f32 factor of an e8m0 scale byte: byte 0 is 2^-127, byte 255 is NaN.
  // CHECK: [[S:%.*]] = llvm.zext {{%.*}} : i8 to i32
  // CHECK: [[SBITS:%.*]] = llvm.shl [[S]], {{%.*}} : i32
  // CHECK: [[SNORM:%.*]] = llvm.intr.umax([[SBITS]], {{%.*}}) : (i32, i32) -> i32
  // CHECK: [[F:%.*]] = llvm.bitcast [[SNORM]] : i32 to f32
  // CHECK: [[IS255:%.*]] = llvm.icmp "eq" [[S]], {{%.*}} : i32
  // CHECK-NEXT: [[NAN:%.*]] = llvm.mlir.constant(0x7FC00000 : f32) : f32
  // CHECK: [[FACTOR:%.*]] = llvm.select [[IS255]], [[NAN]], [[F]] : i1, f32
  // CHECK: llvm.fmul [[VAL]], [[FACTOR]] {triton_gen.keep_fp_math} : f32
  // CHECK-COUNT-64: triton_gen.dpas {{.*}} {pa = bf16, pb = bf16, rc = 8} : (vector<8xf32>, vector<8xi16>, vector<8xi32>) -> vector<8xf32>
  // CHECK-NOT: triton_gen.dpas
  tt.func public @emulated_fp4_dot_scaled(%a: tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %sa: tensor<64x4xi8, #linear>, %b: tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %sb: tensor<64x4xi8, #linear1>, %c: tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #mma> {
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<64x4xi8, #linear> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<64x4xi8, #linear1> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 2]], lane = [[0, 0], [0, 0], [0, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [16, 0], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled_fast_math_op(
  // COM: With fast math on the dot, a scale byte of 255 is not turned into NaN.
  // NEG-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled_fast_math_op(
  // NEG-NOT: 0x7FC00000
  // NEG-NOT: triton_gen.bdpas
  // CHECK: llvm.intr.umax
  // CHECK: triton_gen.dpas
  tt.func public @emulated_fp4_dot_scaled_fast_math_op(%a: tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %sa: tensor<64x4xi8, #linear>, %b: tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %sb: tensor<64x4xi8, #linear1>, %c: tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #mma> {
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e2m1 {fastMath = true} : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<64x4xi8, #linear> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<64x4xi8, #linear1> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}

// -----

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 2]], lane = [[0, 0], [0, 0], [0, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [16, 0], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io} {
  // CHECK-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled_no_scales(
  // COM: Without scales the decoded value goes to DPAS as it is: no scale factor and one multiply per value.
  // NEG-LABEL: llvm.func spir_kernelcc @emulated_fp4_dot_scaled_no_scales(
  // NEG-NOT: llvm.intr.umax
  // NEG-NOT: 0x7FC00000
  // NEG-NOT: triton_gen.bdpas
  // COM: A nibble moved to bits [31:28], shifted right by 6 and masked, is 2^-126 times its e2m1 value.
  // CHECK: [[MASK:%.*]] = llvm.mlir.constant(-2118123520 : i32) : i32
  // CHECK-NEXT: [[SIX:%.*]] = llvm.mlir.constant(6 : i32) : i32
  // CHECK-NEXT: [[SHR:%.*]] = llvm.ashr {{%.*}}, [[SIX]] : i32
  // CHECK-NEXT: [[BITS:%.*]] = llvm.and [[SHR]], [[MASK]] : i32
  // CHECK-NEXT: [[TINY:%.*]] = llvm.bitcast [[BITS]] : i32 to f32
  // CHECK-NEXT: [[VAL:%.*]] = llvm.fmul [[TINY]], {{%.*}} {triton_gen.keep_fp_math} : f32
  // CHECK-NOT: llvm.fmul [[VAL]]
  // CHECK: triton_gen.dpas
  tt.func public @emulated_fp4_dot_scaled_no_scales(%a: tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %b: tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %c: tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #mma> {
    %d = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<64x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
    tt.return %d : tensor<64x64xf32, #mma>
  }
}
