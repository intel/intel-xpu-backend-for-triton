// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: An INT8 dot on the FMA path must be lowered to dp4a instructions rather
// COM: than to a mul/add chain: IGC folds such a chain back into a dp4a and
// COM: mis-pairs its two operands (issue #7854).

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK: llvm.func spir_funccc @llvm.genx.GenISA.dp4a.ss.i32(i32, i32, i32, i1) -> i32
  // CHECK-LABEL: matmul_int8_fmadot
  tt.func @matmul_int8_fmadot(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                              %a: !ttg.memdesc<32x16xi8, #shared, #smem>,
                              %b: !ttg.memdesc<16x32xi8, #shared, #smem>) {
    %cst = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %a_mat = ttg.local_load %a : !ttg.memdesc<32x16xi8, #shared, #smem> -> tensor<32x16xi8, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<16x32xi8, #shared, #smem> -> tensor<16x32xi8, #dot_operand_b>
    %a_i32 = arith.extsi %a_mat : tensor<32x16xi8, #dot_operand_a> to tensor<32x16xi32, #dot_operand_a>
    %b_i32 = arith.extsi %b_mat : tensor<16x32xi8, #dot_operand_b> to tensor<16x32xi32, #dot_operand_b>
    // COM: Both operands are packed from four i8 values in the same order, so
    // COM: component i of one side pairs with component i of the other.
    // CHECK:           llvm.mlir.undef : vector<4xi8>
    // CHECK-COUNT-4:   llvm.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<4xi8>
    // CHECK:           llvm.bitcast %{{.*}} : vector<4xi8> to i32
    // CHECK:           llvm.mlir.undef : vector<4xi8>
    // CHECK-COUNT-4:   llvm.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<4xi8>
    // CHECK:           llvm.bitcast %{{.*}} : vector<4xi8> to i32
    // CHECK:           llvm.call spir_funccc @llvm.genx.GenISA.dp4a.ss.i32(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{.*}} : (i32, i32, i32, i1) -> i32
    %d = tt.dot %a_i32, %b_i32, %cst : tensor<32x16xi32, #dot_operand_a> * tensor<16x32xi32, #dot_operand_b> -> tensor<32x32xi32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<i32>, #blocked> -> tensor<32x32x!tt.ptr<i32>, #blocked>
    tt.store %36, %d : tensor<32x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// COM: dp4a only takes 8-bit components, so a wider integer dot keeps the
// COM: generic mul/add chain. A chain of fewer than four products is not
// COM: foldable into a dp4a, so it is not exposed to #7854.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#blocked}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: matmul_int16_fmadot
  // CHECK-NOT: dp4a
  tt.func @matmul_int16_fmadot(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                               %a: !ttg.memdesc<32x16xi16, #shared, #smem>,
                               %b: !ttg.memdesc<16x32xi16, #shared, #smem>) {
    %cst = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    %a_mat = ttg.local_load %a : !ttg.memdesc<32x16xi16, #shared, #smem> -> tensor<32x16xi16, #dot_operand_a>
    %b_mat = ttg.local_load %b : !ttg.memdesc<16x32xi16, #shared, #smem> -> tensor<16x32xi16, #dot_operand_b>
    %a_i32 = arith.extsi %a_mat : tensor<32x16xi16, #dot_operand_a> to tensor<32x16xi32, #dot_operand_a>
    %b_i32 = arith.extsi %b_mat : tensor<16x32xi16, #dot_operand_b> to tensor<16x32xi32, #dot_operand_b>
    // CHECK: llvm.mul
    %d = tt.dot %a_i32, %b_i32, %cst : tensor<32x16xi32, #dot_operand_a> * tensor<16x32xi32, #dot_operand_b> -> tensor<32x32xi32, #blocked>
    %30 = tt.splat %ptr : !tt.ptr<i32> -> tensor<32x1x!tt.ptr<i32>, #blocked>
    %36 = tt.broadcast %30 : tensor<32x1x!tt.ptr<i32>, #blocked> -> tensor<32x32x!tt.ptr<i32>, #blocked>
    tt.store %36, %d : tensor<32x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
