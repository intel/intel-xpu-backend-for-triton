// RUN: triton-opt %s -split-input-file -triton-intel-simplify-signed-arithmetic | FileCheck %s

// CHECK-LABEL: tt.func @test_divsi_with_assume_nonnegative
// Test that divsi is converted to divui when assumptions prove operands are non-negative
tt.func @test_divsi_with_assume_nonnegative(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // Assume arg0 >= 0
  %cmp0 = arith.cmpi sge, %arg0, %c0 : i32
  llvm.assume %cmp0 : i1

  // Assume arg1 > 0 (strictly positive)
  %cmp1 = arith.cmpi sgt, %arg1, %c0 : i32
  llvm.assume %cmp1 : i1

  // This divsi should be converted to divui
  // CHECK: arith.divui
  // CHECK-NOT: arith.divsi
  %result = arith.divsi %arg0, %arg1 : i32
  tt.return %result : i32
}

// -----

// CHECK-LABEL: @test_remsi_with_assume_nonnegative
// Test that remsi is converted to remui when assumptions prove operands are non-negative
tt.func @test_remsi_with_assume_nonnegative(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // Assume arg0 >= 0
  %cmp0 = arith.cmpi sge, %arg0, %c0 : i32
  llvm.assume %cmp0 : i1

  // Assume arg1 > 0
  %cmp1 = arith.cmpi sgt, %arg1, %c0 : i32
  llvm.assume %cmp1 : i1

  // This remsi should be converted to remui
  // CHECK: arith.remui
  // CHECK-NOT: arith.remsi
  %result = arith.remsi %arg0, %arg1 : i32
  tt.return %result : i32
}

// -----

// CHECK-LABEL: @test_ceildivsi_with_assume_nonnegative
// Test that ceildivsi is converted to ceildivui when assumptions prove operands are non-negative
tt.func @test_ceildivsi_with_assume_nonnegative(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32

  // Assume arg0 >= 0
  %cmp0 = arith.cmpi sge, %arg0, %c0 : i32
  llvm.assume %cmp0 : i1

  // Assume arg1 > 0
  %cmp1 = arith.cmpi sgt, %arg1, %c0 : i32
  llvm.assume %cmp1 : i1

  // This ceildivsi should be converted to ceildivui
  // CHECK: arith.ceildivui
  // CHECK-NOT: arith.ceildivsi
  %result = arith.ceildivsi %arg0, %arg1 : i32
  tt.return %result : i32
}

// -----

// CHECK-LABEL: @test_divsi_without_assume_positive_divisor
// Test that divsi is NOT converted when divisor is not proven positive
tt.func @test_divsi_without_assume_positive_divisor(%arg0: i32, %arg1: i32) -> i32 {
  %c0 = arith.constant 0 : i32

  // Assume arg0 >= 0 (dividend non-negative)
  %cmp0 = arith.cmpi sge, %arg0, %c0 : i32
  llvm.assume %cmp0 : i1

  // Assume arg1 >= 0 (divisor non-negative, NOT strictly positive)
  %cmp1 = arith.cmpi sge, %arg1, %c0 : i32
  llvm.assume %cmp1 : i1

  // This divsi should NOT be converted (divisor must be > 0, not >= 0)
  // CHECK: arith.divsi
  // CHECK-NOT: arith.divui
  %result = arith.divsi %arg0, %arg1 : i32
  tt.return %result : i32
}

// -----

// CHECK-LABEL: @test_divsi_with_range_assumption
// Test with bounded range assumptions (e.g., 0 <= x <= 1024)
tt.func @test_divsi_with_range_assumption(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1024 = arith.constant 1024 : i32
  %c16 = arith.constant 16 : i32

  // Assume 0 <= arg0
  %cmp0 = arith.cmpi sge, %arg0, %c0 : i32
  llvm.assume %cmp0 : i1

  // Assume arg0 <= 1024
  %cmp1 = arith.cmpi sle, %arg0, %c1024 : i32
  llvm.assume %cmp1 : i1

  // With range [0, 1024] and constant positive divisor, should convert
  // CHECK: arith.divui
  // CHECK-NOT: arith.divsi
  %result = arith.divsi %arg0, %c16 : i32
  tt.return %result : i32
}

// -----

// CHECK-LABEL: @test_tensor_divsi_with_assume
// Test with tensor types
tt.func @test_tensor_divsi_with_assume(%arg0: tensor<1024xi32>, %arg1: tensor<1024xi32>) -> tensor<1024xi32> {
  %c0 = arith.constant dense<0> : tensor<1024xi32>

  // Assume all elements of arg0 >= 0
  %cmp0 = arith.cmpi sge, %arg0, %c0 : tensor<1024xi32>
  // Note: llvm.assume expects i1, not tensor<1024xi1>, so this pattern may need
  // element-wise or reduction. For now this tests the scalar pattern.

  // Assume all elements of arg1 > 0
  %cmp1 = arith.cmpi sgt, %arg1, %c0 : tensor<1024xi32>

  // Without tensor assumptions, this relies on constant analysis
  // CHECK: arith.divsi
  %result = arith.divsi %arg0, %arg1 : tensor<1024xi32>
  tt.return %result : tensor<1024xi32>
}
