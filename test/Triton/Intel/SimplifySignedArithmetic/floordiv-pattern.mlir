// RUN: triton-opt %s -split-input-file -triton-intel-simplify-signed-arithmetic | FileCheck %s

// Test the PyTorch Inductor floordiv lowering pattern.
// The floordiv implementation uses bitwise complement (~) to handle negative dividends,
// and abs() for the divisor. SimplifySignedArithmetic should recognize these patterns
// and convert the final divsi to divui.

// CHECK-LABEL: tt.func @floordiv_pattern
tt.func @floordiv_pattern(%a: i32, %b: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c_minus1 = arith.constant -1 : i32

  // Step 1: Guard against division by zero
  %b_zero = arith.cmpi eq, %b, %c0 : i32
  %b_safe = arith.select %b_zero, %c1, %b : i32

  // Step 2: Make b positive by taking absolute value
  %b_neg = arith.cmpi slt, %b_safe, %c0 : i32
  %neg_a = arith.subi %c0, %a : i32
  %a1 = arith.select %b_neg, %neg_a, %a : i32
  %neg_b = arith.subi %c0, %b_safe : i32
  %b_pos = arith.select %b_neg, %neg_b, %b_safe : i32

  // Step 3: Make a non-negative using bitwise complement
  %a_neg = arith.cmpi slt, %a1, %c0 : i32
  %not_a = arith.xori %a1, %c_minus1 : i32
  %a_nonneg = arith.select %a_neg, %not_a, %a1 : i32

  // Step 4: Divide - should be optimized to divui
  // CHECK: arith.divui
  // CHECK-NOT: arith.divsi
  %quot = arith.divsi %a_nonneg, %b_pos : i32

  // Step 5: Transform result back
  %quot_fixed = arith.xori %quot, %c_minus1 : i32
  %result1 = arith.select %a_neg, %quot_fixed, %quot : i32
  %result = arith.select %b_zero, %c0, %result1 : i32

  tt.return %result : i32
}

// -----

// CHECK-LABEL: @floordiv_pattern_tensor
// Test with tensor types (vectorized floordiv)
tt.func @floordiv_pattern_tensor(%a: tensor<1024xi32>, %b: tensor<1024xi32>) -> tensor<1024xi32> {
  %c0 = arith.constant dense<0> : tensor<1024xi32>
  %c1 = arith.constant dense<1> : tensor<1024xi32>
  %c_minus1 = arith.constant dense<-1> : tensor<1024xi32>

  // Step 1: Guard against division by zero
  %b_zero = arith.cmpi eq, %b, %c0 : tensor<1024xi32>
  %b_safe = arith.select %b_zero, %c1, %b : tensor<1024xi1>, tensor<1024xi32>

  // Step 2: Make b positive by taking absolute value
  %b_neg = arith.cmpi slt, %b_safe, %c0 : tensor<1024xi32>
  %neg_a = arith.subi %c0, %a : tensor<1024xi32>
  %a1 = arith.select %b_neg, %neg_a, %a : tensor<1024xi1>, tensor<1024xi32>
  %neg_b = arith.subi %c0, %b_safe : tensor<1024xi32>
  %b_pos = arith.select %b_neg, %neg_b, %b_safe : tensor<1024xi1>, tensor<1024xi32>

  // Step 3: Make a non-negative using bitwise complement
  %a_neg = arith.cmpi slt, %a1, %c0 : tensor<1024xi32>
  %not_a = arith.xori %a1, %c_minus1 : tensor<1024xi32>
  %a_nonneg = arith.select %a_neg, %not_a, %a1 : tensor<1024xi1>, tensor<1024xi32>

  // Step 4: Divide - should be optimized to divui
  // CHECK: arith.divui
  // CHECK-NOT: arith.divsi
  %quot = arith.divsi %a_nonneg, %b_pos : tensor<1024xi32>

  // Step 5: Transform result back
  %quot_fixed = arith.xori %quot, %c_minus1 : tensor<1024xi32>
  %result1 = arith.select %a_neg, %quot_fixed, %quot : tensor<1024xi1>, tensor<1024xi32>
  %result = arith.select %b_zero, %c0, %result1 : tensor<1024xi1>, tensor<1024xi32>

  tt.return %result : tensor<1024xi32>
}

// -----

// CHECK-LABEL: @floordiv_simplified
// Test a simplified version without all the transformations
// This should NOT be optimized because we can't prove the properties
tt.func @floordiv_simplified(%a: i32, %b: i32) -> i32 {
  // Without the pattern, divsi should remain
  // CHECK: arith.divsi
  // CHECK-NOT: arith.divui
  %quot = arith.divsi %a, %b : i32
  tt.return %quot : i32
}

// -----

// CHECK-LABEL: @abs_pattern_positive
// Test that abs pattern is recognized when input is guaranteed non-zero
tt.func @abs_pattern_positive(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // Make x non-zero: select(x == 0, 1, x)
  %x_zero = arith.cmpi eq, %x, %c0 : i32
  %x_nonzero = arith.select %x_zero, %c1, %x : i32

  // Take absolute value: select(x < 0, -x, x)
  %x_neg = arith.cmpi slt, %x_nonzero, %c0 : i32
  %neg_x = arith.subi %c0, %x_nonzero : i32
  %abs_x = arith.select %x_neg, %neg_x, %x_nonzero : i32

  // Use as divisor - should be recognized as strictly positive
  // CHECK: arith.divui
  // CHECK-NOT: arith.divsi
  %result = arith.divsi %c1, %abs_x : i32
  tt.return %result : i32
}

// -----

// CHECK-LABEL: @bitwise_not_nonnegative
// Test that select(x < 0, ~x, x) is recognized as non-negative
tt.func @bitwise_not_nonnegative(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c_minus1 = arith.constant -1 : i32

  // Pattern: select(x < 0, ~x, x)
  %x_neg = arith.cmpi slt, %x, %c0 : i32
  %not_x = arith.xori %x, %c_minus1 : i32
  %x_nonneg = arith.select %x_neg, %not_x, %x : i32

  // Use as dividend with positive divisor
  // CHECK: arith.divui
  // CHECK-NOT: arith.divsi
  %result = arith.divsi %x_nonneg, %c1 : i32
  tt.return %result : i32
}
