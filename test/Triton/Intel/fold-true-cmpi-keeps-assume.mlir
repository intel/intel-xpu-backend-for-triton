// RUN: triton-opt %s -split-input-file -triton-intel-fold-true-cmpi -triton-intel-simplify-signed-arithmetic | FileCheck %s

// COM: The assumed comparison is proven from the assume itself. Folding it to
// COM: `llvm.intr.assume %true` would drop the fact, and the later pass could
// COM: no longer prove the dividend non-negative.
tt.func @assume_survives_for_later_pass(%arg0: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c4 = arith.constant 4 : i32
  %cmp = arith.cmpi sge, %arg0, %c0 : i32
  llvm.intr.assume %cmp : i1
  %r = arith.remsi %arg0, %c4 : i32
  tt.return %r : i32
}

// CHECK-LABEL: tt.func @assume_survives_for_later_pass
// CHECK:         %[[CMP:.*]] = arith.cmpi sge, %{{.*}}, %{{.*}} : i32
// CHECK:         llvm.intr.assume %[[CMP]] : i1
// CHECK-NOT:     arith.remsi
// CHECK:         arith.remui

// -----

// COM: A comparison that merely follows from the assume still folds.
tt.func @derived_comparison_folds(%arg0: i32) -> i1 {
  %c0 = arith.constant 0 : i32
  %cm1 = arith.constant -1 : i32
  %cmp = arith.cmpi sge, %arg0, %c0 : i32
  llvm.intr.assume %cmp : i1
  %o = arith.cmpi sgt, %arg0, %cm1 : i32
  tt.return %o : i1
}

// CHECK-LABEL: tt.func @derived_comparison_folds
// CHECK:         %[[CMP:.*]] = arith.cmpi sge
// CHECK:         llvm.intr.assume %[[CMP]] : i1
// CHECK-NOT:     arith.cmpi
// CHECK:         tt.return %true
