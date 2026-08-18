// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s

// COM: A loop containing only a masked store. Stores are not collected for
// COM: versioning, so the loop is left alone.
tt.func public @store_only_canonical_mask(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c31_i32 = arith.constant 31 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %3 = arith.addi %arg1, %c31_i32 : i32
  %4 = arith.divsi %3, %c32_i32 : i32
  scf.for %arg2 = %c0_i32 to %4 step %c1_i32  : i32 {
    %5 = arith.muli %arg2, %c32_i32 : i32
    %6 = arith.subi %arg1, %5 : i32
    %7 = tt.splat %6 : i32 -> tensor<32xi32>
    %8 = arith.cmpi slt, %0, %7 : tensor<32xi32>
    tt.store %2, %cst, %8 : tensor<32x!tt.ptr<f16>>
  }
  tt.return
}

// CHECK-LABEL: @store_only_canonical_mask
// CHECK-NOT:   scf.if
// CHECK:       scf.for
// CHECK:         tt.store {{%[0-9]+}}, {{%cst}}, {{%[0-9]+}} : tensor<32x!tt.ptr<f16>>
// CHECK-NOT:   scf.if

// -----

// COM: A versionable loop yielding no result (the loaded value is stored in the
// COM: loop body rather than accumulated in an iteration argument). The
// COM: generated 'scf.if' has no result, therefore its regions must use the
// COM: implicit terminator created by the builder rather than an explicit
// COM: 'scf.yield'. Emitting both used to produce invalid IR.
tt.func public @load_only_no_results(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c31_i32 = arith.constant 31 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
  %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %3 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
  %4 = tt.addptr %3, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
  %5 = arith.addi %arg2, %c31_i32 : i32
  %6 = arith.divsi %5, %c32_i32 : i32
  scf.for %arg3 = %c0_i32 to %6 step %c1_i32  : i32 {
    %7 = arith.muli %arg3, %c32_i32 : i32
    %8 = arith.subi %arg2, %7 : i32
    %9 = tt.splat %8 : i32 -> tensor<32xi32>
    %10 = arith.cmpi slt, %0, %9 : tensor<32xi32>
    %11 = tt.load %2, %10, %cst : tensor<32x!tt.ptr<f16>>
    tt.store %4, %11 : tensor<32x!tt.ptr<f16>>
  }
  tt.return
}

// CHECK-LABEL: @load_only_no_results
// CHECK:       [[REM:%[0-9]+]] = arith.remsi %arg2, %c32_i32 : i32
// CHECK:       [[CMP1:%[0-9]+]] = arith.cmpi eq, [[REM]], %c0_i32 : i32
// CHECK:       [[CMP2:%[0-9]+]] = arith.cmpi sgt, %arg2, %c32_i32 : i32
// CHECK:       [[COND:%[0-9]+]] = arith.andi [[CMP1]], [[CMP2]] : i1
// COM: The 'scf.if' has no result and the fast path load is mask free.
// CHECK:       scf.if [[COND]] {
// CHECK:         scf.for
// CHECK:           [[LOAD:%[0-9]+]] = tt.load {{%[0-9]+}} : tensor<32x!tt.ptr<f16>>
// CHECK:           tt.store {{%[0-9]+}}, [[LOAD]] : tensor<32x!tt.ptr<f16>>
// CHECK:         }
// CHECK:       } else {
// CHECK:         scf.for
// CHECK:           [[LOAD2:%[0-9]+]] = tt.load {{%[0-9]+}}, {{%[0-9]+}}, %cst : tensor<32x!tt.ptr<f16>>
// CHECK:           tt.store {{%[0-9]+}}, [[LOAD2]] : tensor<32x!tt.ptr<f16>>
// CHECK:         }
// CHECK:       }
