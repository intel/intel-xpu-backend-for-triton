// REQUIRES: census-debug
// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks -debug-only=triton-intel-remove-masks-census -o /dev/null 2>&1 | FileCheck %s

// COM: The census trace of a versioned loop lists the conjoined conditions. They
// COM: may be defined in different blocks, where program order is undefined.

// CHECK: versioned: loop=three_masks_across_blocks#{{[^/]*}}/L0 unmasked={{.*}} guard={{[^;]*}}arith.cmpi{{[^;]*}};{{[^;]*}}arith.cmpi{{[^;]*}};{{[^;]*}}arith.cmpi
tt.func @three_masks_across_blocks(%ptr: !tt.ptr<f32>, %n: i32, %m: i32, %k: i32, %cnt: i32, %c: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %n : i32 -> tensor<64xi32>
  %ks = tt.splat %k : i32 -> tensor<64xi32>
  %mask1 = arith.cmpi slt, %lane, %ns : tensor<64xi32>
  %mask3 = arith.cmpi slt, %lane, %ks : tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  scf.if %c {
    %ms = tt.splat %m : i32 -> tensor<64xi32>
    %mask2 = arith.cmpi slt, %lane, %ms : tensor<64xi32>
    scf.for %i = %c0 to %cnt step %c1 : i32 {
      %v1 = tt.load %ps, %mask1 : tensor<64x!tt.ptr<f32>>
      %v2 = tt.load %ps, %mask2 : tensor<64x!tt.ptr<f32>>
      %v3 = tt.load %ps, %mask3 : tensor<64x!tt.ptr<f32>>
      %s = arith.addf %v1, %v2 : tensor<64xf32>
      %t = arith.addf %s, %v3 : tensor<64xf32>
      tt.store %ps, %t : tensor<64x!tt.ptr<f32>>
    }
  }
  tt.return
}

// -----

// COM: Conditions in a single block keep program order. The padding pushes the
// COM: cmpi results to %8, %9, %10, so textual order ("%10" first) would differ.

// CHECK:      versioned: loop=three_masks_one_block#{{[^/]*}}/L0 unmasked={{.*}} guard=%[[#N:]] = arith.cmpi slt, {{[^;]*}};
// CHECK-SAME: %[[#N+1]] = arith.cmpi slt, {{[^;]*}};
// CHECK-SAME: %[[#N+2]] = arith.cmpi slt,
tt.func @three_masks_one_block(%ptr: !tt.ptr<f32>, %n: i32, %m: i32, %k: i32, %cnt: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %ns = tt.splat %n : i32 -> tensor<64xi32>
  %ms = tt.splat %m : i32 -> tensor<64xi32>
  %ks = tt.splat %k : i32 -> tensor<64xi32>
  %pad0 = arith.addi %n, %m : i32
  %pad1 = arith.addi %pad0, %k : i32
  %pad2 = arith.addi %pad1, %cnt : i32
  %mask1 = arith.cmpi slt, %lane, %ns : tensor<64xi32>
  %mask2 = arith.cmpi slt, %lane, %ms : tensor<64xi32>
  %mask3 = arith.cmpi slt, %lane, %ks : tensor<64xi32>
  scf.for %i = %c0 to %cnt step %c1 : i32 {
    %v1 = tt.load %ps, %mask1 : tensor<64x!tt.ptr<f32>>
    %v2 = tt.load %ps, %mask2 : tensor<64x!tt.ptr<f32>>
    %v3 = tt.load %ps, %mask3 : tensor<64x!tt.ptr<f32>>
    %s = arith.addf %v1, %v2 : tensor<64xf32>
    %t = arith.addf %s, %v3 : tensor<64xf32>
    tt.store %ps, %t : tensor<64x!tt.ptr<f32>>
  }
  tt.return
}
