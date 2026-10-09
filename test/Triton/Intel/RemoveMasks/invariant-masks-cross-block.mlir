// RUN: triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s
// RUN: env TRITON_INTEL_SYMBOLIC_MASKS=1 triton-opt %s -split-input-file -triton-intel-remove-masks | FileCheck %s --check-prefix=SYM

// COM: Loop-invariant masks defined in different blocks: one before an enclosing
// COM: scf.if, the others inside it. They are still folded into one guard.
// COM: The symbolic guard compares in i64, the legacy one in i32. Its conjunction
// COM: must combine every comparison and be the scf.if condition.

// CHECK-LABEL: tt.func @two_masks_across_blocks
// SYM-LABEL:   tt.func @two_masks_across_blocks
tt.func @two_masks_across_blocks(%ptr: !tt.ptr<f32>, %n: i32, %m: i32, %cnt: i32, %c: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %n : i32 -> tensor<64xi32>
  %mask1 = arith.cmpi slt, %lane, %ns : tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // CHECK-DAG: arith.cmpi sgt, %arg1, %{{.*}} : i32
  // CHECK-DAG: arith.cmpi sgt, %arg2, %{{.*}} : i32
  // CHECK:     scf.if %{{.*}} {
  // CHECK:       scf.for
  // CHECK-NOT:     tt.load %{{.*}}, %{{.*}} :
  // CHECK:         tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:         tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:     } else {
  // CHECK:       scf.for
  // CHECK:         tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:         tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM-DAG:  [[C1:%[0-9]+]] = arith.cmpi {{[a-z]+}}, %{{.*}}, %{{.*}} : i64
  // SYM-DAG:  [[C2:%[0-9]+]] = arith.cmpi {{[a-z]+}}, %{{.*}}, %{{.*}} : i64
  // SYM-DAG:  [[AND:%[0-9]+]] = arith.andi [[C1]], [[C2]] : i1
  // SYM:      scf.if [[AND]] {
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  scf.if %c {
    %ms = tt.splat %m : i32 -> tensor<64xi32>
    %mask2 = arith.cmpi slt, %lane, %ms : tensor<64xi32>
    scf.for %i = %c0 to %cnt step %c1 : i32 {
      %v1 = tt.load %ps, %mask1 : tensor<64x!tt.ptr<f32>>
      %v2 = tt.load %ps, %mask2 : tensor<64x!tt.ptr<f32>>
      %s = arith.addf %v1, %v2 : tensor<64xf32>
      tt.store %ps, %s : tensor<64x!tt.ptr<f32>>
    }
  }
  tt.return
}

// -----

// COM: As above with three masks, two of them in the outer block.

// CHECK-LABEL: tt.func @three_masks_across_blocks
// SYM-LABEL:   tt.func @three_masks_across_blocks
tt.func @three_masks_across_blocks(%ptr: !tt.ptr<f32>, %n: i32, %m: i32, %k: i32, %cnt: i32, %c: i1) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %n : i32 -> tensor<64xi32>
  %ks = tt.splat %k : i32 -> tensor<64xi32>
  %mask1 = arith.cmpi slt, %lane, %ns : tensor<64xi32>
  %mask3 = arith.cmpi slt, %lane, %ks : tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  // CHECK-DAG: arith.cmpi sgt, %arg1, %{{.*}} : i32
  // CHECK-DAG: arith.cmpi sgt, %arg2, %{{.*}} : i32
  // CHECK-DAG: arith.cmpi sgt, %arg3, %{{.*}} : i32
  // CHECK:     scf.if %{{.*}} {
  // CHECK:       scf.for
  // CHECK-NOT:     tt.load %{{.*}}, %{{.*}} :
  // CHECK:         tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:         tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:         tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:     } else {
  // CHECK:       scf.for
  // CHECK:         tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:         tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // CHECK:         tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM-DAG:  [[C1:%[0-9]+]] = arith.cmpi {{[a-z]+}}, %{{.*}}, %{{.*}} : i64
  // SYM-DAG:  [[C2:%[0-9]+]] = arith.cmpi {{[a-z]+}}, %{{.*}}, %{{.*}} : i64
  // SYM-DAG:  [[C3:%[0-9]+]] = arith.cmpi {{[a-z]+}}, %{{.*}}, %{{.*}} : i64
  // SYM-DAG:  [[AND1:%[0-9]+]] = arith.andi [[C1]], [[C2]] : i1
  // SYM-DAG:  [[AND2:%[0-9]+]] = arith.andi [[AND1]], [[C3]] : i1
  // SYM:      scf.if [[AND2]] {
  // SYM:        scf.for
  // SYM-NOT:      tt.load %{{.*}}, %{{.*}} :
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{[^,]*}} : tensor<64x!tt.ptr<f32>>
  // SYM:      } else {
  // SYM:        scf.for
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
  // SYM:          tt.load %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>>
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
