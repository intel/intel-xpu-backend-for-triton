// REQUIRES: census-debug
// RUN: env TRITON_INTEL_SYMBOLIC_MASKS=1 triton-opt %s -triton-intel-remove-masks -debug-only=triton-intel-remove-masks-census -o /dev/null 2>&1 | FileCheck %s

// COM: The second loop's guard condition names the first loop's result, and
// COM: the trace prints it when that loop is versioned. Versioning the first
// COM: loop erases that value, so the driver must version in reverse program
// COM: order: forward order makes the trace read the erased value.

// CHECK: versioned: loop=dependent_loops#{{[^/]*}}/L1 {{.*}} guard=opaque(scf.for) divisible by 64
// CHECK: versioned: loop=dependent_loops#{{[^/]*}}/L0 {{.*}} guard=arg1 divisible by 64

tt.func @dependent_loops(%ptr: !tt.ptr<f32>, %rnumel: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c64 = arith.constant 64 : i32
  %lane = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %ns = tt.splat %rnumel : i32 -> tensor<64xi32>
  %ps = tt.splat %ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
  %n2 = scf.for %r = %c0 to %rnumel step %c64 iter_args(%a = %rnumel) -> (i32) : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %mask : tensor<64x!tt.ptr<f32>>
    %a2 = arith.addi %a, %c1 : i32
    scf.yield %a2 : i32
  }
  %ns2 = tt.splat %n2 : i32 -> tensor<64xi32>
  scf.for %r = %c0 to %n2 step %c64 : i32 {
    %rs = tt.splat %r : i32 -> tensor<64xi32>
    %idx = arith.addi %rs, %lane : tensor<64xi32>
    %mask = arith.cmpi slt, %idx, %ns2 : tensor<64xi32>
    %p = tt.addptr %ps, %idx : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %v = tt.load %p, %mask : tensor<64x!tt.ptr<f32>>
    tt.store %p, %v, %mask : tensor<64x!tt.ptr<f32>>
    scf.yield
  }
  tt.return
}
