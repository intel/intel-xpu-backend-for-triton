// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute | FileCheck %s

// Test: a loop-carried non-accumulator iter_arg (%off, a descriptor-load
// offset advanced every iteration via arith.addi) must be replicated into
// BOTH distributed loops, with its own cloned arith.addi and the incremented
// value yielded -- not the stale %off passed through unchanged.
// Both loops must also start from the ORIGINAL init values: neither is chained
// onto the other's carried result, so the offset advances exactly once per
// iteration in each loop.
// CHECK-LABEL: @carried_offset
// CHECK-DAG: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[OFF_INIT:.*]] = arith.constant 0 : i32
// CHECK: scf.for {{.*}}iter_args(%{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[OFF_INIT]])
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   %[[OFFNEXT1:.*]] = arith.addi
// CHECK:   scf.yield %[[D0]], %{{.*}}, %[[OFFNEXT1]]
// CHECK: scf.for {{.*}}iter_args(%{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[OFF_INIT]])
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   %[[OFFNEXT2:.*]] = arith.addi
// CHECK:   scf.yield %{{.*}}, %[[D1]], %[[OFFNEXT2]]
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_offset(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    // %off is a loop-carried offset (NOT an accumulator) advanced every iteration.
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %off = %c0_i32) -> (tensor<128x128xf32>, tensor<128x128xf32>, i32) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %off] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%off, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%off, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %off_next = arith.addi %off, %c64_i32 : i32
      scf.yield %d0, %d1, %off_next : tensor<128x128xf32>, tensor<128x128xf32>, i32
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: a pure but unrelated nested scf.for loop in the body is itself a
// pure, dot-independent, accumulator-independent loop-carried chain (yielded
// to the carried, non-accumulator index 2) -- just like %off_next in
// @carried_offset above, it must be replicated (cloned) into BOTH
// distributed loops rather than frozen or rejected. Each new loop gets its
// own copy of the inner scf.for and yields that copy's result at index 2.
// Because the carried value (%s, a running sum) is also a function result, the
// wiring is checked too: BOTH loops must start the sum from the original init
// (neither is chained onto the other's result) and the returned sum must come
// from the first loop -- i.e. the sum is accumulated once, not double counted.
// CHECK-LABEL: @nested_pure_loop
// CHECK-DAG: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[S_INIT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[LOOP1:.*]]:3 = scf.for {{.*}}iter_args(%{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[S_INIT]])
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   %[[INNER1:.*]] = scf.for
// CHECK:     arith.addf
// CHECK:     scf.yield
// CHECK:   scf.yield %[[D0]], %{{.*}}, %[[INNER1]]
// CHECK: %[[LOOP2:.*]]:3 = scf.for {{.*}}iter_args(%{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[S_INIT]])
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   %[[INNER2:.*]] = scf.for
// CHECK:     arith.addf
// CHECK:     scf.yield
// CHECK:   scf.yield %{{.*}}, %[[D1]], %[[INNER2]]
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1, %[[LOOP1]]#2
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @nested_pure_loop(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>, f32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %f0 = arith.constant 0.000000e+00 : f32
    %f1 = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %s = %f0) -> (tensor<128x128xf32>, tensor<128x128xf32>, f32) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // A pure inner loop, unrelated to either dot.
      %inner = scf.for %j = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%t = %s) -> (f32) : i32 {
        %n = arith.addf %t, %f1 : f32
        scf.yield %n : f32
      }
      scf.yield %d0, %d1, %inner : tensor<128x128xf32>, tensor<128x128xf32>, f32
    }
    tt.return %0#0, %0#1, %0#2 : tensor<128x128xf32>, tensor<128x128xf32>, f32
  }
}

// -----

// Test: canonical pointer-advance GEMM shape. Operand A is shared by both
// dots and its pointer tensor is carried across iterations via a
// loop-carried tt.addptr (never re-derived from the induction variable, and
// only consumed by the yield). Both distributed loops must contain their own
// cloned tt.addptr and yield the advanced pointer, not the stale one.
// Both loops must also start from the ORIGINAL base pointer tensor, so each
// advances it exactly once per iteration.
// CHECK-LABEL: @carried_ptr_gemm
// CHECK-DAG: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG: %[[PTR_INIT:.*]] = tt.splat %arg0
// CHECK: scf.for {{.*}}iter_args(%{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[PTR_INIT]])
// CHECK:   %[[X1:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<bf16>>
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]]
// CHECK-NOT: tt.dot
// CHECK:   %[[ANEXT1:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<128x64x!tt.ptr<bf16>>
// CHECK:   scf.yield %[[D0]], %{{.*}}, %[[ANEXT1]]
// CHECK: scf.for {{.*}}iter_args(%{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[ACC_INIT]], %{{[^ ]+}} = %[[PTR_INIT]])
// CHECK:   %[[X2:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<bf16>>
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]]
// CHECK-NOT: tt.dot
// CHECK:   %[[ANEXT2:.*]] = tt.addptr %{{.*}}, %{{.*}} : tensor<128x64x!tt.ptr<bf16>>
// CHECK:   scf.yield %{{.*}}, %[[D1]], %[[ANEXT2]]
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_ptr_gemm(%a_ptr: !tt.ptr<bf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %a_off = arith.constant dense<64> : tensor<128x64xi32>
    %a_splat = tt.splat %a_ptr : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %a_ptrs = %a_splat) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x64x!tt.ptr<bf16>>) : i32 {
      %x = tt.load %a_ptrs : tensor<128x64x!tt.ptr<bf16>>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // %a_ptrs is only ever advanced here and consumed by the yield -- never
      // re-derived from %k.
      %a_next = tt.addptr %a_ptrs, %a_off : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
      scf.yield %d0, %d1, %a_next : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x64x!tt.ptr<bf16>>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: a carried, non-accumulator chain (%carry, index 2) reads dot0's
// accumulator block argument (%acc_g). Since it reads exactly ONE
// accumulator, index 2 is OWNED by dot0's loop rather than rejected: the
// chain is cloned into the first loop only, the second loop FREEZES index 2
// (yields its own block argument unchanged so the slot is inert there), and
// the function result for index 2 is wired from the first loop.
//
// Replicating the chain into both loops -- the treatment every other carried
// chain in this file gets -- would be wrong here: the second loop's %acc_g
// slot is a frozen pass-through, so its copy would read the init value.
// CHECK-LABEL: @carried_reads_own_acc_distribute
// CHECK: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[LOOP1:.*]]:3 = scf.for {{.*}}iter_args(%[[L1_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L1_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L1_CARRY:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]], %[[L1_ACCG]]
// CHECK-NOT: tt.dot
// CHECK:   %[[NEXT:.*]] = arith.addf %[[L1_CARRY]], %[[L1_ACCG]]
// CHECK:   scf.yield %[[D0]], %[[L1_ACCFC]], %[[NEXT]]
// CHECK: %[[LOOP2:.*]]:3 = scf.for {{.*}}iter_args(%[[L2_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L2_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L2_CARRY:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]], %[[L2_ACCFC]]
// CHECK-NOT: tt.dot
// The owned chain is NOT replicated here; index 2 is frozen instead.
// CHECK-NOT: arith.addf
// CHECK:   scf.yield %[[L2_ACCG]], %[[D1]], %[[L2_CARRY]]
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1, %[[LOOP1]]#2
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_reads_own_acc_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %carry = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // %carry's update reads dot0's accumulator block argument directly.
      %next = arith.addf %carry, %acc_g : tensor<128x128xf32>
      scf.yield %d0, %d1, %next : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1, %0#2 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: the mirror image of @carried_reads_own_acc_distribute -- the carried
// chain reads dot1's accumulator (%acc_fc), so index 2 is owned by the SECOND
// loop. This direction is the one that actually catches a broken ownership
// analysis: the pre-existing wiring took every carried result from the first
// loop, so index 2 must now be taken from %[[LOOP2]], not %[[LOOP1]].
// CHECK-LABEL: @carried_reads_dot1_acc_distribute
// CHECK: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[LOOP1:.*]]:3 = scf.for {{.*}}iter_args(%[[L1_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L1_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L1_CARRY:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]], %[[L1_ACCG]]
// CHECK-NOT: tt.dot
// Index 2 is frozen in this loop -- no cloned arith.addf.
// CHECK-NOT: arith.addf
// CHECK:   scf.yield %[[D0]], %[[L1_ACCFC]], %[[L1_CARRY]]
// CHECK: %[[LOOP2:.*]]:3 = scf.for {{.*}}iter_args(%[[L2_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L2_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L2_CARRY:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]], %[[L2_ACCFC]]
// CHECK-NOT: tt.dot
// CHECK:   %[[NEXT:.*]] = arith.addf %[[L2_CARRY]], %[[L2_ACCFC]]
// CHECK:   scf.yield %[[L2_ACCG]], %[[D1]], %[[NEXT]]
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1, %[[LOOP2]]#2
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_reads_dot1_acc_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %carry = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // %carry's update reads dot1's accumulator block argument directly.
      %next = arith.addf %carry, %acc_fc : tensor<128x128xf32>
      scf.yield %d0, %d1, %next : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1, %0#2 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: the carried slot's yielded value IS an accumulator block argument, with
// no intervening op -- %lag holds dot1's accumulator delayed by one iteration.
// There is no chain to slice here (a backward slice rooted at a block argument
// is empty), so ownership must be decided from the yielded value itself:
// index 2 is owned by dot1's loop.
//
// This is the case that a slice-only test cannot see, and getting it wrong is
// silently wrong rather than a crash: classifying index 2 as replicable would
// take its result from the FIRST loop, where the %acc_fc slot is a frozen
// pass-through, so the function would return the init value instead of the
// second-to-last accumulator. Hence dot1 -- not dot0 -- is the one to use.
// CHECK-LABEL: @carried_is_dot1_acc_distribute
// CHECK: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[LOOP1:.*]]:3 = scf.for {{.*}}iter_args(%[[L1_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L1_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L1_LAG:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]], %[[L1_ACCG]]
// CHECK-NOT: tt.dot
// Index 2 is frozen here: it yields its OWN block argument, not %[[L1_ACCFC]].
// CHECK:   scf.yield %[[D0]], %[[L1_ACCFC]], %[[L1_LAG]]
// CHECK: %[[LOOP2:.*]]:3 = scf.for {{.*}}iter_args(%[[L2_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L2_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L2_LAG:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]], %[[L2_ACCFC]]
// CHECK-NOT: tt.dot
// Index 2 is live here: it yields THIS loop's live %acc_fc block argument.
// CHECK:   scf.yield %[[L2_ACCG]], %[[D1]], %[[L2_ACCFC]]
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1, %[[LOOP2]]#2
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_is_dot1_acc_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:3 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %lag = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // No op in between: the yielded value IS dot1's accumulator block argument.
      scf.yield %d0, %d1, %acc_fc : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1, %0#2 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// Test: an OWNED carried chain (index 2, reads %acc_g) and a REPLICABLE one
// (index 3, reads only its own slot) coexist and share a pure sub-chain
// (arith.sitofp/tt.splat). The shared ops go into both loops because the
// replicable chain needs them there; only the acc-reading arith.addf is
// confined to the first loop. Index 2's result comes from its owner and index
// 3's from the first loop as before.
// CHECK-LABEL: @carried_mixed_owners_distribute
// CHECK: %[[ACC_INIT:.*]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK: %[[LOOP1:.*]]:4 = scf.for {{.*}}iter_args(%[[L1_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L1_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L1_OWNED:[^ ]+]] = %[[ACC_INIT]], %[[L1_SHARED:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X1:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WG:.*]] = tt.descriptor_load %arg1
// CHECK:   %[[D0:.*]] = tt.dot %[[X1]], %[[WG]], %[[L1_ACCG]]
// CHECK-NOT: tt.dot
// CHECK:   %[[KF1:.*]] = arith.sitofp
// CHECK:   %[[SPLAT1:.*]] = tt.splat %[[KF1]]
// CHECK:   %[[OWNED:.*]] = arith.addf %[[SPLAT1]], %[[L1_ACCG]]
// CHECK:   %[[SHARED1:.*]] = arith.subf %[[SPLAT1]], %[[L1_SHARED]]
// CHECK:   scf.yield %[[D0]], %[[L1_ACCFC]], %[[OWNED]], %[[SHARED1]]
// CHECK: %[[LOOP2:.*]]:4 = scf.for {{.*}}iter_args(%[[L2_ACCG:[^ ]+]] = %[[ACC_INIT]], %[[L2_ACCFC:[^ ]+]] = %[[ACC_INIT]], %[[L2_OWNED:[^ ]+]] = %[[ACC_INIT]], %[[L2_SHARED:[^ ]+]] = %[[ACC_INIT]])
// CHECK:   %[[X2:.*]] = tt.descriptor_load %arg0
// CHECK:   %[[WFC:.*]] = tt.descriptor_load %arg2
// CHECK:   %[[D1:.*]] = tt.dot %[[X2]], %[[WFC]], %[[L2_ACCFC]]
// CHECK-NOT: tt.dot
// The shared pure sub-chain IS replicated here, for index 3's sake...
// CHECK:   %[[KF2:.*]] = arith.sitofp
// CHECK:   %[[SPLAT2:.*]] = tt.splat %[[KF2]]
// ...but the acc-reading arith.addf is not, and index 2 is frozen.
// CHECK-NOT: arith.addf
// CHECK:   %[[SHARED2:.*]] = arith.subf %[[SPLAT2]], %[[L2_SHARED]]
// CHECK:   scf.yield %[[L2_ACCG]], %[[D1]], %[[L2_OWNED]], %[[SHARED2]]
// CHECK: tt.return %[[LOOP1]]#0, %[[LOOP2]]#1, %[[LOOP1]]#2, %[[LOOP1]]#3
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @carried_mixed_owners_distribute(%arg0: !tt.tensordesc<128x64xbf16>, %arg1: !tt.tensordesc<64x128xbf16>, %arg2: !tt.tensordesc<64x128xbf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:4 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc_g = %cst, %acc_fc = %cst, %owned = %cst, %shared_carry = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %x = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16>
      %wg = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %wfc = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16>
      %d0 = tt.dot %x, %wg, %acc_g, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      %d1 = tt.dot %x, %wfc, %acc_fc, inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x128xbf16> -> tensor<128x128xf32>
      // Pure sub-chain feeding BOTH carried chains below.
      %kf = arith.sitofp %k : i32 to f32
      %shared = tt.splat %kf : f32 -> tensor<128x128xf32>
      // Index 2: reads %acc_g, so owned by dot0's loop.
      %next_owned = arith.addf %shared, %acc_g : tensor<128x128xf32>
      // Index 3: reads no accumulator, so replicable into both loops.
      %next_shared = arith.subf %shared, %shared_carry : tensor<128x128xf32>
      scf.yield %d0, %d1, %next_owned, %next_shared : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1, %0#2, %0#3 : tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>, tensor<128x128xf32>
  }
}
