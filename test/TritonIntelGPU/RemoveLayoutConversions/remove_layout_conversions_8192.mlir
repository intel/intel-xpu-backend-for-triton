// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions='max-backward-remat-iterations=10' | FileCheck %s
// RUN: env TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP=1 triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=FOR-LOOP

// COM: https://github.com/intel/intel-xpu-backend-for-triton/issues/8192
// COM: The convert of %x1 survives (its backward slice reaches a function argument) but is recorded as a remat.
// COM: Forward-propagating the %x0 convert rewrites math.exp, which must not reuse that convert: it is defined
// COM: after math.exp, so reusing it produced "operand #0 does not dominate this use".

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel(
  // CHECK: [[ADD:%.*]] = arith.addf
  // CHECK-NEXT: [[CVT:%.*]] = ttg.convert_layout [[ADD]] :
  // CHECK-NEXT: math.exp [[CVT]] :
  // CHECK: tt.return
  tt.func public @kernel(%y: tensor<128xf32, #blocked1>) -> (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %x0 = arith.sitofp %r : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
    %x1 = arith.addf %x0, %y : tensor<128xf32, #blocked1>
    %x2 = math.exp %x1 : tensor<128xf32, #blocked1>
    %a = ttg.convert_layout %x1 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    %b = ttg.convert_layout %x0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    %c = ttg.convert_layout %x2 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    tt.return %a, %b, %c : tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>
  }
}

// -----

// COM: Cross-region variant: the surviving convert of %x1 is in the then-branch of an scf.if (kept there by the
// COM: tt.store), and the forward-propagated math.exp is in the sibling else-branch, which it does not dominate.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_sibling_region(
  // CHECK: [[ADD:%.*]] = arith.addf
  // CHECK-NEXT: [[CVT:%.*]] = ttg.convert_layout [[ADD]] :
  // CHECK: scf.if
  // CHECK: } else {
  // CHECK-NEXT: math.exp [[CVT]] :
  // CHECK: tt.return
  tt.func public @kernel_sibling_region(%y: tensor<128xf32, #blocked1>, %p: !tt.ptr<f32>, %cond: i1) -> (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %x0 = arith.sitofp %r : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
    %x1 = arith.addf %x0, %y : tensor<128xf32, #blocked1>
    %rb = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %pp = tt.addptr %ps, %rb : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %z = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked1>
    %e = scf.if %cond -> (tensor<128xf32, #blocked1>) {
      %t = ttg.convert_layout %x1 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
      tt.store %pp, %t : tensor<128x!tt.ptr<f32>, #blocked>
      scf.yield %z : tensor<128xf32, #blocked1>
    } else {
      %x2 = math.exp %x1 : tensor<128xf32, #blocked1>
      scf.yield %x2 : tensor<128xf32, #blocked1>
    }
    %b = ttg.convert_layout %x0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    %c = ttg.convert_layout %e : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    tt.return %b, %c : tensor<128xf32, #blocked>, tensor<128xf32, #blocked>
  }
}

// -----

// COM: The rejected remat of %e#0 is replaced by a new convert, re-recording the same (value, encoding) pair.
// COM: Rewriting the scf.if (for %d) then remaps that pair; a duplicate encoding entry used to hit an assertion.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_rerecord_remat(
  // CHECK: [[CVT:%.*]] = ttg.convert_layout %arg0 :
  // CHECK-NEXT: arith.addf %{{.*}}, [[CVT]] :
  // CHECK: tt.return
  tt.func public @kernel_rerecord_remat(%y: tensor<128xf32, #blocked1>, %cond: i1) -> (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %x0 = arith.sitofp %r : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
    %e:2 = scf.if %cond -> (tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>) {
      scf.yield %y, %x0 : tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>
    } else {
      scf.yield %y, %x0 : tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>
    }
    %x2 = arith.addf %x0, %e#0 : tensor<128xf32, #blocked1>
    %a = ttg.convert_layout %e#0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    %b = ttg.convert_layout %x0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    %c = ttg.convert_layout %x2 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    %d = ttg.convert_layout %e#1 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    tt.return %a, %b, %c, %d : tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>
  }
}

// -----

// COM: Independent of the dominance fix: the two converts of %e#0 are in sibling branches, so neither dominates the
// COM: other and both are recorded as remats of the same (value, encoding) pair. Rewriting the first scf.if (for %d)
// COM: then remaps that pair; a duplicate encoding entry used to hit an assertion in updateRematMapping.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_duplicate_remat(
  // CHECK: [[X0:%.*]] = arith.sitofp
  // CHECK: scf.if
  // CHECK-NEXT: [[T1:%.*]] = ttg.convert_layout %arg0 :
  // CHECK-NEXT: tt.store %{{.*}}, [[T1]] :
  // CHECK: } else {
  // CHECK-NEXT: [[T2:%.*]] = ttg.convert_layout %arg0 :
  // CHECK-NEXT: tt.store %{{.*}}, [[T2]] :
  // CHECK: tt.return [[X0]] :
  tt.func public @kernel_duplicate_remat(%y: tensor<128xf32, #blocked1>, %p: !tt.ptr<f32>, %cond: i1, %c2: i1) -> tensor<128xf32, #blocked> {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %x0 = arith.sitofp %r : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
    %rb = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %ps = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %pp = tt.addptr %ps, %rb : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %e:2 = scf.if %cond -> (tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>) {
      scf.yield %y, %x0 : tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>
    } else {
      scf.yield %y, %x0 : tensor<128xf32, #blocked1>, tensor<128xf32, #blocked1>
    }
    scf.if %c2 {
      %t1 = ttg.convert_layout %e#0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
      tt.store %pp, %t1 : tensor<128x!tt.ptr<f32>, #blocked>
    } else {
      %t2 = ttg.convert_layout %e#0 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
      tt.store %pp, %t2 : tensor<128x!tt.ptr<f32>, #blocked>
    }
    %d = ttg.convert_layout %e#1 : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    tt.return %d : tensor<128xf32, #blocked>
  }
}

// -----

// COM: Same surviving remat of %x1 as above, reached through forwardPropagateRemat's tt.assert branch: the
// COM: assert must keep %x1 because the recorded remat is defined after it.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_assert(
  // CHECK: [[X1:%.*]] = arith.cmpi sgt, {{.*}} : tensor<128xi32, [[ENC:#[a-z0-9]+]]>
  // CHECK-NEXT: tt.assert [[X1]], "msg" : tensor<128xi1, [[ENC]]>
  // CHECK-NEXT: [[A:%.*]] = ttg.convert_layout [[X1]] :
  // CHECK: tt.return [[A]],
  tt.func public @kernel_assert(%y: tensor<128xi32, #blocked1>) -> (tensor<128xi1, #blocked>, tensor<128xi32, #blocked>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %x0 = arith.addi %r, %r : tensor<128xi32, #blocked1>
    %x1 = arith.cmpi sgt, %x0, %y : tensor<128xi32, #blocked1>
    tt.assert %x1, "msg" : tensor<128xi1, #blocked1>
    %a = ttg.convert_layout %x1 : tensor<128xi1, #blocked1> -> tensor<128xi1, #blocked>
    %b = ttg.convert_layout %x0 : tensor<128xi32, #blocked1> -> tensor<128xi32, #blocked>
    tt.return %a, %b : tensor<128xi1, #blocked>, tensor<128xi32, #blocked>
  }
}

// -----

// COM: As above, through the tt.descriptor_store branch: the store must keep %x1 as its source.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @kernel_desc_store(
  // CHECK: [[X1:%.*]] = arith.addi {{.*}}, %arg0 : tensor<128xi32, [[ENC:#[a-z0-9]+]]>
  // CHECK-NEXT: tt.descriptor_store %arg1[{{.*}}], [[X1]] : !tt.tensordesc<128xi32>, tensor<128xi32, [[ENC]]>
  // CHECK-NEXT: [[A:%.*]] = ttg.convert_layout [[X1]] :
  // CHECK: tt.return [[A]],
  tt.func public @kernel_desc_store(%y: tensor<128xi32, #blocked1>, %desc: !tt.tensordesc<128xi32>) -> (tensor<128xi32, #blocked>, tensor<128xi32, #blocked>) {
    %c0 = arith.constant 0 : i32
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %x0 = arith.addi %r, %r : tensor<128xi32, #blocked1>
    %x1 = arith.addi %x0, %y : tensor<128xi32, #blocked1>
    tt.descriptor_store %desc[%c0], %x1 : !tt.tensordesc<128xi32>, tensor<128xi32, #blocked1>
    %a = ttg.convert_layout %x1 : tensor<128xi32, #blocked1> -> tensor<128xi32, #blocked>
    %b = ttg.convert_layout %x0 : tensor<128xi32, #blocked1> -> tensor<128xi32, #blocked>
    tt.return %a, %b : tensor<128xi32, #blocked>, tensor<128xi32, #blocked>
  }
}

// -----

// COM: With for-loop support, removing %a records the remat of %res both on the old loop result and on the new one,
// COM: so remapping the old result into the new one merged a duplicate encoding. Removing %b then rewrites the new
// COM: loop again, and the duplicate used to hit an assertion in updateRematMapping.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [16], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // FOR-LOOP-LABEL: tt.func public @kernel_loop_remap_twice(
  // FOR-LOOP: [[LOOP:%.*]]:2 = scf.for
  // FOR-LOOP-NOT: ttg.convert_layout
  // FOR-LOOP: tt.store {{.*}}, [[LOOP]]#0 :
  // FOR-LOOP-NOT: ttg.convert_layout
  // FOR-LOOP: tt.store {{.*}}, [[LOOP]]#1 :
  tt.func public @kernel_loop_remap_twice(%lb: index, %ub: index, %st: index, %p: !tt.ptr<f32>) {
    %r = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %init = arith.sitofp %r : tensor<128xi32, #blocked1> to tensor<128xf32, #blocked1>
    %res = scf.for %i = %lb to %ub step %st iter_args(%acc = %init) -> (tensor<128xf32, #blocked1>) {
      %n = arith.addf %acc, %acc : tensor<128xf32, #blocked1>
      scf.yield %n : tensor<128xf32, #blocked1>
    }
    %ra = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %sa = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %pa = tt.addptr %sa, %ra : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
    %a = ttg.convert_layout %res : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked>
    tt.store %pa, %a : tensor<128x!tt.ptr<f32>, #blocked>
    %rb = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
    %sb = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
    %pb = tt.addptr %sb, %rb : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
    %b = ttg.convert_layout %res : tensor<128xf32, #blocked1> -> tensor<128xf32, #blocked2>
    tt.store %pb, %b : tensor<128x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}
