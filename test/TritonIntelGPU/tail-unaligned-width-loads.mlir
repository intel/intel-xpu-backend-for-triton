// RUN: triton-opt %s -split-input-file --tritonintelgpu-tail-unaligned-width-loads | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-tail-unaligned-width-loads --tritonintelgpu-materialize-block-pointer | FileCheck %s --check-prefix=MBP
// RUN: triton-opt %s -split-input-file --tritonintelgpu-tail-unaligned-width-loads --canonicalize | FileCheck %s --check-prefix=FOLD

// COM: The MBP prefix checks that every load through a prefix descriptor is marked for block IO.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#b1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func public @fp16_one_column(
  // CHECK-SAME: %[[BASE:.*]]: !tt.ptr<f16>
  tt.func public @fp16_one_column(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %x: i32 {tt.divisibility = 32 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    // CHECK: %[[FULL:.*]] = tt.make_tensor_descriptor %[[BASE]], [%c64_i32, %c67_i32]
    // CHECK: %[[PREFIX:.*]] = tt.make_tensor_descriptor %[[BASE]], [%c64_i32, %c66_i32]
    // CHECK: %[[TILE:.*]] = tt.descriptor_load %[[PREFIX]][%c0_i32, %arg2]
    // CHECK: scf.if
    // CHECK:   tt.load {{.*}} {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_last>}
    // CHECK-NOT: ttg.convert_layout
    // CHECK:   scf.yield
    // CHECK: } else {
    // CHECK: arith.select {{.*}}, %[[TILE]] : tensor<64x32xi1, #blocked>, tensor<64x32xf16, #blocked>
    // CHECK-NOT: arith.select
    // COM: Other users of the descriptor are untouched.
    // CHECK: tt.descriptor_store %[[FULL]]
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %1 = tt.descriptor_load %0[%c0_i32, %x] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_last>} : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    tt.descriptor_store %0[%c0_i32, %x], %1 : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
    tt.return
  }

  // COM: int8 drops two and three columns; an si8 descriptor over an i8 pointer needs no bitcast.
  // CHECK-LABEL: tt.func public @int8(
  tt.func public @int8(%base: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %x: i32 {tt.divisibility = 64 : i32}, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c66_i32 = arith.constant 66 : i32
    %c67_i32 = arith.constant 67 : i32
    // CHECK: %[[PREFIX:.*]] = tt.make_tensor_descriptor {{.*}}, [%c64_i32, %c64_i32_0]
    // CHECK: tt.descriptor_load %[[PREFIX]]
    // CHECK: scf.if {{.*}} -> (tensor<64xi8{{.*}}, tensor<64xi8
    // CHECK-NOT: tt.bitcast
    // CHECK-COUNT-2: arith.select
    // CHECK-NOT: arith.select
    // CHECK: %[[PREFIX3:.*]] = tt.make_tensor_descriptor {{.*}}, [%c64_i32, %c64_i32_4]
    // CHECK: scf.for
    // CHECK:   tt.descriptor_load %[[PREFIX3]]
    // CHECK-NOT: arith.select
    // CHECK: scf.for
    // CHECK:   tt.descriptor_load %[[PREFIX3]]
    // CHECK-COUNT-3: arith.select
    // CHECK: scf.for
    // CHECK:   tt.descriptor_load %[[PREFIX3]]
    // CHECK-NOT: arith.select
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c66_i32], [%pitch, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<64x64xsi8, #blocked>
    %1 = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x64xsi8, #blocked> -> tensor<64x64xi8, #blocked>
    %2 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<i8>, !tt.tensordesc<64x64xi8, #blocked>
    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %o = arith.muli %i, %c64_i32 : i32
      %t = tt.descriptor_load %2[%c0_i32, %o] : !tt.tensordesc<64x64xi8, #blocked> -> tensor<64x64xi8, #blocked>
    }
    tt.return
  }

  // COM: Left alone: rounded width below 64 bytes, padding not zero, width not constant or already aligned, packed fp4,
  // COM: an index MaterializeBlockPointer cannot prove aligned (also in a loop), a tile that is not the block shape.
  // CHECK-LABEL: tt.func public @not_matched(
  tt.func public @not_matched(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %k: i32, %fp4: !tt.ptr<!tt.fp4e2m1x2> {tt.divisibility = 16 : i32}, %unaligned: i32, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c31_i32 = arith.constant 31 : i32
    %c64_i32 = arith.constant 64 : i32
    %c66_i32 = arith.constant 66 : i32
    %c67_i32 = arith.constant 67 : i32
    // CHECK-NOT: scf.if
    // CHECK-NOT: tt.load
    // CHECK-COUNT-8: tt.descriptor_load {{.*}} : !tt.tensordesc
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c31_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %1 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    %2 = tt.make_tensor_descriptor %base, [%c64_i32, %k], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %3 = tt.descriptor_load %2[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    %4 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] {padding = 2 : i32} : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %5 = tt.descriptor_load %4[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    %6 = tt.make_tensor_descriptor %base, [%c64_i32, %c66_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %7 = tt.descriptor_load %6[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    %8 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %9 = tt.descriptor_load %8[%c0_i32, %unaligned] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    %10 = tt.make_tensor_descriptor %fp4, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<!tt.fp4e2m1x2>, !tt.tensordesc<64x64x!tt.fp4e2m1x2, #blocked>
    %11 = tt.descriptor_load %10[%c0_i32, %c0_i32] : !tt.tensordesc<64x64x!tt.fp4e2m1x2, #blocked> -> tensor<64x64x!tt.fp4e2m1x2, #blocked>
    %12 = tt.descriptor_load %8[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<32x64xf16, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c33_i32 = arith.constant 33 : i32
    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %x = arith.muli %i, %c33_i32 : i32
      %t = tt.descriptor_load %8[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    }
    tt.return
  }

  // COM: Offsets: runtime bounds with a chained accumulator, an iteration argument, decreasing with a step above one,
  // COM: two loads whose hull skips the one that never reaches, and a tensor iteration argument.
  // CHECK-LABEL: tt.func public @split_forms(
  // MBP-LABEL: tt.func public @split_forms(
  tt.func public @split_forms(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %start: i32, %end: i32, %n: i32) -> tensor<64x32xf32, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c11_i32 = arith.constant 11 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    %c200_i32 = arith.constant 200 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked>
    %t0 = arith.constant dense<0> : tensor<1xi32, #b1>
    %t32 = arith.constant dense<32> : tensor<1xi32, #b1>
    // CHECK: %[[FULL:.*]] = tt.make_tensor_descriptor {{.*}}, [%c64_i32, %c67_i32]
    // CHECK: %[[W:.*]] = arith.constant 66 : i32
    // CHECK: %[[PREFIX:.*]] = tt.make_tensor_descriptor {{.*}}, [%c64_i32, %[[W]]]
    // CHECK-NOT: tt.descriptor_load %[[FULL]]
    // CHECK: %[[A:.*]] = scf.for
    // CHECK:   tt.descriptor_load %[[PREFIX]]
    // CHECK-NOT: scf.if
    // CHECK-NOT: arith.select {{.*}}tensor<64x32xi1
    // CHECK: %[[B:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[A]])
    // CHECK:   %[[TILE:.*]] = tt.descriptor_load %[[PREFIX]]
    // CHECK-NOT: scf.if
    // CHECK:   tt.load
    // CHECK:   arith.select {{.*}}, %[[TILE]]
    // CHECK:   scf.yield
    // CHECK: %[[C:.*]] = scf.for {{.*}} iter_args({{.*}} = %[[B]])
    // CHECK:   tt.descriptor_load %[[PREFIX]]
    // CHECK-NOT: arith.select {{.*}}tensor<64x32xi1
    // CHECK: scf.for {{.*}} -> (tensor<64x32xf32, #blocked>, i32)
    // CHECK: scf.for {{.*}} iter_args({{.*}} = %{{.*}}#0, %{{.*}} = %{{[^#]*}})
    // CHECK-COUNT-7: scf.for
    // CHECK-NOT: scf.for
    // CHECK: tt.return
    // MBP-COUNT-15: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %r0 = scf.for %i = %start to %end step %c1_i32 iter_args(%acc = %zero) -> (tensor<64x32xf32, #blocked>) : i32 {
      %x = arith.muli %i, %c32_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      %e = arith.extf %t : tensor<64x32xf16, #blocked> to tensor<64x32xf32, #blocked>
      %s = arith.addf %acc, %e : tensor<64x32xf32, #blocked>
      scf.yield %s : tensor<64x32xf32, #blocked>
    }
    %r1:2 = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %r0, %off = %c0_i32) -> (tensor<64x32xf32, #blocked>, i32) : i32 {
      %t = tt.descriptor_load %0[%c0_i32, %off] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      %e = arith.extf %t : tensor<64x32xf16, #blocked> to tensor<64x32xf32, #blocked>
      %s = arith.addf %acc, %e : tensor<64x32xf32, #blocked>
      %next = arith.addi %off, %c32_i32 : i32
      scf.yield %s, %next : tensor<64x32xf32, #blocked>, i32
    }
    %1 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    scf.for %i = %c3_i32 to %c11_i32 step %c2_i32 : i32 {
      %m = arith.muli %i, %c32_i32 : i32
      %x = arith.subi %c200_i32, %m : i32
      %t = tt.descriptor_load %1[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    }
    // COM: A tensor iteration argument is no seed and no offset.
    scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%v = %t0) -> (tensor<1xi32, #b1>) : i32 {
      %x = arith.muli %i, %c32_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      %u0 = tt.unsplat %v : tensor<1xi32, #b1>
      %u = arith.muli %u0, %c32_i32 : i32
      %t2 = tt.descriptor_load %0[%c0_i32, %u] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      %vn = arith.addi %v, %t32 : tensor<1xi32, #b1>
      scf.yield %vn : tensor<1xi32, #b1>
    }
    tt.return %r1#0 : tensor<64x32xf32, #blocked>
  }

  // COM: The second load never reaches a dropped column and does not widen the middle loop: iterations [2, 3) of 10.
  // FOLD-LABEL: tt.func public @hull(
  tt.func public @hull(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    %c4096_i32 = arith.constant 4096 : i32
    // FOLD: scf.for %{{.*}} = %c0_i32 to %c2_i32
    // FOLD-NOT: scf.for
    // FOLD-COUNT-2: arith.select
    // FOLD: scf.for %{{.*}} = %c3_i32 to %c10_i32
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %1 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    scf.for %i = %c0_i32 to %c10_i32 step %c1_i32 : i32 {
      %x = arith.muli %i, %c32_i32 : i32
      %y = arith.addi %x, %c4096_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      %u = tt.descriptor_load %1[%c0_i32, %y] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      tt.descriptor_store %0[%c0_i32, %x], %t : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
      tt.descriptor_store %1[%c0_i32, %y], %u : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
    }
    tt.return
  }

  // COM: The second loop continues the first loop's tile counter; each merges at x = 64.
  // FOLD-LABEL: tt.func public @chain_iter(
  tt.func public @chain_iter(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    // FOLD: scf.for
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c64_i32]
    // FOLD: tt.load
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c96_i32]
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c64_i32]
    // FOLD: tt.load
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c96_i32]
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %r0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%k = %c0_i32) -> (i32) : i32 {
      %x = arith.muli %k, %c32_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      tt.descriptor_store %0[%c0_i32, %x], %t : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
      %kn = arith.addi %k, %c1_i32 : i32
      scf.yield %kn : i32
    }
    %r1 = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%k = %r0) -> (i32) : i32 {
      %x = arith.muli %k, %c32_i32 : i32
      %y = arith.subi %x, %c64_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %y] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      tt.descriptor_store %0[%c0_i32, %y], %t : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
      %kn = arith.addi %k, %c1_i32 : i32
      scf.yield %kn : i32
    }
    tt.return
  }

  // COM: The second loop starts at the first loop's result.
  // FOLD-LABEL: tt.func public @chain_lb(
  tt.func public @chain_lb(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    // FOLD: scf.for
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c64_i32]
    // FOLD: tt.load
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c96_i32]
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c64_i32]
    // FOLD: tt.load
    // FOLD: tt.descriptor_load {{.*}}[%c0_i32, %c96_i32]
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %r0 = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%k = %c0_i32) -> (i32) : i32 {
      %x = arith.muli %i, %c32_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      tt.descriptor_store %0[%c0_i32, %x], %t : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
      %kn = arith.addi %k, %c1_i32 : i32
      scf.yield %kn : i32
    }
    scf.for %j = %r0 to %c6_i32 step %c1_i32 : i32 {
      %x = arith.muli %j, %c32_i32 : i32
      %y = arith.subi %x, %c64_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %y] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      tt.descriptor_store %0[%c0_i32, %y], %t : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
    }
    tt.return
  }

  // COM: Only the innermost loop is split.
  // CHECK-LABEL: tt.func public @nested(
  tt.func public @nested(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %n: i32) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    // CHECK: scf.for
    // CHECK-COUNT-3: scf.for
    // CHECK-NOT: scf.for
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    scf.for %j = %c0_i32 to %n step %c1_i32 : i32 {
      %o = arith.muli %j, %c32_i32 : i32
      %t0 = tt.descriptor_load %0[%c0_i32, %o] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
        %x = arith.muli %i, %c32_i32 : i32
        %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      }
    }
    tt.return
  }

  // COM: Not affine, may trap or may wrap: the loop stays whole and the load merges under a branch.
  // CHECK-LABEL: tt.func public @not_affine(
  tt.func public @not_affine(%base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %n: i32, %a: i32, %d: i32, %n8: i8) {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %c32_i8 = arith.constant 32 : i8
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c67_i32 = arith.constant 67 : i32
    // CHECK: scf.for
    // CHECK:   scf.if
    // CHECK:   arith.select
    // CHECK: scf.for
    // CHECK:   scf.if
    // CHECK:     arith.divsi
    // CHECK:     tt.descriptor_load
    // CHECK: scf.for
    // CHECK:   scf.if
    // CHECK:   arith.select
    // CHECK-NOT: scf.for
    %0 = tt.make_tensor_descriptor %base, [%c64_i32, %c67_i32], [%pitch, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      %sq = arith.muli %i, %i : i32
      %x = arith.muli %sq, %c32_i32 : i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    }
    // COM: The division runs only when d is not zero.
    %nz = arith.cmpi ne, %d, %c0_i32 : i32
    scf.for %i = %c0_i32 to %n step %c1_i32 : i32 {
      scf.if %nz {
        %q = arith.divsi %a, %d : i32
        %qm = arith.muli %q, %c32_i32 : i32
        %m = arith.muli %i, %c32_i32 : i32
        %x = arith.addi %m, %qm : i32
        %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      }
    }
    // COM: The offset wraps in 8 bits.
    scf.for %i = %c0_i8 to %n8 step %c1_i8 : i8 {
      %x8 = arith.muli %i, %c32_i8 : i8
      %x = arith.extsi %x8 : i8 to i32
      %t = tt.descriptor_load %0[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    }
    tt.return
  }
}

// -----

// COM: A load feeding a DPAS operand: the columns go through a sub-group shuffle, no shared memory.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_subgroup_matrix_multiply_accumulate} {
  // CHECK-LABEL: tt.func public @dpas_gemm(
  // MBP-LABEL: tt.func public @dpas_gemm(
  tt.func public @dpas_gemm(%a: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<64x32xf32, #dpas> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %K = arith.constant 7347 : i32
    %lda = arith.constant 7424 : i64
    %ldb = arith.constant 32 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %da = tt.make_tensor_descriptor %a, [%c64_i32, %K], [%lda, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %db = tt.make_tensor_descriptor %b, [%K, %c32_i32], [%ldb, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<32x32xf16, #blocked>
    // CHECK: scf.for
    // CHECK-NOT: tt.load
    // CHECK: scf.for
    // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f16>, #linear>
    // CHECK: ttg.convert_layout {{.*}} : tensor<64xf16, #linear> -> tensor<64xf16, #ttg.slice<{dim = 1, parent = #ttg.dot_op<
    // CHECK: scf.for
    // CHECK-NOT: tt.load
    // MBP-COUNT-6: tt.descriptor_load {{.*}} {ttig.block_io = "row_major"
    %r = scf.for %k = %c0_i32 to %K step %c32_i32 iter_args(%acc = %zero) -> (tensor<64x32xf32, #dpas>) : i32 {
      %ta = tt.descriptor_load %da[%c0_i32, %k] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
      %tb = tt.descriptor_load %db[%k, %c0_i32] : !tt.tensordesc<32x32xf16, #blocked> -> tensor<32x32xf16, #blocked>
      %ca = ttg.convert_layout %ta : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %cb = ttg.convert_layout %tb : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dot1>
      %d = tt.dot %ca, %cb, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<64x32xf32, #dpas>
      scf.yield %d : tensor<64x32xf32, #dpas>
    }
    tt.return %r : tensor<64x32xf32, #dpas>
  }

  // COM: The columns are read at the load, before a store to the same tile.
  // CHECK-LABEL: tt.func public @store_between(
  tt.func public @store_between(%a: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %x: i32 {tt.divisibility = 32 : i32}, %tb: tensor<32x32xf16, #blocked>) -> tensor<64x32xf32, #dpas> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %K = arith.constant 67 : i32
    %lda = arith.constant 128 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %zt = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #blocked>
    // CHECK: tt.descriptor_load
    // CHECK: tt.load
    // CHECK: tt.descriptor_store
    // CHECK: ttg.convert_layout
    // CHECK: arith.select
    %da = tt.make_tensor_descriptor %a, [%c64_i32, %K], [%lda, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %ta = tt.descriptor_load %da[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    tt.descriptor_store %da[%c0_i32, %x], %zt : !tt.tensordesc<64x32xf16, #blocked>, tensor<64x32xf16, #blocked>
    %ca = ttg.convert_layout %ta : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
    %cb = ttg.convert_layout %tb : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dot1>
    %d = tt.dot %ca, %cb, %zero, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<64x32xf32, #dpas>
    tt.return %d : tensor<64x32xf32, #dpas>
  }

  // COM: A dot_op convert inside a loop does not repeat the merge.
  // CHECK-LABEL: tt.func public @convert_in_loop(
  tt.func public @convert_in_loop(%a: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %x: i32 {tt.divisibility = 32 : i32}, %tb: tensor<32x32xf16, #blocked>, %n: i32) -> tensor<64x32xf32, #dpas> {
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %K = arith.constant 67 : i32
    %lda = arith.constant 128 : i64
    %zero = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    // CHECK: tt.load
    // CHECK: scf.for
    // CHECK-NOT: scf.if
    // CHECK-NOT: tt.load
    %da = tt.make_tensor_descriptor %a, [%c64_i32, %K], [%lda, %c1_i64] : !tt.ptr<f16>, !tt.tensordesc<64x32xf16, #blocked>
    %ta = tt.descriptor_load %da[%c0_i32, %x] : !tt.tensordesc<64x32xf16, #blocked> -> tensor<64x32xf16, #blocked>
    %r = scf.for %i = %c0_i32 to %n step %c1_i32 iter_args(%acc = %zero) -> (tensor<64x32xf32, #dpas>) : i32 {
      %ca = ttg.convert_layout %ta : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #dot0>
      %cb = ttg.convert_layout %tb : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #dot1>
      %d = tt.dot %ca, %cb, %acc, inputPrecision = tf32 : tensor<64x32xf16, #dot0> * tensor<32x32xf16, #dot1> -> tensor<64x32xf32, #dpas>
      scf.yield %d : tensor<64x32xf32, #dpas>
    }
    tt.return %r : tensor<64x32xf32, #dpas>
  }
}
