// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=ttig.2d_block_load

// COM: tt.descriptor_load that cannot use a 2D block load is lowered to a masked tt.load.

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
// COM: Shapes not divisible by the block: per-element bounds on each dim, NaN fill.
  // CHECK-LABEL: tt.func @per_element_pad_nan(
  // CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f16>, %[[M:[^:]+]]: i32, %[[N:[^:]+]]: i32, %[[STRIDE:[^:]+]]: i64, %[[I:[^:]+]]: i32, %[[J:[^:]+]]: i32
  // CHECK-NOT: ttig.extract_desc
  // CHECK: %[[M64:.*]] = arith.extsi %[[M]] : i32 to i64
  // CHECK: %[[I64:.*]] = arith.extsi %[[I]] : i32 to i64
  // CHECK: %[[ROWOFF:.*]] = arith.muli %[[I64]], %[[STRIDE]] : i64
  // CHECK: tt.addptr %[[BASE]], %[[ROWOFF]] : !tt.ptr<f16>, i64
  // CHECK: tt.splat %{{.*}} : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #[[$BLOCKED]]>
  // CHECK: %[[RANGE:.*]] = tt.expand_dims {{.*}} -> tensor<64x1xi64, #[[$BLOCKED]]>
  // CHECK: tt.splat %[[STRIDE]] : i64 -> tensor<64x1xi64
  // CHECK: %[[ROWIDX:.*]] = tt.splat %[[I64]] : i64 -> tensor<64x1xi64
  // CHECK: %[[ROWS:.*]] = arith.addi %[[ROWIDX]], %[[RANGE]]
  // CHECK: arith.cmpi sge, %[[ROWS]]
  // CHECK: %[[MSPLAT:.*]] = tt.splat %[[M64]] : i64 -> tensor<64x1xi64
  // CHECK: arith.cmpi slt, %[[ROWS]], %[[MSPLAT]]
  // CHECK: arith.cmpi sge, {{.*}} : tensor<1x32xi64
  // CHECK: arith.cmpi slt, {{.*}} : tensor<1x32xi64
  // CHECK: %[[NAN:.*]] = arith.constant dense<0x7E00> : tensor<64x32xf16
  // CHECK: tt.load {{.*}}, {{.*}}, %[[NAN]]
  // CHECK-NOT: tt.descriptor_load
  tt.func @per_element_pad_nan(%base: !tt.ptr<f16>, %m: i32, %n: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%m, %n], [%stride, %c1_i64] {padding = 2 : i32} : <f16>, <64x32xf16>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: Shapes and indices divisible by the block: one scalar check per dim, splatted.
  // CHECK-LABEL: tt.func @uniform_mask(
  // CHECK-NOT: arith.cmpi {{.*}} : tensor<
  // CHECK: arith.cmpi sge, {{.*}} : i64
  // CHECK: arith.cmpi slt, {{.*}} : i64
  // CHECK: %[[PRED:.*]] = tt.splat {{.*}} : i1 -> tensor<64x32xi1
  // CHECK: tt.load {{.*}}, %[[PRED]],
  tt.func @uniform_mask(%base: !tt.ptr<f16>, %stride: i64) -> tensor<64x32xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%c128_i32, %c128_i32], [%stride, %c1_i64] : <f16>, <64x32xf16>
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: column_major: result dim 0 walks descriptor dim 1 (stride 1, bound K),
// COM: result dim 1 walks descriptor dim 0 (runtime stride, bound N).
  // CHECK-LABEL: tt.func @column_major(
  // CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f16>, %[[N:[^:]+]]: i32, %[[K:[^:]+]]: i32, %[[STRIDE:[^:]+]]: i64
  // CHECK: %[[C1:.*]] = arith.constant 1 : i64
  // CHECK: %[[N64:.*]] = arith.extsi %[[N]] : i32 to i64
  // CHECK: %[[K64:.*]] = arith.extsi %[[K]] : i32 to i64
  // CHECK: tt.splat %[[C1]] : i64 -> tensor<64x1xi64
  // CHECK: tt.splat %[[K64]] : i64 -> tensor<64x1xi64
  // CHECK: tt.splat %[[STRIDE]] : i64 -> tensor<1x32xi64
  // CHECK: tt.splat %[[N64]] : i64 -> tensor<1x32xi64
  // CHECK: tt.load
  tt.func @column_major(%base: !tt.ptr<f16>, %n: i32, %k: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%n, %k], [%stride, %c1_i64] : <f16>, <32x64xf16>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 1 : i32, ttig.block_io = "column_major"} : !tt.tensordesc<32x64xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: Rank-reducing: the dropped dim's index is checked once for the whole tile.
  // CHECK-LABEL: tt.func @rank_reducing(
  // CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f16>, %[[B:[^:]+]]: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %[[SB:[^:]+]]: i64, %{{[^:]+}}: i64, %[[IB:[^:]+]]: i32,
  // CHECK: %[[B64:.*]] = arith.extsi %[[B]] : i32 to i64
  // CHECK: %[[IB64:.*]] = arith.extsi %[[IB]] : i32 to i64
  // CHECK: %[[OFF:.*]] = arith.muli %[[IB64]], %[[SB]] : i64
  // CHECK: %[[NEWBASE:.*]] = tt.addptr %[[BASE]], %[[OFF]] : !tt.ptr<f16>, i64
  // CHECK: arith.cmpi sge, %[[IB64]], {{.*}} : i64
  // CHECK: arith.cmpi slt, %[[IB64]], %[[B64]] : i64
  // CHECK: tt.addptr %[[NEWBASE]]
  // CHECK: tt.splat {{.*}} : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>
  // CHECK: tt.splat {{.*}} : i1 -> tensor<64x32xi1
  // CHECK: tt.load
  tt.func @rank_reducing(%base: !tt.ptr<f16>, %b: i32, %m: i32, %n: i32, %sb: i64, %sm: i64, %ib: i32, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%b, %m, %n], [%sb, %sm, %c1_i64] : <f16>, <1x64x32xf16>
    %0 = tt.descriptor_load %desc[%ib, %i, %j] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<1x64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: Bools are loaded as bytes and compared against zero.
  // CHECK-LABEL: tt.func @bool_load(
  // CHECK: tt.bitcast {{.*}} : !tt.ptr<i1> -> !tt.ptr<i8>
  // CHECK: %[[LOAD:.*]] = tt.load {{.*}} : tensor<64x32x!tt.ptr<i8>
  // CHECK: arith.cmpi ne, %[[LOAD]], {{.*}} : tensor<64x32xi8
  tt.func @bool_load(%base: !tt.ptr<i1>, %m: i32, %n: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xi1, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%m, %n], [%stride, %c1_i64] : <i1>, <64x32xi1>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xi1> -> tensor<64x32xi1, #blocked>
    tt.return %0 : tensor<64x32xi1, #blocked>
  }
// COM: scf.if descriptors share the shape and stride but not the base: shared
// COM: operands are used directly, the base is read from the descriptor.
  // CHECK-LABEL: tt.func @if_shared_stride(
  // CHECK-SAME: %{{[^:]+}}: !tt.ptr<f16>, %{{[^:]+}}: !tt.ptr<f16>, %[[M:[^:]+]]: i32, %{{[^:]+}}: i32, %[[STRIDE:[^:]+]]: i64
  // CHECK: %[[DESC:.*]] = scf.if
  // CHECK: %[[BASE:.*]] = ttig.extract_desc %[[DESC]][4]
  // CHECK-NOT: ttig.extract_desc
  // CHECK: arith.extsi %[[M]] : i32 to i64
  // CHECK-NOT: ttig.extract_desc
  // CHECK: tt.addptr %[[BASE]]
  // CHECK: tt.splat %[[STRIDE]] : i64 -> tensor<64x1xi64
  // CHECK: tt.load
  tt.func @if_shared_stride(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %m: i32, %n: i32, %stride: i64, %cond: i1) -> tensor<64x32xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = scf.if %cond -> (!tt.tensordesc<64x32xf16>) {
      %d1 = tt.make_tensor_descriptor %a, [%m, %n], [%stride, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d1 : !tt.tensordesc<64x32xf16>
    } else {
      %d2 = tt.make_tensor_descriptor %b, [%m, %n], [%stride, %c1_i64] : <f16>, <64x32xf16>
      scf.yield %d2 : !tt.tensordesc<64x32xf16>
    }
    %0 = tt.descriptor_load %desc[%c0_i32, %c0_i32] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
}

// -----

// COM: Without support_2d_block_io the descriptor loads are still lowered.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
// COM: An untraceable descriptor reads every field from the descriptor, and the
// COM: stamped ttig.desc_padding selects the fill.
  // CHECK-LABEL: tt.func @untraceable_stamped_nan(
  // CHECK-DAG: ttig.extract_desc %{{.*}}[0]
  // CHECK-DAG: ttig.extract_desc %{{.*}}[1]
  // CHECK-DAG: ttig.extract_desc %{{.*}}[2]
  // CHECK-DAG: ttig.extract_desc %{{.*}}[3]
  // CHECK-DAG: ttig.extract_desc %{{.*}}[4]
  // CHECK: %[[NAN:.*]] = arith.constant dense<0x7E00> : tensor<64x32xf16
  // CHECK: tt.load {{.*}}, {{.*}}, %[[NAN]]
  tt.func @untraceable_stamped_nan(%desc: !tt.tensordesc<64x32xf16>, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }

// COM: A loop index stepping by the block size keeps the single bounds check.
  // CHECK-LABEL: tt.func @loop_iv_uniform(
  // CHECK: scf.for
  // CHECK-NOT: arith.cmpi {{.*}} : tensor<
  // CHECK: %[[PRED:.*]] = tt.splat {{.*}} : i1 -> tensor<64x32xi1
  // CHECK: tt.load {{.*}}, %[[PRED]],
  tt.func @loop_iv_uniform(%base: !tt.ptr<f16>, %stride: i64, %ub: i32) -> tensor<64x32xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant dense<0.0> : tensor<64x32xf16, #blocked>
    %desc = tt.make_tensor_descriptor %base, [%c128_i32, %c128_i32], [%stride, %c1_i64] : <f16>, <64x32xf16>
    %r = scf.for %iv = %c0_i32 to %ub step %c64_i32 iter_args(%acc = %zero) -> (tensor<64x32xf16, #blocked>) : i32 {
      %v = tt.descriptor_load %desc[%iv, %c0_i32] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
      %s = arith.addf %acc, %v : tensor<64x32xf16, #blocked>
      scf.yield %s : tensor<64x32xf16, #blocked>
    }
    tt.return %r : tensor<64x32xf16, #blocked>
  }
}
