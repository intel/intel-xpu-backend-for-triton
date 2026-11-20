// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=ttig.2d_block_load
// RUN: env TRITON_INTEL_DISABLE_DESCRIPTOR_LOAD_TO_LOAD=1 triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --check-prefix=KEEP

// COM: tt.descriptor_load that cannot use a 2D block load is lowered to a masked tt.load.

// CHECK: #[[$BLOCKED:.+]] = #ttg.blocked
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
// COM: Shapes not divisible by the block: per-element bounds on each dim, NaN fill.
  // CHECK-LABEL: tt.func @per_element_pad_nan(
  // CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f16>, %[[M:[^:]+]]: i32, %[[N:[^:]+]]: i32, %[[STRIDE:[^:]+]]: i64, %[[I:[^:]+]]: i32, %[[J:[^:]+]]: i32
  // CHECK-NOT: ttig.extract_desc
  // CHECK: %[[I64:.*]] = arith.extsi %[[I]] : i32 to i64
  // CHECK: %[[ROWOFF:.*]] = arith.muli %[[I64]], %[[STRIDE]] : i64
  // CHECK: tt.addptr %[[BASE]], %[[ROWOFF]] : !tt.ptr<f16>, i64
  // CHECK: tt.splat %{{.*}} : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #[[$BLOCKED]]>
  // CHECK: %[[RANGE:.*]] = tt.expand_dims {{.*}} -> tensor<64x1xi32, #[[$BLOCKED]]>
  // CHECK: tt.splat %[[STRIDE]] : i64 -> tensor<64x1xi64
  // CHECK: %[[ROWIDX:.*]] = tt.splat %[[I]] : i32 -> tensor<64x1xi32
  // CHECK: %[[ROWS:.*]] = arith.addi %[[ROWIDX]], %[[RANGE]]
  // CHECK: arith.cmpi sge, %[[ROWS]]
  // CHECK: %[[MSPLAT:.*]] = tt.splat %[[M]] : i32 -> tensor<64x1xi32
  // CHECK: arith.cmpi slt, %[[ROWS]], %[[MSPLAT]]
  // COM: The unit-stride dim is addressed by index + range directly, in i32.
  // CHECK: %[[COLIDX:.*]] = tt.splat %[[J]] : i32 -> tensor<1x32xi32
  // CHECK: %[[COLS:.*]] = arith.addi %[[COLIDX]], %{{.*}} : tensor<1x32xi32
  // CHECK: tt.addptr {{.*}} : tensor<64x32x!tt.ptr<f16>, #[[$BLOCKED]]>, tensor<64x32xi32
  // CHECK: arith.cmpi sge, %[[COLS]], {{.*}} : tensor<1x32xi32
  // CHECK: %[[NSPLAT:.*]] = tt.splat %[[N]] : i32 -> tensor<1x32xi32
  // CHECK: arith.cmpi slt, %[[COLS]], %[[NSPLAT]]
  // CHECK: %[[MASK:.*]] = arith.andi {{.*}} : tensor<64x32xi1
  // CHECK: %[[NAN:.*]] = arith.constant dense<0x7E00> : tensor<64x32xf16
  // CHECK: tt.load %{{.*}}, %[[MASK]], %[[NAN]]
  // CHECK-NOT: tt.descriptor_load
  // COM: With the env var the load is left to DescriptorLoadOpConversion.
  // KEEP-LABEL: tt.func @per_element_pad_nan(
  // KEEP-NOT: tt.load
  // KEEP: tt.descriptor_load
  tt.func @per_element_pad_nan(%base: !tt.ptr<f16>, %m: i32, %n: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%m, %n], [%stride, %c1_i64] {padding = 2 : i32} : <f16>, <64x32xf16>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: A block-multiple shape with a runtime index is checked per element.
  // CHECK-LABEL: tt.func @runtime_index(
  // CHECK: arith.cmpi sge, {{.*}} : tensor<64x1xi32
  // CHECK: arith.cmpi slt, {{.*}} : tensor<64x1xi32
  // CHECK-NOT: arith.cmpi {{.*}} : tensor<1x32
  // CHECK: tt.load
  tt.func @runtime_index(%base: !tt.ptr<f16>, %stride: i64, %i: i32) -> tensor<64x32xf16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%c128_i32, %c128_i32], [%stride, %c1_i64] : <f16>, <64x32xf16>
    %0 = tt.descriptor_load %desc[%i, %c0_i32] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: NaN padding fills fp8 with NaN, but e8m0 and integers with zero bits.
  // CHECK-LABEL: tt.func @pad_nan_fp8(
  // CHECK: %[[NAN8:.*]] = arith.constant dense<0x7F> : tensor<64x32xf8E4M3FN
  // CHECK: tt.load {{.*}}, {{.*}}, %[[NAN8]]
  tt.func @pad_nan_fp8(%base: !tt.ptr<f8E4M3FN>, %m: i32, %n: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xf8E4M3FN, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%m, %n], [%stride, %c1_i64] {padding = 2 : i32} : <f8E4M3FN>, <64x32xf8E4M3FN>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf8E4M3FN> -> tensor<64x32xf8E4M3FN, #blocked>
    tt.return %0 : tensor<64x32xf8E4M3FN, #blocked>
  }
  // CHECK-LABEL: tt.func @pad_nan_e8m0(
  // CHECK: %[[Z8:.*]] = arith.constant 0 : i8
  // CHECK: %[[ZBITS:.*]] = tt.splat %[[Z8]] : i8 -> tensor<64x32xi8
  // CHECK: %[[ZE8M0:.*]] = tt.bitcast %[[ZBITS]] : tensor<64x32xi8, #[[$BLOCKED]]> -> tensor<64x32xf8E8M0FNU
  // CHECK: tt.load {{.*}}, {{.*}}, %[[ZE8M0]]
  tt.func @pad_nan_e8m0(%base: !tt.ptr<f8E8M0FNU>, %m: i32, %n: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xf8E8M0FNU, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%m, %n], [%stride, %c1_i64] {padding = 2 : i32} : <f8E8M0FNU>, <64x32xf8E8M0FNU>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf8E8M0FNU> -> tensor<64x32xf8E8M0FNU, #blocked>
    tt.return %0 : tensor<64x32xf8E8M0FNU, #blocked>
  }
  // CHECK-LABEL: tt.func @pad_nan_i32(
  // CHECK: %[[ZI32:.*]] = tt.splat %{{.*}} : i32 -> tensor<64x32xi32
  // CHECK: tt.load {{.*}}, {{.*}}, %[[ZI32]]
  tt.func @pad_nan_i32(%base: !tt.ptr<i32>, %m: i32, %n: i32, %stride: i64, %i: i32, %j: i32) -> tensor<64x32xi32, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %desc = tt.make_tensor_descriptor %base, [%m, %n], [%stride, %c1_i64] {padding = 2 : i32} : <i32>, <64x32xi32>
    %0 = tt.descriptor_load %desc[%i, %j] {ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xi32> -> tensor<64x32xi32, #blocked>
    tt.return %0 : tensor<64x32xi32, #blocked>
  }
// COM: Shapes and indices divisible by the block: one scalar check per dim, splatted.
  // CHECK-LABEL: tt.func @uniform_mask(
  // CHECK-NOT: arith.cmpi {{.*}} : tensor<
  // CHECK: arith.cmpi sge, {{.*}} : i32
  // CHECK: arith.cmpi slt, {{.*}} : i32
  // CHECK: %[[PRED:.*]] = tt.splat {{.*}} : i1 -> tensor<64x32xi1
  // CHECK: %[[Z16:.*]] = arith.constant 0.000000e+00 : f16
  // CHECK: %[[ZERO:.*]] = tt.splat %[[Z16]] : f16 -> tensor<64x32xf16
  // CHECK: tt.load {{.*}}, %[[PRED]], %[[ZERO]]
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
  // CHECK-SAME: %[[BASE:[^:]+]]: !tt.ptr<f16>, %[[N:[^:]+]]: i32, %[[K:[^:]+]]: i32, %[[STRIDE:[^:]+]]: i64, %{{[^:]+}}: i32, %[[J:[^:]+]]: i32
  // CHECK: tt.splat %[[J]] : i32 -> tensor<64x1xi32
  // CHECK: tt.addptr {{.*}}, tensor<64x32xi32
  // CHECK: tt.splat %[[K]] : i32 -> tensor<64x1xi32
  // CHECK: tt.splat %[[STRIDE]] : i64 -> tensor<1x32xi64
  // CHECK: tt.splat %[[N]] : i32 -> tensor<1x32xi32
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
  // CHECK: %[[IB64:.*]] = arith.extsi %[[IB]] : i32 to i64
  // CHECK: %[[OFF:.*]] = arith.muli %[[IB64]], %[[SB]] : i64
  // CHECK: %[[NEWBASE:.*]] = tt.addptr %[[BASE]], %[[OFF]] : !tt.ptr<f16>, i64
  // CHECK: arith.cmpi sge, %[[IB]], {{.*}} : i32
  // CHECK: arith.cmpi slt, %[[IB]], %[[B]] : i32
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
  // CHECK: tt.addptr %[[BASE]]
  // CHECK: tt.splat %[[STRIDE]] : i64 -> tensor<64x1xi64
  // CHECK: tt.splat %[[M]] : i32 -> tensor<64x1xi32
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
// COM: Definitions that disagree on an operand ([%a, %b, %b] here) do not share
// COM: it, even when the last two agree: the base is read from the descriptor.
  // CHECK-LABEL: tt.func @three_defs_one_differs(
  // CHECK-SAME: %[[A:[^:]+]]: !tt.ptr<f16>, %[[B:[^:]+]]: !tt.ptr<f16>
  // CHECK: ttig.extract_desc %{{.*}}[4]
  // CHECK-NOT: tt.addptr %[[B]]
  // CHECK-NOT: tt.splat %[[B]]
  // CHECK: tt.load
  tt.func @three_defs_one_differs(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %m: i32, %n: i32, %s: i64, %c1: i1, %c2: i1, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %da = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1_i64] : <f16>, <64x32xf16>
    %db1 = tt.make_tensor_descriptor %b, [%m, %n], [%s, %c1_i64] : <f16>, <64x32xf16>
    %db2 = tt.make_tensor_descriptor %b, [%m, %n], [%s, %c1_i64] : <f16>, <64x32xf16>
    %y = arith.select %c2, %db1, %db2 : !tt.tensordesc<64x32xf16>
    %x = arith.select %c1, %da, %y : !tt.tensordesc<64x32xf16>
    %0 = tt.descriptor_load %x[%i, %j] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
    tt.return %0 : tensor<64x32xf16, #blocked>
  }
// COM: Through control flow, a shared operand defined inside a loop around the
// COM: load is read from the descriptor: a loop-carried descriptor could come
// COM: from an earlier iteration. Loop-invariant shared operands are used directly.
  // CHECK-LABEL: tt.func @loop_select_in_loop_shape(
  // CHECK-SAME: %{{[^:]+}}: !tt.ptr<f16>, %{{[^:]+}}: !tt.ptr<f16>, %[[N:[^:]+]]: i32, %[[STRIDE:[^:]+]]: i64
  // CHECK: scf.for
  // CHECK: %[[X:.*]] = arith.select
  // CHECK-DAG: ttig.extract_desc %[[X]][4]
  // CHECK-DAG: ttig.extract_desc %[[X]][0]
  // CHECK-NOT: ttig.extract_desc
  // CHECK-DAG: tt.splat %[[STRIDE]] : i64 -> tensor<64x1xi64
  // CHECK-DAG: tt.splat %[[N]] : i32 -> tensor<1x32xi32
  // CHECK: tt.load
  tt.func @loop_select_in_loop_shape(%a: !tt.ptr<f16>, %b: !tt.ptr<f16>, %n: i32, %s: i64, %ub: i32, %c: i1, %i: i32, %j: i32) -> tensor<64x32xf16, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %zero = arith.constant dense<0.0> : tensor<64x32xf16, #blocked>
    %r = scf.for %iv = %c0 to %ub step %c1 iter_args(%acc = %zero) -> (tensor<64x32xf16, #blocked>) : i32 {
      %m = arith.addi %iv, %c64 : i32
      %d1 = tt.make_tensor_descriptor %a, [%m, %n], [%s, %c1_i64] : <f16>, <64x32xf16>
      %d2 = tt.make_tensor_descriptor %b, [%m, %n], [%s, %c1_i64] : <f16>, <64x32xf16>
      %x = arith.select %c, %d1, %d2 : !tt.tensordesc<64x32xf16>
      %v = tt.descriptor_load %x[%i, %j] {ttig.desc_padding = 1 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, #blocked>
      %sum = arith.addf %acc, %v : tensor<64x32xf16, #blocked>
      scf.yield %sum : tensor<64x32xf16, #blocked>
    }
    tt.return %r : tensor<64x32xf16, #blocked>
  }
}

// -----

// COM: column_major with mixed divisibility: result dim 0 walks descriptor dim 1
// COM: (shape 16, index 8: one check), result dim 1 walks descriptor dim 0
// COM: (shape 17: per element).
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @colmaj_mixed_inner_div(
  // CHECK-DAG: %[[C17:.*]] = arith.constant 17 : i32
  // CHECK-DAG: %[[C16:.*]] = arith.constant 16 : i32
  // CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
  // CHECK: arith.cmpi sge, %[[C8]], %{{.*}} : i32
  // CHECK: arith.cmpi slt, %[[C8]], %[[C16]] : i32
  // CHECK: arith.cmpi sge, {{.*}} : tensor<1x16xi32
  // CHECK: %[[S17:.*]] = tt.splat %[[C17]] : i32 -> tensor<1x16xi32
  // CHECK: arith.cmpi slt, %{{.*}}, %[[S17]] : tensor<1x16xi32
  // CHECK: tt.load
  tt.func public @colmaj_mixed_inner_div(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<8x16xf32, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c17_i32 = arith.constant 17 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c17_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x8xf32>
    %0 = tt.descriptor_load %desc[%c0_i32, %c8_i32] {ttig.block_io = "column_major", ttig.desc_padding = 1 : i32} : !tt.tensordesc<16x8xf32> -> tensor<8x16xf32, #blocked>
    tt.return %0 : tensor<8x16xf32, #blocked>
  }
}
