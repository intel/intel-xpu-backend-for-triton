// RUN: triton-opt %s -split-input-file --tritongpu-fuse-nested-loops -canonicalize --tritonintelgpu-lower-to-2d-block-load | FileCheck %s --implicit-check-not=tt.load

// COM: A persistent loop (tl.range(..., flatten=True)) over 64-row tiles, with a
// COM: block-IO load advanced along K in the inner loop. Loop fusion carries the
// COM: tile pointer through iter_args with a ub.poison init (checked, so the test
// COM: fails if fusion stops firing); the load must still get its pitch after
// COM: fusion (issue #8687).
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @fused_constant_stride
  tt.func @fused_constant_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ntiles: i32, %ktiles: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst32 = arith.constant dense<32> : tensor<64x32xi32, #dot0>
    %stride = arith.constant dense<4096> : tensor<64x1xi32, #dot0>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %c2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %colb = tt.broadcast %c2 : tensor<1x32xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %obase = tt.splat %out : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: %[[POISON:.*]] = ub.poison
    // CHECK: scf.for {{.*}} iter_args({{.*}} = %[[POISON]]
    // CHECK: %[[PITCH:.*]] = arith.constant 8192 : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %{{.*}}, %{{.*}} {row_major}
    scf.for %t = %c0_i32 to %ntiles step %c1_i32 : i32 {
      %m = arith.muli %t, %c64_i32 : i32
      %ms = tt.splat %m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
      %r = arith.addi %ms, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
      %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
      %r2 = arith.muli %r1, %stride : tensor<64x1xi32, #dot0>
      %rowb = tt.broadcast %r2 : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
      %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot0>
      %ptr0 = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
      %res:2 = scf.for %k = %c0_i32 to %ktiles step %c1_i32 iter_args(%p = %ptr0, %acc = %zero) -> (tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>) : i32 {
        %ld = tt.load %p {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
        %acc2 = arith.addf %acc, %ld : tensor<64x32xf16, #dot0>
        %pn = tt.addptr %p, %cst32 : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
        scf.yield %pn, %acc2 : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>
      }
      %optr = tt.addptr %obase, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
      tt.store %optr, %res#1 : tensor<64x32x!tt.ptr<f16>, #dot0>
    } {tt.flatten}
    tt.return
  }
}

// -----

// COM: Same, with a runtime row stride.
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io} {
  // CHECK-LABEL: tt.func @fused_runtime_stride
  // CHECK-SAME: %[[LDA:[a-z0-9_]+]]: i32 {tt.divisibility = 16 : i32})
  tt.func @fused_runtime_stride(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %ntiles: i32, %ktiles: i32, %lda: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst32 = arith.constant dense<32> : tensor<64x32xi32, #dot0>
    %stride = tt.splat %lda : i32 -> tensor<64x1xi32, #dot0>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x32xf16, #dot0>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>>
    %c2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot0}>> -> tensor<1x32xi32, #dot0>
    %colb = tt.broadcast %c2 : tensor<1x32xi32, #dot0> -> tensor<64x32xi32, #dot0>
    %base = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    %obase = tt.splat %out : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #dot0>
    // CHECK: %[[POISON:.*]] = ub.poison
    // CHECK: scf.for {{.*}} iter_args({{.*}} = %[[POISON]]
    // CHECK: %[[C2:.*]] = arith.constant 2 : i32
    // CHECK: %[[PITCH:.*]] = arith.muli %[[LDA]], %[[C2]] : i32
    // CHECK: ttig.2d_block_load_from_ptr %{{.*}}, %[[PITCH]], %{{.*}}, %{{.*}} {row_major}
    scf.for %t = %c0_i32 to %ntiles step %c1_i32 : i32 {
      %m = arith.muli %t, %c64_i32 : i32
      %ms = tt.splat %m : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
      %r = arith.addi %ms, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>>
      %r1 = tt.expand_dims %r {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #dot0}>> -> tensor<64x1xi32, #dot0>
      %r2 = arith.muli %r1, %stride : tensor<64x1xi32, #dot0>
      %rowb = tt.broadcast %r2 : tensor<64x1xi32, #dot0> -> tensor<64x32xi32, #dot0>
      %off = arith.addi %rowb, %colb : tensor<64x32xi32, #dot0>
      %ptr0 = tt.addptr %base, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
      %res:2 = scf.for %k = %c0_i32 to %ktiles step %c1_i32 iter_args(%p = %ptr0, %acc = %zero) -> (tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>) : i32 {
        %ld = tt.load %p {ttig.block_io = "row_major"} : tensor<64x32x!tt.ptr<f16>, #dot0>
        %acc2 = arith.addf %acc, %ld : tensor<64x32xf16, #dot0>
        %pn = tt.addptr %p, %cst32 : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
        scf.yield %pn, %acc2 : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xf16, #dot0>
      }
      %optr = tt.addptr %obase, %off : tensor<64x32x!tt.ptr<f16>, #dot0>, tensor<64x32xi32, #dot0>
      tt.store %optr, %res#1 : tensor<64x32x!tt.ptr<f16>, #dot0>
    } {tt.flatten}
    tt.return
  }
}
