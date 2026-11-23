// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm
// RUN: env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s --check-prefixes=GEN,SPV
// RUN: sed 's/ttig.support_predicated_io}/ttig.support_predicated_io, ttig.use_genisa_2d_block_io}/' %s | env TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=1 triton-opt -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s --check-prefixes=GEN,GENISA

// COM: A column bound that alone rejects a block store tile clamps the surface; other masks stay generic.

#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [2, 2], A = [16, 16], B = [16, 32], C = [16, 32]}>
#ba = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [1, 16], warpsPerCTA = [8, 2], order = [1, 0]}>
#bq = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [16, 1], order = [1, 0]}>
#r = #ttg.slice<{dim = 1, parent = #mma}>
#c = #ttg.slice<{dim = 0, parent = #mma}>
#b1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [16, 1], order = [1, 0]}>
#r1 = #ttg.slice<{dim = 1, parent = #b1}>
#c1 = #ttg.slice<{dim = 0, parent = #b1}>
module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_2d_block_io, ttig.support_predicated_io} {
  // COM: Bounds 8448x504 (#8813): W = clamp(504 - colBase, 0, 128) * 2, s = max(64 - W, 0).
  // CHECK-LABEL: llvm.func spir_kernelcc @col_clamp(
  // CHECK: %[[CB:.*]] = llvm.mul %arg3, %{{.*}} : i32
  // CHECK: %[[PITCH:.*]] = llvm.mlir.constant(1008 : i32) : i32
  // CHECK: %[[Z:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[E:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: %[[B:.*]] = llvm.mlir.constant(504 : i64) : i64
  // CHECK-NEXT: %[[CB64:.*]] = llvm.sext %[[CB]] : i32 to i64
  // CHECK-NEXT: %[[D:.*]] = llvm.sub %[[B]], %[[CB64]] : i64
  // CHECK-NEXT: %[[MX:.*]] = llvm.intr.smax(%[[D]], %[[Z]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[MN:.*]] = llvm.intr.smin(%[[MX]], %[[E]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[T:.*]] = llvm.trunc %[[MN]] : i64 to i32
  // CHECK-NEXT: %[[EB:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT: %[[W:.*]] = llvm.mul %[[T]], %[[EB]] : i32
  // CHECK: %[[K:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-NEXT: %[[NS:.*]] = llvm.sub %[[K]], %[[W]] : i32
  // CHECK: %[[S:.*]] = llvm.intr.smax(%[[NS]], %{{.*}}) : (i32, i32) -> i32
  // CHECK: %[[SE:.*]] = llvm.udiv %[[S]], %{{.*}} : i32
  // CHECK: %[[X:.*]] = llvm.add %[[X0:.*]], %[[SE]] : i32
  // CHECK: %[[NX:.*]] = llvm.sub %{{.*}}, %[[X]] : i32
  // CHECK-NEXT: %[[A:.*]] = llvm.getelementptr %{{.*}}[%[[NX]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f16
  // CHECK: %[[XB:.*]] = llvm.mul %[[X0]], %{{.*}} : i32
  // CHECK-NEXT: %[[W0:.*]] = llvm.add %{{.*}}, %[[XB]] : i32
  // CHECK: %[[UX:.*]] = llvm.intr.umax(%[[W0]], %{{.*}}) : (i32, i32) -> i32
  // CHECK-NEXT: %[[UN:.*]] = llvm.intr.umin(%[[UX]], %[[W]]) : (i32, i32) -> i32
  // CHECK-NEXT: %[[BW:.*]] = llvm.add %[[UN]], %[[S]] : i32
  // CHECK: %[[P:.*]] = llvm.trunc %{{.*}} : i8 to i1
  // CHECK-NEXT: %[[Y:.*]] = llvm.select %[[P]], %{{.*}}, %{{.*}} : i1, i32
  // CHECK: %[[XU:.*]] = llvm.udiv %[[X]], %{{.*}} : i32
  // CHECK-NEXT: triton_gen.2Dblockstore %[[A]], %[[BW]], %{{.*}}, %[[PITCH]], %[[XU]], %[[Y]], %{{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1
  // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}}tile_height = 8,
  // CHECK-NOT: triton_gen.{{2Dblockstore|predicated_store}}
  // GEN-LABEL: llvm.func spir_kernelcc @col_clamp(
  // GEN: %[[S:.*]] = llvm.intr.smax(%{{.*}}) : (i32, i32) -> i32
  // GEN: %[[SE:.*]] = llvm.udiv %[[S]], %{{.*}} : i32
  // GEN: %[[X:.*]] = llvm.add %{{.*}}, %[[SE]] : i32
  // GEN: %[[UN:.*]] = llvm.intr.umin(
  // GEN-NEXT: %[[BW:.*]] = llvm.add %[[UN]], %[[S]] : i32
  // GEN: %[[XU:.*]] = llvm.udiv %[[X]], %{{.*}} : i32
  // GEN: %[[W2:.*]] = llvm.add %[[BW]], %{{.*}} : i32
  // GEN: %[[X2:.*]] = llvm.add %[[XU]], %{{.*}} : i32
  // SPV: %[[V0:.*]] = llvm.insertelement %[[X2]], %{{.*}} : vector<2xi32>
  // SPV: %[[V:.*]] = llvm.insertelement %{{.*}}, %[[V0]]{{.*}} : vector<2xi32>
  // SPV: llvm.call spir_funccc @_Z33__spirv_Subgroup2DBlockStoreINTEL{{.*}}, %[[W2]], %{{[0-9]+}}, %{{[0-9]+}}, %[[V]])
  // GENISA: %[[WM1:.*]] = llvm.sub %[[W2]], %{{.*}} : i32
  // GENISA: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i16(%{{[0-9]+}}, %[[WM1]], %{{[0-9]+}}, %{{[0-9]+}}, %[[X2]],
  tt.func public @col_clamp(%c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %val: tensor<64x128xf16, #mma>, %pid_m: i32, %pid_n: i32) {
    %cst = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %cst_0 = arith.constant dense<504> : tensor<1x128xi32, #mma>
    %cst_1 = arith.constant dense<8448> : tensor<64x1xi32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %acc = arith.muli %pid_m, %c64_i32 : i32
    %acc_9 = arith.muli %pid_n, %c128_i32 : i32
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %rm_11 = tt.splat %acc : i32 -> tensor<64xi32, #r>
    %rm_12 = arith.addi %rm_11, %rm : tensor<64xi32, #r>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %rn_13 = tt.splat %acc_9 : i32 -> tensor<128xi32, #c>
    %rn_14 = arith.addi %rn_13, %rn : tensor<128xi32, #c>
    %mask = tt.expand_dims %rm_12 {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %mask_15 = arith.cmpi slt, %mask, %cst_1 : tensor<64x1xi32, #mma>
    %mask_16 = tt.expand_dims %rn_14 {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mask_17 = arith.cmpi slt, %mask_16, %cst_0 : tensor<1x128xi32, #mma>
    %mask_18 = tt.broadcast %mask_15 : tensor<64x1xi1, #mma> -> tensor<64x128xi1, #mma>
    %mask_19 = tt.broadcast %mask_17 : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %mask_20 = arith.andi %mask_18, %mask_19 : tensor<64x128xi1, #mma>
    %0 = arith.muli %mask, %cst : tensor<64x1xi32, #mma>
    %1 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %2 = tt.addptr %1, %0 : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %3 = tt.broadcast %2 : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %4 = tt.broadcast %mask_16 : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %5 = tt.addptr %3, %4 : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %5, %val, %mask_20 {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.return
  }

  // COM: Bounds 8450x504: only the column is relaxed, so the 2-row constancy still sets the tile.
  // CHECK-LABEL: llvm.func spir_kernelcc @col_clamp_two_row_tile(
  // GEN-LABEL: llvm.func spir_kernelcc @col_clamp_two_row_tile(
  // CHECK: llvm.intr.umin
  // CHECK-COUNT-16: triton_gen.2Dblockstore {{.*}}tile_height = 2,
  tt.func @col_clamp_two_row_tile(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf16, #mma>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %mb = arith.constant dense<8450> : tensor<64x1xi32, #mma>
    %nb = arith.constant dense<504> : tensor<1x128xi32, #mma>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mr = arith.cmpi slt, %er, %mb : tensor<64x1xi32, #mma>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #mma>
    %br = tt.broadcast %mr : tensor<64x1xi1, #mma> -> tensor<64x128xi1, #mma>
    %bc = tt.broadcast %mc : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %m = arith.andi %br, %bc : tensor<64x128xi1, #mma>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #mma>
    %b = tt.splat %p : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.return
  }

  // COM: An f32 width granule is one element, so the odd bound 121 still clamps.
  // CHECK-LABEL: llvm.func spir_kernelcc @col_clamp_f32_odd_bound(
  // CHECK: llvm.intr.umin
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}}elem_size_in_bits = 32,
  tt.func @col_clamp_f32_odd_bound(%p: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf32, #mma>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %nb = arith.constant dense<121> : tensor<1x128xi32, #mma>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #mma>
    %m = tt.broadcast %mc : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #mma>
    %b = tt.splat %p : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #mma>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f32>, #mma>, tensor<64x1xi32, #mma>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f32>, #mma> -> tensor<64x128x!tt.ptr<f32>, #mma>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f32>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f32>, #mma>
    tt.return
  }

  // COM: A blocked layout whose unmasked tile is valid clamps as well.
  // CHECK-LABEL: llvm.func spir_kernelcc @col_clamp_blocked(
  // CHECK: llvm.intr.umin
  // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}}tile_height = 8,
  tt.func @col_clamp_blocked(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf16, #ba>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #ba>
    %nb = arith.constant dense<120> : tensor<1x128xi32, #ba>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ba}>>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ba}>>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #ba}>> -> tensor<64x1xi32, #ba>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ba}>> -> tensor<1x128xi32, #ba>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #ba>
    %m = tt.broadcast %mc : tensor<1x128xi1, #ba> -> tensor<64x128xi1, #ba>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #ba>
    %b = tt.splat %p : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #ba>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f16>, #ba>, tensor<64x1xi32, #ba>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f16>, #ba> -> tensor<64x128x!tt.ptr<f16>, #ba>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #ba> -> tensor<64x128xi32, #ba>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f16>, #ba>, tensor<64x128xi32, #ba>
    tt.store %ptr, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #ba>
    tt.return
  }

  // COM: f16 packed into 64-bit elements cannot shift by a 2-element granule, half a packed element.
  // CHECK-LABEL: llvm.func spir_kernelcc @no_clamp_packed_64bit(
  // CHECK-COUNT-8: triton_gen.predicated_store
  tt.func @no_clamp_packed_64bit(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf16, #bq>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #bq>
    %nb = arith.constant dense<120> : tensor<1x128xi32, #bq>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #bq}>>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #bq}>>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #bq}>> -> tensor<64x1xi32, #bq>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #bq}>> -> tensor<1x128xi32, #bq>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #bq>
    %m = tt.broadcast %mc : tensor<1x128xi1, #bq> -> tensor<64x128xi1, #bq>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #bq>
    %b = tt.splat %p : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #bq>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f16>, #bq>, tensor<64x1xi32, #bq>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f16>, #bq> -> tensor<64x128x!tt.ptr<f16>, #bq>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #bq> -> tensor<64x128xi32, #bq>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f16>, #bq>, tensor<64x128xi32, #bq>
    tt.store %ptr, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #bq>
    tt.return
  }

  // COM: A row constancy of 3 still rejects the tile once the column is relaxed.
  // CHECK-LABEL: llvm.func spir_kernelcc @no_clamp_row_constancy_3(
  // CHECK-COUNT-32: triton_gen.predicated_store
  tt.func @no_clamp_row_constancy_3(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf16, #mma>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %nb = arith.constant dense<504> : tensor<1x128xi32, #mma>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #mma>
    %m = tt.broadcast %mc {tt.constancy = dense<[3, 8]> : tensor<2xi32>} : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #mma>
    %b = tt.splat %p : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.return
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @no_clamp_cache_modifier_or_hint(
  // CHECK-NOT: triton_gen.2Dblockstore
  // COM: The cg store: a cache modifier keeps the generic path, which honours it.
  // CHECK-COUNT-32: triton_gen.predicated_store {{.*}}cache_control = L1UC_L3WB
  // COM: The hinted store: a non-power-of-two column constancy at or above the tile width is not relaxed.
  // CHECK-COUNT-32: triton_gen.predicated_store {{.*}}cache_control = Default
  tt.func @no_clamp_cache_modifier_or_hint(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf16, #mma>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %nb = arith.constant dense<504> : tensor<1x128xi32, #mma>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #mma>
    %m = tt.broadcast %mc : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %mg = tt.broadcast %mc {tt.constancy = dense<[64, 24]> : tensor<2xi32>} : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #mma>
    %b = tt.splat %p : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr, %v, %m {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>, ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.store %ptr, %v, %mg {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.return
  }

  // COM: A second store of the same mask at pitch 240, below row bytes + 63, lowers as it would alone.
  // CHECK-LABEL: llvm.func spir_kernelcc @clamp_pitch_per_store(
  // CHECK-COUNT-4: triton_gen.2Dblockstore
  // CHECK-COUNT-32: triton_gen.predicated_store
  tt.func @clamp_pitch_per_store(%p: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %v: tensor<64x128xf16, #mma>) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %ld2 = arith.constant dense<120> : tensor<64x1xi32, #mma>
    %nb = arith.constant dense<120> : tensor<1x128xi32, #mma>
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %er = tt.expand_dims %ar {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %ec = tt.expand_dims %ac {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #mma>
    %m = tt.broadcast %mc : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #mma>
    %b = tt.splat %p : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %cb = tt.broadcast %ec : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %ptr = tt.addptr %pb, %cb : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    %o2 = arith.muli %er, %ld2 : tensor<64x1xi32, #mma>
    %pr2 = tt.addptr %b, %o2 : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %pb2 = tt.broadcast %pr2 : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %ptr2 = tt.addptr %pb2, %cb : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr2, %v, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.return
  }

  // COM: Bounds 8449x504: row constancy 1 relaxes the row as well, H = clamp(8449 - rowBase, 0, 64).
  // COM: Each message gets base_height = max(H - rowOffset, 1), next to @col_clamp's column clamp.
  // CHECK-LABEL: llvm.func spir_kernelcc @row_and_col_clamp(
  // CHECK: %[[RB:.*]] = llvm.mul %arg2, %{{.*}} : i32
  // CHECK: %[[PITCH:.*]] = llvm.mlir.constant(1008 : i32) : i32
  // CHECK: %[[K:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-NEXT: %[[NS:.*]] = llvm.sub %[[K]], %[[W:.*]] : i32
  // CHECK: %[[S:.*]] = llvm.intr.smax(%[[NS]], %{{.*}}) : (i32, i32) -> i32
  // CHECK: %[[Z:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[E:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-NEXT: %[[MB:.*]] = llvm.mlir.constant(8449 : i64) : i64
  // CHECK-NEXT: %[[RB64:.*]] = llvm.sext %[[RB]] : i32 to i64
  // CHECK-NEXT: %[[D:.*]] = llvm.sub %[[MB]], %[[RB64]] : i64
  // CHECK-NEXT: %[[MX:.*]] = llvm.intr.smax(%[[D]], %[[Z]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[MN:.*]] = llvm.intr.smin(%[[MX]], %[[E]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[H:.*]] = llvm.trunc %[[MN]] : i64 to i32
  // CHECK: %[[UN:.*]] = llvm.intr.umin(%{{.*}}, %[[W]]) : (i32, i32) -> i32
  // CHECK-NEXT: %[[BW:.*]] = llvm.add %[[UN]], %[[S]] : i32
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[HR:.*]] = llvm.sub %[[H]], %{{.*}} : i32
  // CHECK: %[[BH:.*]] = llvm.intr.smax(%[[HR]], %[[ONE]]) : (i32, i32) -> i32
  // CHECK: %[[Y:.*]] = llvm.select %{{.*}}, %{{.*}}, %[[BH]] : i1, i32
  // CHECK: triton_gen.2Dblockstore %{{.*}}, %[[BW]], %[[BH]], %[[PITCH]], %{{.*}}, %[[Y]], %{{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1
  // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}}tile_height = 8,
  // CHECK-NOT: triton_gen.{{2Dblockstore|predicated_store}}
  tt.func public @row_and_col_clamp(%c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %val: tensor<64x128xf16, #mma>, %pid_m: i32, %pid_n: i32) {
    %cst = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %cst_0 = arith.constant dense<504> : tensor<1x128xi32, #mma>
    %cst_1 = arith.constant dense<8449> : tensor<64x1xi32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %acc = arith.muli %pid_m, %c64_i32 : i32
    %acc_9 = arith.muli %pid_n, %c128_i32 : i32
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %rm_11 = tt.splat %acc : i32 -> tensor<64xi32, #r>
    %rm_12 = arith.addi %rm_11, %rm : tensor<64xi32, #r>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %rn_13 = tt.splat %acc_9 : i32 -> tensor<128xi32, #c>
    %rn_14 = arith.addi %rn_13, %rn : tensor<128xi32, #c>
    %mask = tt.expand_dims %rm_12 {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %mask_15 = arith.cmpi slt, %mask, %cst_1 : tensor<64x1xi32, #mma>
    %mask_16 = tt.expand_dims %rn_14 {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mask_17 = arith.cmpi slt, %mask_16, %cst_0 : tensor<1x128xi32, #mma>
    %mask_18 = tt.broadcast %mask_15 : tensor<64x1xi1, #mma> -> tensor<64x128xi1, #mma>
    %mask_19 = tt.broadcast %mask_17 : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %mask_20 = arith.andi %mask_18, %mask_19 : tensor<64x128xi1, #mma>
    %0 = arith.muli %mask, %cst : tensor<64x1xi32, #mma>
    %1 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #mma>
    %2 = tt.addptr %1, %0 : tensor<64x1x!tt.ptr<f16>, #mma>, tensor<64x1xi32, #mma>
    %3 = tt.broadcast %2 : tensor<64x1x!tt.ptr<f16>, #mma> -> tensor<64x128x!tt.ptr<f16>, #mma>
    %4 = tt.broadcast %mask_16 : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %5 = tt.addptr %3, %4 : tensor<64x128x!tt.ptr<f16>, #mma>, tensor<64x128xi32, #mma>
    tt.store %5, %val, %mask_20 {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #mma>
    tt.return
  }

  // COM: Bounds 8449x512: a layout that caps the tile at one row keeps one-row tiles and gets no row clamp.
  // CHECK-LABEL: llvm.func spir_kernelcc @one_row_layout_no_row_clamp(
  // CHECK-NOT: llvm.intr.smax
  // CHECK-COUNT-32: triton_gen.2Dblockstore {{.*}}tile_height = 1,
  tt.func public @one_row_layout_no_row_clamp(%c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %val: tensor<64x128xf16, #b1>, %pid_m: i32, %pid_n: i32) {
    %cst = arith.constant dense<1024> : tensor<64x1xi32, #b1>
    %cst_0 = arith.constant dense<512> : tensor<1x128xi32, #b1>
    %cst_1 = arith.constant dense<8449> : tensor<64x1xi32, #b1>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %acc = arith.muli %pid_m, %c64_i32 : i32
    %acc_9 = arith.muli %pid_n, %c128_i32 : i32
    %rm = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r1>
    %rm_11 = tt.splat %acc : i32 -> tensor<64xi32, #r1>
    %rm_12 = arith.addi %rm_11, %rm : tensor<64xi32, #r1>
    %rn = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c1>
    %rn_13 = tt.splat %acc_9 : i32 -> tensor<128xi32, #c1>
    %rn_14 = arith.addi %rn_13, %rn : tensor<128xi32, #c1>
    %mask = tt.expand_dims %rm_12 {axis = 1 : i32} : tensor<64xi32, #r1> -> tensor<64x1xi32, #b1>
    %mask_15 = arith.cmpi slt, %mask, %cst_1 : tensor<64x1xi32, #b1>
    %mask_16 = tt.expand_dims %rn_14 {axis = 0 : i32} : tensor<128xi32, #c1> -> tensor<1x128xi32, #b1>
    %mask_17 = arith.cmpi slt, %mask_16, %cst_0 : tensor<1x128xi32, #b1>
    %mask_18 = tt.broadcast %mask_15 : tensor<64x1xi1, #b1> -> tensor<64x128xi1, #b1>
    %mask_19 = tt.broadcast %mask_17 : tensor<1x128xi1, #b1> -> tensor<64x128xi1, #b1>
    %mask_20 = arith.andi %mask_18, %mask_19 : tensor<64x128xi1, #b1>
    %0 = arith.muli %mask, %cst : tensor<64x1xi32, #b1>
    %1 = tt.splat %c_ptr : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #b1>
    %2 = tt.addptr %1, %0 : tensor<64x1x!tt.ptr<f16>, #b1>, tensor<64x1xi32, #b1>
    %3 = tt.broadcast %2 : tensor<64x1x!tt.ptr<f16>, #b1> -> tensor<64x128x!tt.ptr<f16>, #b1>
    %4 = tt.broadcast %mask_16 : tensor<1x128xi32, #b1> -> tensor<64x128xi32, #b1>
    %5 = tt.addptr %3, %4 : tensor<64x128x!tt.ptr<f16>, #b1>, tensor<64x128xi32, #b1>
    tt.store %5, %val, %mask_20 {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f16>, #b1>
    tt.return
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @runtime_bounds(
  // COM: Argument bounds clamp the row and column; f32, as the lowering cannot prove an argument divisible.
  // CHECK: %[[RB:.*]] = llvm.mul %arg2, %{{.*}} : i32
  // CHECK-NEXT: %[[CB:.*]] = llvm.mul %arg3, %{{.*}} : i32
  // CHECK: %[[PITCH:.*]] = llvm.mlir.constant(2016 : i32) : i32
  // CHECK-NEXT: %[[ZW:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[EW:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK-NEXT: %[[NB:.*]] = llvm.sext %arg5 : i32 to i64
  // CHECK-NEXT: %[[CB64:.*]] = llvm.sext %[[CB]] : i32 to i64
  // CHECK-NEXT: %[[DW:.*]] = llvm.sub %[[NB]], %[[CB64]] : i64
  // CHECK-NEXT: %[[MXW:.*]] = llvm.intr.smax(%[[DW]], %[[ZW]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[MNW:.*]] = llvm.intr.smin(%[[MXW]], %[[EW]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[T:.*]] = llvm.trunc %[[MNW]] : i64 to i32
  // CHECK-NEXT: %[[EB:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-NEXT: %[[W:.*]] = llvm.mul %[[T]], %[[EB]] : i32
  // CHECK: %[[K:.*]] = llvm.mlir.constant(64 : i32) : i32
  // CHECK-NEXT: %[[NS:.*]] = llvm.sub %[[K]], %[[W]] : i32
  // CHECK: %[[S:.*]] = llvm.intr.smax(%[[NS]], %{{.*}}) : (i32, i32) -> i32
  // CHECK: %[[ZH:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[EH:.*]] = llvm.mlir.constant(64 : i64) : i64
  // CHECK-NEXT: %[[MB:.*]] = llvm.sext %arg4 : i32 to i64
  // CHECK-NEXT: %[[RB64:.*]] = llvm.sext %[[RB]] : i32 to i64
  // CHECK-NEXT: %[[DH:.*]] = llvm.sub %[[MB]], %[[RB64]] : i64
  // CHECK-NEXT: %[[MXH:.*]] = llvm.intr.smax(%[[DH]], %[[ZH]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[MNH:.*]] = llvm.intr.smin(%[[MXH]], %[[EH]]) : (i64, i64) -> i64
  // CHECK-NEXT: %[[H:.*]] = llvm.trunc %[[MNH]] : i64 to i32
  // CHECK: %[[UN:.*]] = llvm.intr.umin(%{{.*}}, %[[W]]) : (i32, i32) -> i32
  // CHECK-NEXT: %[[BW:.*]] = llvm.add %[[UN]], %[[S]] : i32
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[HR:.*]] = llvm.sub %[[H]], %{{.*}} : i32
  // CHECK: %[[BH:.*]] = llvm.intr.smax(%[[HR]], %[[ONE]]) : (i32, i32) -> i32
  // CHECK: %[[Y:.*]] = llvm.select %{{.*}}, %{{.*}}, %[[BH]] : i1, i32
  // CHECK: triton_gen.2Dblockstore %{{.*}}, %[[BW]], %[[BH]], %[[PITCH]], %{{.*}}, %[[Y]], %{{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1
  // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}}tile_height = 8,
  // CHECK-NOT: triton_gen.{{2Dblockstore|predicated_store}}
  // COM: A row-only argument bound clamps the row alone and still relaxes the tile to 8 rows.
  // CHECK: %[[MB2:.*]] = llvm.sext %arg4 : i32 to i64
  // CHECK-NEXT: %[[RB2:.*]] = llvm.sext %[[RB]] : i32 to i64
  // CHECK-NEXT: %[[DH2:.*]] = llvm.sub %[[MB2]], %[[RB2]] : i64
  // CHECK-NEXT: %[[MX2:.*]] = llvm.intr.smax(%[[DH2]], %{{.*}}) : (i64, i64) -> i64
  // CHECK-NEXT: %[[MN2:.*]] = llvm.intr.smin(%[[MX2]], %{{.*}}) : (i64, i64) -> i64
  // CHECK-NEXT: %[[RH2:.*]] = llvm.trunc %[[MN2]] : i64 to i32
  // CHECK: %[[HR2:.*]] = llvm.sub %[[RH2]], %{{.*}} : i32
  // CHECK: %[[BH2:.*]] = llvm.intr.smax(%[[HR2]], %{{.*}}) : (i32, i32) -> i32
  // CHECK: triton_gen.2Dblockstore %{{[^,]*}}, %{{[^,]*}}, %[[BH2]], {{.*}}tile_height = 8,
  // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}}tile_height = 8,
  // CHECK-NOT: triton_gen.{{2Dblockstore|predicated_store}}
  // CHECK: llvm.return
  tt.func public @runtime_bounds(%c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %val: tensor<64x128xf32, #mma>, %pid_m: i32, %pid_n: i32, %M: i32, %N: i32) {
    %ld = arith.constant dense<504> : tensor<64x1xi32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %rb = arith.muli %pid_m, %c64_i32 : i32
    %cb = arith.muli %pid_n, %c128_i32 : i32
    %ar = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #r>
    %rs = tt.splat %rb : i32 -> tensor<64xi32, #r>
    %rows = arith.addi %rs, %ar : tensor<64xi32, #r>
    %ac = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #c>
    %cs = tt.splat %cb : i32 -> tensor<128xi32, #c>
    %cols = arith.addi %cs, %ac : tensor<128xi32, #c>
    %er = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #r> -> tensor<64x1xi32, #mma>
    %ec = tt.expand_dims %cols {axis = 0 : i32} : tensor<128xi32, #c> -> tensor<1x128xi32, #mma>
    %mb = tt.splat %M : i32 -> tensor<64x1xi32, #mma>
    %nb = tt.splat %N : i32 -> tensor<1x128xi32, #mma>
    %mr = arith.cmpi slt, %er, %mb : tensor<64x1xi32, #mma>
    %mc = arith.cmpi slt, %ec, %nb : tensor<1x128xi32, #mma>
    %br = tt.broadcast %mr : tensor<64x1xi1, #mma> -> tensor<64x128xi1, #mma>
    %bc = tt.broadcast %mc : tensor<1x128xi1, #mma> -> tensor<64x128xi1, #mma>
    %m = arith.andi %br, %bc : tensor<64x128xi1, #mma>
    %o = arith.muli %er, %ld : tensor<64x1xi32, #mma>
    %b = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #mma>
    %pr = tt.addptr %b, %o : tensor<64x1x!tt.ptr<f32>, #mma>, tensor<64x1xi32, #mma>
    %pb = tt.broadcast %pr : tensor<64x1x!tt.ptr<f32>, #mma> -> tensor<64x128x!tt.ptr<f32>, #mma>
    %co = tt.broadcast %ec : tensor<1x128xi32, #mma> -> tensor<64x128xi32, #mma>
    %ptr = tt.addptr %pb, %co : tensor<64x128x!tt.ptr<f32>, #mma>, tensor<64x128xi32, #mma>
    tt.store %ptr, %val, %m {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f32>, #mma>
    tt.store %ptr, %val, %br {ttig.block_io = "row_major"} : tensor<64x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
