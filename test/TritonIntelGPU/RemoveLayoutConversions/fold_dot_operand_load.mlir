// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

// COM: A dot-operand convert that the hoist leaves on a single-use tile load is folded into the load.
// COM: The scale's convert is kept: its target is a slice, not a dot operand.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @scaled_tile_load_folded
  // CHECK: scf.for
  // CHECK-NOT: ttg.convert_layout
  // CHECK: %[[BETA:.*]] = tt.load {{.*}} : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK: ttg.convert_layout %[[BETA]] : {{.*}} -> tensor<64xbf16, #ttg.slice<{dim = 1, parent = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK-NOT: tt.load
  // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<bf16>, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK-NOT: tt.load
  // CHECK: scf.yield
  tt.func public @scaled_tile_load_folded(%a: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %v_ptr: !tt.ptr<bf16>, %beta_ptr: !tt.ptr<bf16>, %n: i32, %out: tensor<64x64x!tt.ptr<f32>, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xbf16, #blocked>
    %stride = arith.constant dense<128> : tensor<64x1xi32, #blocked>
    %cols = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %roff = arith.muli %rows2, %stride : tensor<64x1xi32, #blocked>
    %vsplat = tt.splat %v_ptr : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
    %vrow = tt.addptr %vsplat, %roff : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
    %vbase = tt.broadcast %vrow : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x64x!tt.ptr<bf16>, #blocked>
    %nsplat = tt.splat %n : i32 -> tensor<64x1xi32, #blocked>
    %rmask = arith.cmpi slt, %rows2, %nsplat : tensor<64x1xi32, #blocked>
    %mask = tt.broadcast %rmask : tensor<64x1xi1, #blocked> -> tensor<64x64xi1, #blocked>
    %r = scf.for %iv = %c0 to %c128 step %c64 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #dpas>) : i32 {
      %offs = tt.splat %iv : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %c = arith.addi %offs, %cols : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %koffs = tt.splat %iv : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %rk = arith.addi %koffs, %rows : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %bsplat = tt.splat %beta_ptr : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %bptrs = tt.addptr %bsplat, %rk : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %beta = tt.load %bptrs : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %beta2 = tt.expand_dims %beta {axis = 1 : i32} : tensor<64xbf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xbf16, #blocked>
      %scale = tt.broadcast %beta2 : tensor<64x1xbf16, #blocked> -> tensor<64x64xbf16, #blocked>
      %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
      %cb = tt.broadcast %c2d : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %ptrs = tt.addptr %vbase, %cb : tensor<64x64x!tt.ptr<bf16>, #blocked>, tensor<64x64xi32, #blocked>
      %tile = tt.load %ptrs, %mask, %zero {ttig.block_io = "row_major"} : tensor<64x64x!tt.ptr<bf16>, #blocked>
      %scaled = arith.mulf %tile, %scale : tensor<64x64xbf16, #blocked>
      %b = ttg.convert_layout %scaled : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x64xf32, #dpas>
      scf.yield %d : tensor<64x64xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x64x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: The same fold applies to a descriptor load feeding dot operand A.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @scaled_descriptor_load_folded
  // CHECK: tt.descriptor_load {{.*}} -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.dot
  tt.func public @scaled_descriptor_load_folded(%desc: !tt.tensordesc<64x64xbf16>, %b: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %beta_ptr: !tt.ptr<bf16>, %out: tensor<64x64x!tt.ptr<f32>, #dpas>) {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bsplat = tt.splat %beta_ptr : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bptrs = tt.addptr %bsplat, %rows : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %beta = tt.load %bptrs : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %beta2 = tt.expand_dims %beta {axis = 1 : i32} : tensor<64xbf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xbf16, #blocked>
    %scale = tt.broadcast %beta2 : tensor<64x1xbf16, #blocked> -> tensor<64x64xbf16, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #blocked>
    %scaled = arith.mulf %tile, %scale : tensor<64x64xbf16, #blocked>
    %a = ttg.convert_layout %scaled : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x64xf32, #dpas>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: A tile load with a second live use keeps its convert, since folding would duplicate the load.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [4, 2]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @tile_load_with_second_use_kept
  // CHECK: %[[TILE:.*]] = tt.descriptor_load {{.*}} -> tensor<64x64xbf16, #blocked>
  // CHECK: ttg.convert_layout %[[TILE]] : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
  // CHECK-NOT: tt.descriptor_load
  tt.func public @tile_load_with_second_use_kept(%desc: !tt.tensordesc<64x64xbf16>, %b: tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %beta_ptr: !tt.ptr<bf16>, %out: tensor<64x64x!tt.ptr<f32>, #dpas>, %copy: tensor<64x64x!tt.ptr<bf16>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #dpas>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bsplat = tt.splat %beta_ptr : !tt.ptr<bf16> -> tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %bptrs = tt.addptr %bsplat, %rows : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %beta = tt.load %bptrs : tensor<64x!tt.ptr<bf16>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %beta2 = tt.expand_dims %beta {axis = 1 : i32} : tensor<64xbf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xbf16, #blocked>
    %scale = tt.broadcast %beta2 : tensor<64x1xbf16, #blocked> -> tensor<64x64xbf16, #blocked>
    %tile = tt.descriptor_load %desc[%c0, %c0] {ttig.block_io = "row_major"} : !tt.tensordesc<64x64xbf16> -> tensor<64x64xbf16, #blocked>
    tt.store %copy, %tile : tensor<64x64x!tt.ptr<bf16>, #blocked>
    %scaled = arith.mulf %tile, %scale : tensor<64x64xbf16, #blocked>
    %a = ttg.convert_layout %scaled : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>
    %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x64xf32, #dpas>
    tt.store %out, %d : tensor<64x64x!tt.ptr<f32>, #dpas>
    tt.return
  }
}
