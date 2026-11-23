// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// COM: The row scale load is relabeled into the DPAS slice.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @row_scale_relabeled_into_dpas_slice
  // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: scf.for
  // CHECK-NOT: ttg.convert_layout
  // CHECK: scf.yield
  tt.func @row_scale_relabeled_into_dpas_slice(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %scale_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %scale = tt.load %ptrs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x32xf32, #dpas>) : i32 {
      %d = tt.dot %a, %b, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xf32, #dpas>
      %kp = tt.addptr %k_ptr, %iv : !tt.ptr<f32>, i32
      %k = tt.load %kp : !tt.ptr<f32>
      %ks = tt.splat %k : f32 -> tensor<64x1xf32, #blocked>
      %p = arith.mulf %row, %ks : tensor<64x1xf32, #blocked>
      %pb = tt.broadcast %p : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %pc = ttg.convert_layout %pb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dpas>
      %m = arith.mulf %d, %pc : tensor<64x32xf32, #dpas>
      %acc2 = arith.addf %acc, %m : tensor<64x32xf32, #dpas>
      scf.yield %acc2 : tensor<64x32xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: The scale is also stored, so it is converted once before the loop.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @shared_row_scale_converted_once_before_loop
  // CHECK: %[[SCALE:.*]] = tt.load {{.*}} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK: ttg.convert_layout %[[SCALE]] : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #{{.*}}}>>
  // CHECK: scf.for
  // CHECK-NOT: ttg.convert_layout
  // CHECK: scf.yield
  tt.func @shared_row_scale_converted_once_before_loop(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %scale_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #dpas>, %copy: tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %scale = tt.load %ptrs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.store %copy, %scale : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x32xf32, #dpas>) : i32 {
      %d = tt.dot %a, %b, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xf32, #dpas>
      %kp = tt.addptr %k_ptr, %iv : !tt.ptr<f32>, i32
      %k = tt.load %kp : !tt.ptr<f32>
      %ks = tt.splat %k : f32 -> tensor<64x1xf32, #blocked>
      %p = arith.mulf %row, %ks : tensor<64x1xf32, #blocked>
      %pb = tt.broadcast %p : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %pc = ttg.convert_layout %pb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dpas>
      %m = arith.mulf %d, %pc : tensor<64x32xf32, #dpas>
      %acc2 = arith.addf %acc, %m : tensor<64x32xf32, #dpas>
      scf.yield %acc2 : tensor<64x32xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: A row scale loaded inside the loop is relabeled too.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @loop_variant_row_scale_relabeled_into_dpas_slice
  // CHECK-NOT: ttg.convert_layout
  // CHECK: scf.for
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #mma}>>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: scf.yield
  tt.func @loop_variant_row_scale_relabeled_into_dpas_slice(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %scale_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #dpas>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x32xf32, #dpas>) : i32 {
      %d = tt.dot %a, %b, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xf32, #dpas>
      %off = arith.muli %iv, %c64 : i32
      %offs = tt.splat %off : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %p = tt.addptr %ptrs, %offs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %scale = tt.load %p : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %rb = tt.broadcast %row : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %rc = ttg.convert_layout %rb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dpas>
      %m = arith.mulf %d, %rc : tensor<64x32xf32, #dpas>
      %acc2 = arith.addf %acc, %m : tensor<64x32xf32, #dpas>
      scf.yield %acc2 : tensor<64x32xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: The target is a blocked layout, which is not final yet, so the convert
// COM: stays in the loop.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @row_scale_blocked_target_keeps_convert_in_loop
  // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f32>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: scf.for
  // CHECK: ttg.convert_layout
  // CHECK: tt.store
  tt.func @row_scale_blocked_target_keeps_convert_in_loop(%scale_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #blocked1>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %scale = tt.load %ptrs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    scf.for %iv = %c0 to %n step %c1 : i32 {
      %kp = tt.addptr %k_ptr, %iv : !tt.ptr<f32>, i32
      %k = tt.load %kp : !tt.ptr<f32>
      %ks = tt.splat %k : f32 -> tensor<64x1xf32, #blocked>
      %p = arith.mulf %row, %ks : tensor<64x1xf32, #blocked>
      %pb = tt.broadcast %p : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %pc = ttg.convert_layout %pb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #blocked1>
      tt.store %out, %pc : tensor<64x32x!tt.ptr<f32>, #blocked1>
    }
    tt.return
  }
}

// -----

// COM: A load inside the loop with a second use cannot be relabeled, so the convert stays.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @two_use_row_scale_in_loop_keeps_convert
  // CHECK: scf.for
  // CHECK: ttg.convert_layout
  // CHECK: scf.yield
  tt.func @two_use_row_scale_in_loop_keeps_convert(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %scale_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #dpas>, %copy: tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x32xf32, #dpas>) : i32 {
      %scale = tt.load %ptrs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
      tt.store %copy, %scale : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
      %d = tt.dot %a, %b, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xf32, #dpas>
      %kp = tt.addptr %k_ptr, %iv : !tt.ptr<f32>, i32
      %k = tt.load %kp : !tt.ptr<f32>
      %ks = tt.splat %k : f32 -> tensor<64x1xf32, #blocked>
      %p = arith.mulf %row, %ks : tensor<64x1xf32, #blocked>
      %pb = tt.broadcast %p : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %pc = ttg.convert_layout %pb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dpas>
      %m = arith.mulf %d, %pc : tensor<64x32xf32, #dpas>
      %acc2 = arith.addf %acc, %m : tensor<64x32xf32, #dpas>
      scf.yield %acc2 : tensor<64x32xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: The scale pointer is also stored with another layout, so the load keeps its layout.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @stored_row_scale_pointer_keeps_convert
  // CHECK: scf.for
  // CHECK: ttg.convert_layout
  // CHECK: scf.yield
  tt.func @stored_row_scale_pointer_keeps_convert(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %scale_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #dpas>, %val: tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %scale = tt.load %ptrs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    %range0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %splat0 = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    %wptrs = tt.addptr %splat0, %range0 : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.store %wptrs, %val : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x32xf32, #dpas>) : i32 {
      %d = tt.dot %a, %b, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xf32, #dpas>
      %kp = tt.addptr %k_ptr, %iv : !tt.ptr<f32>, i32
      %k = tt.load %kp : !tt.ptr<f32>
      %ks = tt.splat %k : f32 -> tensor<64x1xf32, #blocked>
      %p = arith.mulf %row, %ks : tensor<64x1xf32, #blocked>
      %pb = tt.broadcast %p : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %pc = ttg.convert_layout %pb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dpas>
      %m = arith.mulf %d, %pc : tensor<64x32xf32, #dpas>
      %acc2 = arith.addf %acc, %m : tensor<64x32xf32, #dpas>
      scf.yield %acc2 : tensor<64x32xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: The expanded scale is also stored, so relabeling would load it twice.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @forked_row_scale_keeps_convert
  // CHECK: tt.load {{.*}} : tensor<64x!tt.ptr<f32>
  // CHECK-NOT: tt.load {{.*}} : tensor<64x!tt.ptr<f32>
  // CHECK: scf.for
  // CHECK: ttg.convert_layout
  // CHECK: scf.yield
  tt.func @forked_row_scale_keeps_convert(%a: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>, %b: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>, %scale_ptr: !tt.ptr<f32>, %k_ptr: !tt.ptr<f32>, %n: i32, %out: tensor<64x32x!tt.ptr<f32>, #dpas>, %rowout: tensor<64x1x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #dpas>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %splat = tt.splat %scale_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %ptrs = tt.addptr %splat, %range : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %scale = tt.load %ptrs : tensor<64x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row = tt.expand_dims %scale {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xf32, #blocked>
    tt.store %rowout, %row : tensor<64x1x!tt.ptr<f32>, #blocked>
    %r = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<64x32xf32, #dpas>) : i32 {
      %d = tt.dot %a, %b, %cst : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>> -> tensor<64x32xf32, #dpas>
      %kp = tt.addptr %k_ptr, %iv : !tt.ptr<f32>, i32
      %k = tt.load %kp : !tt.ptr<f32>
      %ks = tt.splat %k : f32 -> tensor<64x1xf32, #blocked>
      %p = arith.mulf %row, %ks : tensor<64x1xf32, #blocked>
      %pb = tt.broadcast %p : tensor<64x1xf32, #blocked> -> tensor<64x32xf32, #blocked>
      %pc = ttg.convert_layout %pb : tensor<64x32xf32, #blocked> -> tensor<64x32xf32, #dpas>
      %m = arith.mulf %d, %pc : tensor<64x32xf32, #dpas>
      %acc2 = arith.addf %acc, %m : tensor<64x32xf32, #dpas>
      scf.yield %acc2 : tensor<64x32xf32, #dpas>
    }
    tt.store %out, %r : tensor<64x32x!tt.ptr<f32>, #dpas>
    tt.return
  }
}

// -----

// COM: The scales of a tt.dot_scaled are loaded in the K loop: both loads are relabeled into the
// COM: scale layouts of the block scaled DPAS and no convert is left in the loop (#8931).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [16, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[8, 0], [16, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [0, 0]], warp = [[0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[16, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>

module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-DAG: #[[$SCALE_A_LAYOUT:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
  // CHECK-DAG: #[[$SCALE_B_LAYOUT:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
  // CHECK-LABEL: @dot_scaled_scale_loads_relabeled
  // CHECK: scf.for
  // CHECK-NOT: ttg.convert_layout
  // CHECK-DAG: %[[SCALE_A:.*]] = tt.load {{.*}} : tensor<128x2x!tt.ptr<i8>, #[[$SCALE_A_LAYOUT]]>
  // CHECK-DAG: %[[SCALE_B:.*]] = tt.load {{.*}} : tensor<128x2x!tt.ptr<i8>, #[[$SCALE_B_LAYOUT]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.dot_scaled %{{.*}} scale %[[SCALE_A]], %{{.*}} scale %[[SCALE_B]]
  // CHECK: scf.yield
  tt.func @dot_scaled_scale_loads_relabeled(%a: tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %b: tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %A_scale: !tt.ptr<i8>, %B_scale: !tt.ptr<i8>, %K: i32, %groups: i32, %out: tensor<128x128x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_g = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %pitch = tt.splat %groups : i32 -> tensor<128x1xi32, #blocked>
    %row_off = arith.muli %rows2, %pitch : tensor<128x1xi32, #blocked>
    %a_base = tt.splat %A_scale : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
    %a_rows = tt.addptr %a_base, %row_off : tensor<128x1x!tt.ptr<i8>, #blocked>, tensor<128x1xi32, #blocked>
    %a_rows2 = tt.broadcast %a_rows : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x2x!tt.ptr<i8>, #blocked>
    %b_base = tt.splat %B_scale : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
    %b_rows = tt.addptr %b_base, %row_off : tensor<128x1x!tt.ptr<i8>, #blocked>, tensor<128x1xi32, #blocked>
    %b_rows2 = tt.broadcast %b_rows : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x2x!tt.ptr<i8>, #blocked>
    %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
      %g = arith.divui %k, %c32 : i32
      %gs = tt.splat %g : i32 -> tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %gk = arith.addi %gs, %offs_g : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %gk2 = tt.expand_dims %gk {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x2xi32, #blocked>
      %gkb = tt.broadcast %gk2 : tensor<1x2xi32, #blocked> -> tensor<128x2xi32, #blocked>
      %a_ptr = tt.addptr %a_rows2, %gkb : tensor<128x2x!tt.ptr<i8>, #blocked>, tensor<128x2xi32, #blocked>
      %a_sc = tt.load %a_ptr : tensor<128x2x!tt.ptr<i8>, #blocked>
      %b_ptr = tt.addptr %b_rows2, %gkb : tensor<128x2x!tt.ptr<i8>, #blocked>, tensor<128x2xi32, #blocked>
      %b_sc = tt.load %b_ptr : tensor<128x2x!tt.ptr<i8>, #blocked>
      %a_scl = ttg.convert_layout %a_sc : tensor<128x2xi8, #blocked> -> tensor<128x2xi8, #linear>
      %b_scl = ttg.convert_layout %b_sc : tensor<128x2xi8, #blocked> -> tensor<128x2xi8, #linear1>
      %d = tt.dot_scaled %a scale %a_scl, %b scale %b_scl, %acc lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x2xi8, #linear> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<128x2xi8, #linear1> -> tensor<128x128xf32, #mma>
      scf.yield %d : tensor<128x128xf32, #mma>
    }
    tt.store %out, %r : tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// COM: The A scale is also stored from the loop, so relabeling its load would read it twice:
// COM: its convert stays. The B scale load is still relabeled.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [16, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[8, 0], [16, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [0, 0]], warp = [[0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[16, 0], [0, 1]], lane = [[1, 0], [2, 0], [4, 0], [8, 0]], warp = [[32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 4], repCluster = [4, 2], A = [32, 32], B = [32, 32], C = [32, 32]}>

module attributes {"ttg.num-warps" = 16 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-DAG: #[[$SCALE_A_LAYOUT:.+]] = #ttg.linear<{register = {{\[\[}}8, 0], [16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [0, 0]], warp = {{\[\[}}0, 0], [0, 0], [32, 0], [64, 0]], block = []}>
  // CHECK-DAG: #[[$SCALE_B_LAYOUT:.+]] = #ttg.linear<{register = {{\[\[}}16, 0], [0, 1]], lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0]], warp = {{\[\[}}32, 0], [64, 0], [0, 0], [0, 0]], block = []}>
  // CHECK-LABEL: @two_use_dot_scaled_scale_keeps_convert
  // CHECK: scf.for
  // CHECK-DAG: %[[SCALE_A:.*]] = tt.load {{.*}} : tensor<128x2x!tt.ptr<i8>, #blocked>
  // CHECK-DAG: %[[SCALE_B:.*]] = tt.load {{.*}} : tensor<128x2x!tt.ptr<i8>, #[[$SCALE_B_LAYOUT]]>
  // CHECK: %[[SCALE_A_CVT:.*]] = ttg.convert_layout %[[SCALE_A]] : tensor<128x2xi8, #blocked> -> tensor<128x2xi8, #[[$SCALE_A_LAYOUT]]>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.dot_scaled %{{.*}} scale %[[SCALE_A_CVT]], %{{.*}} scale %[[SCALE_B]]
  // CHECK: scf.yield
  tt.func @two_use_dot_scaled_scale_keeps_convert(%a: tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %b: tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %A_scale: !tt.ptr<i8>, %B_scale: !tt.ptr<i8>, %dbg: !tt.ptr<i8>, %K: i32, %out: tensor<128x128x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %a_ptr = tt.splat %A_scale : !tt.ptr<i8> -> tensor<128x2x!tt.ptr<i8>, #blocked>
    %b_ptr = tt.splat %B_scale : !tt.ptr<i8> -> tensor<128x2x!tt.ptr<i8>, #blocked>
    %dbg_ptr = tt.splat %dbg : !tt.ptr<i8> -> tensor<128x2x!tt.ptr<i8>, #blocked>
    %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x128xf32, #mma>) : i32 {
      %ks = tt.splat %k : i32 -> tensor<128x2xi32, #blocked>
      %a_p = tt.addptr %a_ptr, %ks : tensor<128x2x!tt.ptr<i8>, #blocked>, tensor<128x2xi32, #blocked>
      %a_sc = tt.load %a_p : tensor<128x2x!tt.ptr<i8>, #blocked>
      tt.store %dbg_ptr, %a_sc : tensor<128x2x!tt.ptr<i8>, #blocked>
      %b_p = tt.addptr %b_ptr, %ks : tensor<128x2x!tt.ptr<i8>, #blocked>, tensor<128x2xi32, #blocked>
      %b_sc = tt.load %b_p : tensor<128x2x!tt.ptr<i8>, #blocked>
      %a_scl = ttg.convert_layout %a_sc : tensor<128x2xi8, #blocked> -> tensor<128x2xi8, #linear>
      %b_scl = ttg.convert_layout %b_sc : tensor<128x2xi8, #blocked> -> tensor<128x2xi8, #linear1>
      %d = tt.dot_scaled %a scale %a_scl, %b scale %b_scl, %acc lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x2xi8, #linear> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<128x2xi8, #linear1> -> tensor<128x128xf32, #mma>
      scf.yield %d : tensor<128x128xf32, #mma>
    }
    tt.store %out, %r : tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}
