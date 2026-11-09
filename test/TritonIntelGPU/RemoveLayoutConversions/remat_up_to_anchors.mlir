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
