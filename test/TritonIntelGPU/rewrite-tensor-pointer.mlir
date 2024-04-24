// RUN: triton-opt %s -split-input-file -tritonintelgpu-rewrite-tensor-pointer=device-architecture=PVC | FileCheck %s

// COM: Case1:
// COM: Block Pointers satisfy 3 conditions will not be rewrited
// COM:  - has triton_intel_gpu.dpas layout attribute
// COM:  - is row major: strides[order[0]] == 1
// COM:  - pitch is divisable by QW: strides[order[1]] % (64 / elemTypeBitWidth) == 0
// CHECK: #[[BLOCKED:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
// CHECK: #[[DPAS:.+]] = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 16], order = [1, 0]}>
#dpas = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [16, 4], A = [8, 16], B = [16, 16], C = [8, 16]}>
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK_LABLE:  @matmul_kernel_with_block_pointers
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c4_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c256_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK: tt.make_tensor_ptr {{.*}}, {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}], {{\[}}{{.*}}, {{.*}}] {order = array<i32: 1, 0>} : <tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>)  : i32 {
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
      // CHECK: tt.load {{.*}} {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      // CHECK:  tt.dot {{.*}}, {{.*}}, {{.*}}, inputPrecision = tf32 : tensor<256x32xf32, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>> * tensor<32x256xf32, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>> -> tensor<256x256xf32, #[[DPAS]]>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<256x32xf16, {{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>>
      // CHECK:  tt.advance {{.*}}, {{\[}}{{.*}}, {{.*}}] : <tensor<32x256xf16, {{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>>
      %30 = tt.fp_to_fp %28 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> -> tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>
      %31 = tt.fp_to_fp %29 : tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<256x256xf32, #dpas>
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      scf.yield %32, %33, %34 : tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    }
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #dpas> to tensor<256x256xf16, #dpas>
    %25 = triton_gpu.convert_layout %24 : tensor<256x256xf16, #dpas> -> tensor<256x256xf16, #blocked>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked>>
    // CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked>>
    tt.return
  }
}

// COM: Case2:
// COM: Block Pointers with no divisibility will be rewrited
module attributes {"triton_gpu.compute-capability" = 0 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_block_pointers_indivisible(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK_LABLE:  @matmul_kernel_with_block_pointers_indivisible
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c4_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c256_i32 : i32
    %15 = arith.extsi %arg3 : i32 to i64
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = arith.extsi %arg6 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %18 = tt.make_tensor_ptr %arg0, [%15, %16], [%17, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
    %19 = arith.muli %13, %c256_i32 : i32
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %22 = tt.make_tensor_ptr %arg1, [%16, %20], [%21, %c1_i64], [%c0_i32, %19] {order = array<i32: 1, 0>} : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    %23:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>)  : i32 {
      // CHECK: tt.load {{.*}}, {{.*}} : tensor<256x32x!tt.ptr<f16>, #{{.*}}<{opIdx = 0, parent = #[[DPAS]]}>>
      // CHECK: tt.load {{.*}}, {{.*}} : tensor<32x256x!tt.ptr<f16>, #{{.*}}<{opIdx = 1, parent = #[[DPAS]]}>>
      %28 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      %29 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      %30 = tt.fp_to_fp %28 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> -> tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>
      %31 = tt.fp_to_fp %29 : tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>
      %32 = tt.dot %30, %31, %arg10, inputPrecision = tf32 : tensor<256x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>> * tensor<32x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>> -> tensor<256x256xf32, #dpas>
      // CHECK-NOT: tt.advance
      %33 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>
      // CHECK-NOT: tt.advance
      %34 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
      scf.yield %32, %33, %34 : tensor<256x256xf32, #dpas>, !tt.ptr<tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #dpas}>>>, !tt.ptr<tensor<32x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #dpas}>>>
    }
    %24 = arith.truncf %23#0 : tensor<256x256xf32, #dpas> to tensor<256x256xf16, #dpas>
    %25 = triton_gpu.convert_layout %24 : tensor<256x256xf16, #dpas> -> tensor<256x256xf16, #blocked>
    %26 = arith.extsi %arg8 : i32 to i64
    // CHECK-NOT: tt.make_tensor_ptr
    %27 = tt.make_tensor_ptr %arg2, [%15, %20], [%26, %c1_i64], [%14, %19] {order = array<i32: 1, 0>} : <tensor<256x256xf16, #blocked>>
    // CHECK: tt.store {{.*}}, {{.*}}, {{.*}} : tensor<256x256x!tt.ptr<f16>, #[[BLOCKED]]>
    tt.store %27, %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<256x256xf16, #blocked>>
    tt.return
  }
}

// COM: Case3:
// COM: Tensor Pointer without layout attr will be rewrited
tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
  %c31_i32 = arith.constant 31 : i32
  %c127_i32 = arith.constant 127 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<128x32xf32>
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %c128_i32 = arith.constant 128 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = arith.addi %arg3, %c127_i32 : i32
  %3 = arith.divsi %2, %c128_i32 : i32
  %4 = arith.addi %arg4, %c31_i32 : i32
  %5 = arith.divsi %4, %c32_i32 : i32
  %6 = arith.muli %5, %c8_i32 : i32
  %7 = arith.divsi %0, %6 : i32
  %8 = arith.muli %7, %c8_i32 : i32
  %9 = arith.subi %3, %8 : i32
  %10 = arith.cmpi slt, %9, %c8_i32 : i32
  %11 = arith.select %10, %9, %c8_i32 : i32
  %12 = arith.remsi %0, %11 : i32
  %13 = arith.addi %8, %12 : i32
  %14 = arith.remsi %0, %6 : i32
  %15 = arith.divsi %14, %11 : i32
  %16 = arith.muli %13, %c128_i32 : i32
  %17 = arith.muli %1, %c32_i32 : i32
  %18 = arith.extsi %arg3 : i32 to i64
  %19 = arith.extsi %arg5 : i32 to i64
  %20 = arith.extsi %arg6 : i32 to i64
  // CHECK-NOT: tt.make_tensor_ptr
  %21 = tt.make_tensor_ptr %arg0, [%18, %19], [%20, %c1_i64], [%16, %17] {order = array<i32: 1, 0>} : !tt.ptr<tensor<128x32xf16>>
  %22 = arith.muli %15, %c32_i32 : i32
  %23 = arith.extsi %arg4 : i32 to i64
  %24 = arith.extsi %arg7 : i32 to i64
  // CHECK-NOT: tt.make_tensor_ptr
  %25 = tt.make_tensor_ptr %arg1, [%19, %23], [%24, %c1_i64], [%17, %22] {order = array<i32: 1, 0>} : !tt.ptr<tensor<32x32xf16>>
  %26 = arith.addi %arg5, %c31_i32 : i32
  %27 = arith.divsi %26, %c32_i32 : i32
  %28 = arith.index_cast %27 : i32 to index
  %29:3 = scf.for %arg9 = %c0 to %28 step %c1 iter_args(%arg10 = %cst, %arg11 = %21, %arg12 = %25) -> (tensor<128x32xf32>, !tt.ptr<tensor<128x32xf16>>, !tt.ptr<tensor<32x32xf16>>) {
    // CHECK: tt.load %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<128x32x!tt.ptr<f16>>
    %55 = tt.load %arg11 {boundaryCheck = array<i32: 1>, padding = 2 : i32} : !tt.ptr<tensor<128x32xf16>>
    // CHECK: tt.load %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}} : tensor<32x32x!tt.ptr<f16>>
    %56 = tt.load %arg12 {boundaryCheck = array<i32: 0>, padding = 2 : i32} : !tt.ptr<tensor<32x32xf16>>
    %57 = tt.dot %55, %56, %arg10 : tensor<128x32xf16> * tensor<32x32xf16> -> tensor<128x32xf32>
    // CHECK-NOT: tt.advance
    %58 = tt.advance %arg11, [%c0_i32, %c32_i32] : !tt.ptr<tensor<128x32xf16>>
    // CHECK-NOT: tt.advance
    %59 = tt.advance %arg12, [%c32_i32, %c0_i32] : !tt.ptr<tensor<32x32xf16>>
    scf.yield %57, %58, %59 : tensor<128x32xf32>, !tt.ptr<tensor<128x32xf16>>, !tt.ptr<tensor<32x32xf16>>
  }
  %30 = arith.truncf %29#0 : tensor<128x32xf32> to tensor<128x32xf16>
  %31 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %32 = tt.splat %16 : i32 -> tensor<128xi32>
  %33 = arith.addi %32, %31 : tensor<128xi32>
  %34 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %35 = tt.splat %22 : i32 -> tensor<32xi32>
  %36 = arith.addi %35, %34 : tensor<32xi32>
  %37 = tt.expand_dims %33 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %38 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
  %39 = arith.muli %37, %38 : tensor<128x1xi32>
  %40 = tt.expand_dims %36 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
  %41 = tt.broadcast %39 : tensor<128x1xi32> -> tensor<128x32xi32>
  %42 = tt.broadcast %40 : tensor<1x32xi32> -> tensor<128x32xi32>
  %43 = arith.addi %41, %42 : tensor<128x32xi32>
  %44 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
  %45 = tt.addptr %44, %43 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
  %46 = tt.splat %arg3 : i32 -> tensor<128xi32>
  %47 = arith.cmpi slt, %33, %46 : tensor<128xi32>
  %48 = tt.expand_dims %47 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
  %49 = tt.splat %arg4 : i32 -> tensor<32xi32>
  %50 = arith.cmpi slt, %36, %49 : tensor<32xi32>
  %51 = tt.expand_dims %50 {axis = 0 : i32} : tensor<32xi1> -> tensor<1x32xi1>
  %52 = tt.broadcast %48 : tensor<128x1xi1> -> tensor<128x32xi1>
  %53 = tt.broadcast %51 : tensor<1x32xi1> -> tensor<128x32xi1>
  %54 = arith.andi %52, %53 : tensor<128x32xi1>
  tt.store %45, %30, %54 : tensor<128x32x!tt.ptr<f16>>
  tt.return
}
