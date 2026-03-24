// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s

// COM: Test that descriptor loads for a tt.dot operation produce ttig.descriptor_prefetch ops
// COM: when pipelined with 3 stages.
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  tt.func public @matmul_kernel_descriptor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64) {
    // CHECK-LABEL:   tt.func public @matmul_kernel_descriptor
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64

    // COM: Create tensor descriptors for A [M, K] and B [K, N].
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <tensor<128x64xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <tensor<64x256xf16>>

    // COM: 3-stage pipeline: 2 prefetching stages before the loop, 1 inside.
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<64x256xf16>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:      ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<64x256xf16>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK:        ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:        ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<tensor<64x256xf16>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // CHECK:        scf.yield
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<64x256xf16>> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

// COM: Reproducer for issue #3810, using tensor descriptors.
// COM: Tests pipelining of a persistent kernel with nested loops.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 1, 4], threadsPerWarp = [1, 2, 16], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [4, 4, 1], threadsPerWarp = [1, 16, 2], warpsPerCTA = [4, 1, 1], order = [1, 2, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 8, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
module attributes {ttig.min_sg_size = 16 : i32, ttig.support_bfloat16_conversion, ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_descriptor_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL: tt.func public @matmul_kernel_descriptor_persistent
    // CHECK-SAME: ([[PARAM_0:%.*]]: !tt.ptr<f16> {{.*}}, [[PARAM_1:%.*]]: !tt.ptr<f16> {{.*}}, [[PARAM_2:%.*]]: !tt.ptr<f16> {{.*}}, [[PARAM_3:%.*]]: i32 {{.*}}, [[PARAM_4:%.*]]: i32 {{.*}}, [[PARAM_5:%.*]]: i32 {{.*}})
    %c448_i32 = arith.constant 448 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg4 : i32 to i64
    %10 = arith.subi %0, %c448_i32 : i32
    %11 = arith.muli %4, %c8_i32 : i32

    // COM: Create tensor descriptors outside both loops.
    // COM:   A has shape [M, K] with strides [K, 1].
    // COM:   B has shape [N, K] with strides [K, 1].
    // COM:   C has shape [M, N] with strides [N, 1].
    // CHECK:      %[[DESC_A:.*]] = tt.make_tensor_descriptor [[PARAM_0]], {{.*}} : <f16>, <tensor<128x64xf16>>
    // CHECK:      %[[DESC_B:.*]] = tt.make_tensor_descriptor [[PARAM_1]], {{.*}} : <f16>, <tensor<128x64xf16>>
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%8, %c1_i64] : <f16>, <tensor<128x64xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%8, %c1_i64] : <f16>, <tensor<128x64xf16>>
    %descC = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%9, %c1_i64] : <f16>, <tensor<128x64xf16>>

    // COM: 3-stage pipeline: 2 prefetching stages before the inner loop, 1 inside.
    // CHECK:      scf.for %[[OUTER_IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (i32)
    // CHECK:        ttig.descriptor_prefetch %[[DESC_A]][{{.*}}] {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:        ttig.descriptor_prefetch %[[DESC_B]][{{.*}}] {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:        ttig.descriptor_prefetch %[[DESC_A]][{{.*}}] {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:        ttig.descriptor_prefetch %[[DESC_B]][{{.*}}] {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK-NEXT:   scf.for %[[INNER_IV:.*]] = {{.*}} to {{.*}} step {{.*}}
    // CHECK:          ttig.descriptor_prefetch %[[DESC_A]][{{.*}}] {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    // CHECK:          ttig.descriptor_prefetch %[[DESC_B]][{{.*}}] {{.*}} : !tt.tensordesc<tensor<128x64xf16>>
    %13 = scf.for %arg6 = %0 to %7 step %c448_i32 iter_args(%arg7 = %10) -> (i32)  : i32 {
      %14 = arith.divsi %arg6, %11 : i32
      %15 = arith.muli %14, %c8_i32 : i32
      %16 = arith.subi %2, %15 : i32
      %17 = arith.minsi %16, %c8_i32 : i32
      %18 = arith.remsi %arg6, %17 : i32
      %19 = arith.addi %15, %18 : i32
      %20 = arith.remsi %arg6, %11 : i32
      %21 = arith.divsi %20, %17 : i32
      %22 = arith.muli %19, %c128_i32 : i32
      %23 = arith.muli %21, %c128_i32 : i32
      %24 = scf.for %arg8 = %c0_i32 to %6 step %c1_i32 iter_args(%arg9 = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
        %44 = arith.muli %arg8, %c64_i32 : i32
        %46 = tt.descriptor_load %descA[%22, %44] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked1>
        %48 = tt.descriptor_load %descB[%23, %44] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<128x64xf16>> -> tensor<128x64xf16, #blocked1>
        %49 = tt.trans %48 {order = array<i32: 1, 0>} : tensor<128x64xf16, #blocked1> -> tensor<64x128xf16, #blocked2>
        %50 = tt.fp_to_fp %46 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf32, #blocked1>
        %51 = ttg.convert_layout %50 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
        %52 = tt.fp_to_fp %49 : tensor<64x128xf16, #blocked2> -> tensor<64x128xf32, #blocked2>
        %53 = ttg.convert_layout %52 : tensor<64x128xf32, #blocked2> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
        %54 = tt.dot %51, %53, %arg9, inputPrecision = tf32 : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
        scf.yield %54 : tensor<128x128xf32, #blocked>
      }
      %25 = arith.addi %arg7, %c448_i32 : i32
      %26 = arith.divsi %25, %11 : i32
      %27 = arith.muli %26, %c8_i32 : i32
      %28 = arith.subi %2, %27 : i32
      %29 = arith.minsi %28, %c8_i32 : i32
      %30 = arith.remsi %25, %29 : i32
      %31 = arith.addi %27, %30 : i32
      %32 = arith.remsi %25, %11 : i32
      %33 = arith.divsi %32, %29 : i32
      %34 = arith.muli %31, %c128_i32 : i32
      %35 = arith.muli %33, %c128_i32 : i32
      %36 = tt.reshape %24 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
      %37 = tt.trans %36 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
      %38 = ttg.convert_layout %37 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64x2xf32, #blocked5>
      %outLHS, %outRHS = tt.split %38 : tensor<128x64x2xf32, #blocked5> -> tensor<128x64xf32, #blocked1>
      %39 = arith.truncf %outLHS : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      tt.descriptor_store %descC[%34, %35], %39 : !tt.tensordesc<tensor<128x64xf16>>, tensor<128x64xf16, #blocked1>
      %41 = arith.truncf %outRHS : tensor<128x64xf32, #blocked1> to tensor<128x64xf16, #blocked1>
      %42 = arith.addi %35, %c64_i32 : i32
      tt.descriptor_store %descC[%34, %42], %41 : !tt.tensordesc<tensor<128x64xf16>>, tensor<128x64xf16, #blocked1>
      scf.yield %25 : i32
    } {tt.flatten}
    tt.return
  }
}

// -----

// COM: Test that rank-3 descriptor loads inside scf.for loops get prefetched when their
// COM: transitive uses flow through blocked encodings to dot operand encodings.
// COM: Load A: 3D blocked -> reshape -> 2D blocked -> convert_layout -> dot_op (opIdx=0)
// COM: Load B: 3D blocked -> reshape -> 2D blocked -> convert_layout -> dot_op (opIdx=1)

#blocked3d = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 4], warpsPerCTA = [1, 2, 4], order = [2, 1, 0]}>
#blocked2d = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 4], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 4], repCluster = [4, 2], A = [32, 16], B = [16, 32], C = [32, 32]}>

module attributes {ttig.support_subgroup_matrix_multiply_accumulate, ttig.support_2d_block_io, ttig.target_arch = "spir64", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: tt.func public @rank3_descriptor_prefetch
  tt.func public @rank3_descriptor_prefetch(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32, %arg6: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.extsi %arg6 : i32 to i64

    // COM: Create 3D tensor descriptors outside the loop.
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg3, %arg6], [%0, %0, %c1_i64] : <f16>, <tensor<1x128x64xf16>>
    %descB = tt.make_tensor_descriptor %arg1, [%arg3, %arg6, %arg6], [%0, %0, %c1_i64] : <f16>, <tensor<1x64x256xf16>>

    // COM: 3-stage pipeline: prefetch iterations 0 and 1 before the loop,
    // COM: iteration i+2 inside the loop.
    // CHECK-DAG:  %[[DESCA:.*]] = tt.make_tensor_descriptor %arg0{{.*}}<tensor<1x128x64xf16>>
    // CHECK-DAG:  %[[DESCB:.*]] = tt.make_tensor_descriptor %arg1{{.*}}<tensor<1x64x256xf16>>
    // COM: Iteration 0: prefetch at initial offset.
    // CHECK:      ttig.descriptor_prefetch %[[DESCA]][%[[C0:.*]], %[[C0]], %[[C0]]] {{.*}} : !tt.tensordesc<tensor<1x128x64xf16>>
    // CHECK:      ttig.descriptor_prefetch %[[DESCB]][%[[C0]], %[[C0]], %[[C0]]] {{.*}} : !tt.tensordesc<tensor<1x64x256xf16>>
    // COM: Iteration 1: prefetch at advanced offset. descA advances dim 2, descB advances dim 1.
    // COM: Capture OFF1 from descA prefetch, verify same value used in descB prefetch.
    // CHECK:      ttig.descriptor_prefetch %[[DESCA]][%[[C0]], %[[C0]], %[[OFF1:.*]]] {{.*}} : !tt.tensordesc<tensor<1x128x64xf16>>
    // CHECK:      ttig.descriptor_prefetch %[[DESCB]][%[[C0]], %[[OFF1]], %[[C0]]] {{.*}} : !tt.tensordesc<tensor<1x64x256xf16>>
    // COM: Inside the loop: prefetch uses an advanced offset computed from arith.addi,
    // COM: load uses the current iteration offset — verifying prefetch is ahead.
    // CHECK:      scf.for %[[IV:.*]] = {{.*}}
    // CHECK:        %[[OFFN:.*]] = arith.addi
    // CHECK:        ttig.descriptor_prefetch %[[DESCA]][%[[C0]], %[[C0]], %[[OFFN]]] {{.*}} : !tt.tensordesc<tensor<1x128x64xf16>>
    // CHECK:        ttig.descriptor_prefetch %[[DESCB]][%[[C0]], %[[OFFN]], %[[C0]]] {{.*}} : !tt.tensordesc<tensor<1x64x256xf16>>
    // CHECK:        tt.descriptor_load %[[DESCA]][%[[C0]], %[[C0]], %[[CURR:.*]]] {{.*}} : !tt.tensordesc<tensor<1x128x64xf16>>
    // CHECK:        tt.descriptor_load %[[DESCB]][%[[C0]], %[[CURR]], %[[C0]]] {{.*}} : !tt.tensordesc<tensor<1x64x256xf16>>
    // CHECK:        tt.dot
    // CHECK:        scf.yield
    %result:2 = scf.for %k = %c0_i32 to %c64_i32 step %c1_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #mma>, i32) : i32 {
      %a3d = tt.descriptor_load %descA[%c0_i32, %c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<1x128x64xf16>> -> tensor<1x128x64xf16, #blocked3d>
      %a2d = tt.reshape %a3d : tensor<1x128x64xf16, #blocked3d> -> tensor<128x64xf16, #blocked2d>
      %a = ttg.convert_layout %a2d : tensor<128x64xf16, #blocked2d> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %b3d = tt.descriptor_load %descB[%c0_i32, %koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<tensor<1x64x256xf16>> -> tensor<1x64x256xf16, #blocked3d>
      %b2d = tt.reshape %b3d : tensor<1x64x256xf16, #blocked3d> -> tensor<64x256xf16, #blocked2d>
      %b = ttg.convert_layout %b2d : tensor<64x256xf16, #blocked2d> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x256xf32, #mma>
      %next_koff = arith.addi %koff, %c1_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #mma>, i32
    }
    tt.return
  }
}
