// RUN: triton-opt %s -split-input-file -tritonintelgpu-match-target-size | FileCheck %s

#warp = #triton_intel_gpu.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#dot0_ = #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>
#dot1_ = #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>

// COM: Test code generation for the 'tritonintelgpu-match-target-size' transformation.
module {
  tt.func public @matmul_kernel_with_block_pointers_without_convertlayout(
    %arg0: !tt.ptr<f16, 1>, %arg1: !tt.ptr<f16, 1>, %arg2: !tt.ptr<f16, 1>, %arg3: i32, %arg4: i32,
    %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    // CHECK-LABEL: @matmul_kernel_with_block_pointers_without_convertlayout
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #warp>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.addi %arg3, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg4, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %2, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %4, %9 : i32
    %11 = arith.minsi %10, %c8_i32 : i32
    %12 = arith.remsi %2, %11 : i32
    %13 = arith.addi %9, %12 : i32
    %14 = arith.remsi %2, %7 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = arith.extsi %arg3 : i32 to i64
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = arith.extsi %arg6 : i32 to i64
    %20 = arith.divsi %1, %c4_i32 : i32
    %21 = arith.remsi %20, %c8_i32 : i32
    %22 = arith.muli %21, %c32_i32 : i32
    %23 = arith.addi %22, %16 : i32
    %24 = tt.make_tensor_ptr %arg0, [%17, %18], [%19, %c1_i64], [%23, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16, #dot0_>>
    %25 = arith.muli %15, %c128_i32 : i32
    %26 = arith.extsi %arg4 : i32 to i64
    %27 = arith.extsi %arg7 : i32 to i64
    %28 = arith.remsi %1, %c4_i32 : i32
    %29 = arith.remsi %28, %c4_i32 : i32
    %30 = arith.muli %29, %c64_i32 : i32
    %31 = arith.addi %30, %25 : i32
    %32 = tt.make_tensor_ptr %arg1, [%18, %26], [%27, %c1_i64], [%c0_i32, %31] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #dot1_>>
    %33:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10=%cst, %arg11=%24, %arg12=%32)
          -> (tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #dot0_>>, !tt.ptr<tensor<32x64xf16, #dot1_>>) : i32 {
      %37 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16, #dot0_>>
      %38 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x64xf16, #dot1_>>
      %39 = tt.dot %37, %38, %arg10 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf16, #dot0_> * tensor<32x64xf16, #dot1_> -> tensor<32x64xf32, #warp>
      // CHECK: scf.for
      // CHECK: [[A:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x32xf16>>
      // CHECK: [[B0:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x32xf16>>
      // CHECK: [[B1:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x32xf16>>
      // CHECK: [[subA0:%.*]] = triton_intel_gpu.extract [[A]][0] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB0:%.*]] = triton_intel_gpu.extract [[B0]][0] : tensor<32x32xf16> -> tensor<16x16xf16>
      // CHECK: [[subC0:%.*]] = tt.dot [[subA0]], [[subB0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      // CHECK: [[subA1:%.*]] = triton_intel_gpu.extract [[A]][4] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB1:%.*]] = triton_intel_gpu.extract [[B0]][1] : tensor<32x32xf16> -> tensor<16x16xf16>
      // CHECK: [[subC1:%.*]] = tt.dot [[subA1]], [[subB1]], [[subC0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %40 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #dot0_>>
      %41 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x64xf16, #dot1_>>
      scf.yield %39, %40, %41 : tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #dot0_>>, !tt.ptr<tensor<32x64xf16, #dot1_>>
    }
    %34 = arith.truncf %33#0 : tensor<32x64xf32, #warp> to tensor<32x64xf16, #warp>
    %35 = arith.extsi %arg8 : i32 to i64
    %36 = tt.make_tensor_ptr %arg2, [%17, %26], [%35, %c1_i64], [%23, %31] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #warp>>
    tt.store %36, %34 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x64xf16, #warp>>
    tt.return
  }
}

// -----

// COM: Test SCF canonicalization: ensure result of loop (containing simplification opportunities) can be
//      consumed by a extract operations.
tt.func public @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                                 %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK-NOT: triton_intel_gpu.glue
  // CHECK:      [[RES:%.*]]:2 = scf.for {{.*}} iter_args([[INIT1:%.*]] = %arg0, [[INIT2:%.*]] = %arg1)
  // CHECK-SAME:               -> (tensor<16x8xf16>, tensor<16x8xf16>) : i32 {
  // CHECK-NEXT:   scf.yield [[INIT2]], [[INIT1]] : tensor<16x8xf16>, tensor<16x8xf16>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[GLUE:%.*]] = triton_intel_gpu.glue [[RES]]#1, [[RES]]#0
  // CHECK-SAME:              : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  // CHECK-NEXT: [[PTR:%.*]] = tt.make_tensor_ptr %arg2
  // CHECK-NEXT: tt.store [[PTR]], [[GLUE]]
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = triton_intel_gpu.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = triton_intel_gpu.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = triton_intel_gpu.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = triton_intel_gpu.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %e3 = triton_intel_gpu.extract %res[0] : tensor<16x16xf16> -> tensor<16x8xf16>
  %e4 = triton_intel_gpu.extract %res[1] : tensor<16x16xf16> -> tensor<16x8xf16>
  %g2 = triton_intel_gpu.glue %e4, %e3 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
  tt.store %ptr, %g2 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

// COM: Test SCF canonicalization: ensure loop is not modified when the result is not used by just 'extract' operations.
tt.func public @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                                 %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK:      [[GLUE:%.*]] = triton_intel_gpu.glue
  // CHECK-NEXT: [[RES:%.*]] = scf.for {{.*}} iter_args([[INIT1:%.*]] = [[GLUE]]) -> (tensor<16x16xf16>) : i32 {
  // CHECK:        scf.yield {{.*}} : tensor<16x16xf16>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[PTR:%.*]] = tt.make_tensor_ptr %arg2
  // CHECK-NEXT: tt.store [[PTR]], [[RES]]
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = triton_intel_gpu.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = triton_intel_gpu.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = triton_intel_gpu.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = triton_intel_gpu.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
  tt.store %ptr, %res {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

// COM: Test SCF canonicalization: ensure loop is not modified if any user of a 'glue' init value is not an 'extract' operation.
tt.func public @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                                 %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK:      [[GLUE:%.*]] = triton_intel_gpu.glue
  // CHECK-NEXT: [[RES:%.*]] = scf.for {{.*}} iter_args([[INIT1:%.*]] = [[GLUE]]) -> (tensor<16x16xf16>) : i32 {
  // CHECK:        scf.yield {{.*}} : tensor<16x16xf16>
  // CHECK-NEXT: }
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = triton_intel_gpu.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = triton_intel_gpu.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = triton_intel_gpu.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = triton_intel_gpu.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
    tt.store %ptr, %arg {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
    scf.yield %g1 : tensor<16x16xf16>
  }
  tt.return
}

// -----

// COM: Test transformation for int8 datatype

// CHECK-LABEL: @matmul_kernel_with_block_pointers
#warp = #triton_intel_gpu.warp<{sizePerThread = [8, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg5: i32) {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
  %cst = arith.constant dense<0> : tensor<8x32xi32, #warp>
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  // CHECK: %[[TPTR_A:.*]] = tt.make_tensor_ptr %arg0, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%{{.*}}, %[[C0]]]
  // CHECK: %[[TPTR_B1:.*]] = tt.make_tensor_ptr %arg1, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%[[C0]], %{{.*}}]
  // CHECK: %[[TPTR_B2:.*]] = tt.make_tensor_ptr %arg1, [%{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}], [%[[C32]], %{{.*}}]
  %tptr_a = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>>
  %tptr_b = tt.make_tensor_ptr %arg1, [%c0_i64,%c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>>
  // CHECK: %[[LOOP_RES:.*]]:5 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER_1:.*]] = %{{.*}}, %[[ITER_2:.*]] = %{{.*}}, %[[TPTR_A_ITER:.*]] = %[[TPTR_A]], %[[TPTR_B1_ITER:.*]] = %[[TPTR_B1]], %[[TPTR_B2_ITER:.*]] = %[[TPTR_B2]])
  %35:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %tptr_a, %arg12 = %tptr_b) -> (tensor<8x32xi32, #warp>, !tt.ptr<tensor<8x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>>, !tt.ptr<tensor<64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>>)  : i32 {
    // CHECK: %[[LD_A:.*]] = tt.load %[[TPTR_A_ITER]] {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x64xi8>>
    // CHECK: %[[LD_B1:.*]] = tt.load %[[TPTR_B1_ITER]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xi8>>
    // CHECK: %[[LD_B2:.*]] = tt.load %[[TPTR_B2_ITER]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xi8>>
    %46 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>>
    %47 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>>
    // CHECK: %[[EX_A_0:.*]] = triton_intel_gpu.extract %[[LD_A]][0] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: %[[EX_B1_0:.*]] = triton_intel_gpu.extract %[[LD_B1]][0] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: %[[DOT_1:.*]] = tt.dot %[[EX_A_0]], %[[EX_B1_0]], %[[ITER_1]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    // CHECK: %[[EX_A_1:.*]] = triton_intel_gpu.extract %[[LD_A]][1] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: %[[EX_B2_0:.*]] = triton_intel_gpu.extract %[[LD_B2]][0] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: %[[DOT_2:.*]] = tt.dot %[[EX_A_1]], %[[EX_B2_0]], %[[DOT_1]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    // CHECK: %[[EX_A_0:.*]] = triton_intel_gpu.extract %[[LD_A]][0] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: %[[EX_B1_1:.*]] = triton_intel_gpu.extract %[[LD_B1]][1] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: %[[DOT_3:.*]] = tt.dot %[[EX_A_0]], %[[EX_B1_1]], %[[ITER_2]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    // CHECK: %[[EX_A_1:.*]] = triton_intel_gpu.extract %[[LD_A]][1] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: %[[EX_B2_1:.*]] = triton_intel_gpu.extract %[[LD_B2]][1] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: %[[DOT_4:.*]] = tt.dot %[[EX_A_1]], %[[EX_B2_1]], %[[DOT_3]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    %48 = tt.dot %46, %47, %arg10, inputPrecision = tf32 : tensor<8x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>> * tensor<64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<8x32xi32, #warp>
    // CHECK: %[[ADV_A:.*]] = tt.advance %[[TPTR_A_ITER]],
    // CHECK: %[[ADV_B1:.*]] = tt.advance %[[TPTR_B1_ITER]],
    // CHECK: %[[ADV_B2:.*]] = tt.advance %[[TPTR_B2_ITER]],
    %49 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<8x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>>
    %50 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>>
    // CHECK: scf.yield %[[DOT_2]], %[[DOT_4]], %[[ADV_A]], %[[ADV_B1]], %[[ADV_B2]]
    scf.yield %48, %49, %50 : tensor<8x32xi32, #warp>, !tt.ptr<tensor<8x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #warp}>>>, !tt.ptr<tensor<64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #warp}>>>
  } {triton_gpu.workload = 3 : i32}
  // CHECK: %[[TPTR_C1:.*]] = tt.make_tensor_ptr %arg2,
  // CHECK: %[[TPTR_C2:.*]] = tt.make_tensor_ptr %arg2,
  %tptr_c = tt.make_tensor_ptr %arg2, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xi32, #warp>>
  // CHECK: tt.store %[[TPTR_C1:.*]], %[[LOOP_RES]]#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xi32>>
  // CHECK: tt.store %[[TPTR_C2:.*]], %[[LOOP_RES]]#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xi32>>
  tt.store %tptr_c, %35#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x32xi32, #warp>>
  tt.return
}
