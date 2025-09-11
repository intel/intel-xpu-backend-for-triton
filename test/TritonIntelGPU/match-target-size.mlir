// RUN: env TRITON_INTEL_ADVANCED_PATH=1 TRITON_INTEL_REDUCE_TRANSPOSE=1 \
// RUN: triton-opt %s -split-input-file -tritonintelgpu-match-target-size | FileCheck %s --check-prefixes=CHECK,CHECK-TR-RED
// RUN: env TRITON_INTEL_ADVANCED_PATH=1 triton-opt %s -split-input-file -tritonintelgpu-match-target-size | FileCheck %s --check-prefixes=CHECK,CHECK-SG-RED

#warp = #ttig.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#dot0_ = #ttg.dot_op<{opIdx = 0, parent = #warp}>
#dot1_ = #ttg.dot_op<{opIdx = 1, parent = #warp}>

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
      // CHECK: [[B0:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x16xf16>>
      // CHECK: [[B1:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x16xf16>>
      // CHECK: [[B2:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x16xf16>>
      // CHECK: [[B3:%.*]] = tt.load {{.*}} : !tt.ptr<tensor<32x16xf16>>
      // CHECK: [[subA0:%.*]] = ttig.extract [[A]][0] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB0:%.*]] = ttig.extract [[B0]][0] : tensor<32x16xf16> -> tensor<16x16xf16>
      // CHECK: [[subC0:%.*]] = tt.dot [[subA0]], [[subB0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      // CHECK: [[subA1:%.*]] = ttig.extract [[A]][4] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB1:%.*]] = ttig.extract [[B0]][1] : tensor<32x16xf16> -> tensor<16x16xf16>
      // CHECK: [[subC1:%.*]] = tt.dot [[subA1]], [[subB1]], [[subC0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %40 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #dot0_>>
      %41 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x64xf16, #dot1_>>
      scf.yield %39, %40, %41 : tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #dot0_>>, !tt.ptr<tensor<32x64xf16, #dot1_>>
    } {ttg.workload = 4 : i32}
    %34 = arith.truncf %33#0 : tensor<32x64xf32, #warp> to tensor<32x64xf16, #warp>
    %35 = arith.extsi %arg8 : i32 to i64
    %36 = tt.make_tensor_ptr %arg2, [%17, %26], [%35, %c1_i64], [%23, %31] {order = array<i32: 1, 0>} : <tensor<32x64xf16, #warp>>
    tt.store %36, %34 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<32x64xf16, #warp>>
    tt.return
  }
}

// -----

#warp = #ttig.warp<{sizePerThread = [32, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#dot0_ = #ttg.dot_op<{opIdx = 0, parent = #warp}>
#dot1_ = #ttg.dot_op<{opIdx = 1, parent = #warp}>

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
      // CHECK: [[subA0:%.*]] = ttig.extract [[A]][0] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB0:%.*]] = ttig.extract [[B0]][0] : tensor<32x32xf16> -> tensor<16x16xf16>
      // CHECK: [[subC0:%.*]] = tt.dot [[subA0]], [[subB0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      // CHECK: [[subA1:%.*]] = ttig.extract [[A]][4] : tensor<32x32xf16> -> tensor<8x16xf16>
      // CHECK: [[subB1:%.*]] = ttig.extract [[B0]][1] : tensor<32x32xf16> -> tensor<16x16xf16>
      // CHECK: [[subC1:%.*]] = tt.dot [[subA1]], [[subB1]], [[subC0]], {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
      %40 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<32x32xf16, #dot0_>>
      %41 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x64xf16, #dot1_>>
      scf.yield %39, %40, %41 : tensor<32x64xf32, #warp>, !tt.ptr<tensor<32x32xf16, #dot0_>>, !tt.ptr<tensor<32x64xf16, #dot1_>>
    } {ttg.workload = 3 : i32}
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
  // CHECK-NOT: ttig.glue
  // CHECK:      [[RES:%.*]]:2 = scf.for {{.*}} iter_args([[INIT1:%.*]] = %arg0, [[INIT2:%.*]] = %arg1)
  // CHECK-SAME:               -> (tensor<16x8xf16>, tensor<16x8xf16>) : i32 {
  // CHECK-NEXT:   scf.yield [[INIT2]], [[INIT1]] : tensor<16x8xf16>, tensor<16x8xf16>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[GLUE:%.*]] = ttig.glue [[RES]]#1, [[RES]]#0
  // CHECK-SAME:              : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  // CHECK-NEXT: [[PTR:%.*]] = tt.make_tensor_ptr %arg2
  // CHECK-NEXT: tt.store [[PTR]], [[GLUE]]
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = ttig.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = ttig.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = ttig.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = ttig.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %e3 = ttig.extract %res[0] : tensor<16x16xf16> -> tensor<16x8xf16>
  %e4 = ttig.extract %res[1] : tensor<16x16xf16> -> tensor<16x8xf16>
  %g2 = ttig.glue %e4, %e3 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
  tt.store %ptr, %g2 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

// COM: Test SCF canonicalization: ensure loop is not modified if any user of a 'glue' init value is not an 'extract' operation.
tt.func public @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                                 %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK:      [[GLUE:%.*]] = ttig.glue
  // CHECK-NEXT: [[RES:%.*]] = scf.for {{.*}} iter_args([[INIT1:%.*]] = [[GLUE]]) -> (tensor<16x16xf16>) : i32 {
  // CHECK:        scf.yield {{.*}} : tensor<16x16xf16>
  // CHECK-NEXT: }
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = ttig.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = ttig.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = ttig.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = ttig.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
    tt.store %ptr, %arg {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
    scf.yield %g1 : tensor<16x16xf16>
  }
  tt.return
}

// -----

// COM: Test SCF canonicalization: ensure loop canonicalization can be applied to dependendent loops
tt.func public @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                                 %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK-NOT: ttig.glue
  // CHECK:      [[RES:%.*]]:2 = scf.for {{.*}} iter_args([[INIT1:%.*]] = %arg0, [[INIT2:%.*]] = %arg1)
  // CHECK-SAME:               -> (tensor<16x8xf16>, tensor<16x8xf16>) : i32 {
  // CHECK-NEXT:   scf.yield [[INIT2]], [[INIT1]] : tensor<16x8xf16>, tensor<16x8xf16>
  // CHECK-NEXT: }
  // CHECK:      [[RES2:%.*]]:2 = scf.for {{.*}} iter_args([[INIT3:%.*]] = [[RES]]#0, [[INIT4:%.*]] = [[RES]]#1)
  // CHECK-SAME:               -> (tensor<16x8xf16>, tensor<16x8xf16>) : i32 {
  // CHECK-NEXT:   scf.yield [[INIT3]], [[INIT4]] : tensor<16x8xf16>, tensor<16x8xf16>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[GLUE:%.*]] = ttig.glue [[RES2]]#1, [[RES2]]#0
  // CHECK-SAME:              : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  // CHECK-NEXT: [[PTR:%.*]] = tt.make_tensor_ptr %arg2
  // CHECK-NEXT: tt.store [[PTR]], [[GLUE]]
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<42.0> : tensor<16x16xf16>
  %glue = ttig.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = ttig.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = ttig.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = ttig.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %res2 = scf.for %iv = %lb to %ub step %st iter_args(%arg = %res) -> (tensor<16x16xf16>) : i32 {
    %e1 = ttig.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = ttig.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = ttig.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %e3 = ttig.extract %res2[0] : tensor<16x16xf16> -> tensor<16x8xf16>
  %e4 = ttig.extract %res2[1] : tensor<16x16xf16> -> tensor<16x8xf16>
  %g2 = ttig.glue %e4, %e3 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
  tt.store %ptr, %g2 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

// COM: Test transformation for int8 datatype

// CHECK-LABEL: @matmul_kernel_with_block_pointers_int8
#warp = #ttig.warp<{sizePerThread = [8, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
tt.func public @matmul_kernel_with_block_pointers_int8(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg5: i32) {
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : i32
  // CHECK-DAG: [[C32:%.*]] = arith.constant 32 : i32
  %cst = arith.constant dense<0> : tensor<8x32xi32, #warp>
  %c0_i32 = arith.constant 0 : i32
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  // CHECK: [[TPTR_A:%.*]] = tt.make_tensor_ptr %arg0, [{{.*}}, {{.*}}], [{{.*}}, {{.*}}], [{{.*}}, [[C0]]]
  // CHECK: [[TPTR_B1:%.*]] = tt.make_tensor_ptr %arg1, [{{.*}}, {{.*}}], [{{.*}}, {{.*}}], [[[C0]], {{.*}}]
  // CHECK: [[TPTR_B2:%.*]] = tt.make_tensor_ptr %arg1, [{{.*}}, {{.*}}], [{{.*}}, {{.*}}], [[[C32]], {{.*}}]
  %tptr_a = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x64xi8, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
  %tptr_b = tt.make_tensor_ptr %arg1, [%c0_i64,%c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  // CHECK: [[LOOP_RES:%.*]]:5 = scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args([[ITER_1:%.*]] = {{.*}}, [[ITER_2:%.*]] = {{.*}}, [[TPTR_A_ITER:%.*]] = [[TPTR_A]], [[TPTR_B1_ITER:%.*]] = [[TPTR_B1]], [[TPTR_B2_ITER:%.*]] = [[TPTR_B2]])
  %35:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %tptr_a, %arg12 = %tptr_b) -> (tensor<8x32xi32, #warp>, !tt.ptr<tensor<8x64xi8, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>, !tt.ptr<tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>)  : i32 {
    // CHECK: [[LD_A:%.*]] = tt.load [[TPTR_A_ITER]] {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x64xi8>>
    // CHECK: [[LD_B1:%.*]] = tt.load [[TPTR_B1_ITER]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xi8>>
    // CHECK: [[LD_B2:%.*]] = tt.load [[TPTR_B2_ITER]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xi8>>
    %46 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x64xi8, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
    %47 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    // CHECK: [[EX_A_0:%.*]] = ttig.extract [[LD_A]][0] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: [[EX_B1_0:%.*]] = ttig.extract [[LD_B1]][0] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: [[DOT_1:%.*]] = tt.dot [[EX_A_0]], [[EX_B1_0]], [[ITER_1]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    // CHECK: [[EX_A_1:%.*]] = ttig.extract [[LD_A]][1] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: [[EX_B2_0:%.*]] = ttig.extract [[LD_B2]][0] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: [[DOT_2:%.*]] = tt.dot [[EX_A_1]], [[EX_B2_0]], [[DOT_1]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    // CHECK: [[EX_A_0:%.*]] = ttig.extract [[LD_A]][0] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: [[EX_B1_1:%.*]] = ttig.extract [[LD_B1]][1] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: [[DOT_3:%.*]] = tt.dot [[EX_A_0]], [[EX_B1_1]], [[ITER_2]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    // CHECK: [[EX_A_1:%.*]] = ttig.extract [[LD_A]][1] : tensor<8x64xi8> -> tensor<8x32xi8>
    // CHECK: [[EX_B2_1:%.*]] = ttig.extract [[LD_B2]][1] : tensor<32x32xi8> -> tensor<32x16xi8>
    // CHECK: [[DOT_4:%.*]] = tt.dot [[EX_A_1]], [[EX_B2_1]], [[DOT_3]], inputPrecision = tf32 : tensor<8x32xi8> * tensor<32x16xi8> -> tensor<8x16xi32>
    %48 = tt.dot %46, %47, %arg10, inputPrecision = tf32 : tensor<8x64xi8, #ttg.dot_op<{opIdx = 0, parent = #warp}>> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<8x32xi32, #warp>
    // CHECK: [[ADV_A:%.*]] = tt.advance [[TPTR_A_ITER]],
    // CHECK: [[ADV_B1:%.*]] = tt.advance [[TPTR_B1_ITER]],
    // CHECK: [[ADV_B2:%.*]] = tt.advance [[TPTR_B2_ITER]],
    %49 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<8x64xi8, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
    %50 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    // CHECK: scf.yield [[DOT_2]], [[DOT_4]], [[ADV_A]], [[ADV_B1]], [[ADV_B2]]
    scf.yield %48, %49, %50 : tensor<8x32xi32, #warp>, !tt.ptr<tensor<8x64xi8, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>, !tt.ptr<tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  } {ttg.workload = 3 : i32}
  // CHECK: [[TPTR_C1:%.*]] = tt.make_tensor_ptr %arg2,
  // CHECK: [[TPTR_C2:%.*]] = tt.make_tensor_ptr %arg2,
  %tptr_c = tt.make_tensor_ptr %arg2, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xi32, #warp>>
  // CHECK: tt.store [[TPTR_C1:%.*]], [[LOOP_RES]]#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xi32>>
  // CHECK: tt.store [[TPTR_C2:%.*]], [[LOOP_RES]]#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xi32>>
  tt.store %tptr_c, %35#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x32xi32, #warp>>
  tt.return
}

// -----

// COM: Test transformation for tf32 datatype

// CHECK-LABEL: @matmul_kernel_with_block_pointers_tf32
#warp = #ttig.warp<{sizePerThread = [8, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
tt.func public @matmul_kernel_with_block_pointers_tf32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: i32) {
  // CHECK: [[TZERO:%.*]] = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
  %cst = arith.constant dense<0.000000e+00> : tensor<8x32xf32, #warp>
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c0_i64 = arith.constant 0 : i64
  %c32_i32 = arith.constant 32 : i32
  // CHECK-COUNT-4: {{.*}} = tt.make_tensor_ptr %arg0
  // CHECK-COUNT-8: {{.*}} = tt.make_tensor_ptr %arg1
  %tptr_a = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
  %tptr_b = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  // CHECK: [[LOOP_RES:%.*]]:14 = scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args([[ITER_1:%.*]] = [[TZERO]], [[ITER_2:%.*]] = [[TZERO]], {{.*}})
  %35:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c32_i32 iter_args(%arg10 = %cst, %arg11 = %tptr_a, %arg12 = %tptr_b) -> (tensor<8x32xf32, #warp>, !tt.ptr<tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>, !tt.ptr<tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>)  : i32 {
    // CHECK: [[LD_A1:%.*]] = tt.load %arg[[#first_ptr:]] {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    // CHECK: [[LD_A2:%.*]] = tt.load %arg[[#first_ptr+1]] {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    // CHECK: [[LD_A3:%.*]] = tt.load %arg[[#first_ptr+2]] {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    // CHECK: [[LD_A4:%.*]] = tt.load %arg[[#first_ptr+3]] {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    // CHECK: [[LD_B1:%.*]] = tt.load %arg[[#first_ptr+4]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B2:%.*]] = tt.load %arg[[#first_ptr+5]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B3:%.*]] = tt.load %arg[[#first_ptr+6]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B4:%.*]] = tt.load %arg[[#first_ptr+7]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B5:%.*]] = tt.load %arg[[#first_ptr+8]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B6:%.*]] = tt.load %arg[[#first_ptr+9]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B7:%.*]] = tt.load %arg[[#first_ptr+10]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    // CHECK: [[LD_B8:%.*]] = tt.load %arg[[#first_ptr+11]] {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    %46 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
    %47 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    // CHECK: [[DOT_1:%.*]] = tt.dot [[LD_A1]], [[LD_B1]], [[ITER_1]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_2:%.*]] = tt.dot [[LD_A2]], [[LD_B2]], [[DOT_1]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_3:%.*]] = tt.dot [[LD_A3]], [[LD_B3]], [[DOT_2]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_4:%.*]] = tt.dot [[LD_A4]], [[LD_B4]], [[DOT_3]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_5:%.*]] = tt.dot [[LD_A1]], [[LD_B5]], [[ITER_2]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_6:%.*]] = tt.dot [[LD_A2]], [[LD_B6]], [[DOT_5]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_7:%.*]] = tt.dot [[LD_A3]], [[LD_B7]], [[DOT_6]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    // CHECK: [[DOT_8:%.*]] = tt.dot [[LD_A4]], [[LD_B8]], [[DOT_7]], inputPrecision = tf32 : tensor<8x8xf32> * tensor<8x16xf32> -> tensor<8x16xf32>
    %48 = tt.dot %46, %47, %arg10, inputPrecision = tf32 : tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #warp}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<8x32xf32, #warp>
    // CHECK-COUNT-12: {{.*}} = tt.advance
    %49 = tt.advance %arg11, [%c0_i32, %c32_i32] : <tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
    %50 = tt.advance %arg12, [%c32_i32, %c0_i32] : <tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    scf.yield %48, %49, %50 : tensor<8x32xf32, #warp>, !tt.ptr<tensor<8x32xf32, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>, !tt.ptr<tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  } {ttg.workload = 3 : i32}
  // CHECK: [[TPTR_C1:%.*]] = tt.make_tensor_ptr %arg2,
  // CHECK: [[TPTR_C2:%.*]] = tt.make_tensor_ptr %arg2,
  %tptr_c = tt.make_tensor_ptr %arg2, [%c0_i64, %c0_i64], [%c0_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xf32, #warp>>
  // CHECK: tt.store [[TPTR_C1:%.*]], [[LOOP_RES]]#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
  // CHECK: tt.store [[TPTR_C2:%.*]], [[LOOP_RES]]#1 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
  tt.store %tptr_c, %35#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x32xf32, #warp>>
  tt.return
}

// -----

// COM: Test Attention Related Ops
#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 1 : i32} {

// Transpose reduction requires local memory.
// CHECK-TR-RED: ttg.shared = 8192 : index

// CHECK-LABEL:   tt.func public @attn_fwd(
// CHECK-SAME:                             %{{.*}}: !tt.ptr<f16>, %{{.*}}: !tt.ptr<f16>, %{{.*}}: !tt.ptr<f16>, %{{.*}}: f32, %{{.*}}: !tt.ptr<f32>, %{{.*}}: !tt.ptr<f32>
// CHECK-TR-RED-SAME:                      %[[LOCAL_BUFFER:.*]]: !tt.ptr<f32, 3>) {
tt.func public @attn_fwd(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: f32, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>) {
  %c16_i32 = arith.constant 16 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1024_i64 = arith.constant 1024 : i64
  %c64_i64 = arith.constant 64 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 1.44269502 : f32
  %c3145728_i64 = arith.constant 3145728 : i64
  %c65536_i64 = arith.constant 65536 : i64
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #warp>
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %cst_1 = arith.constant dense<0xFF800000> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
  %cst_2 = arith.constant dense<1.000000e+00> : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
  %0 = gpu.subgroup_id : index
  %1 = arith.index_cast %0 : index to i32
  %2 = tt.get_program_id z : i32
  %3 = tt.get_program_id x : i32
  %4 = tt.get_program_id y : i32
  %5 = arith.extsi %3 : i32 to i64
  %6 = arith.muli %5, %c3145728_i64 : i64
  %7 = arith.extsi %4 : i32 to i64
  %8 = arith.muli %7, %c65536_i64 : i64
  %9 = arith.addi %6, %8 : i64
  %10 = tt.addptr %arg0, %9 : !tt.ptr<f16>, i64
  %11 = arith.muli %2, %c128_i32 : i32
  %12 = arith.muli %1, %c16_i32 : i32
  %13 = arith.addi %12, %11 : i32
  %14 = tt.make_tensor_ptr %10, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
  %15 = tt.addptr %arg2, %9 : !tt.ptr<f16>, i64
  %16 = tt.make_tensor_ptr %15, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  %17 = tt.addptr %arg1, %9 : !tt.ptr<f16>, i64
  %18 = tt.make_tensor_ptr %17, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  %19 = tt.addptr %arg5, %9 : !tt.ptr<f32>, i64
  %20 = tt.make_tensor_ptr %19, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf32, #warp>>
  %21 = arith.mulf %arg3, %cst : f32
  %22 = tt.load %14 : !tt.ptr<tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
  //         CHECK: tt.splat {{.*}} : f32 -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
  // CHECK-COUNT-8: tt.splat {{.*}} : f32 -> tensor<8x16xf32>
  %23 = tt.splat %21 : f32 -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
  %24 = tt.splat %21 : f32 -> tensor<16x64xf32, #warp>
  // CHECK: [[SCF:%.*]] = scf.for
  %25:5 = scf.for %arg6 = %c0_i32 to %c1024_i32 step %c64_i32 iter_args(%arg7 = %cst_2, %arg8 = %cst_0, %arg9 = %cst_1, %arg10 = %16, %arg11 = %18) -> (tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>, tensor<16x64xf32, #warp>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>)  : i32 {
    // CHECK-COUNT-16: tt.load {{.*}} {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
    // CHECK-COUNT-32: tt.dot {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
    %29 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    %30 = tt.dot %22, %29, %cst_0, inputPrecision = tf32 : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<16x64xf32, #warp>

    // CHECK-TR-RED:             %[[VAL_211:.*]] = arith.maxnumf %{{.*}}, %{{.*}} : tensor<16x16xf32>
    // CHECK-TR-RED:             %[[VAL_212:.*]] = arith.maxnumf %{{.*}}, %{{.*}} : tensor<16x16xf32>
    // CHECK-TR-RED:             %[[MAX:.*]] = arith.maxnumf %[[VAL_211]], %[[VAL_212]] : tensor<16x16xf32>
    // CHECK-TR-RED:             %[[MAXT:.*]] = ttig.sub_group_transpose %[[LOCAL_BUFFER]], %[[MAX]] : tensor<16x16xf32>
    // CHECK-TR-RED:             %[[RED:.*]] = "tt.reduce"(%[[MAXT]]) <{axis = 1 : i32}> ({
    // CHECK-TR-RED:             ^bb0(%[[VAL_204:.*]]: f32, %[[VAL_205:.*]]: f32):
    // CHECK-TR-RED:               %[[VAL_206:.*]] = arith.maxnumf %[[VAL_204]], %[[VAL_205]] : f32
    // CHECK-TR-RED:               tt.reduce.return %[[VAL_206]] : f32
    // CHECK-TR-RED:             }) : (tensor<16x16xf32>) -> tensor<16xf32>
    // CHECK-TR-RED:             %[[RES:.*]] = ttg.convert_layout %[[RED]] : tensor<16xf32> -> tensor<16xf32, #ttg.slice

    // CHECK-SG-RED-COUNT-2:     arith.maxnumf {{.*}} : tensor<8x16xf32>
    // CHECK-SG-RED:             [[MAX:%.*]] = arith.maxnumf {{.*}} : tensor<8x16xf32>
    // CHECK-SG-RED-NEXT:        [[EXTRACT0:%.*]] = ttig.extract [[MAX]][0] : tensor<8x16xf32> -> tensor<16xf32>
    // CHECK-SG-RED-NEXT:          "tt.reduce"([[EXTRACT0]]) <{axis = 0 : i32}> ({
    // CHECK-SG-RED:             }) : (tensor<16xf32>) -> f32
    %31 = "tt.reduce"(%30) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %53 = arith.maxnumf %arg12, %arg13 : f32
      tt.reduce.return %53 : f32
    }) : (tensor<16x64xf32, #warp>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %32 = arith.mulf %31, %23 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %33 = arith.maxnumf %arg9, %32 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %34 = arith.mulf %30, %24 : tensor<16x64xf32, #warp>

    // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : tensor<16xf32
    // CHECK: ttig.broadcast {{.*}} -> tensor<16x16xf32>
    %35 = tt.expand_dims %33 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xf32, #warp>
    %36 = ttig.broadcast %35 : tensor<16x1xf32, #warp> -> tensor<16x64xf32, #warp>
    %37 = arith.subf %34, %36 : tensor<16x64xf32, #warp>
    %38 = math.exp2 %37 : tensor<16x64xf32, #warp>
    %39 = "tt.reduce"(%38) <{axis = 1 : i32}> ({
    ^bb0(%arg12: f32, %arg13: f32):
      %53 = arith.addf %arg12, %arg13 : f32
      tt.reduce.return %53 : f32
    }) : (tensor<16x64xf32, #warp>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %40 = arith.subf %arg9, %33 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %41 = math.exp2 %40 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %42 = arith.mulf %arg7, %41 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %43 = arith.addf %42, %39 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %44 = tt.expand_dims %41 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xf32, #warp>
    %45 = ttig.broadcast %44 : tensor<16x1xf32, #warp> -> tensor<16x64xf32, #warp>
    %46 = arith.mulf %arg8, %45 : tensor<16x64xf32, #warp>
    %47 = tt.load %arg10 : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    %48 = arith.truncf %38 : tensor<16x64xf32, #warp> to tensor<16x64xf16, #warp>
    %49 = ttg.convert_layout %48 : tensor<16x64xf16, #warp> -> tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>>

    // CHECK-COUNT-32: tt.dot {{.*}} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
    // CHECK-COUNT-4: tt.advance {{.*}} : <tensor<32x16xf16>>
    // CHECK-COUNT-16: tt.advance {{.*}} : <tensor<16x16xf16>>
    // CHECK: scf.yield
    %50 = tt.dot %49, %47, %46, inputPrecision = tf32 : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<16x64xf32, #warp>
    %51 = tt.advance %arg10, [%c64_i32, %c0_i32] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    %52 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    scf.yield %43, %50, %33, %51, %52 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>, tensor<16x64xf32, #warp>, tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
  } {ttg.workload = 4 : i32, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
  %26 = tt.expand_dims %25#0 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xf32, #warp>
  %27 = ttig.broadcast %26 : tensor<16x1xf32, #warp> -> tensor<16x64xf32, #warp>
  %28 = arith.divf %25#1, %27 : tensor<16x64xf32, #warp>
  tt.store %20, %28 : !tt.ptr<tensor<16x64xf32, #warp>>
  tt.return
}
}

// -----

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @_attn_fwd(%arg0: i32, %arg1: !tt.ptr<i32>) {
    // COM: This op primes the map of known layouts
    %cst = arith.constant dense<1> : tensor<16x64xi32, #warp>

    // CHECK: %[[CST_48:.*]] = arith.constant dense<48> : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    // CHECK: %[[CST_32:.*]] = arith.constant dense<32> : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    // CHECK: %[[CST_16:.*]] = arith.constant dense<16> : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>

    // CHECK: %[[MR1:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #warp}>>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #warp}>>

    // CHECK: %[[MR2:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    // CHECK: %[[MR2_PLUS_16:.*]] = arith.addi %[[MR2]], %[[CST_16]] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    // CHECK: %[[MR2_PLUS_32:.*]] = arith.addi %[[MR2]], %[[CST_32]] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    // CHECK: %[[MR2_PLUS_48:.*]] = arith.addi %[[MR2]], %[[CST_48]] : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    // CHECK: %[[GLUE:.*]] = ttig.glue %[[MR2]], %[[MR2_PLUS_16]], %[[MR2_PLUS_32]], %[[MR2_PLUS_48]] : (tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>) -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #warp}>>

    // CHECK: %[[ED1:.*]] = tt.expand_dims %[[MR1]] {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xi32, #warp>
    // CHECK: %[[ED2:.*]] = tt.expand_dims %[[GLUE]] {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #warp}>> -> tensor<1x64xi32, #warp>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xi32, #warp>
    %3 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #warp}>> -> tensor<1x64xi32, #warp>

    // CHECK: %[[BC1:.*]] = ttig.broadcast %[[ED1]] : tensor<16x1xi32, #warp> -> tensor<16x16xi32>
    %4 = ttig.broadcast %2 : tensor<16x1xi32, #warp> -> tensor<16x64xi32, #warp>

    // CHECK: %[[EX0:.*]] = ttig.extract %[[ED2]][0] : tensor<1x64xi32, #warp> -> tensor<1x16xi32, #warp>
    // CHECK: %[[BC20:.*]] = ttig.broadcast %[[EX0]] : tensor<1x16xi32, #warp> -> tensor<16x16xi32>
    // CHECK: %[[EX1:.*]] = ttig.extract %[[ED2]][1] : tensor<1x64xi32, #warp> -> tensor<1x16xi32, #warp>
    // CHECK: %[[BC21:.*]] = ttig.broadcast %[[EX1]] : tensor<1x16xi32, #warp> -> tensor<16x16xi32>
    // CHECK: %[[EX2:.*]] = ttig.extract %[[ED2]][2] : tensor<1x64xi32, #warp> -> tensor<1x16xi32, #warp>
    // CHECK: %[[BC22:.*]] = ttig.broadcast %[[EX2]] : tensor<1x16xi32, #warp> -> tensor<16x16xi32>
    // CHECK: %[[EX3:.*]] = ttig.extract %[[ED2]][3] : tensor<1x64xi32, #warp> -> tensor<1x16xi32, #warp>
    // CHECK: %[[BC23:.*]] = ttig.broadcast %[[EX3]] : tensor<1x16xi32, #warp> -> tensor<16x16xi32>
    %5 = ttig.broadcast %3 : tensor<1x64xi32, #warp> -> tensor<16x64xi32, #warp>

    // CHECK: arith.addi %[[BC1]], %[[BC20]] : tensor<16x16xi32>
    // CHECK: arith.addi %[[BC1]], %[[BC21]] : tensor<16x16xi32>
    // CHECK: arith.addi %[[BC1]], %[[BC22]] : tensor<16x16xi32>
    // CHECK: arith.addi %[[BC1]], %[[BC23]] : tensor<16x16xi32>
    %6 = arith.addi %4, %5 : tensor<16x64xi32, #warp>

    // COM: Prevent DCE
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %7 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xi32, #warp>>
    tt.store %7, %6 : !tt.ptr<tensor<16x64xi32, #warp>>
    tt.return
  }
}

// -----

// COM: This test checks that the tt.load/tt.advance ops in _both_ loops are detected as being transposed and hence having the 16x16 shape (would be 32x16 otherwise).

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
#warp1 = #ttig.warp<{sizePerThread = [16, 32], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 1 : i32, ttig.min_sg_size = 16 : i32, ttig.support_bf16_conversion, ttig.support_dpas, ttig.support_sg_2d_block} {
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: f32, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<16x64xf32, #warp>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id z : i32
    %3 = tt.get_program_id x : i32
    %4 = tt.get_program_id y : i32
    %5 = arith.extsi %3 : i32 to i64
    %6 = arith.muli %5, %c131072_i64 : i64
    %7 = arith.extsi %4 : i32 to i64
    %8 = arith.muli %7, %c65536_i64 : i64
    %9 = arith.addi %6, %8 : i64
    %10 = tt.addptr %arg0, %9 : !tt.ptr<f16>, i64
    %11 = arith.muli %2, %c128_i32 : i32
    %12 = arith.muli %1, %c16_i32 : i32
    %13 = arith.addi %12, %11 : i32
    %14 = tt.make_tensor_ptr %10, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
    %28 = tt.addptr %arg1, %9 : !tt.ptr<f16>, i64
    %34 = tt.make_tensor_ptr %28, [%c64_i64, %c1024_i64], [%c1_i64, %c64_i64], [%c0_i32, %c0_i32] {order = array<i32: 0, 1>} : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    %35 = tt.addptr %arg5, %9 : !tt.ptr<f32>, i64
    %36 = tt.make_tensor_ptr %35, [%c1024_i64, %c64_i64], [%c64_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf32, #warp>>
    %44 = tt.load %14 : !tt.ptr<tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>>>
    %47:2 = scf.for %arg6 = %c0_i32 to %11 step %c64_i32 iter_args(%arg7 = %cst_0, %arg11 = %34) -> (tensor<16x64xf32, #warp>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>)  : i32 {
      // CHECK-COUNT-16: tt.load {{%.*}} {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %60 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
      %61 = tt.dot %44, %60, %cst_0, inputPrecision = tf32 : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<16x64xf32, #warp>
      // CHECK-COUNT-16: tt.advance {{%.*}}, [%c0_i32, %c64_i32] {DotIdx = 1 : i32} : <tensor<16x16xf16>>
      %85 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
      scf.yield %61, %85 : tensor<16x64xf32, #warp>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    } {ttg.workload = 4 : i32, tt.divisibility_arg1 = dense<64> : tensor<1xi32>}
    // CHECK: gpu.barrier
    gpu.barrier
    %48 = arith.muli %2, %c128_i32 {tt.divisibility = dense<128> : tensor<1xi32>} : i32
    %49 = arith.addi %2, %c1_i32 : i32
    %50 = arith.muli %49, %c128_i32 : i32
    %51 = tt.advance %34, [%c0_i32, %48] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    %56:2 = scf.for %arg6 = %48 to %50 step %c64_i32 iter_args(%arg7 = %47#0, %arg11 = %51) -> (tensor<16x64xf32, #warp>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>)  : i32 {
      // CHECK-COUNT-16: tt.load {{%.*}} {DotIdx = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
      %60 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
      %61 = tt.dot %44, %60, %cst_0, inputPrecision = tf32 : tensor<16x64xf16, #ttg.dot_op<{opIdx = 0, parent = #warp}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>> -> tensor<16x64xf32, #warp>
      // CHECK-COUNT-16: tt.advance {{%.*}}, [%c0_i32, %c64_i32] {DotIdx = 1 : i32} : <tensor<16x16xf16>>
      %88 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
      scf.yield %61, %88 : tensor<16x64xf32, #warp>, !tt.ptr<tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #warp}>>>
    } {ttg.workload = 4 : i32}
    tt.store %36, %56#0 : !tt.ptr<tensor<16x64xf32, #warp>>
    tt.return
  }
}
