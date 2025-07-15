// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

tt.func @ttig.glue(%tensor1 : tensor<16x8xf16>, %tensor2 : tensor<16x8xf16>,
                               %ptr1 : !tt.ptr<tensor<16x8xf16>>, %ptr2 : !tt.ptr<tensor<16x8xf16>>) {
  // CHECK-LABEL: @ttig.glue
  // CHECK: %0 = ttig.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  // CHECK: %1 = ttig.glue %arg2, %arg3 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<16x16xf16>>
  %tensor = ttig.glue %tensor1, %tensor2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %ptr = ttig.glue %ptr1, %ptr2 : (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x8xf16>>) -> !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

tt.func @ttig.extract(%tensor : tensor<16x16xf16>, %ptr : !tt.ptr<tensor<16x8xf16>>) {
  // CHECK-LABEL: @ttig.extract
  // CHECK: %0 = ttig.extract %arg0[0] : tensor<16x16xf16> -> tensor<4x4xf16>
  // CHECK: %1 = ttig.extract %arg1[1] : !tt.ptr<tensor<16x8xf16>> -> !tt.ptr<tensor<8x8xf16>>
  %tensorRes = ttig.extract %tensor[0] : tensor<16x16xf16> -> tensor<4x4xf16>
  %ptrRes = ttig.extract %ptr[1] : !tt.ptr<tensor<16x8xf16>> -> !tt.ptr<tensor<8x8xf16>>
  tt.return
}

// -----

tt.func @simplify_scf_for(%arg0: tensor<16x8xf16>, %arg1: tensor<16x8xf16>, %arg2: !tt.ptr<f16, 1>,
                          %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i32, %arg7: i32) {
  // CHECK-LABEL: @simplify_scf_for
  // CHECK:      [[GLUE:%.*]] = ttig.glue
  // CHECK-NEXT: [[RES:%.*]] = scf.for {{.*}} iter_args([[INIT1:%.*]] = [[GLUE]]) -> (tensor<16x16xf16>) : i32 {
  // CHECK:        scf.yield {{.*}} : tensor<16x16xf16>
  // CHECK-NEXT: }
  // CHECK-NEXT: [[PTR:%.*]] = tt.make_tensor_ptr %arg2
  // CHECK-NEXT: tt.store [[PTR]], [[RES]]
  %lb = arith.constant 0 : i32
  %ub = arith.constant 32 : i32
  %st = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %glue = ttig.glue %arg0, %arg1 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
  %res = scf.for %iv = %lb to %ub step %st iter_args(%arg = %glue) -> (tensor<16x16xf16>) : i32 {
    %e1 = ttig.extract %arg[0] : tensor<16x16xf16> -> tensor<16x8xf16>
    %e2 = ttig.extract %arg[1] : tensor<16x16xf16> -> tensor<16x8xf16>
    %g1 = ttig.glue %e1, %e2 : (tensor<16x8xf16>, tensor<16x8xf16>) -> tensor<16x16xf16>
    scf.yield %g1 : tensor<16x16xf16>
  }
  %ptr = tt.make_tensor_ptr %arg2, [%arg3, %arg4], [%arg5, %c1_i64], [%arg6, %arg7] {order = array<i32: 1, 0>} : <tensor<16x16xf16>>
  tt.store %ptr, %res {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<16x16xf16>>
  tt.return
}

// -----

tt.func @ttig.prefetch(%arg0: !tt.ptr<tensor<2x32xf32>>, %arg1: tensor<2x32xi1>) {
  // CHECK-LABEL: @ttig.prefetch
  // CHECK:         ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  ttig.prefetch %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  // CHECK:         ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  ttig.prefetch %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<2x32xf32>>
  tt.return
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  tt.func @ttig.sub_group_transpose(%local_buffer : !tt.ptr<f16, 3>, %src : tensor<16x16xf16>) -> tensor<16x16xf16> {
    // CHECK-LABEL: @ttig.sub_group_transpose
    // CHECK:         ttig.sub_group_transpose %arg0, %arg1 : tensor<16x16xf16>
    %res = ttig.sub_group_transpose %local_buffer, %src : tensor<16x16xf16>
    tt.return %res : tensor<16x16xf16>
  }
}
