// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3 split-barriers-scope=workgroup" | FileCheck %s --check-prefixes=CHECK,WORKGROUP_SCOPE
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3 split-barriers-scope=subgroup" | FileCheck %s --check-prefixes=CHECK,SUBGROUP_SCOPE

// CHECK: #[[$BLOCK:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel_nested(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel_nested
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<0> : tensor<1x256xi64, #blocked>
    %18 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #dot0>>
    %22 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #dot1>>
    // CHECK:      scf.for %[[V:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #mma>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK-NEXT: ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK-NEXT: ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #mma>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)
    // WORKGROUP_SCOPE-NEXT: spirv.INTEL.ControlBarrierArrive <Workgroup> <Workgroup> <None>
    // SUBGROUP_SCOPE-NEXT: spirv.INTEL.ControlBarrierArrive <Subgroup> <Subgroup> <None>
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // WORKGROUP_SCOPE:   spirv.INTEL.ControlBarrierWait <Workgroup> <Workgroup> <None>
    // SUBGROUP_SCOPE:   spirv.INTEL.ControlBarrierWait <Subgroup> <Subgroup> <None>
    // CHECK-NEXT:   scf.yield
    %23:3 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst, %arg4 = %18, %arg5 = %22) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      %55:3 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      	    %56 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #dot0>>
      	    %57 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x256xf16, #dot1>>
      	    %58 = tt.dot %56, %57, %arg10, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      	    %59 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
      	    %60 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      	    scf.yield %58, %59, %60 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>
      }
      scf.yield %55#0, %55#1, %55#2 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>
    }
    tt.return
  }
}

// -----

// CHECK: #[[$BLOCK:.+]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_sg_2d_block"} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // CHECK-LABEL:   tt.func public @matmul_kernel
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<0> : tensor<1x256xi64, #blocked>
    %18 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #dot0>>
    %22 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x256xf16, #dot1>>

    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK-NEXT: ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK-NEXT: ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>>
    // CHECK:      scf.for %[[IV:.*]] = {{.*}} to {{.*}} step {{.*}} iter_args({{.*}}) -> (tensor<128x256xf32, #mma>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>, !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>>, !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>>)
    // WORKGROUP_SCOPE-NEXT: spirv.INTEL.ControlBarrierArrive <Workgroup> <Workgroup> <None>
    // SUBGROUP_SCOPE-NEXT: spirv.INTEL.ControlBarrierArrive <Subgroup> <Subgroup> <None>
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>>>
    // CHECK:        ttig.prefetch {{.*}} : !tt.ptr<tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>>
    // CHECK:        tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
    // WORKGROUP_SCOPE:   spirv.INTEL.ControlBarrierWait <Workgroup> <Workgroup> <None>
    // SUBGROUP_SCOPE:   spirv.INTEL.ControlBarrierWait <Subgroup> <Subgroup> <None>
    // CHECK-NEXT:   scf.yield
    %23:3 = scf.for %arg9 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %18, %arg12 = %22) -> (tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>)  : i32 {
      %56 = tt.load %arg11 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<128x64xf16, #dot0>>
      %57 = tt.load %arg12 {boundaryCheck = array<i32: 0, 1>, ttig.block_io = "row_major"} : !tt.ptr<tensor<64x256xf16, #dot1>>
      %58 = tt.dot %56, %57, %arg10, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %59 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>>>
      %60 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>>>
      scf.yield %58, %59, %60 : tensor<128x256xf32, #dpas>, !tt.ptr<tensor<128x64xf16, #dot0>>, !tt.ptr<tensor<64x256xf16, #dot1>>
    }
    tt.return
  }
}
