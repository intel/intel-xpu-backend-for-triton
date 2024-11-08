// RUN: triton-opt %s --split-input-file -tritonintelgpu-optimize-elementwise-parallelism | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// CHECK-LABEL:   tt.func @test_two_convert_layout(
// CHECK-SAME:                                     %[[VAL_0:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
tt.func @test_two_convert_layout(%arg0: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> {
  %0 = triton_gpu.convert_layout %arg0 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %1 = triton_gpu.convert_layout %arg1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           %[[VAL_2:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
  // CHECK:           %[[VAL_3:.*]] = triton_gpu.convert_layout %[[VAL_2]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  %2 = arith.addf %0, %1 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           tt.return %[[VAL_3]] : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  tt.return %2 : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// CHECK-LABEL:   tt.func @test_convert_layout_splat(
// CHECK-SAME:                                       %[[VAL_0:.*]]: tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                                       %[[VAL_1:.*]]: f32
tt.func @test_convert_layout_splat(%arg0: tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: f32) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> {
  %0 = triton_gpu.convert_layout %arg0 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           %[[VAL_2:.*]] = tt.splat %[[VAL_1]] : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
  %1 = tt.splat %arg1 : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           %[[VAL_3:.*]] = arith.addf %[[VAL_0]], %[[VAL_2]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
  // CHECK:           %[[VAL_4:.*]] = triton_gpu.convert_layout %[[VAL_3]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  %2 = arith.addf %0, %1 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           tt.return %[[VAL_4]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  tt.return %2 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// CHECK-LABEL:   tt.func @test_chain(
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>, %[[VAL_1:.*]]: tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                        %[[VAL_2:.*]]: f32
tt.func @test_chain(%arg0: tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg1: tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, %arg2: f32) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> {
  // CHECK:           %[[VAL_3:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
  %0 = triton_gpu.convert_layout %arg0 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %1 = triton_gpu.convert_layout %arg1 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           %[[VAL_4:.*]] = tt.splat %[[VAL_2]] : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
  %2 = tt.splat %arg2 : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %3 = arith.addf %0, %1 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           %[[VAL_5:.*]] = arith.addf %[[VAL_4]], %[[VAL_3]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>>
  // CHECK:           %[[VAL_6:.*]] = triton_gpu.convert_layout %[[VAL_5]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_0]]}>> -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  %4 = arith.addf %2, %3 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  // CHECK:           tt.return %[[VAL_6]] : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  tt.return %4 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
}

// -----

// CHECK: #[[$ATTR_0:.+]] = #triton_gpu.blocked<{sizePerThread = [1, 2, 16], threadsPerWarp = [16, 1, 1], warpsPerCTA = [1, 1, 1], order = [0, 1, 2]}>
// CHECK: #[[$ATTR_1:.+]] = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK: #[[$ATTR_2:.+]] = #triton_gpu.blocked<{sizePerThread = [16, 2, 1], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 1, 1], order = [0, 1, 2]}>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 2, 16], threadsPerWarp = [16, 1, 1], warpsPerCTA = [1, 1, 1], order = [0, 1, 2]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [16, 2, 1], threadsPerWarp = [1, 1, 16], warpsPerCTA = [1, 1, 1], order = [0, 1, 2]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// CHECK-LABEL:   tt.func @test_convert_layout_with_reshape(
// CHECK-SAME:                                              %[[VAL_0:.*]]: tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #[[$ATTR_0]]}>>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: f32) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>> {
tt.func @test_convert_layout_with_reshape(%arg0: tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>>, %arg1: f32) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> {
  // CHECK:           %[[VAL_2:.*]] = tt.splat %[[VAL_1]] : f32 -> tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #[[$ATTR_0]]}>>
  // CHECK:           %[[VAL_3:.*]] = arith.addf %[[VAL_0]], %[[VAL_2]] : tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #[[$ATTR_0]]}>>
  %0 = triton_gpu.convert_layout %arg0 : tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #blocked}>> -> tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #blocked1}>>
  %2 = tt.reshape %0 allow_reorder efficient_layout : tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #blocked1}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %3 = tt.splat %arg1 : f32 -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  %4 = arith.addf %2, %3 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
  // CHECK:           %[[VAL_4:.*]] = triton_gpu.convert_layout %[[VAL_3]] : tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #[[$ATTR_0]]}>> -> tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #[[$ATTR_2]]}>>
  // CHECK:           %[[VAL_5:.*]] = tt.reshape %[[VAL_4]] allow_reorder efficient_layout : tensor<16x2xf32, #triton_gpu.slice<{dim = 2, parent = #[[$ATTR_2]]}>> -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  // CHECK:           tt.return %[[VAL_5]] : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #[[$ATTR_1]]}>>
  tt.return %4 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
}
