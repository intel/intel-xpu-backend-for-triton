// RUN: triton-opt %s -split-input-file --tritonintelgpu-widen-store-encoding | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-widen-store-encoding --tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=E2E

// COM: POSITIVE case — 128b -> 256b widening when all gates pass.
// COM: 4 f32 per thread * 32 bits = 128 bits; divisibility=16 elements >= 8 elements needed;
// COM: shape 1024 == newSpt(8) * threadsPerWarp(32) * warpsPerCTA(4).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: @widen_128b_to_256b
  tt.func @widen_128b_to_256b(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %val = arith.constant dense<1.0> : tensor<1024xf32, #blocked>

    // CHECK-DAG: ttg.convert_layout %{{.*}} : tensor<1024x!tt.ptr<f32>, #[[$OLD:blocked]]> -> tensor<1024x!tt.ptr<f32>, #[[$NEW:blocked[0-9]*]]>
    // CHECK-DAG: ttg.convert_layout %{{.*}} : tensor<1024xf32, #[[$OLD]]> -> tensor<1024xf32, #[[$NEW]]>
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<1024x!tt.ptr<f32>, #[[$NEW]]>
    tt.store %ptrs, %val : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: NEGATIVE case — module attribute absent.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @no_widen_missing_attr
  tt.func @no_widen_missing_attr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %val = arith.constant dense<1.0> : tensor<1024xf32, #blocked>

    // CHECK-NOT: convert_layout
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %val : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: NEGATIVE case — insufficient divisibility (4 elements < 8 needed).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: @no_widen_low_divisibility
  tt.func @no_widen_low_divisibility(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %val = arith.constant dense<1.0> : tensor<1024xf32, #blocked>

    // CHECK-NOT: convert_layout
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %val : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: NEGATIVE case — already at 256b (8 f32 per thread * 32 bits = 256 bits).
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: @no_widen_already_256b
  tt.func @no_widen_already_256b(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %val = arith.constant dense<1.0> : tensor<1024xf32, #blocked>

    // CHECK-NOT: convert_layout
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %val : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: NEGATIVE case — shape too small (512 < newSpt(8) * threadsPerWarp(32) * warpsPerCTA(4) = 1024).
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: @no_widen_shape_too_small
  tt.func @no_widen_shape_too_small(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    %val = arith.constant dense<1.0> : tensor<512xf32, #blocked>

    // CHECK-NOT: convert_layout
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<512x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %val : tensor<512x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// COM: END-TO-END: confirm remove-layout-conversions absorbs the CLOs inserted by widening.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // E2E-LABEL: @widen_then_rlc
  tt.func @widen_then_rlc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %val = arith.constant dense<1.0> : tensor<1024xf32, #blocked>

    // E2E-NOT: ttg.convert_layout
    // E2E: tt.store
    tt.store %ptrs, %val : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
