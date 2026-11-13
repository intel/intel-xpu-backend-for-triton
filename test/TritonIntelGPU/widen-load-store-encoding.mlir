// RUN: triton-opt %s -split-input-file --tritonintelgpu-widen-load-store-encoding | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritonintelgpu-widen-load-store-encoding --tritonintelgpu-remove-layout-conversions | FileCheck %s --check-prefix=E2E

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

// -----

// COM: POSITIVE case — tt.load 128b -> 256b widening.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: @widen_load_128b_to_256b
  tt.func @widen_load_128b_to_256b(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<1024xf32, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>

    // CHECK-DAG: ttg.convert_layout %{{.*}} : tensor<1024x!tt.ptr<f32>, #[[$OLD:blocked]]> -> tensor<1024x!tt.ptr<f32>, #[[$NEW:blocked[0-9]*]]>
    // CHECK: tt.load %{{.*}} : tensor<1024x!tt.ptr<f32>, #[[$NEW]]>
    // CHECK: ttg.convert_layout %{{.*}} : tensor<1024xf32, #[[$NEW]]> -> tensor<1024xf32, #[[$OLD]]>
    %val = tt.load %ptrs : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %val : tensor<1024xf32, #blocked>
  }
}

// -----

// COM: NEGATIVE case — tt.load with low divisibility.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK-LABEL: @no_widen_load_low_divisibility
  tt.func @no_widen_load_low_divisibility(%arg0: !tt.ptr<f32> {tt.divisibility = 4 : i32}) -> tensor<1024xf32, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>

    // CHECK-NOT: convert_layout
    // CHECK: tt.load %{{.*}} : tensor<1024x!tt.ptr<f32>, #blocked>
    %val = tt.load %ptrs : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %val : tensor<1024xf32, #blocked>
  }
}

// -----

// COM: END-TO-END: tt.load widening + RLC absorbs conversions.
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // E2E-LABEL: @widen_load_then_rlc
  tt.func @widen_load_then_rlc(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<1024xf32, #blocked> {
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %ptr_splat, %range : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>

    // E2E-NOT: ttg.convert_layout
    // E2E: tt.load
    %val = tt.load %ptrs : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %val : tensor<1024xf32, #blocked>
  }
}

// -----

// COM: ttig.descriptor_gather widens when the descriptor (32 B base, 512 B row pitch) and y prove
// COM: 16 bf16 per lane; the offsets follow the new layout and a convert restores the old type.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#rows = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#x = #ttg.slice<{dim = 1, parent = #blocked}>
#xr = #ttg.slice<{dim = 1, parent = #rows}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_256b_load_store} {
  // CHECK: #[[$W:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: @widen_gather
  // CHECK: %[[X:.*]] = ttg.convert_layout %arg1 {{.*}} -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #[[$W]]}>>
  // CHECK: %[[G:.*]] = ttig.descriptor_gather %{{.*}}[%[[X]], %arg2] {{.*}} -> tensor<64x256xbf16, #[[$W]]>
  // CHECK: ttg.convert_layout %[[G]] : tensor<64x256xbf16, #[[$W]]> -> tensor<64x256xbf16, #blocked>
  tt.func @widen_gather(%base: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %x: tensor<64xi32, #x>, %y: i32 {tt.divisibility = 16 : i32}) -> tensor<64x256xbf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d = tt.make_tensor_descriptor %base, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <bf16>, <1x256xbf16>
    %g = ttig.descriptor_gather %d[%x, %y] : (!tt.tensordesc<1x256xbf16>, tensor<64xi32, #x>, i32) -> tensor<64x256xbf16, #blocked>
    tt.return %g : tensor<64x256xbf16, #blocked>
  }
  // CHECK-LABEL: @no_widen_gather_base_16b
  // CHECK-NOT: convert_layout
  tt.func @no_widen_gather_base_16b(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %x: tensor<64xi32, #x>, %y: i32 {tt.divisibility = 16 : i32}) -> tensor<64x256xbf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d = tt.make_tensor_descriptor %base, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <bf16>, <1x256xbf16>
    %g = ttig.descriptor_gather %d[%x, %y] : (!tt.tensordesc<1x256xbf16>, tensor<64xi32, #x>, i32) -> tensor<64x256xbf16, #blocked>
    tt.return %g : tensor<64x256xbf16, #blocked>
  }
  // CHECK-LABEL: @no_widen_gather_y_8
  // CHECK-NOT: convert_layout
  tt.func @no_widen_gather_y_8(%base: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %x: tensor<64xi32, #x>, %y: i32 {tt.divisibility = 8 : i32}) -> tensor<64x256xbf16, #blocked> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d = tt.make_tensor_descriptor %base, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <bf16>, <1x256xbf16>
    %g = ttig.descriptor_gather %d[%x, %y] : (!tt.tensordesc<1x256xbf16>, tensor<64xi32, #x>, i32) -> tensor<64x256xbf16, #blocked>
    tt.return %g : tensor<64x256xbf16, #blocked>
  }
  // COM: The 128 bits per lane run along the rows, which a gather does not keep contiguous.
  // CHECK-LABEL: @no_widen_gather_rows
  // CHECK-NOT: convert_layout
  tt.func @no_widen_gather_rows(%base: !tt.ptr<bf16> {tt.divisibility = 32 : i32}, %x: tensor<256xi32, #xr>, %y: i32 {tt.divisibility = 16 : i32}) -> tensor<256x16xbf16, #rows> {
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %d = tt.make_tensor_descriptor %base, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <bf16>, <1x16xbf16>
    %g = ttig.descriptor_gather %d[%x, %y] : (!tt.tensordesc<1x16xbf16>, tensor<256xi32, #xr>, i32) -> tensor<256x16xbf16, #rows>
    tt.return %g : tensor<256x16xbf16, #rows>
  }
}
