// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check no scratch memory is allocated for sub-group shuffle-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK: tt.func @test_sub_group_shuffle
  // CHECK-NOT: llvm.ptr<3>
  tt.func @test_sub_group_shuffle(%arg0: tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> {
    %0 = ttg.convert_layout %arg0 : tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return %0 : tensor<16xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check no scratch memory is allocated for sub-group shuffle-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK: tt.func @test_sub_group_shuffle
  // CHECK-NOT: llvm.ptr<3>
  tt.func @test_sub_group_shuffle(%arg0: tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> {
    %0 = ttg.convert_layout %arg0 : tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return %0 : tensor<32xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 544 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_f16(%arg0: tensor<16x16xf16, #blocked>) -> tensor<16x16xf16, #blocked1> {
    // Scratch buffers carry both the offset and the allocated size.
    // CHECK: ttg.convert_layout {{.*}}allocation.offset = 0 : i32, allocation.size = 544 : i32
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    tt.return %0 : tensor<16x16xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 1088 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_f32(%arg0: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.return %0 : tensor<16x16xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 34816 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_f32(%arg0: tensor<128x64xf32, #blocked>) -> tensor<128x64xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<128x64xf32, #blocked> -> tensor<128x64xf32, #blocked1>
    tt.return %0 : tensor<128x64xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [16, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [1, 16], warpsPerCTA = [2, 4], order = [0, 1]}>

// Check scracth memory configuration for different sub-group transpose-like layout conversions.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 17408 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @test_contiguous(%arg0: tensor<32x128xf32, #blocked>) -> tensor<32x128xf32, #blocked1> {
    %0 = ttg.convert_layout %arg0 : tensor<32x128xf32, #blocked> -> tensor<32x128xf32, #blocked1>
    tt.return %0 : tensor<32x128xf32, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [1, 1], order = [0, 1]}>

// Check a call site gets the virtual buffer reserving the callee's scratch memory.
//
// Deliberately do not check `allocation.size` on the `tt.call`: whether virtual
// buffers carry it depends on the upstream revision. Upstream 18a3c9740 (#11268)
// replaced the `!isVirtualBuffer(bufferId)` guard in
// `attachAllocationSizeAndOffsetAttr` with `!isa<FunctionOpInterface>(op)`, which
// starts emitting it on call sites. Only the offset is stable across both.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 544 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func private @test_callee(%arg0: tensor<16x16xf16, #blocked>) {
    // CHECK: ttg.convert_layout {{.*}}allocation.offset = 0 : i32, allocation.size = 544 : i32
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #blocked1>
    tt.return
  }
  tt.func public @test_caller(%arg0: tensor<16x16xf16, #blocked>) {
    // CHECK: tt.call @test_callee{{.*}}allocation.offset = 0 : i32
    tt.call @test_callee(%arg0) : (tensor<16x16xf16, #blocked>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 1, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [2, 1], A = [16, 8], B = [8, 16], C = [16, 16]}>

// CHECK-LABEL: module attributes
// CHECK-SAME: ttg.shared = 1024 : i32
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @test_reinterpret(%arg0: tensor<16x16xf32, #mma>, %arg1: tensor<16x16xf16, #mma>)  -> (tensor<16x16xf32, #blocked>, tensor<16x16xf16, #blocked>) {
    // The reinterpret bitcast is used for this convert layout op without using share local memory.
    // CHECK: ttg.convert_layout {{.*}} : tensor<16x16xf16
    %1 = ttg.convert_layout %arg1 : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked>
    // The reinterpret bitcast cannot be used. The share local memory is allocated for this convert layout op.
    // CHECK: ttg.convert_layout {{.*}} {allocation.offset = 0 : i32} : tensor<16x16xf32
    %0 = ttg.convert_layout %arg0 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    tt.return %0, %1 : tensor<16x16xf32, #blocked>, tensor<16x16xf16, #blocked>
  }
}
