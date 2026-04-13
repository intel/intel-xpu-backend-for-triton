// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --cse -canonicalize | FileCheck %s

// COM: Regression test for #6138. Cross-warp blocked->blocked ConvertLayoutOp
// COM: previously fell through to the generic NVIDIA-tuned XOR swizzle path,
// COM: producing @llvm.bitreverse.i4 intrinsics that crash IGC (SIGFPE).
// COM: With the flat SLM interception, these conversions use store -> barrier -> load
// COM: without any XOR swizzle address computation.

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: convert_layout_cross_warp
  tt.func @convert_layout_cross_warp(%arg0: tensor<4x32xf16, #blocked0>, %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    // COM: Flat SLM path: stores to shared memory, barrier, loads from shared memory.
    // COM: No XOR/swizzle address computation should appear.
    // CHECK-COUNT-2: llvm.store {{.*}} : vector<2xf16>, !llvm.ptr<3>
    // CHECK:     llvm.call spir_funccc @_Z7barrierj({{.*}}) {convergent, no_unwind, will_return} : (i32) -> ()
    // CHECK:     llvm.load {{.*}} : !llvm.ptr<3> -> vector<8xf16>
    // CHECK-NOT: llvm.intr.bitreverse
    %0 = ttg.convert_layout %arg0 : tensor<4x32xf16, #blocked0> -> tensor<4x32xf16, #blocked1>
    %ptr = tt.splat %out : !tt.ptr<f16> -> tensor<4x1x!tt.ptr<f16>, #blocked1>
    %ptrs = tt.broadcast %ptr : tensor<4x1x!tt.ptr<f16>, #blocked1> -> tensor<4x32x!tt.ptr<f16>, #blocked1>
    tt.store %ptrs, %0 : tensor<4x32x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
