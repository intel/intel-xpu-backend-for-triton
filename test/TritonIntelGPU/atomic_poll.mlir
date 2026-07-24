// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: 16-bit poll must not lower to a raw `load atomic i16` (issue #7390).
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_relaxed
  tt.func @poll_i16_relaxed(%ptr: !tt.ptr<i16, 1>, %expected: i16) {
    // CHECK-NOT: llvm.load {{.*}} atomic {{.*}} : !llvm.ptr<1> -> i16
    // CHECK: %[[WORD:.*]] = llvm.load %{{.*}} atomic syncscope("device") monotonic {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: %[[VEC:.*]] = llvm.bitcast %[[WORD]] : i32 to vector<2xi16>
    // CHECK: %[[HALF:.*]] = llvm.extractelement %[[VEC]][%{{.*}} : i32] : vector<2xi16>
    // CHECK: %{{.*}} = llvm.icmp "eq" %[[HALF]], %arg1 : i16
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i16, 1>, i16 -> i1
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_acquire
  tt.func @poll_i16_acquire(%ptr: !tt.ptr<i16, 1>, %expected: i16) {
    // CHECK-NOT: llvm.load {{.*}} atomic {{.*}} : !llvm.ptr<1> -> i16
    // CHECK: llvm.load %{{.*}} atomic syncscope("workgroup") monotonic {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xi16>
    // CHECK: llvm.fence syncscope("workgroup") acquire
    %0 = tt.atomic_poll acquire, cta, %ptr, %expected : !tt.ptr<i16, 1>, i16 -> i1
    tt.return
  }
}

// -----

// COM: 32-bit polls defer to the upstream lowering: native 32-bit atomic load.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i32
  tt.func @poll_i32(%ptr: !tt.ptr<i32, 1>, %expected: i32) {
    // CHECK-NOT: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: %[[LOADED:.*]] = llvm.load %arg0 atomic syncscope("device") monotonic {alignment = 4 : i64} : !llvm.ptr<1> -> i32
    // CHECK: %{{.*}} = llvm.icmp "eq" %[[LOADED]], %arg1 : i32
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i32, 1>, i32 -> i1
    tt.return
  }
}

// -----

// COM: With ttig.support_16bit_atomics the override defers to upstream: native i16.
module attributes {ttig.support_16bit_atomics = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_hw_support
  tt.func @poll_i16_hw_support(%ptr: !tt.ptr<i16, 1>, %expected: i16) {
    // CHECK-NOT: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: %[[LOADED:.*]] = llvm.load %arg0 atomic syncscope("device") monotonic {alignment = 2 : i64} : !llvm.ptr<1> -> i16
    // CHECK: %{{.*}} = llvm.icmp "eq" %[[LOADED]], %arg1 : i16
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i16, 1>, i16 -> i1
    tt.return
  }
}
