// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: 16-bit poll must not lower to a raw `load atomic i16` (issue #7390).
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_relaxed
  tt.func @poll_i16_relaxed(%ptr: !tt.ptr<i16>, %expected: i16) {
    // CHECK-NOT: llvm.load {{.*}} atomic {{.*}} : !llvm.ptr<1> -> i16
    // CHECK: %[[WORD:.*]] = llvm.load %{{.*}} atomic syncscope("device") monotonic <alignment = 4> : !llvm.ptr<1> -> i32
    // CHECK: %[[VEC:.*]] = llvm.bitcast %[[WORD]] : i32 to vector<2xi16>
    // CHECK: %[[HALF:.*]] = llvm.extractelement %[[VEC]][%{{.*}} : i32] : vector<2xi16>
    // CHECK: %{{.*}} = llvm.icmp "eq" %[[HALF]], %arg1 : i16
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i16>, i16 -> i1
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_acquire
  tt.func @poll_i16_acquire(%ptr: !tt.ptr<i16>, %expected: i16) {
    // CHECK-NOT: llvm.load {{.*}} atomic {{.*}} : !llvm.ptr<1> -> i16
    // CHECK: llvm.load %{{.*}} atomic syncscope("workgroup") monotonic <alignment = 4> : !llvm.ptr<1> -> i32
    // CHECK: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: llvm.extractelement %{{.*}}[%{{.*}} : i32] : vector<2xi16>
    // CHECK: llvm.fence syncscope("workgroup") acquire
    // CHECK: triton_gen.barrier {mem_fence = LocalAndGlobal}
    %0 = tt.atomic_poll acquire, cta, %ptr, %expected : !tt.ptr<i16>, i16 -> i1
    tt.return
  }

  // -----

  // COM: Global ordering must reach every warp on both the emulated and native
  // COM: acquire-poll paths, including unused and shared-memory results.
  module attributes {ttig.core_clock_rate_khz = 1000000 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
    // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_acquire_timeout_unused
    tt.func @poll_i16_acquire_timeout_unused(%ptr: !tt.ptr<i16>, %expected: i16, %timeout: i64) {
      // CHECK: llvm.fence syncscope("device") acquire
      // CHECK: triton_gen.barrier {mem_fence = LocalAndGlobal}
      %0 = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : !tt.ptr<i16>, i16 -> i1
      tt.return
    }

    // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_acquire_timeout_result
    tt.func @poll_i16_acquire_timeout_result(%ptr: !tt.ptr<i16>, %expected: i16, %timeout: i64) -> i1 {
      // CHECK: llvm.fence syncscope("device") acquire
      // CHECK: llvm.store %{{.*}}, %{{.*}} : i1, !llvm.ptr<3>
      // CHECK: triton_gen.barrier {mem_fence = LocalAndGlobal}
      // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i1
      %0 = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : !tt.ptr<i16>, i16 -> i1
      tt.return %0 : i1
    }

    // CHECK-LABEL: llvm.func spir_kernelcc @poll_i32_acquire
    tt.func @poll_i32_acquire(%ptr: !tt.ptr<i32>, %expected: i32) {
      // CHECK: llvm.fence syncscope("device") acquire
      // CHECK: triton_gen.barrier {mem_fence = LocalAndGlobal}
      %0 = tt.atomic_poll acquire, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
      tt.return
    }

    // CHECK-LABEL: llvm.func spir_kernelcc @poll_i32_acquire_timeout_result
    tt.func @poll_i32_acquire_timeout_result(%ptr: !tt.ptr<i32>, %expected: i32, %timeout: i64) -> i1 {
      // CHECK: llvm.fence syncscope("device") acquire
      // CHECK: llvm.store %{{.*}}, %{{.*}} : i1, !llvm.ptr<3>
      // CHECK: triton_gen.barrier {mem_fence = LocalAndGlobal}
      // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i1
      %0 = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : !tt.ptr<i32>, i32 -> i1
      tt.return %0 : i1
    }
  }
}

// -----

// COM: 32-bit polls defer to the upstream lowering: native 32-bit atomic load.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i32
  tt.func @poll_i32(%ptr: !tt.ptr<i32>, %expected: i32) {
    // CHECK-NOT: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: %[[LOADED:.*]] = llvm.load %arg0 atomic syncscope("device") monotonic <alignment = 4> : !llvm.ptr<1> -> i32
    // CHECK: llvm.br ^[[MERGE:bb[0-9]+]](%[[LOADED]] : i32)
    // CHECK: ^[[MERGE]](%[[POLLED:.*]]: i32):
    // CHECK: %{{.*}} = llvm.icmp "eq" %[[POLLED]], %arg1 : i32
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

#poll = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_tensor_i32
  tt.func @poll_tensor_i32(%ptr: tensor<128x!tt.ptr<i32>, #poll>, %expected: tensor<128xi32, #poll>) {
    // CHECK-NOT: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: llvm.load %{{.*}} atomic syncscope("device") monotonic <alignment = 4> : !llvm.ptr<1> -> i32
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : tensor<128x!tt.ptr<i32>, #poll>, tensor<128xi32, #poll> -> tensor<128xi1, #poll>
    tt.return
  }
}

// -----

// COM: With ttig.support_16bit_atomics the override defers to upstream: native i16.
module attributes {ttig.support_16bit_atomics = true, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @poll_i16_hw_support
  tt.func @poll_i16_hw_support(%ptr: !tt.ptr<i16>, %expected: i16) {
    // CHECK-NOT: llvm.bitcast %{{.*}} : i32 to vector<2xi16>
    // CHECK: %[[LOADED:.*]] = llvm.load %arg0 atomic syncscope("device") monotonic <alignment = 2> : !llvm.ptr<1> -> i16
    // CHECK: llvm.br ^[[MERGE:bb[0-9]+]](%[[LOADED]] : i16)
    // CHECK: ^[[MERGE]](%[[POLLED:.*]]: i16):
    // CHECK: %{{.*}} = llvm.icmp "eq" %[[POLLED]], %arg1 : i16
    %0 = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i16>, i16 -> i1
    tt.return
  }
}
