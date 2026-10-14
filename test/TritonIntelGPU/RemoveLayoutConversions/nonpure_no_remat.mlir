// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions='max-backward-remat-iterations=10' | FileCheck %s

// COM: Ops with write effects must not be rematerialized (cloned) by RLC; the
// COM: second layout must come from a ttg.convert_layout of the single op.
// COM: @side_effecting_inline_asm is adapted from upstream triton-lang/triton#11001 (combine.mlir):
// COM: threadsPerWarp 32 -> 16, both orders [1, 0], #blocked1 sizePerThread [1, 2].

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @side_effecting_inline_asm
  tt.func @side_effecting_inline_asm() -> (tensor<1x32xi32, #blocked1>, tensor<1x32xi32, #blocked>) {
    %cst = arith.constant dense<0> : tensor<1x32xi32, #blocked>
    // CHECK: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}}pure = false
    %asm = tt.elementwise_inline_asm "mov.u32 $0, %clock;" {constraints = "=r,r", packed_element = 1 : i32, pure = false} %cst : tensor<1x32xi32, #blocked> -> tensor<1x32xi32, #blocked>
    // CHECK-NEXT: %[[CONVERT:.+]] = ttg.convert_layout %[[ASM]]
    %converted = ttg.convert_layout %asm : tensor<1x32xi32, #blocked> -> tensor<1x32xi32, #blocked1>
    // CHECK-NEXT: tt.return %[[CONVERT]], %[[ASM]]
    tt.return %converted, %asm : tensor<1x32xi32, #blocked1>, tensor<1x32xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_load_no_remat
  // CHECK: %[[V:.+]] = tt.atomic_load
  // CHECK-NOT: tt.atomic_load
  // CHECK: %[[C:.+]] = ttg.convert_layout %[[V]]
  // CHECK-NOT: tt.atomic_load
  // CHECK: tt.return %[[C]], %[[V]]
  tt.func @atomic_load_no_remat(%p: !tt.ptr<i32>) -> (tensor<1x32xi32, #blocked1>, tensor<1x32xi32, #blocked>) {
    %ptr = tt.splat %p : !tt.ptr<i32> -> tensor<1x32x!tt.ptr<i32>, #blocked>
    %v = tt.atomic_load acquire, gpu, %ptr : (tensor<1x32x!tt.ptr<i32>, #blocked>) -> tensor<1x32xi32, #blocked>
    %c = ttg.convert_layout %v : tensor<1x32xi32, #blocked> -> tensor<1x32xi32, #blocked1>
    tt.return %c, %v : tensor<1x32xi32, #blocked1>, tensor<1x32xi32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @atomic_poll_no_remat
  // CHECK: %[[V:.+]] = tt.atomic_poll
  // CHECK-NOT: tt.atomic_poll
  // CHECK: %[[C:.+]] = ttg.convert_layout %[[V]]
  // CHECK-NOT: tt.atomic_poll
  // CHECK: tt.return %[[C]], %[[V]]
  tt.func @atomic_poll_no_remat(%p: !tt.ptr<i32>) -> (tensor<1x32xi1, #blocked1>, tensor<1x32xi1, #blocked>) {
    %ptr = tt.splat %p : !tt.ptr<i32> -> tensor<1x32x!tt.ptr<i32>, #blocked>
    %e = arith.constant dense<1> : tensor<1x32xi32, #blocked>
    %m = tt.atomic_poll acquire, gpu, %ptr, %e : tensor<1x32x!tt.ptr<i32>, #blocked>, tensor<1x32xi32, #blocked> -> tensor<1x32xi1, #blocked>
    %c = ttg.convert_layout %m : tensor<1x32xi1, #blocked> -> tensor<1x32xi1, #blocked1>
    tt.return %c, %m : tensor<1x32xi1, #blocked1>, tensor<1x32xi1, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @nonpure_extern_no_remat
  // CHECK: %[[V:.+]] = tt.extern_elementwise
  // CHECK-NOT: tt.extern_elementwise
  // CHECK: %[[C:.+]] = ttg.convert_layout %[[V]]
  // CHECK-NOT: tt.extern_elementwise
  // CHECK: tt.return %[[C]], %[[V]]
  tt.func @nonpure_extern_no_remat() -> (tensor<1x32xi32, #blocked1>, tensor<1x32xi32, #blocked>) {
    %cst = arith.constant dense<0> : tensor<1x32xi32, #blocked>
    %e = tt.extern_elementwise %cst {libname = "", libpath = "", pure = false, symbol = "foo"} : (tensor<1x32xi32, #blocked>) -> tensor<1x32xi32, #blocked>
    %c = ttg.convert_layout %e : tensor<1x32xi32, #blocked> -> tensor<1x32xi32, #blocked1>
    tt.return %c, %e : tensor<1x32xi32, #blocked1>, tensor<1x32xi32, #blocked>
  }
}
