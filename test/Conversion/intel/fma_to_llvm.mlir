// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// COM: f32 math.fma is lowered to a contractable fmul/fadd pair (IGC W/A, #8626).
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fma_f32_contract_pair
  tt.func @fma_f32_contract_pair(%a: tensor<32xf32, #blocked>, %b: tensor<32xf32, #blocked>, %c: tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    // CHECK-NOT: llvm.intr.fma
    // CHECK: llvm.fmul {{.*}}<contract>
    // CHECK: llvm.fadd {{.*}}<contract>
    // CHECK-NOT: llvm.intr.fma
    %0 = math.fma %a, %b, %c : tensor<32xf32, #blocked>
    tt.return %0 : tensor<32xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

// COM: When the addend is a separately rounded multiply, IGC would fold it into
// COM: the mad and skip the rounding, so a real llvm.fma must be used instead.
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fma_f32_mulf_addend
  tt.func @fma_f32_mulf_addend(%a: tensor<32xf32, #blocked>, %b: tensor<32xf32, #blocked>, %x: tensor<32xf32, #blocked>, %y: tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked> {
    // CHECK: llvm.fmul
    %c = arith.mulf %x, %y : tensor<32xf32, #blocked>
    // CHECK-NOT: <contract>
    // CHECK: llvm.intr.fma
    %0 = math.fma %a, %b, %c : tensor<32xf32, #blocked>
    tt.return %0 : tensor<32xf32, #blocked>
  }
}
