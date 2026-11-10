// RUN: triton-opt %s -split-input-file --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: bf16 fma is computed in f32 (no llvm.fma.bf16), with or without native bf16
// COM: arithmetic.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: @fma_bf16_lts
  // CHECK-NOT: llvm.intr.fma
  // CHECK: %[[MUL:.*]] = llvm.fmul {{.*}} fastmath<contract> : f32
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[MUL]], {{.*}} fastmath<contract> : f32
  // CHECK: llvm.bitcast %[[ADD]] : f32 to i32
  // CHECK-NOT: llvm.intr.fma
  tt.func public @fma_bf16_lts(%a: tensor<16xbf16, #blocked>, %b: tensor<16xbf16, #blocked>, %c: tensor<16xbf16, #blocked>) -> tensor<16xbf16, #blocked> {
    %0 = math.fma %a, %b, %c : tensor<16xbf16, #blocked>
    tt.return %0 : tensor<16xbf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_bfloat16_arithmetic, ttig.support_bfloat16_conversion} {
  // CHECK-LABEL: @fma_bf16
  // CHECK-NOT: llvm.intr.fma
  // CHECK: %[[MUL:.*]] = llvm.fmul {{.*}} fastmath<contract> : f32
  // CHECK: %[[ADD:.*]] = llvm.fadd %[[MUL]], {{.*}} fastmath<contract> : f32
  // CHECK: llvm.call spir_funccc @_Z27__spirv_ConvertFToBF16INTELf(%[[ADD]])
  // CHECK-NOT: llvm.intr.fma
  tt.func public @fma_bf16(%a: tensor<16xbf16, #blocked>, %b: tensor<16xbf16, #blocked>, %c: tensor<16xbf16, #blocked>) -> tensor<16xbf16, #blocked> {
    %0 = math.fma %a, %b, %c : tensor<16xbf16, #blocked>
    tt.return %0 : tensor<16xbf16, #blocked>
  }

  // CHECK-LABEL: @fma_f16
  // CHECK: llvm.intr.fma({{.*}}) : (f16, f16, f16) -> f16
  tt.func public @fma_f16(%a: tensor<16xf16, #blocked>, %b: tensor<16xf16, #blocked>, %c: tensor<16xf16, #blocked>) -> tensor<16xf16, #blocked> {
    %0 = math.fma %a, %b, %c : tensor<16xf16, #blocked>
    tt.return %0 : tensor<16xf16, #blocked>
  }

  // CHECK-LABEL: @fma_f32
  // CHECK-NOT: llvm.intr.fma
  // CHECK: %[[MUL:.*]] = llvm.fmul {{.*}} fastmath<contract> : f32
  // CHECK: llvm.fadd %[[MUL]], {{.*}} fastmath<contract> : f32
  tt.func public @fma_f32(%a: tensor<16xf32, #blocked>, %b: tensor<16xf32, #blocked>, %c: tensor<16xf32, #blocked>) -> tensor<16xf32, #blocked> {
    %0 = math.fma %a, %b, %c : tensor<16xf32, #blocked>
    tt.return %0 : tensor<16xf32, #blocked>
  }
}
