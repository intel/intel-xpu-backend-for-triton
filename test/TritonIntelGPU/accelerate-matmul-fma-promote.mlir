// RUN: triton-opt %s -split-input-file --tritonintelgpu-accelerate-matmul | FileCheck %s

// FMA-path promotion of fp8 operands is applied before the layout conversion
// into the dot operand layout, which replicates elements across threads.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @fp8_to_bf16_fma
  tt.func public @fp8_to_bf16_fma(
    %arg0: tensor<128x64xf8E4M3FN, #blocked1>,
    %arg1: tensor<64x128xf8E4M3FN, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<bf16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xbf16, #blocked>
    // CHECK-DAG: [[A:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E4M3FN, #[[$SRC]]> -> tensor<128x64xf32, #[[$SRC]]>
    // CHECK-DAG: [[B:%.*]] = tt.fp_to_fp %arg1 : tensor<64x128xf8E4M3FN, #[[$SRC]]> -> tensor<64x128xf32, #[[$SRC]]>
    // CHECK-DAG: [[CA:%.*]] = ttg.convert_layout [[A]] : tensor<128x64xf32, #[[$SRC]]> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[CB:%.*]] = ttg.convert_layout [[B]] : tensor<64x128xf32, #[[$SRC]]> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-NOT: ttg.convert_layout {{.*}}xf8E4M3FN
    // CHECK: tt.dot [[CA]], [[CB]], {{.*}} -> tensor<128x128xf32, #[[$BL]]>
    // CHECK: tt.fp_to_fp {{.*}}, rounding = rtne : tensor<128x128xf32, #[[$BL]]> -> tensor<128x128xbf16, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf8E4M3FN, #blocked1> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xbf16, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

// -----

// Other operand types are still promoted in the dot operand layout: their cast
// is cheap, and the wider replicated operand would cost more registers.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @i8_to_i32_fma
  tt.func public @i8_to_i32_fma(
    %arg0: tensor<128x64xi8, #blocked1>,
    %arg1: tensor<64x128xi8, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<i32>, #blocked>) {
    %cst = arith.constant dense<0> : tensor<128x128xi32, #blocked>
    // CHECK-DAG: [[A:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xi8, #[[$SRC]]> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = ttg.convert_layout %arg1 : tensor<64x128xi8, #[[$SRC]]> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[EA:%.*]] = arith.extsi [[A]] : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> to tensor<128x64xi32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[EB:%.*]] = arith.extsi [[B]] : tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> to tensor<64x128xi32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[EA]], [[EB]], {{.*}} -> tensor<128x128xi32, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xi8, #blocked1> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xi32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @f16_to_f32_fma
  tt.func public @f16_to_f32_fma(
    %arg0: tensor<128x64xf16, #blocked1>,
    %arg1: tensor<64x128xf16, #blocked1>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: [[A:%.*]] = ttg.convert_layout %arg0 : tensor<128x64xf16, #[[$SRC]]> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[B:%.*]] = ttg.convert_layout %arg1 : tensor<64x128xf16, #[[$SRC]]> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK-DAG: [[EA:%.*]] = tt.fp_to_fp [[A]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK-DAG: [[EB:%.*]] = tt.fp_to_fp [[B]] : tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[EA]], [[EB]], {{.*}} -> tensor<128x128xf32, #[[$BL]]>
    %a = ttg.convert_layout %arg0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %b = ttg.convert_layout %arg1 : tensor<64x128xf16, #blocked1> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %result = tt.dot %a, %b, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %result : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A layout conversion shared by two dots is promoted for each and dropped once
// unused. An operand that is not a layout conversion is promoted in place.
// CHECK-DAG: #[[$BL:.+]] = #ttg.blocked<{sizePerThread = [4, 4]
// CHECK-DAG: #[[$SRC:.+]] = #ttg.blocked<{sizePerThread = [1, 16]
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.min_sg_size" = 16 : i32} {
  // CHECK-LABEL: tt.func public @shared_convert_fma
  tt.func public @shared_convert_fma(
    %arg0: tensor<128x64xf8E5M2, #blocked1>,
    %arg1: tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
    %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>,
    %arg3: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst16 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    // CHECK-NOT: ttg.convert_layout {{.*}}xf8E5M2
    // CHECK: [[A16:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E5M2, #[[$SRC]]> -> tensor<128x64xf16, #[[$SRC]]>
    // CHECK: [[A32:%.*]] = tt.fp_to_fp %arg0 : tensor<128x64xf8E5M2, #[[$SRC]]> -> tensor<128x64xf32, #[[$SRC]]>
    // CHECK-NOT: ttg.convert_layout {{.*}}xf8E5M2
    // CHECK: [[CA32:%.*]] = ttg.convert_layout [[A32]] : tensor<128x64xf32, #[[$SRC]]> -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK: [[B32:%.*]] = tt.fp_to_fp %arg1 : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[CA32]], [[B32]], {{.*}} -> tensor<128x128xf32, #[[$BL]]>
    // CHECK-NOT: ttg.convert_layout {{.*}}xf8E5M2
    // CHECK: [[CA16:%.*]] = ttg.convert_layout [[A16]] : tensor<128x64xf16, #[[$SRC]]> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$BL]]}>>
    // CHECK: [[B16:%.*]] = tt.fp_to_fp %arg1 : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$BL]]}>>
    // CHECK: tt.dot [[CA16]], [[B16]], {{.*}} -> tensor<128x128xf16, #[[$BL]]>
    // CHECK-NOT: ttg.convert_layout {{.*}}xf8E5M2
    %a = ttg.convert_layout %arg0 : tensor<128x64xf8E5M2, #blocked1> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %r32 = tt.dot %a, %arg1, %cst : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    %r16 = tt.dot %a, %arg1, %cst16 : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf16, #blocked>
    tt.store %arg2, %r32 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.store %arg3, %r16 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
