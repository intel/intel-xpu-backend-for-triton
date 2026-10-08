// RUN: triton-opt %s -split-input-file --triton-intel-rewrite-tensor-descriptor-to-pointer --cse --convert-triton-to-tritongpu='target=xpu num-warps=4 threads-per-warp=16 num-ctas=1' --tritonintelgpu-materialize-block-pointer | FileCheck %s --implicit-check-not=unrealized_conversion_cast --implicit-check-not=tt.make_tensor_descriptor --implicit-check-not=tt.descriptor_load --implicit-check-not=tt.load

// COM: Integration starts with unencoded TTIR and executes the actual descriptor
// COM: rewrite, immediate CSE, TTGPU conversion, and materialization. Do not run
// COM: TTGPU conversion over the encoded materialize-tensor-descriptor fixtures.
// COM: The direct PAD_NAN load must remain native and receive row-major block I/O
// COM: and padding annotations; the select-connected load must use pointers.
// COM: These annotations are IR evidence, not proof of final hardware lowering.
// COM: Implicit negatives require exactly one maker and one access of each
// COM: representation, and prohibit casts throughout the function.
module attributes {ttig.support_2d_block_io} {
  tt.func public @descriptor_native_use_split(%incoming: !tt.tensordesc<64x32xf16>, %base: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %pitch: i64 {tt.divisibility = 16 : i32}, %choose_local: i1) -> (tensor<64x32xf16>, tensor<64x32xf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %desc = tt.make_tensor_descriptor %base, [%c64_i32, %c32_i32], [%pitch, %c1_i64] {padding = 2 : i32} : <f16>, <64x32xf16>
    %direct = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
    %selected = arith.select %choose_local, %desc, %incoming : !tt.tensordesc<64x32xf16>
    %indirect = tt.descriptor_load %selected[%c0_i32, %c0_i32] : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16>
    tt.return %direct, %indirect : tensor<64x32xf16>, tensor<64x32xf16>
  }
}

// CHECK-LABEL: @descriptor_native_use_split
// CHECK-SAME: %[[INCOMING_PTR:[^:]*]]: !tt.ptr<f16>
// CHECK-SAME: %[[PAD:[^:]*]]: i1, %[[ROUND:[^:]*]]: i1, %[[BASE:[^:]*]]: !tt.ptr<f16>
// CHECK-SAME: %[[PITCH:[^:]*]]: i64
// CHECK-SAME: %[[COND:[^:]*]]: i1)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : i32
// CHECK-DAG: %[[C64:.*]] = arith.constant 64 : i32
// CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[BASE]], [%[[C64]], %[[C32]]], [%[[PITCH]], %[[C1]]] {padding = 2 : i32} :
// CHECK-NEXT: %[[DIRECT:.*]] = tt.descriptor_load %[[DESC]][%[[C0]], %[[C0]]] {ttig.block_io = "row_major", ttig.desc_padding = 2 : i32} : !tt.tensordesc<64x32xf16> -> tensor<64x32xf16, {{.*}}>
// CHECK: %[[SELECTED:.*]]:7 = scf.if %[[COND]] -> (!tt.ptr<f16>, i64, i64, i64, i64, i1, i1) {
// CHECK: scf.yield %[[BASE]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// CHECK: } else {
// CHECK: scf.yield %[[INCOMING_PTR]], {{.*}} : !tt.ptr<f16>, i64, i64, i64, i64, i1, i1
// CHECK: }
// CHECK: %[[PTRS:.*]] = tt.splat %[[SELECTED]]#0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, {{.*}}>
// CHECK: %[[ROW_PTRS:.*]] = tt.addptr %[[PTRS]], %{{.*}} :
// CHECK: %[[COL_PTRS:.*]] = tt.addptr %[[ROW_PTRS]], %{{.*}} :
// CHECK: %[[INDIRECT:.*]] = tt.load %[[COL_PTRS]], %{{.*}}, %{{.*}} {{.*}}: tensor<64x32x!tt.ptr<f16>, {{.*}}>
// CHECK-NEXT: tt.return %[[DIRECT]], %[[INDIRECT]] : tensor<64x32xf16, {{.*}}>, tensor<64x32xf16, {{.*}}>
