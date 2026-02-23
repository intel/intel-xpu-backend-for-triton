// RUN: triton-opt %s -tritonintelgpu-remove-layout-conversions 2>&1 | FileCheck %s

// CHECK-LABEL: tt.func public @update_advance_op_chain
// CHECK-NOT: ttg.convert_layout
// CHECK: [[MAKETENSORPTR:%.*]] = tt.make_tensor_ptr
// CHECK: [[ADVANCE:%.*]] = tt.advance [[MAKETENSORPTR]]
// CHECK: [[ADVANCE1:%.*]] = tt.advance [[ADVANCE]]
// CHECK: tt.store [[ADVANCE1]]

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 2 : i32, ttig.support_2d_block_io} {
  tt.func public @update_advance_op_chain(%ptr: !tt.ptr<f16>, %val: tensor<4x128xf16, #blocked>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i64
    %c128 = arith.constant 128 : i64
    %xoffset = tt.get_program_id x : i32
    %desc = tt.make_tensor_ptr %ptr, [%c128, %c128], [%c128, %c1], [%c0, %c0] {order = array<i32: 1, 0>} : <tensor<4x128xf16, #blocked1>>
    %adv = tt.advance %desc, [%xoffset, %c0] : <tensor<4x128xf16, #blocked1>>
    %adv1 = tt.advance %adv, [%xoffset, %c0] : <tensor<4x128xf16, #blocked1>>
    %0 = ttg.convert_layout %val : tensor<4x128xf16, #blocked> -> tensor<4x128xf16, #blocked1>
    tt.store %adv1, %0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<4x128xf16, #blocked1>>
    tt.return
  }
}
