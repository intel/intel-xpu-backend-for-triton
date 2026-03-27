// RUN: triton-opt %s -split-input-file -tritonintelgpu-remove-layout-conversions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>

// COM: ============================================================
// COM: Test 6: Descriptor index-update chain remains intact while
// COM: a trailing convert_layout is removed before descriptor_store.
// COM: This mirrors the old advance-op-chain regression intent but
// COM: for tensor descriptors.
// COM: ============================================================

// CHECK-LABEL: tt.func public @update_descriptor_index_chain
// CHECK-NOT: ttg.convert_layout
// CHECK: %[[PID:.*]] = tt.get_program_id x : i32
// CHECK: %[[OFF:.*]] = arith.addi %{{.*}}, %[[PID]] : i32
// CHECK: tt.descriptor_store %{{.*}}[%[[OFF]], {{.*}}], {{.*}} : !tt.tensordesc<tensor<4x128xf16>>, tensor<4x128xf16, {{.*}}>
module attributes {"ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32, ttig.support_2d_block_io} {
  tt.func public @update_descriptor_index_chain(%desc: !tt.tensordesc<tensor<4x128xf16>>, %val: f16) {
    %c0_i32 = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %off0 = arith.addi %pid, %c0_i32 : i32
    %off1 = arith.addi %off0, %pid : i32
    %splat = tt.splat %val : f16 -> tensor<4x128xf16, #blocked>
    %cvt = ttg.convert_layout %splat : tensor<4x128xf16, #blocked> -> tensor<4x128xf16, #blocked1>
    tt.descriptor_store %desc[%off1, %c0_i32], %cvt : !tt.tensordesc<tensor<4x128xf16>>, tensor<4x128xf16, #blocked1>
    tt.return
  }
}
