// RUN: triton-opt %s -split-input-file -test-register-pressure | FileCheck %s

// The test pass prints the bare function name, then the analysis report, then
// the IR. Anchor CHECK-LABEL on the pass's bare-name line so the report checks
// are contiguous.
// CHECK-LABEL: loop_with_dpas_accumulator
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Block {{.*}} in scf.for: peak = 256 bytes
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 32 bytes
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32} {
  tt.func @loop_with_dpas_accumulator(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>) {
    // COM: A loop holding a DPAS accumulator. At the tt.dot in the loop body the
    // COM: live values (per-thread bytes) are: operand A tensor<8x64xf16,#dot0>
    // COM: = 32 elems * 2B = 64; operand B tensor<64x16xf16,#dot1> = 64 * 2B =
    // COM: 128; accumulator iter_arg tensor<8x16xf32,#dpas> = 8 * 4B = 32; dot
    // COM: result (same type) = 32. Peak of the loop body block = 256 bytes.
    // COM: The outer func block only has the scf.for result (32 bytes) live.
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32, #dpas>
    %0 = tt.make_tensor_descriptor %arg0, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <8x64xf16>
    %1 = tt.make_tensor_descriptor %arg1, [%c0_i32, %c0_i32], [%c0_i64, %c0_i64] : <f16>, <64x16xf16>
    %2 = scf.for %arg2 = %c0_i32 to %c64_i32 step %c64_i32 iter_args(%arg3 = %cst) -> (tensor<8x16xf32, #dpas>) : i32 {
      %3 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<8x64xf16> -> tensor<8x64xf16, #dot0>
      %4 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<64x16xf16> -> tensor<64x16xf16, #dot1>
      %5 = tt.dot %3, %4, %arg3, inputPrecision = tf32 : tensor<8x64xf16, #dot0> * tensor<64x16xf16, #dot1> -> tensor<8x16xf32, #dpas>
      scf.yield %5 : tensor<8x16xf32, #dpas>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: block_with_excluded_ops
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 0 bytes
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @block_with_excluded_ops(%arg0: !tt.ptr<f32>) {
    // COM: This test checks that rematerializable values (constants, splat, make_range)
    // COM: are excluded from register pressure.
    // COM: Every value here is rematerializable and excluded: the constant
    // COM: tensor (arith.constant), the make_range, and the splat (whose source
    // COM: %arg0 is a block argument — but the splat result itself is excluded
    // COM: because it is a splat of a value with no live cost here). The i32
    // COM: constant is also a constant. Peak = 0 bytes.
    %c1024_i32 = arith.constant 1024 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x256xf32, #blocked>
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: scalar_only_block
// CHECK-NEXT: Register Pressure Analysis (per-thread bytes):
// CHECK-NEXT: Block {{.*}} in tt.func: peak = 16 bytes
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @scalar_only_block(%arg0: i32, %arg1: i32) -> i32 {
    // COM: Scalars only (i32 = 4 bytes each). currentlyLiveValues takes an
    // COM: expansive view (a value defined-by or consumed-by the op is live), so
    // COM: peak is set at %1 = muli where {%arg0, %arg1, %0, %1} are all live =
    // COM: 4 * 4B = 16 bytes.
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg1 : i32
    %2 = arith.addi %1, %arg0 : i32
    tt.return %2 : i32
  }
}
