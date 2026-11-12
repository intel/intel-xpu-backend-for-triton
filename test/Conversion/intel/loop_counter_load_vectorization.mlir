// RUN: triton-opt %s --convert-scf-to-cf --convert-triton-intel-gpu-to-llvm | FileCheck %s

// COM: `make_llir` runs `convert-scf-to-cf` before the LLVM lowering builds its
// COM: own AxisInfo, so the sign check behind the `arith.remsi` contiguity
// COM: deduction has to recognize a counter in block-argument form too.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: @rem_loop_counter_load
  tt.func @rem_loop_counter_load(%in: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %rnumel: i32, %HW: i32 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : i32
    %c128 = arith.constant 128 : i32
    scf.for %roffset = %c0 to %rnumel step %c128 : i32 {
      %base = tt.splat %roffset : i32 -> tensor<128xi32, #blocked>
      %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
      %rindex = arith.addi %base, %range : tensor<128xi32, #blocked>
      %hwv = tt.splat %HW : i32 -> tensor<128xi32, #blocked>
      %hw = arith.remsi %rindex, %hwv : tensor<128xi32, #blocked>
      %ptrs = tt.splat %in : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
      %p = tt.addptr %ptrs, %hw : tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xi32, #blocked>
      // CHECK: llvm.load {{.*}}<alignment = 16> : !llvm.ptr<1> -> vector<4xi32>
      // CHECK-NOT: llvm.load
      %v = tt.load %p : tensor<128x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}
