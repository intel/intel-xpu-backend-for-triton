// Boundary table for the profitability cost model:
//
//   distribute iff sum(accumulator bytes of the loop's dots) / num-warps
//                  > grf_bytes_per_thread(grf-mode)
//
// where grf_bytes_per_thread is 4096 for "128", 8192 for "256", 16384 for
// "512" and 8192 for "default"/"auto". Each RUN line below pins one property
// of that formula; every function in this file is above the budget for the
// DIST runs and at-or-below it for the NODIST runs.

// COM: Above the budget: distribute.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=4 grf-mode=256" | FileCheck %s --check-prefixes=CHECK,DIST

// COM: Exactly at the budget: do not distribute (the comparison is strict).
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=16 grf-mode=256" | FileCheck %s --check-prefixes=CHECK,NODIST

// COM: Same num-warps, half the budget: the verdict flips, so the threshold
// COM: really is the register file size and not a constant.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=16 grf-mode=128" | FileCheck %s --check-prefixes=CHECK,DIST

// COM: Same pair one budget up: 256 GRF rejects nothing here, 512 GRF does.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=8 grf-mode=256" | FileCheck %s --check-prefixes=CHECK,DIST
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=8 grf-mode=512" | FileCheck %s --check-prefixes=CHECK,NODIST

// COM: "default" and "auto" use the 256-GRF budget: a spilling 128-GRF kernel
// COM: is rebuilt at 256 GRF by the driver, so that is the effective budget.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=16 grf-mode=default" | FileCheck %s --check-prefixes=CHECK,NODIST
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=true num-warps=16 grf-mode=auto" | FileCheck %s --check-prefixes=CHECK,NODIST

// COM: Without the cost model every legal loop is distributed, whatever the
// COM: model would have said -- including with no options at all, which is how
// COM: the rest of this directory's tests invoke the pass.
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute="use-cost-model=false num-warps=16 grf-mode=256" | FileCheck %s --check-prefixes=CHECK,DIST
// RUN: triton-opt %s -split-input-file -tritonintelgpu-loop-distribute | FileCheck %s --check-prefixes=CHECK,DIST

// COM: Note that none of the modules below carry a "ttg.num-warps" attribute:
// COM: the pass runs before the module is annotated, so num-warps and grf-mode
// COM: can only come from the pass options.

// COM: Two 128x128xf32 accumulators: 2 * 128 * 128 * 4 = 131072 bytes, i.e.
// COM: 32768 bytes/thread at 4 warps, 16384 at 8 and 8192 at 16.
// CHECK-LABEL: @gate_boundary
// DIST: scf.for
// DIST: scf.for
// DIST-NOT: scf.for
// NODIST: scf.for
// NODIST-NOT: scf.for
module {
  tt.func @gate_boundary(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x128xf16>, %arg2: !tt.tensordesc<64x128xf16>) -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc0 = %cst, %acc1 = %cst) -> (tensor<128x128xf32>, tensor<128x128xf32>) : i32 {
      %a = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
      %b0 = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
      %b1 = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
      %d0 = tt.dot %a, %b0, %acc0, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      %d1 = tt.dot %a, %b1, %acc1, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x128xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x128xf32>, tensor<128x128xf32>
  }
}

// -----

// COM: Two 256x128xf16 accumulators: twice the elements of @gate_boundary but
// COM: the same 131072 bytes, so the verdicts must match it. A model that
// COM: assumed 4-byte accumulator elements would compute 262144 bytes and
// COM: distribute on the NODIST runs.
// CHECK-LABEL: @gate_uses_accumulator_element_size
// DIST: scf.for
// DIST: scf.for
// DIST-NOT: scf.for
// NODIST: scf.for
// NODIST-NOT: scf.for
module {
  tt.func @gate_uses_accumulator_element_size(%arg0: !tt.tensordesc<256x64xf16>, %arg1: !tt.tensordesc<64x128xf16>, %arg2: !tt.tensordesc<64x128xf16>) -> (tensor<256x128xf16>, tensor<256x128xf16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf16>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc0 = %cst, %acc1 = %cst) -> (tensor<256x128xf16>, tensor<256x128xf16>) : i32 {
      %a = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<256x64xf16> -> tensor<256x64xf16>
      %b0 = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
      %b1 = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
      %d0 = tt.dot %a, %b0, %acc0, inputPrecision = tf32 : tensor<256x64xf16> * tensor<64x128xf16> -> tensor<256x128xf16>
      %d1 = tt.dot %a, %b1, %acc1, inputPrecision = tf32 : tensor<256x64xf16> * tensor<64x128xf16> -> tensor<256x128xf16>
      scf.yield %d0, %d1 : tensor<256x128xf16>, tensor<256x128xf16>
    }
    tt.return %0#0, %0#1 : tensor<256x128xf16>, tensor<256x128xf16>
  }
}

// -----

// COM: Accumulators of different shapes: 128x64xf32 (32768 bytes) plus
// COM: 128x128xf32 (65536 bytes) = 98304 bytes, i.e. 24576 bytes/thread at 4
// COM: warps, 12288 at 8 and 6144 at 16. Both dots are counted: a model
// COM: doubling the first dot alone would get 65536 bytes and would not
// COM: distribute on the 16-warp / 128-GRF run.
// CHECK-LABEL: @gate_sums_both_dots
// DIST: scf.for
// DIST: scf.for
// DIST-NOT: scf.for
// NODIST: scf.for
// NODIST-NOT: scf.for
module {
  tt.func @gate_sums_both_dots(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x64xf16>, %arg2: !tt.tensordesc<64x128xf16>) -> (tensor<128x64xf32>, tensor<128x128xf32>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %0:2 = scf.for %k = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc0 = %cst0, %acc1 = %cst1) -> (tensor<128x64xf32>, tensor<128x128xf32>) : i32 {
      %a = tt.descriptor_load %arg0[%c0_i32, %k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16>
      %b0 = tt.descriptor_load %arg1[%k, %c0_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16>
      %b1 = tt.descriptor_load %arg2[%k, %c0_i32] : !tt.tensordesc<64x128xf16> -> tensor<64x128xf16>
      %d0 = tt.dot %a, %b0, %acc0, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x64xf16> -> tensor<128x64xf32>
      %d1 = tt.dot %a, %b1, %acc1, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x128xf16> -> tensor<128x128xf32>
      scf.yield %d0, %d1 : tensor<128x64xf32>, tensor<128x128xf32>
    }
    tt.return %0#0, %0#1 : tensor<128x64xf32>, tensor<128x128xf32>
  }
}
