// RUN: triton-opt %s -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s
// RUN: triton-opt %s -tritonintelgpu-pipeline="num-stages=3 tile-barrier=false" | FileCheck %s --check-prefix=OFF
// RUN: triton-opt %s -tritonintelgpu-pipeline="num-stages=3 use-barrier" | FileCheck %s --check-prefix=UB

// COM: A pipelined loop nested in a work-group-uniform loop gets one `ttg.barrier none` at the top of the
// COM: enclosing loop body, so a persistent kernel re-aligns its warps at every tile (issue #8719).

// OFF-NOT: ttg.barrier

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: tt.func public @persistent_two_loops
  // CHECK:         scf.for
  // CHECK-NEXT:      ttg.barrier none
  // CHECK-NOT:       ttg.barrier
  // CHECK:         tt.return
  // UB-LABEL:    tt.func public @persistent_two_loops
  // UB-NOT:        ttg.barrier
  // UB:            triton_gen.split_barrier_arrive
  tt.func public @persistent_two_loops(%descA: !tt.tensordesc<128x64xf16>, %descB: !tt.tensordesc<64x256xf16>, %K: i32, %tiles: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %np = tt.get_num_programs x : i32
    scf.for %t = %pid to %tiles step %np : i32 {
      %r0 = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x256xf32, #dpas>) : i32 {
        %a = tt.descriptor_load %descA[%t, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
        %b = tt.descriptor_load %descB[%k, %t] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
        %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
        scf.yield %d : tensor<128x256xf32, #dpas>
      }
      %r1 = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %r0) -> (tensor<128x256xf32, #dpas>) : i32 {
        %a = tt.descriptor_load %descA[%t, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
        %b = tt.descriptor_load %descB[%k, %t] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
        %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
        scf.yield %d : tensor<128x256xf32, #dpas>
      }
    }
    tt.return
  }

  // CHECK-LABEL: tt.func public @not_nested
  // CHECK-NOT:     ttg.barrier
  // CHECK:         tt.return
  tt.func public @not_nested(%descA: !tt.tensordesc<128x64xf16>, %descB: !tt.tensordesc<64x256xf16>, %K: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      scf.yield %d : tensor<128x256xf32, #dpas>
    }
    tt.return
  }

  // COM: A bound that differs between warps would leave some warps waiting at the barrier. Here it is an
  // COM: iter_arg seeded with the warp id: unlike an induction variable, it is not uniform by its loop's bounds.
  // CHECK-LABEL: tt.func public @nonuniform_bound
  // CHECK-NOT:     ttg.barrier
  // CHECK:         tt.return
  tt.func public @nonuniform_bound(%descA: !tt.tensordesc<128x64xf16>, %descB: !tt.tensordesc<64x256xf16>, %K: i32, %tiles: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %np = tt.get_num_programs x : i32
    %w = ttg.warp_id
    %u = scf.for %bat = %c0 to %c1 step %c1 iter_args(%lb = %w) -> (i32) : i32 {
      scf.for %t = %lb to %tiles step %np : i32 {
        %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x256xf32, #dpas>) : i32 {
          %a = tt.descriptor_load %descA[%t, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
          %b = tt.descriptor_load %descB[%k, %t] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
          %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
          scf.yield %d : tensor<128x256xf32, #dpas>
        }
      }
      scf.yield %lb : i32
    }
    tt.return
  }

  // CHECK-LABEL: tt.func public @nonuniform_ancestor
  // CHECK-NOT:     ttg.barrier
  // CHECK:         tt.return
  tt.func public @nonuniform_ancestor(%descA: !tt.tensordesc<128x64xf16>, %descB: !tt.tensordesc<64x256xf16>, %K: i32, %tiles: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %np = tt.get_num_programs x : i32
    %w = ttg.warp_id
    %first = arith.cmpi eq, %w, %c0 : i32
    scf.if %first {
      scf.for %t = %pid to %tiles step %np : i32 {
        %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x256xf32, #dpas>) : i32 {
          %a = tt.descriptor_load %descA[%t, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
          %b = tt.descriptor_load %descB[%k, %t] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
          %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
          scf.yield %d : tensor<128x256xf32, #dpas>
        }
      }
    }
    tt.return
  }

  // COM: Without `ttig.block_io` the loads get no prefetch, so the inner loop is not pipelined.
  // CHECK-LABEL: tt.func public @not_pipelined
  // CHECK-NOT:     ttg.barrier
  // CHECK:         tt.return
  tt.func public @not_pipelined(%descA: !tt.tensordesc<128x64xf16>, %descB: !tt.tensordesc<64x256xf16>, %K: i32, %tiles: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %np = tt.get_num_programs x : i32
    scf.for %t = %pid to %tiles step %np : i32 {
      %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x256xf32, #dpas>) : i32 {
        %a = tt.descriptor_load %descA[%t, %k] : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
        %b = tt.descriptor_load %descB[%k, %t] : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
        %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
        scf.yield %d : tensor<128x256xf32, #dpas>
      }
    }
    tt.return
  }

  // COM: Bounds built from a uniform outer loop's induction variable are uniform; only the innermost
  // COM: enclosing loop gets the barrier.
  // CHECK-LABEL: tt.func public @iv_uniform_innermost
  // CHECK:         scf.for
  // CHECK-NOT:       ttg.barrier
  // CHECK:           scf.for
  // CHECK-NEXT:        ttg.barrier none
  // CHECK-NOT:       ttg.barrier
  // CHECK:         tt.return
  tt.func public @iv_uniform_innermost(%descA: !tt.tensordesc<128x64xf16>, %descB: !tt.tensordesc<64x256xf16>, %K: i32, %tiles: i32, %batches: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %np = tt.get_num_programs x : i32
    scf.for %bat = %c0 to %batches step %c1 : i32 {
      %lb = arith.addi %pid, %bat : i32
      scf.for %t = %lb to %tiles step %np : i32 {
        %r = scf.for %k = %c0 to %K step %c64 iter_args(%acc = %cst) -> (tensor<128x256xf32, #dpas>) : i32 {
          %a = tt.descriptor_load %descA[%t, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
          %b = tt.descriptor_load %descB[%k, %t] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
          %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
          scf.yield %d : tensor<128x256xf32, #dpas>
        }
      }
    }
    tt.return
  }
}
