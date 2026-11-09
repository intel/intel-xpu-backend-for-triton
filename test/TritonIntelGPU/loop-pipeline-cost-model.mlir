// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=4 device-arch=bmg core-clock-khz=2850000 mem-clock-mhz=19000" | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=4 core-clock-khz=2850000 mem-clock-mhz=19000" | FileCheck %s --check-prefix=KEEP
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=4 device-arch=bmg mem-clock-mhz=19000" | FileCheck %s --check-prefix=KEEP
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=4 device-arch=bmg core-clock-khz=2850000" | FileCheck %s --check-prefix=KEEP
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3 device-arch=bmg core-clock-khz=2850000 mem-clock-mhz=19000" | FileCheck %s --check-prefix=DROP
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=7 device-arch=bmg core-clock-khz=2850000 mem-clock-mhz=19000" | FileCheck %s --check-prefix=DROP
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=2 device-arch=bmg core-clock-khz=2850000 mem-clock-mhz=19000" | FileCheck %s --check-prefix=KEEP
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=4 use-barrier=true device-arch=bmg core-clock-khz=2850000 mem-clock-mhz=19000" | FileCheck %s --check-prefix=KEEP

// COM: The prefetch cost model drops the prefetch of the one gapped dot operand (A: 64-byte rows read
// COM: 256 bytes apart) in a long loop, and keeps every prefetch outside its calibrated regime. KEEP
// COM: runs leave it (no device-arch or clock; split barriers) or run 2 stages, whose predicted gain
// COM: at 224 iterations is under the 15% margin (the drop starts at 259); 3 and 7 stages drop (DROP).

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#dpas8 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 4, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 32], B = [32, 16], C = [8, 16]}>
#dpas3 = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 4, 8], repCluster = [1, 1, 1], A = [1, 8, 16], B = [1, 16, 16], C = [1, 8, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 16], warpsPerCTA = [8, 4], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // COM: 224 iterations of the gapped A walk: A's prefetch is dropped and B's is kept.
  // CHECK-LABEL: tt.func public @gapped_long_drops_a
  // CHECK-NOT:     ttig.descriptor_prefetch
  // CHECK-COUNT-3: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch
  // CHECK:         scf.for
  // CHECK-NOT:       ttig.descriptor_prefetch
  // CHECK:           ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:       ttig.descriptor_prefetch
  // CHECK:           tt.descriptor_load {{.*}} : !tt.tensordesc<256x32xbf16>
  // CHECK-NOT:       ttig.descriptor_prefetch
  // CHECK:           scf.yield
  // KEEP-LABEL: tt.func public @gapped_long_drops_a
  // KEEP:         scf.for
  // KEEP-NOT:     scf.yield
  // KEEP:           ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  // DROP-LABEL: tt.func public @gapped_long_drops_a
  // DROP-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  // DROP:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // DROP-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  // DROP:         scf.yield
  tt.func public @gapped_long_drops_a(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: 32 iterations end before the walk loses its shared reads: keep both.
  // CHECK-LABEL: tt.func public @gapped_short_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @gapped_short_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %r = scf.for %k = %c0_i32 to %c4096_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: An unknown trip count keeps both.
  // CHECK-LABEL: tt.func public @dynamic_trip_count_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @dynamic_trip_count_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>, %ub: i32) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %r = scf.for %k = %c0_i32 to %ub step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A walks backwards (subi), and B, loaded first, reaches the dot through a layout conversion:
  // COM: the step's magnitude decides, and the dropped candidate is A, not the first one.
  // CHECK-LABEL: tt.func public @negative_step_long_drops_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @negative_step_long_drops_a(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #blocked>
      %kr = arith.subi %c28672_i32, %k : i32
      %a = tt.descriptor_load %descA[%c0_i32, %kr] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %bc = ttg.convert_layout %b : tensor<32x256xbf16, #blocked> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %bc, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A stored K x M and B stored N x K, both loaded column-major: only B is gapped, and dropping a
  // COM: gapped B was never measured, so keep both.
  // CHECK-LABEL: tt.func public @gapped_b_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @gapped_b_keeps_both(%descA: !tt.tensordesc<32x256xbf16>, %descB: !tt.tensordesc<256x32xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%k, %c0_i32] {ttig.block_io = "column_major"} : !tt.tensordesc<32x256xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%c0_i32, %k] {ttig.block_io = "column_major"} : !tt.tensordesc<256x32xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: f16 operands, and the indices are a unit-step index induction variable times 128, cast to i32,
  // COM: plus a loop-invariant base: drop A.
  // CHECK-LABEL: tt.func public @tile_count_loop_drops_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xf16>
  tt.func public @tile_count_loop_drops_a(%descA: !tt.tensordesc<256x32xf16>, %descB: !tt.tensordesc<32x256xf16>, %base: i32) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c224 = arith.constant 224 : index
    %r = scf.for %t = %c0 to %c224 step %c1 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) {
      %tk = arith.muli %t, %c128 : index
      %ki = arith.index_cast %tk : index to i32
      %k = arith.addi %ki, %base : i32
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xf16> -> tensor<256x32xf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xf16> -> tensor<32x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xf16, #dot0> * tensor<32x256xf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A's index is an iter_arg advanced by addi: drop A.
  // CHECK-LABEL: tt.func public @iter_arg_index_drops_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @iter_arg_index_drops_a(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %off = %c0_i32) -> (tensor<256x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %off] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %next = arith.addi %off, %c128_i32 : i32
      scf.yield %d, %next : tensor<256x256xf32, #dpas>, i32
    }
    tt.return %r#0 : tensor<256x256xf32, #dpas>
  }

  // COM: B's index is an iter_arg advanced by subi, a step the pass does not derive: keep both.
  // CHECK-LABEL: tt.func public @nonconst_step_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @nonconst_step_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %n = %c28672_i32) -> (tensor<256x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %n] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %next = arith.subi %n, %c128_i32 : i32
      scf.yield %d, %next : tensor<256x256xf32, #dpas>, i32
    }
    tt.return %r#0 : tensor<256x256xf32, #dpas>
  }

  // COM: A's i32 index (k * 65536) * 65536 always wraps to 0: a step that overflows i32 keeps both.
  // CHECK-LABEL: tt.func public @wrapped_index_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @wrapped_index_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %p = arith.muli %k, %c65536_i32 : i32
      %off = arith.muli %p, %c65536_i32 : i32
      %a = tt.descriptor_load %descA[%c0_i32, %off] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A's i16 index o * 256 is always 0 before its extsi to i32: a wrapping operand keeps both.
  // CHECK-LABEL: tt.func public @narrow_wrap_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @narrow_wrap_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %c0_i16 = arith.constant 0 : i16
    %c256_i16 = arith.constant 256 : i16
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %o = %c0_i16) -> (tensor<256x256xf32, #dpas>, i16) : i32 {
      %w = arith.muli %o, %c256_i16 : i16
      %off = arith.extsi %w : i16 to i32
      %a = tt.descriptor_load %descA[%c0_i32, %off] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %next = arith.addi %o, %c256_i16 : i16
      scf.yield %d, %next : tensor<256x256xf32, #dpas>, i16
    }
    tt.return %r#0 : tensor<256x256xf32, #dpas>
  }

  // COM: A loop-variant descriptor keeps both: this one's base moves back as A's index moves on.
  // CHECK-LABEL: tt.func public @loop_variant_desc_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @loop_variant_desc_keeps_both(%baseA: !tt.ptr<bf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %c256_i32 = arith.constant 256 : i32
      %c1_i64 = arith.constant 1 : i64
      %cs = arith.constant 28672 : i64
      %back = arith.subi %c28672_i32, %k : i32
      %pA = tt.addptr %baseA, %back : !tt.ptr<bf16>, i32
      %descA = tt.make_tensor_descriptor %pA, [%c256_i32, %c28672_i32], [%cs, %c1_i64] : <bf16>, <256x32xbf16>
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A descriptor built in the loop from loop-invariant values is invariant: drop A.
  // CHECK-LABEL: tt.func public @invariant_desc_in_loop_drops_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @invariant_desc_in_loop_drops_a(%baseA: !tt.ptr<bf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %c256_i32 = arith.constant 256 : i32
      %c1_i64 = arith.constant 1 : i64
      %cs = arith.constant 28672 : i64
      %pA = tt.addptr %baseA, %c28672_i32 : !tt.ptr<bf16>, i32
      %descA = tt.make_tensor_descriptor %pA, [%c256_i32, %c28672_i32], [%cs, %c1_i64] : <bf16>, <256x32xbf16>
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: int8 DPAS runs at twice the 16-bit rate, so this BK=32 walk is fill-bound: drop A. Priced at
  // COM: the 16-bit rate it would be compute-bound and keep.
  // CHECK-LABEL: tt.func public @int8_operands_drop_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xi8>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xi8>
  tt.func public @int8_operands_drop_a(%descA: !tt.tensordesc<256x32xi8>, %descB: !tt.tensordesc<32x256xi8>) -> tensor<256x256xi32, #dpas8> {
    %cst = arith.constant dense<0> : tensor<256x256xi32, #dpas8>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xi32, #dpas8>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xi8> -> tensor<256x32xi8, #ttg.dot_op<{opIdx = 0, parent = #dpas8, kWidth = 2}>>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xi8> -> tensor<32x256xi8, #ttg.dot_op<{opIdx = 1, parent = #dpas8, kWidth = 4}>>
      %d = tt.dot %a, %b, %acc : tensor<256x32xi8, #ttg.dot_op<{opIdx = 0, parent = #dpas8, kWidth = 2}>> * tensor<32x256xi8, #ttg.dot_op<{opIdx = 1, parent = #dpas8, kWidth = 4}>> -> tensor<256x256xi32, #dpas8>
      scf.yield %d : tensor<256x256xi32, #dpas8>
    }
    tt.return %r : tensor<256x256xi32, #dpas8>
  }

  // COM: A non-default cache modifier keeps both.
  // CHECK-LABEL: tt.func public @cg_policy_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @cg_policy_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {cachePolicy = #tt.cache_policy<cache_modifier = cg, eviction_policy = evict_normal>, ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A non-default eviction policy keeps both.
  // CHECK-LABEL: tt.func public @evict_policy_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @evict_policy_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {cachePolicy = #tt.cache_policy<cache_modifier = none, eviction_policy = evict_last>, ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: B from a tensor-of-pointers block load keeps both: only descriptor walks are priced.
  // CHECK-LABEL: tt.func public @nondesc_operand_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @nondesc_operand_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %ptrsB: tensor<32x256x!tt.ptr<bf16>, #blocked>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %offB = arith.constant dense<8192> : tensor<32x256xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %pB = %ptrsB) -> (tensor<256x256xf32, #dpas>, tensor<32x256x!tt.ptr<bf16>, #blocked>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.load %pB {ttig.block_io = "row_major"} : tensor<32x256x!tt.ptr<bf16>, #blocked>
      %bc = ttg.convert_layout %b : tensor<32x256xbf16, #blocked> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %bc, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %pNext = tt.addptr %pB, %offB : tensor<32x256x!tt.ptr<bf16>, #blocked>, tensor<32x256xi32, #blocked>
      scf.yield %d, %pNext : tensor<256x256xf32, #dpas>, tensor<32x256x!tt.ptr<bf16>, #blocked>
    }
    tt.return %r#0 : tensor<256x256xf32, #dpas>
  }

  // COM: Two dots fed by the two prefetches keep both: their summed work makes the loop compute-bound,
  // COM: while one of them alone would drop A.
  // CHECK-LABEL: tt.func public @two_dots_keep_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @two_dots_keep_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> (tensor<256x256xf32, #dpas>, tensor<256x256xf32, #dpas>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %acc2 = %cst) -> (tensor<256x256xf32, #dpas>, tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %d2 = tt.dot %a, %b, %acc2, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d, %d2 : tensor<256x256xf32, #dpas>, tensor<256x256xf32, #dpas>
    }
    tt.return %r#0, %r#1 : tensor<256x256xf32, #dpas>, tensor<256x256xf32, #dpas>
  }

  // COM: Two dots sharing A with a half-width B2 (three prefetches) leave the loop fill-bound: drop A.
  // CHECK-LABEL: tt.func public @two_dots_three_prefetches_drop_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @two_dots_three_prefetches_drop_a(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>, %descB2: !tt.tensordesc<32x128xbf16>) -> (tensor<256x256xf32, #dpas>, tensor<256x128xf32, #dpas>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %acc2 = %cst2) -> (tensor<256x256xf32, #dpas>, tensor<256x128xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %b2 = tt.descriptor_load %descB2[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x128xbf16> -> tensor<32x128xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %d2 = tt.dot %a, %b2, %acc2, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x128xbf16, #dot1> -> tensor<256x128xf32, #dpas>
      scf.yield %d, %d2 : tensor<256x256xf32, #dpas>, tensor<256x128xf32, #dpas>
    }
    tt.return %r#0, %r#1 : tensor<256x256xf32, #dpas>, tensor<256x128xf32, #dpas>
  }

  // COM: A dot under an scf.if is priced as if it always ran: drop A.
  // CHECK-LABEL: tt.func public @conditional_dot_drops_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @conditional_dot_drops_a(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>, %p: i1) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = scf.if %p -> (tensor<256x256xf32, #dpas>) {
        %dd = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
        scf.yield %dd : tensor<256x256xf32, #dpas>
      } else {
        scf.yield %acc : tensor<256x256xf32, #dpas>
      }
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: A tt.dot_scaled beside the dot keeps both, even nested in an scf.if.
  // CHECK-LABEL: tt.func public @nested_dot_scaled_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @nested_dot_scaled_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>, %x: tensor<128x64xf8E4M3FN, #blocked>, %xs: tensor<128x2xi8, #blocked>, %y: tensor<64x128xf8E5M2, #blocked>, %ys: tensor<128x2xi8, #blocked>, %p: i1) -> (tensor<256x256xf32, #dpas>, tensor<128x128xf32, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %cst2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r:2 = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst, %acc2 = %cst2) -> (tensor<256x256xf32, #dpas>, tensor<128x128xf32, #blocked>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      %s = scf.if %p -> (tensor<128x128xf32, #blocked>) {
        %ss = tt.dot_scaled %x scale %xs, %y scale %ys, %acc2 lhs = e4m3 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E4M3FN, #blocked>, tensor<128x2xi8, #blocked> * tensor<64x128xf8E5M2, #blocked>, tensor<128x2xi8, #blocked> -> tensor<128x128xf32, #blocked>
        scf.yield %ss : tensor<128x128xf32, #blocked>
      } else {
        scf.yield %acc2 : tensor<128x128xf32, #blocked>
      }
      scf.yield %d, %s : tensor<256x256xf32, #dpas>, tensor<128x128xf32, #blocked>
    }
    tt.return %r#0, %r#1 : tensor<256x256xf32, #dpas>, tensor<128x128xf32, #blocked>
  }

  // COM: B upcast from fp8 in the loop keeps both: only layout conversions are looked through.
  // CHECK-LABEL: tt.func public @fp8_upcast_operand_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @fp8_upcast_operand_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xf8E4M3FN>) -> tensor<256x256xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #dpas>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #dot0>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xf8E4M3FN> -> tensor<32x256xf8E4M3FN, #dot1>
      %bu = tt.fp_to_fp %b : tensor<32x256xf8E4M3FN, #dot1> -> tensor<32x256xbf16, #dot1>
      %d = tt.dot %a, %bu, %acc, inputPrecision = tf32 : tensor<256x32xbf16, #dot0> * tensor<32x256xbf16, #dot1> -> tensor<256x256xf32, #dpas>
      scf.yield %d : tensor<256x256xf32, #dpas>
    }
    tt.return %r : tensor<256x256xf32, #dpas>
  }

  // COM: An FMA (blocked) dot keeps both.
  // CHECK-LABEL: tt.func public @fma_dot_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<256x32xbf16>
  tt.func public @fma_dot_keeps_both(%descA: !tt.tensordesc<256x32xbf16>, %descB: !tt.tensordesc<32x256xbf16>) -> tensor<256x256xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<256x256xf32, #blocked>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<256x32xbf16> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b = tt.descriptor_load %descB[%k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x256xbf16> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %d = tt.dot %a, %b, %acc : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<256x256xf32, #blocked>
      scf.yield %d : tensor<256x256xf32, #blocked>
    }
    tt.return %r : tensor<256x256xf32, #blocked>
  }

  // COM: A rank-3 dot counts its batch: two 256x512 products make this gap-2 walk compute-bound, so
  // COM: keep both. One product would drop A.
  // CHECK-LABEL: tt.func public @rank3_dot_keeps_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<2x256x32xbf16>
  tt.func public @rank3_dot_keeps_both(%descA: !tt.tensordesc<2x256x32xbf16>, %descB: !tt.tensordesc<2x32x512xbf16>) -> tensor<2x256x512xf32, #dpas3> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x256x512xf32, #dpas3>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c14336_i32 = arith.constant 14336 : i32
    %r = scf.for %k = %c0_i32 to %c14336_i32 step %c64_i32 iter_args(%acc = %cst) -> (tensor<2x256x512xf32, #dpas3>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<2x256x32xbf16> -> tensor<2x256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>>
      %b = tt.descriptor_load %descB[%c0_i32, %k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<2x32x512xbf16> -> tensor<2x32x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<2x256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>> * tensor<2x32x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>> -> tensor<2x256x512xf32, #dpas3>
      scf.yield %d : tensor<2x256x512xf32, #dpas3>
    }
    tt.return %r : tensor<2x256x512xf32, #dpas3>
  }

  // COM: A rank-3 dot is priced like a 2-D one: this gap-4 walk at batch 2 drops A.
  // CHECK-LABEL: tt.func public @rank3_gap4_drops_a
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<2x32x256xbf16>
  // CHECK-NOT:     ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<2x256x32xbf16>
  tt.func public @rank3_gap4_drops_a(%descA: !tt.tensordesc<2x256x32xbf16>, %descB: !tt.tensordesc<2x32x256xbf16>) -> tensor<2x256x256xf32, #dpas3> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x256x256xf32, #dpas3>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c28672_i32 = arith.constant 28672 : i32
    %r = scf.for %k = %c0_i32 to %c28672_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<2x256x256xf32, #dpas3>) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<2x256x32xbf16> -> tensor<2x256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>>
      %b = tt.descriptor_load %descB[%c0_i32, %k, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<2x32x256xbf16> -> tensor<2x32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<2x256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #dpas3, kWidth = 1}>> * tensor<2x32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #dpas3, kWidth = 2}>> -> tensor<2x256x256xf32, #dpas3>
      scf.yield %d : tensor<2x256x256xf32, #dpas3>
    }
    tt.return %r : tensor<2x256x256xf32, #dpas3>
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth = 1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 16], warpsPerCTA = [32, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", "ttig.support_prefetch_256b"} {
  // COM: A self-product plus a prefetched elementwise load (support_prefetch_256b) keeps both: two
  // COM: candidates, but only one feeds the dot.
  // CHECK-LABEL: tt.func public @self_product_and_elementwise_keep_both
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x32xbf16>
  tt.func public @self_product_and_elementwise_keep_both(%desc: !tt.tensordesc<32x32xbf16>, %descE: !tt.tensordesc<32x16xbf16>) -> (tensor<32x32xf32, #dpas>, tensor<32x16xbf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %cste = arith.constant dense<0.000000e+00> : tensor<32x16xbf16, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256000_i32 = arith.constant 256000 : i32
    %r:2 = scf.for %k = %c0_i32 to %c256000_i32 step %c128_i32 iter_args(%acc = %cst, %sum = %cste) -> (tensor<32x32xf32, #dpas>, tensor<32x16xbf16, #blocked>) : i32 {
      %x = tt.descriptor_load %desc[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xbf16> -> tensor<32x32xbf16, #blocked>
      %e = tt.descriptor_load %descE[%c0_i32, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<32x16xbf16> -> tensor<32x16xbf16, #blocked>
      %a = ttg.convert_layout %x : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #dot0>
      %b = ttg.convert_layout %x : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<32x32xbf16, #dot0> * tensor<32x32xbf16, #dot1> -> tensor<32x32xf32, #dpas>
      %s = arith.addf %sum, %e : tensor<32x16xbf16, #blocked>
      scf.yield %d, %s : tensor<32x32xf32, #dpas>, tensor<32x16xbf16, #blocked>
    }
    tt.return %r#0, %r#1 : tensor<32x32xf32, #dpas>, tensor<32x16xbf16, #blocked>
  }

  // COM: A lone prefetch is never dropped, even when it feeds both operands of a gapped long loop.
  // COM: The loop is long enough that the model alone would drop this self-product.
  // CHECK-LABEL: tt.func public @lone_prefetch_kept
  // CHECK:         ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<32x32xbf16>
  tt.func public @lone_prefetch_kept(%desc: !tt.tensordesc<32x32xbf16>) -> tensor<32x32xf32, #dpas> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #dpas>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c256000_i32 = arith.constant 256000 : i32
    %r = scf.for %k = %c0_i32 to %c256000_i32 step %c128_i32 iter_args(%acc = %cst) -> (tensor<32x32xf32, #dpas>) : i32 {
      %x = tt.descriptor_load %desc[%c0_i32, %k] {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xbf16> -> tensor<32x32xbf16, #blocked>
      %a = ttg.convert_layout %x : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #dot0>
      %b = ttg.convert_layout %x : tensor<32x32xbf16, #blocked> -> tensor<32x32xbf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<32x32xbf16, #dot0> * tensor<32x32xbf16, #dot1> -> tensor<32x32xf32, #dpas>
      scf.yield %d : tensor<32x32xf32, #dpas>
    }
    tt.return %r : tensor<32x32xf32, #dpas>
  }
}
