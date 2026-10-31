// RUN: triton-opt %s -split-input-file -triton-intel-simplify-signed-arithmetic | FileCheck %s

// COM: `8 - (pid & 7)` is in [1, 8]. Only the range analysis can prove it:
// COM: the pattern matcher does not look through arith.subi.
tt.func @remsi_dividend_from_range(%out: !tt.ptr<i32>) {
  %c4 = arith.constant 4 : i32
  %c7 = arith.constant 7 : i32
  %c8 = arith.constant 8 : i32
  %pid = tt.get_program_id x : i32
  %a = arith.andi %pid, %c7 : i32
  %b = arith.subi %c8, %a : i32
  %r = arith.remsi %b, %c4 : i32
  tt.store %out, %r : !tt.ptr<i32>
  tt.return
}

// CHECK-LABEL: tt.func @remsi_dividend_from_range
// CHECK-NOT:     arith.remsi
// CHECK:         arith.remui

// -----

// COM: `(pid & 7) + 1` is in [1, 8], so the divisor is strictly positive.
tt.func @divsi_divisor_from_range(%out: !tt.ptr<i32>) {
  %c1 = arith.constant 1 : i32
  %c7 = arith.constant 7 : i32
  %pid = tt.get_program_id x : i32
  %a = arith.andi %pid, %c7 : i32
  %d = arith.addi %a, %c1 : i32
  %q = arith.divsi %pid, %d : i32
  tt.store %out, %q : !tt.ptr<i32>
  tt.return
}

// CHECK-LABEL: tt.func @divsi_divisor_from_range
// CHECK-NOT:     arith.divsi
// CHECK:         arith.divui

// -----

// COM: The grouped program-id swizzle of the matmul tutorial, with the grid
// COM: size (4 x 4 tiles) assumed. The assume bounds `pid`, which makes
// COM: `group_size_m` strictly positive and `pid_n` non-negative, so the
// COM: column offsets `% N` become unsigned.
tt.func @grouped_pid_swizzle(%out: !tt.ptr<i32>) {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %c256 = arith.constant 256 : i32
  %cN = arith.constant dense<1024> : tensor<256xi32>
  %pid = tt.get_program_id x : i32
  %in_grid = arith.cmpi slt, %pid, %c16 : i32
  llvm.intr.assume %in_grid : i1
  %group_id = arith.divsi %pid, %c16 : i32
  %first_pid_m = arith.muli %group_id, %c4 : i32
  %rows_left = arith.subi %c4, %first_pid_m : i32
  %group_size_m = arith.minsi %rows_left, %c4 : i32
  %pid_in_group = arith.remsi %pid, %c16 : i32
  %pid_n = arith.divsi %pid_in_group, %group_size_m : i32
  %start_n = arith.muli %pid_n, %c256 : i32
  %splat = tt.splat %start_n : i32 -> tensor<256xi32>
  %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
  %offs = arith.addi %splat, %range : tensor<256xi32>
  %offs_bn = arith.remsi %offs, %cN : tensor<256xi32>
  %ptr = tt.splat %out : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
  %ptrs = tt.addptr %ptr, %range : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
  tt.store %ptrs, %offs_bn : tensor<256x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @grouped_pid_swizzle
// CHECK-NOT:     arith.divsi
// CHECK-NOT:     arith.remsi
// CHECK:         arith.divui %{{.*}}, %{{.*}} : i32
// CHECK:         arith.remui %{{.*}}, %{{.*}} : tensor<256xi32>

// -----

// COM: Without the assume `group_size_m` may be negative for an out-of-grid
// COM: `pid`, so the division stays signed.
tt.func @grouped_pid_swizzle_unbounded(%out: !tt.ptr<i32>) {
  %c4 = arith.constant 4 : i32
  %c16 = arith.constant 16 : i32
  %pid = tt.get_program_id x : i32
  %group_id = arith.divsi %pid, %c16 : i32
  %first_pid_m = arith.muli %group_id, %c4 : i32
  %rows_left = arith.subi %c4, %first_pid_m : i32
  %group_size_m = arith.minsi %rows_left, %c4 : i32
  %pid_in_group = arith.remsi %pid, %c16 : i32
  %pid_n = arith.divsi %pid_in_group, %group_size_m : i32
  tt.store %out, %pid_n : !tt.ptr<i32>
  tt.return
}

// CHECK-LABEL: tt.func @grouped_pid_swizzle_unbounded
// CHECK:         arith.divsi %{{.*}}, %{{.*}} : i32
