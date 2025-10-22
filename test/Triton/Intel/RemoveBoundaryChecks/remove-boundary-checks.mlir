// RUN: triton-opt %s -split-input-file -triton-intel-remove-boundary-checks | FileCheck %s

module {
tt.func public @simple_load(%load_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %store_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c512_i64 = arith.constant 512 : i64
  %c1024_i64 = arith.constant 1024 : i64
  %c0_i32 = arith.constant 0 : i32
  %x = arith.constant 10 : i32
  %in = tt.make_tensor_ptr %load_ptr, [%c1_i64, %c64_i64, %c1024_i64], [%c512_i64, %c64_i64, %c1_i64], [%c0_i32, %c0_i32, %x] {order = array<i32: 2, 1, 0>} : <tensor<1x64x64xf16>>
  %load = tt.load %in {boundaryCheck = array<i32: 2>} : !tt.ptr<tensor<1x64x64xf16>>
  tt.return
}
// CHECK-LABEL: simple_load
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr
// CHECK: tt.load [[PTR]] : !tt.ptr<tensor<1x64x64xf16>>
}

// -----

module {
tt.func public @load_in_for_loop(%load_ptr0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %load_ptr1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %store_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c20_i32 = arith.constant 20 : i32
  %c64_i32 = arith.constant 64 : i32
  %c1024_i32 = arith.constant 1024 : i32
  scf.for %x = %c0_i32 to %c20_i32 step %c1_i32 : i32 {
    %pid = tt.get_program_id x : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c64_i64 = arith.constant 64 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %ptr0 = tt.make_tensor_ptr %load_ptr0, [%c512_i64, %c1024_i64, %c64_i64], [%c65536_i64, %c64_i64, %c1_i64], [%x, %pid, %c0_i32] {order = array<i32: 2, 1, 0>} : <tensor<1x512x64xf16>>
    %load0 = tt.load %ptr0 {boundaryCheck = array<i32: 1, 2>, padding = 1 : i32} : !tt.ptr<tensor<1x512x64xf16>>
    %9 = arith.bitcast %c0_i32 : i32 to i32
    %10 = arith.bitcast %c1024_i32 : i32 to i32
    %11 = arith.bitcast %c64_i32 : i32 to i32
    scf.for %z = %9 to %10 step %11 iter_args() -> ()  : i32 {
      %ptr1 = tt.make_tensor_ptr %load_ptr1, [%c512_i64, %c64_i64, %c1024_i64], [%c65536_i64, %c1_i64, %c64_i64], [%x, %c0_i32, %z] {order = array<i32: 2, 0, 1>} : <tensor<1x64x64xf16>>
      //   a. boundaryCheck = 1 checks the block ptr offset at index 2 (%z)
      //   b. boundaryCheck = 2 checks the block ptr offset at index 1 (%y)
      // Check (a) is unnecessary because max(%z) = 920 which is less than %s2 (1024)
      // Check (a) is trivially unnecessary because %y(zero) < %s1(64)
      %load1 = tt.load %ptr1 {boundaryCheck = array<i32: 1, 2>} : !tt.ptr<tensor<1x64x64xf16>>
    }
  }
  tt.return
}
// CHECK-LABEL: load_in_for_loop
// CHECK-COUNT-2: scf.for
// CHECK: [[PTR:%.*]] = tt.make_tensor_ptr
// CHECK: tt.load [[PTR]] : !tt.ptr<tensor<1x64x64xf16>>
}
