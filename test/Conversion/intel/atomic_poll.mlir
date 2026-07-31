// RUN: split-file %s %t
// RUN: triton-opt %t/rate.mlir -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %t/rate.mlir
// RUN: not --crash triton-opt %t/no-rate.mlir --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm 2>&1 | FileCheck %t/no-rate.mlir --check-prefix=NO-RATE

//--- rate.mlir

// COM: The clock is read twice and the two reads must not be merged. Cycles are
//      converted to ns with ttig.core_clock_rate_khz, 1600000 kHz gives 5/8.

// CHECK: llvm.func spir_funccc @_Z27__spirv_ReadClockKHR_Rulongi(i32) -> i64
// CHECK-LABEL: llvm.func spir_kernelcc @poll_timeout
// CHECK: llvm.call spir_funccc @_Z27__spirv_ReadClockKHR_Rulongi
// CHECK-DAG: %[[DIV:.*]] = llvm.mlir.constant(8 : i64) : i64
// CHECK-DAG: %[[MUL:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK: %[[T1:.*]] = llvm.mul %{{.*}}, %[[MUL]] : i64
// CHECK: llvm.udiv %[[T1]], %[[DIV]] : i64
// CHECK: llvm.call spir_funccc @_Z27__spirv_ReadClockKHR_Rulongi
// CHECK: %[[T2:.*]] = llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: llvm.udiv %[[T2]], %{{.*}} : i64
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, "ttig.core_clock_rate_khz" = 1600000 : i32} {
  tt.func public @poll_timeout(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i1>, %timeout: i64) {
    %expected = arith.constant 1 : i32
    %matched = tt.atomic_poll relaxed, gpu, %arg0, %expected timeout %timeout : !tt.ptr<i32>, i32 -> i1
    tt.store %arg1, %matched : !tt.ptr<i1>
    tt.return
  }
}

// -----

// COM: Without a timeout there is no clock read at all.

// CHECK-LABEL: llvm.func spir_kernelcc @poll_no_timeout
// CHECK-NOT: __spirv_ReadClockKHR
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32, "ttig.core_clock_rate_khz" = 1600000 : i32} {
  tt.func public @poll_no_timeout(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i1>) {
    %expected = arith.constant 1 : i32
    %matched = tt.atomic_poll relaxed, gpu, %arg0, %expected : !tt.ptr<i32>, i32 -> i1
    tt.store %arg1, %matched : !tt.ptr<i1>
    tt.return
  }
}

//--- no-rate.mlir

// COM: A timeout without ttig.core_clock_rate_khz cannot be expressed in ns.

// NO-RATE: getGlobalTimer needs a positive ttig.core_clock_rate_khz
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "xpu", "ttg.threads-per-warp" = 16 : i32} {
  tt.func public @poll_timeout_no_core_clock_rate(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i1>, %timeout: i64) {
    %expected = arith.constant 1 : i32
    %matched = tt.atomic_poll relaxed, gpu, %arg0, %expected timeout %timeout : !tt.ptr<i32>, i32 -> i1
    tt.store %arg1, %matched : !tt.ptr<i1>
    tt.return
  }
}
