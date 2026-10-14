// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3 use-barrier" | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonintelgpu-pipeline="num-stages=3" | FileCheck %s --check-prefix=DISABLED --implicit-check-not=triton_gen.split_barrier
// DISABLED: module

// COM: Descriptor equivalent of the removed block-pointer split-barrier test.
// COM: Checks that descriptor prefetches are emitted around a pipelined dot loop
// COM: and that the barrier-enabled pipeline inserts split barrier arrive/wait.

// CHECK: #[[$DPAS:.+]] = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: tt.func public @matmul_kernel_descriptor_split_barrier
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<64x256xf16>
  // CHECK: scf.for {{.*}} -> (tensor<128x256xf32, #[[$DPAS]]>, i32, i32)
  // CHECK: %[[BDATA:.*]] = triton_gen.split_barrier_arrive
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<128x64xf16>
  // CHECK: ttig.descriptor_prefetch {{.*}} : !tt.tensordesc<64x256xf16>
  // CHECK: tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DPAS]], kWidth = 1}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DPAS]], kWidth = 2}>> -> tensor<128x256xf32, #[[$DPAS]]>
  // CHECK: triton_gen.split_barrier_wait %[[BDATA]]
  tt.func public @matmul_kernel_descriptor_split_barrier(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @uniform_workgroup
  // CHECK: triton_gen.split_barrier_arrive {workgroup_candidate}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @uniform_workgroup(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @uniform_loaded_bound
  // CHECK: triton_gen.split_barrier_arrive {workgroup_candidate}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @uniform_loaded_bound(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %bound = tt.load %bound_ptr : !tt.ptr<i32>
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @nested_control_flow
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @nested_control_flow(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %condition: i1) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    scf.if %condition {
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    }
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @opaque_kernel_call
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func private @opaque()
  tt.func public @opaque_kernel_call(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    tt.call @opaque() : () -> ()
    %result:2 = scf.for %k = %c0_i32 to %arg5 step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}

// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @volatile_loaded_bound
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @volatile_loaded_bound(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %bound = tt.load %bound_ptr {isVolatile = true} : !tt.ptr<i32>
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}


// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @load_after_write
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @load_after_write(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    tt.store %bound_ptr, %arg5 : !tt.ptr<i32>
    %bound = tt.load %bound_ptr : !tt.ptr<i32>
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}


// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @masked_undefined_bound
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @masked_undefined_bound(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %mask: i1, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %bound = tt.load %bound_ptr, %mask : !tt.ptr<i32>
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}


// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @uniform_metadata_while_early_return
  // CHECK: triton_gen.split_barrier_arrive {workgroup_candidate}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @uniform_metadata_while_early_return(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %c1_i32 = arith.constant 1 : i32
    %bound = scf.while (%i = %c0_i32) : (i32) -> i32 {
      %offset = tt.addptr %bound_ptr, %i : !tt.ptr<i32>, i32
      %limit = tt.load %offset : !tt.ptr<i32>
      %cond = arith.cmpi slt, %i, %limit : i32
      scf.condition(%cond) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %c1_i32 : i32
      scf.yield %next : i32
    }
    %empty = arith.cmpi eq, %bound, %c0_i32 : i32
    cf.cond_br %empty, ^exit, ^compute
  ^exit:
    tt.return
  ^compute:
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}


// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @nonuniform_metadata_while
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @nonuniform_metadata_while(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %c1_i32 = arith.constant 1 : i32
    %bound = scf.while (%i = %c0_i32) : (i32) -> i32 {
      %offset = tt.addptr %bound_ptr, %i : !tt.ptr<i32>, i32
      %limit = tt.load %offset : !tt.ptr<i32>
      %cond = arith.cmpi slt, %i, %limit : i32
      scf.condition(%cond) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %c1_i32 : i32
      %bad = tt.load %bound_ptr {isVolatile = true} : !tt.ptr<i32>
      scf.yield %bad : i32
    }
    %empty = arith.cmpi eq, %bound, %c0_i32 : i32
    cf.cond_br %empty, ^exit, ^compute
  ^exit:
    tt.return
  ^compute:
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}


// -----

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 8], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #dpas, kWidth=2}>

module attributes {"ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io", ttig.support_split_work_group_barrier} {
  // CHECK-LABEL: tt.func public @unknown_early_return
  // CHECK: triton_gen.split_barrier_arrive{{$}}
  // CHECK: triton_gen.split_barrier_wait
  tt.func public @unknown_early_return(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i64, %arg7: i64, %bound_ptr: !tt.ptr<i32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #dpas>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %descA = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%arg6, %c1_i64] : <f16>, <128x64xf16>
    %descB = tt.make_tensor_descriptor %arg1, [%arg5, %arg4], [%arg7, %c1_i64] : <f16>, <64x256xf16>
    %c1_i32 = arith.constant 1 : i32
    %bound = scf.while (%i = %c0_i32) : (i32) -> i32 {
      %offset = tt.addptr %bound_ptr, %i : !tt.ptr<i32>, i32
      %limit = tt.load %offset : !tt.ptr<i32>
      %cond = arith.cmpi slt, %i, %limit : i32
      scf.condition(%cond) %i : i32
    } do {
    ^bb0(%i: i32):
      %next = arith.addi %i, %c1_i32 : i32
      scf.yield %next : i32
    }
    %unknown = tt.load %bound_ptr {isVolatile = true} : !tt.ptr<i32>
    %empty = arith.cmpi eq, %unknown, %c0_i32 : i32
    cf.cond_br %empty, ^exit, ^compute
  ^exit:
    tt.return
  ^compute:
    %result:2 = scf.for %k = %c0_i32 to %bound step %c64_i32 iter_args(%acc = %cst, %koff = %c0_i32) -> (tensor<128x256xf32, #dpas>, i32) : i32 {
      %a = tt.descriptor_load %descA[%c0_i32, %koff] {ttig.block_io = "row_major"} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #dot0>
      %b = tt.descriptor_load %descB[%koff, %c0_i32] {ttig.block_io = "row_major"} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #dot1>
      %d = tt.dot %a, %b, %acc, inputPrecision = tf32 : tensor<128x64xf16, #dot0> * tensor<64x256xf16, #dot1> -> tensor<128x256xf32, #dpas>
      %next_koff = arith.addi %koff, %c64_i32 : i32
      scf.yield %d, %next_koff : tensor<128x256xf32, #dpas>, i32
    }
    tt.return
  }
}
