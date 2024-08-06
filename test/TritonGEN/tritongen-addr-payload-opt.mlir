// RUN: TRITON_INTEL_ENABLE_ADDRESS_PAYLOAD_OPT=1 triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s --implicit-check-not=llvm.inline_asm

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 16], warpsPerCTA = [16, 2], order = [1, 0]}>
#mma = #triton_intel_gpu.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [8, 4], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot1 = #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// COM: Test that, instead of 2D block reads, the compiler generates address payload create/set/load builtins.
// CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {passthrough = ["convergent"]}
// CHECK-DAG: llvm.func spir_funccc @__builtin_IB_subgroup_block_read_ap_transform_u16_m16k16v1(!llvm.ptr {llvm.nonnull}, i32, i32, i32) -> vector<8xi32> attributes {passthrough = ["nounwind", ["memory", "1"]]}
// CHECK-DAG: llvm.func spir_funccc @__builtin_IB_subgroup_block_read_ap_u16_m8k16v1(!llvm.ptr {llvm.nonnull}, i32, i32, i32) -> vector<8xi16> attributes {passthrough = ["nounwind", ["memory", "1"]]}
// CHECK-DAG: llvm.func spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(!llvm.ptr {llvm.nonnull}, i32) attributes {passthrough = ["nounwind", ["memory", "2"]]}
// CHECK-DAG: llvm.func spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(!llvm.ptr {llvm.nonnull}, i32) attributes {passthrough = ["nounwind", ["memory", "2"]]}
// CHECK-DAG: llvm.func spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload(i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr attributes {passthrough = ["nounwind", ["memory", "1"]]}

module attributes {"triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 33792 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  tt.func public @matmul_kernel_with_addr_payload_opt(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) {
    // CHECK-LABEL: @matmul_kernel_with_addr_payload_opt
    // CHECK: [[CMP:%.*]] = llvm.icmp "slt" {{.*}}, %arg4 : i64
    // CHECK: llvm.cond_br [[CMP]], ^bb2, ^bb3
    // CHECK: ^bb2:
    // CHECK:     [[PTRTOINT_1:%.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr<1> to i64
    // CHECK:     [[ADDR_PAYLOAD_1:%.*]] = llvm.call spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload([[PTRTOINT_1]], {{.*}}) {{.*}} : (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr
    // CHECK-DAG: llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX([[ADDR_PAYLOAD_1]], {{.*}}) {{.*}} : (!llvm.ptr, i32) -> ()
    // CHECK:     [[ZERO_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_u16_m8k16v1([[ADDR_PAYLOAD_1]], [[ZERO_1]], [[ZERO_1]], [[ZERO_1]]) {{.*}} : (!llvm.ptr, i32, i32, i32) -> vector<8xi16>
    //
    // CHECK:     [[PTRTOINT_2:%.*]] = llvm.ptrtoint {{.*}} : !llvm.ptr<1> to i64
    // CHECK:     [[ADDR_PAYLOAD_2:%.*]] = llvm.call spir_funccc @__builtin_IB_subgroup_createBlock2DAddressPayload([[PTRTOINT_2]], {{.*}}) {{.*}} : (i64, i32, i32, i32, i32, i32, i32, i32, i32) -> !llvm.ptr
    // CHECK-DAG: llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockX([[ADDR_PAYLOAD_2]], {{.*}}) {{.*}} : (!llvm.ptr, i32) -> ()
    // CHECK-DAG: llvm.call spir_funccc @__builtin_IB_subgroup_setBlock2DAddressPayloadBlockY([[ADDR_PAYLOAD_2]], {{.*}}) {{.*}} : (!llvm.ptr, i32) -> ()
    // CHECK:     [[ZERO_2:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK:     llvm.call spir_funccc @__builtin_IB_subgroup_block_read_ap_transform_u16_m16k16v1([[ADDR_PAYLOAD_2]], [[ZERO_2]], [[ZERO_2]], [[ZERO_2]]) {{.*}} : (!llvm.ptr, i32, i32, i32) -> vector<8xi32>
    // CHECK: ^bb3:
    // CHECK: llvm.return

    %cst = arith.constant dense<0.000000e+00> : tensor<8x8xf32, #mma>
    %c32_i32 = arith.constant 32 : i32
    %c32_i64 = arith.constant 32 : i64
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %18 = tt.make_tensor_ptr %arg0, [%arg3, %arg5], [%arg6, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf16, #dot0>>
    %22 = tt.make_tensor_ptr %arg1, [%arg5, %arg4], [%arg7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x8xf16, #dot1>>
    cf.br ^bb1(%c0_i64, %cst, %18, %22 : i64, tensor<8x8xf32, #mma>, !tt.ptr<tensor<8x16xf16, #dot0>>, !tt.ptr<tensor<16x8xf16, #dot1>>)
  ^bb1(%23: i64, %24: tensor<8x8xf32, #mma>, %25: !tt.ptr<tensor<8x16xf16, #dot0>>, %26: !tt.ptr<tensor<16x8xf16, #dot1>>):  // 2 preds: ^bb0, ^bb2
    %27 = arith.cmpi slt, %23, %arg5 : i64
    cf.cond_br %27, ^bb2, ^bb3
  ^bb2:
    %28 = tt.load %25 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf16, #dot0>>
    %29 = tt.load %26 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<16x8xf16, #dot1>>
    %30 = tt.dot %28, %29, %24, inputPrecision = tf32 : tensor<8x16xf16, #dot0> * tensor<16x8xf16, #dot1> -> tensor<8x8xf32, #mma>
    %31 = tt.advance %25, [%c0_i32, %c32_i32] : <tensor<8x16xf16, #dot0>>
    %32 = tt.advance %26, [%c32_i32, %c0_i32] : <tensor<16x8xf16, #dot1>>
    %33 = arith.addi %23, %c32_i64 : i64
    cf.br ^bb1(%33, %30, %31, %32 : i64, tensor<8x8xf32, #mma>, !tt.ptr<tensor<8x16xf16, #dot0>>, !tt.ptr<tensor<16x8xf16, #dot1>>)
  ^bb3:
   tt.return
  }
}
