// RUN: triton-opt %s -split-input-file --tritonintelgpu-lower-to-2d-block-load --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --canonicalize | FileCheck %s

// COM: Bytes of 32-bit loads (column-major B) are read from the loaded dwords;
// COM: 8-bit loads return bytes, which take the generic lowering.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @fp8_upcast_from_block_load
  tt.func public @fp8_upcast_from_block_load(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32, %arg2: i32) -> (tensor<64x16xf16, #dot1>, tensor<32x16xf16, #dot1>, tensor<32x16xf16, #dot1>) {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %k64_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <16x64xf8E4M3FN>
    %k32_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <16x32xf8E4M3FN>
    %row_desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <32x16xf8E4M3FN>
    // CHECK-DAG: [[MASK:%.*]] = llvm.mlir.constant(-16512 : i16) : i16
    // CHECK-DAG: [[C1:%.*]] = llvm.mlir.constant(1 : i16) : i16
    // CHECK-DAG: [[C8:%.*]] = llvm.mlir.constant(8 : i16) : i16
    // CHECK-DAG: [[C16:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-DAG: [[CA:%.*]] = llvm.mlir.constant(3.686400e+04 : f16) : f16
    // CHECK-DAG: [[CB:%.*]] = llvm.mlir.constant(6.942750e-03 : f16) : f16
    // CHECK-DAG: [[C0:%.*]] = llvm.mlir.constant(0.000000e+00 : f16) : f16
    // CHECK-COUNT-3: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true, {{.*}} -> vector<8xi32>
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, {{.*}} transpose = false, {{.*}} -> vector<32xi8>
    %k64 = tt.descriptor_load %k64_desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<16x64xf8E4M3FN> -> tensor<64x16xf8E4M3FN, #dot1>
    %k32 = tt.descriptor_load %k32_desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<16x32xf8E4M3FN> -> tensor<32x16xf8E4M3FN, #dot1>
    %row = tt.descriptor_load %row_desc[%arg1, %arg2] {ttig.block_io = "row_major"} : !tt.tensordesc<32x16xf8E4M3FN> -> tensor<32x16xf8E4M3FN, #dot1>
    // CHECK-NOT: vector<4xi8> to vector<2xi16>
    // COM: Bytes 0-3 of a dword: even bytes are shifted into the high byte,
    // COM: bytes 2-3 come from the upper half.
    // CHECK:      [[D0:%.*]] = llvm.extractelement [[V:%.*]][{{.*}}] : vector<8xi32>
    // CHECK-NEXT: [[T0:%.*]] = llvm.trunc [[D0]] : i32 to i16
    // CHECK-NEXT: [[W0:%.*]] = llvm.shl [[T0]], [[C8]] : i16
    // CHECK-NEXT: [[S0:%.*]] = llvm.ashr [[W0]], [[C1]] : i16
    // CHECK-NEXT: [[A0:%.*]] = llvm.and [[S0]], [[MASK]] : i16
    // CHECK-NEXT: [[H0:%.*]] = llvm.bitcast [[A0]] : i16 to f16
    // CHECK-NEXT: [[M0:%.*]] = llvm.fmul [[H0]], [[CA]] : f16
    // CHECK-NEXT: [[M1:%.*]] = llvm.fmul [[M0]], [[CB]] : f16
    // CHECK-NEXT: [[Z0:%.*]] = llvm.fmul [[M1]], [[C0]] : f16
    // CHECK-NEXT: llvm.fadd [[M1]], [[Z0]] : f16
    // CHECK-NEXT: [[D1:%.*]] = llvm.extractelement [[V]]
    // CHECK-NEXT: [[T1:%.*]] = llvm.trunc [[D1]] : i32 to i16
    // CHECK-NEXT: llvm.ashr [[T1]], [[C1]] : i16
    // CHECK:      [[D2:%.*]] = llvm.extractelement [[V]]
    // CHECK-NEXT: [[U2:%.*]] = llvm.lshr [[D2]], [[C16]] : i32
    // CHECK-NEXT: [[T2:%.*]] = llvm.trunc [[U2]] : i32 to i16
    // CHECK-NEXT: llvm.shl [[T2]], [[C8]] : i16
    // CHECK:      [[D3:%.*]] = llvm.extractelement [[V]]
    // CHECK-NEXT: [[U3:%.*]] = llvm.lshr [[D3]], [[C16]] : i32
    // CHECK-NEXT: [[T3:%.*]] = llvm.trunc [[U3]] : i32 to i16
    // CHECK-NEXT: llvm.ashr [[T3]], [[C1]] : i16
    // CHECK-COUNT-92: llvm.trunc {{.*}} : i32 to i16
    // CHECK-NOT: llvm.trunc {{.*}} : i32 to i16
    // CHECK-COUNT-16: llvm.bitcast {{.*}} : vector<4xi8> to vector<2xi16>
    // CHECK-NOT: vector<4xi8> to vector<2xi16>
    %k64_f16 = tt.fp_to_fp %k64 : tensor<64x16xf8E4M3FN, #dot1> -> tensor<64x16xf16, #dot1>
    %k32_f16 = tt.fp_to_fp %k32 : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    %row_f16 = tt.fp_to_fp %row : tensor<32x16xf8E4M3FN, #dot1> -> tensor<32x16xf16, #dot1>
    tt.return %k64_f16, %k32_f16, %row_f16 : tensor<64x16xf16, #dot1>, tensor<32x16xf16, #dot1>, tensor<32x16xf16, #dot1>
  }
}

// -----

// COM: Any layout whose bytes come in dwords, here column-major A.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @fp8_upcast_dot_a_from_block_load
  tt.func public @fp8_upcast_dot_a_from_block_load(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32, %arg2: i32) -> tensor<8x64xf16, #dot0> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <64x8xf8E4M3FN>
    // CHECK: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true
    %x = tt.descriptor_load %desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<64x8xf8E4M3FN> -> tensor<8x64xf8E4M3FN, #dot0>
    // CHECK-COUNT-32: llvm.trunc {{.*}} : i32 to i16
    // CHECK-NOT: vector<4xi8> to vector<2xi16>
    %y = tt.fp_to_fp %x : tensor<8x64xf8E4M3FN, #dot0> -> tensor<8x64xf16, #dot0>
    tt.return %y : tensor<8x64xf16, #dot0>
  }
}

// -----

// COM: 8-bit loads return bytes even with K = 64.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @fp8_upcast_from_8bit_block_load
  tt.func public @fp8_upcast_from_8bit_block_load(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32, %arg2: i32) -> tensor<64x16xf16, #dot1> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <64x16xf8E4M3FN>
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 8, {{.*}} -> vector<32xi8>
    %x = tt.descriptor_load %desc[%arg1, %arg2] {ttig.block_io = "row_major"} : !tt.tensordesc<64x16xf8E4M3FN> -> tensor<64x16xf8E4M3FN, #dot1>
    // CHECK-NOT: llvm.trunc {{.*}} : i32 to i16
    // CHECK-COUNT-32: llvm.bitcast {{.*}} : vector<4xi8> to vector<2xi16>
    // CHECK-NOT: llvm.trunc {{.*}} : i32 to i16
    %y = tt.fp_to_fp %x : tensor<64x16xf8E4M3FN, #dot1> -> tensor<64x16xf16, #dot1>
    tt.return %y : tensor<64x16xf16, #dot1>
  }
}

// -----

// COM: LTS drivers keep the integer-domain converter.
#mma = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.is_lts, ttig.min_sg_size = 16 : i32, ttig.support_2d_block_io, ttig.target_arch = "spir64"} {
  // CHECK-LABEL: @fp8_upcast_from_block_load_lts
  tt.func public @fp8_upcast_from_block_load_lts(%arg0: !tt.ptr<f8E4M3FN>, %arg1: i32, %arg2: i32) -> tensor<64x16xf16, #dot1> {
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c64_i32 = arith.constant 64 : i32
    %desc = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <f8E4M3FN>, <16x64xf8E4M3FN>
    // CHECK-DAG: llvm.mlir.constant(8323199 : i32) : i32
    // CHECK-COUNT-2: triton_gen.2Dblockload {{.*}} {elem_size_in_bits = 32, {{.*}} transpose = true, {{.*}} -> vector<8xi32>
    %x = tt.descriptor_load %desc[%arg1, %arg2] {ttig.block_io = "column_major"} : !tt.tensordesc<16x64xf8E4M3FN> -> tensor<64x16xf8E4M3FN, #dot1>
    // CHECK-NOT: llvm.trunc {{.*}} : i32 to i16
    // CHECK-NOT: llvm.fadd
    %y = tt.fp_to_fp %x : tensor<64x16xf8E4M3FN, #dot1> -> tensor<64x16xf16, #dot1>
    tt.return %y : tensor<64x16xf16, #dot1>
  }
}
