// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --intel-allocate-shared-memory --convert-triton-intel-gpu-to-llvm --convert-tritongen-to-llvm | FileCheck %s --check-prefix=GENISA

// Two N-adjacent 8x16 DPAS tiles of an 8-row band fold into one 8x32 store
// (64 B rows). The payload interleaves their rows, registers 0, 8, 1, 9, ...,
// and lowers to GenISA, since the SPIR-V builtin takes another order.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @f16_descriptor_store
  // GENISA-LABEL: llvm.func spir_kernelcc @f16_descriptor_store
  // GENISA-COUNT-4: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v16i16(
  // GENISA-NOT: Subgroup2DBlockStoreINTEL
  tt.func public @f16_descriptor_store(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: tensor<32x32xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK: %[[R0:[0-9]+]] = llvm.extractvalue %arg4[0] :
    // CHECK: %[[R1:[0-9]+]] = llvm.extractvalue %arg4[1] :
    // CHECK: %[[R2:[0-9]+]] = llvm.extractvalue %arg4[2] :
    // CHECK: %[[R3:[0-9]+]] = llvm.extractvalue %arg4[3] :
    // CHECK: %[[R4:[0-9]+]] = llvm.extractvalue %arg4[4] :
    // CHECK: %[[R5:[0-9]+]] = llvm.extractvalue %arg4[5] :
    // CHECK: %[[R6:[0-9]+]] = llvm.extractvalue %arg4[6] :
    // CHECK: %[[R7:[0-9]+]] = llvm.extractvalue %arg4[7] :
    // CHECK: %[[R8:[0-9]+]] = llvm.extractvalue %arg4[8] :
    // CHECK: %[[R9:[0-9]+]] = llvm.extractvalue %arg4[9] :
    // CHECK: %[[R10:[0-9]+]] = llvm.extractvalue %arg4[10] :
    // CHECK: %[[R11:[0-9]+]] = llvm.extractvalue %arg4[11] :
    // CHECK: %[[R12:[0-9]+]] = llvm.extractvalue %arg4[12] :
    // CHECK: %[[R13:[0-9]+]] = llvm.extractvalue %arg4[13] :
    // CHECK: %[[R14:[0-9]+]] = llvm.extractvalue %arg4[14] :
    // CHECK: %[[R15:[0-9]+]] = llvm.extractvalue %arg4[15] :
    // CHECK: llvm.mlir.undef : vector<16xf16>
    // CHECK: llvm.insertelement %[[R0]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R8]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R1]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R9]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R2]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R10]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R3]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R11]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R4]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R12]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R5]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R13]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R6]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R14]], {{.*}} : vector<16xf16>
    // CHECK: llvm.insertelement %[[R7]], {{.*}} : vector<16xf16>
    // CHECK: %[[PAYLOAD:.*]] = llvm.insertelement %[[R15]], {{.*}} : vector<16xf16>
    // CHECK: %[[CAST:.*]] = llvm.bitcast %[[PAYLOAD]] : vector<16xf16> to vector<16xi16>
    // CHECK: triton_gen.2Dblockstore {{.*}}, %[[CAST]] {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default}
    // CHECK-COUNT-3: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<16xi16>)
    // CHECK-NOT: triton_gen.2Dblockstore
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <32x32xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %arg4 {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf16, #dpas>, tensor<32x32xf16, #dpas>
    tt.return
  }

  // The pointer store takes the same 8x32 tile unless a mask can change along
  // it: a mask column constancy of 16 keeps two 8x16 stores, 32 allows 8x32.
  // CHECK-LABEL: llvm.func spir_kernelcc @f16_pointer_store
  // GENISA-LABEL: llvm.func spir_kernelcc @f16_pointer_store
  // GENISA-COUNT-4: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v16i16(
  // GENISA-COUNT-8: llvm.call spir_funccc @_Z33__spirv_Subgroup2DBlockStoreINTEL
  // GENISA-COUNT-4: llvm.call spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v16i16(
  tt.func public @f16_pointer_store(%arg0: !tt.ptr<f16>, %n16: i32 {tt.divisibility = 16 : i32}, %n32: i32 {tt.divisibility = 32 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #dpas>
    %stride = arith.constant dense<64> : tensor<32x1xi32, #dpas>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dpas}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dpas}>> -> tensor<32x1xi32, #dpas>
    %2 = arith.muli %1, %stride : tensor<32x1xi32, #dpas>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dpas}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dpas}>> -> tensor<1x32xi32, #dpas>
    %5 = tt.broadcast %2 : tensor<32x1xi32, #dpas> -> tensor<32x32xi32, #dpas>
    %6 = tt.broadcast %4 : tensor<1x32xi32, #dpas> -> tensor<32x32xi32, #dpas>
    %7 = arith.addi %5, %6 : tensor<32x32xi32, #dpas>
    %8 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #dpas>
    %addr = tt.addptr %8, %7 : tensor<32x32x!tt.ptr<f16>, #dpas>, tensor<32x32xi32, #dpas>
    %9 = tt.splat %n16 : i32 -> tensor<1x32xi32, #dpas>
    %10 = arith.cmpi slt, %4, %9 : tensor<1x32xi32, #dpas>
    %mask16 = tt.broadcast %10 : tensor<1x32xi1, #dpas> -> tensor<32x32xi1, #dpas>
    %11 = tt.splat %n32 : i32 -> tensor<1x32xi32, #dpas>
    %12 = arith.cmpi slt, %4, %11 : tensor<1x32xi32, #dpas>
    %mask32 = tt.broadcast %12 : tensor<1x32xi1, #dpas> -> tensor<32x32xi1, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<16xi16>)
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.store %addr, %cst {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<f16>, #dpas>
    // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<8xi16>)
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.store %addr, %cst, %mask16 {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<f16>, #dpas>
    // CHECK-COUNT-4: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 32, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<16xi16>)
    // CHECK-NOT: triton_gen.2Dblockstore
    tt.store %addr, %cst, %mask32 {ttig.block_io = "row_major"} : tensor<32x32x!tt.ptr<f16>, #dpas>
    tt.return
  }
}

// -----

// A 4-byte element already has 64 B rows at 8x16: the same layout keeps eight
// 8x16 stores with the payload in register order.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [4, 2]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 16 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @f32_descriptor_store
  tt.func public @f32_descriptor_store(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: tensor<32x32xf32, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK: %[[R0:[0-9]+]] = llvm.extractvalue %arg4[0] :
    // CHECK: %[[R1:[0-9]+]] = llvm.extractvalue %arg4[1] :
    // CHECK: llvm.mlir.undef : vector<8xf32>
    // CHECK: llvm.insertelement %[[R0]], {{.*}} : vector<8xf32>
    // CHECK-NEXT: llvm.mlir.constant(1 : i32)
    // CHECK-NEXT: llvm.insertelement %[[R1]], {{.*}} : vector<8xf32>
    // CHECK-COUNT-8: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 32, tile_width = 16, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<8xi32>)
    // CHECK-NOT: triton_gen.2Dblockstore
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f32>, <32x32xf32, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %arg4 {ttig.block_io = "row_major"} : !tt.tensordesc<32x32xf32, #dpas>, tensor<32x32xf32, #dpas>
    tt.return
  }
}

// -----

// Only sub-group 16 folds: at sub-group 8 the v-block pair keeps two 8x8 stores.

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 8, opsPerChan = 2, threadsPerWarp = 8, warpsPerCTA = [1, 1], repCluster = [1, 2]}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 8 : i32, "ttig.support_2d_block_io"} {
  // CHECK-LABEL: llvm.func spir_kernelcc @f16_descriptor_store_sg8
  tt.func public @f16_descriptor_store_sg8(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: tensor<8x16xf16, #dpas>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    // CHECK-COUNT-2: triton_gen.2Dblockstore {{.*}} {elem_size_in_bits = 16, tile_width = 8, tile_height = 8, v_blocks = 1, cache_control = Default} : ({{.*}}, vector<8xi16>)
    // CHECK-NOT: triton_gen.2Dblockstore
    %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%arg3, %c1_i64] : <f16>, <8x16xf16, #dpas>
    tt.descriptor_store %desc[%c0_i32, %c0_i32], %arg4 {ttig.block_io = "row_major"} : !tt.tensordesc<8x16xf16, #dpas>, tensor<8x16xf16, #dpas>
    tt.return
  }
}
