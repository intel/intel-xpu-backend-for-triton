// RUN: TRITON_INTEL_ENABLE_BLOCK_PTR=1 triton-opt %s --convert-triton-intel-gpu-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 32 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  // CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, vector<8xi32>)
  // CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {passthrough = ["convergent"]}
  // CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v32i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<32xi32>
  // CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockRead.v64i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) -> vector<64xi16>
  // CHECK-DAG: llvm.func spir_funccc @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32)

  tt.func public @matmul_kernel_with_block_pointers(%arg0: !tt.ptr<f16, 1>, %arg1: !tt.ptr<f16, 1>, %arg2: !tt.ptr<f32, 1>, %arg3: i32, %arg4: i32, %arg5: i32) {
    // CHECK-LABEL: @matmul_kernel_with_block_pointers
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %c63_i32 = arith.constant 63 : i32
    %c48_i32 = arith.constant 48 : i32
    %c24_i32 = arith.constant 24 : i32
    %c64_i32 = arith.constant 64 : i32
    %c16_i32 = arith.constant 16 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
    %c256_i32 = arith.constant 256 : i32
    %c4096_i64 = arith.constant 4096 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf32>
    %0 = gpu.subgroup_id : index
    %1 = arith.index_cast %0 : index to i32
    %2 = tt.get_program_id x : i32
    %3 = arith.divsi %2, %c64_i32 : i32
    %4 = arith.muli %3, %c4_i32 : i32
    %5 = arith.subi %c16_i32, %4 : i32
    %6 = arith.minsi %5, %c4_i32 : i32
    %7 = arith.remsi %2, %6 : i32
    %8 = arith.addi %4, %7 : i32
    %9 = arith.andi %2, %c63_i32 : i32
    %10 = arith.divsi %9, %6 : i32
    %11 = arith.muli %8, %c256_i32 : i32
    %12 = arith.muli %1, %c8_i32 : i32
    %13 = arith.addi %12, %11 : i32
    // CHECK:      [[UNDEF:%.*]] = llvm.mlir.undef : vector<2xi32>
    // CHECK-NEXT: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT: [[INSERT0:%.*]] = llvm.insertelement {{.*}}, [[UNDEF]][[[ZERO]] : i32] : vector<2xi32>
    // CHECK-NEXT: [[ONE:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: [[INSERT1:%.*]] = llvm.insertelement {{.*}}, [[INSERT0]][[[ONE]] : i32] : vector<2xi32>
    %14 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%13, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x32xf16>, 1>

    // CHECK: [[PTR:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid([[PTR]], {{.*}})
    triton_intel_gpu.prefetch %14 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xf16>, 1>
    %18 = arith.divsi %1, %c4_i32 : i32
    %19 = arith.andi %18, %c7_i32 : i32
    %20 = arith.muli %19, %c32_i32 : i32
    %21 = arith.addi %20, %11 : i32
    %22 = tt.make_tensor_ptr %arg0, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%21, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x32xf16>, 1>
    %23 = arith.muli %10, %c256_i32 : i32
    %34 = arith.andi %1, %c3_i32 : i32
    %35 = arith.muli %34, %c64_i32 : i32
    %36 = arith.addi %35, %23 : i32
    %37 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %36] {order = array<i32: 1, 0>} : <tensor<32x32xf16>, 1>
    %38 = arith.addi %36, %c32_i32 : i32
    %39 = tt.make_tensor_ptr %arg1, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%c0_i32, %38] {order = array<i32: 1, 0>} : <tensor<32x32xf16>, 1>
    cf.br ^bb1(%c0_i32, %cst, %22, %37, %39 : i32, tensor<8x16xf32>, !tt.ptr<tensor<32x32xf16>, 1>, !tt.ptr<tensor<32x32xf16>, 1>, !tt.ptr<tensor<32x32xf16>, 1>)
  ^bb1(%40: i32, %41: tensor<8x16xf32>, %57: !tt.ptr<tensor<32x32xf16>, 1>, %58: !tt.ptr<tensor<32x32xf16>, 1>, %59: !tt.ptr<tensor<32x32xf16>, 1>):
    %62 = arith.cmpi slt, %40, %c4096_i32 : i32
    cf.cond_br %62, ^bb2, ^bb3
  ^bb2:
    // CHECK: [[A_PTR:%.*]] = llvm.ptrtoint %arg0 : !llvm.ptr<1> to i64
    // CHECK: [[A:%.*]] = llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v64i16([[A_PTR]], {{.*}} -> vector<64xi16>
    // CHECK-NEXT: [[castA:%.*]] = llvm.bitcast [[A]] : vector<64xi16> to vector<64xf16>
    // CHECK: [[B_PTR:%.*]] = llvm.ptrtoint %arg1 : !llvm.ptr<1> to i64
    // CHECK: [[B0:%.*]] = llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v32i32([[B_PTR]], {{.*}} -> vector<32xi32>
    // CHECK-NEXT: [[castB:%.*]] = llvm.bitcast [[B0]] : vector<32xi32> to vector<64xf16>
    // CHECK: [[B1:%.*]] = llvm.call @llvm.genx.GenISA.LSC2DBlockRead.v32i32({{.*}} -> vector<32xi32>
    // CHECK: [[subA1:%.*]] = llvm.shufflevector [[castA]], [[castA]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<64xf16>
    // CHECK: [[subB1:%.*]] = llvm.shufflevector [[castB]], [[castB]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xf16>
    // CHECK-NEXT: [[castDotA1:%.*]] = llvm.bitcast [[subA1]] : vector<8xf16> to vector<8xi16>
    // CHECK-NEXT: [[castDotB1:%.*]] = llvm.bitcast [[subB1]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f([[castDotA1]], [[castDotB1]], {{.*}} -> vector<8xf32>
    // CHECK: [[subA2:%.*]] = llvm.shufflevector [[castA]], [[castA]] [32, 33, 34, 35, 36, 37, 38, 39] : vector<64xf16>
    // CHECK: [[subB2:%.*]] = llvm.shufflevector [[castB]], [[castB]] [16, 17, 18, 19, 20, 21, 22,  23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf16>
    // CHECK-NEXT: [[castDotA2:%.*]] = llvm.bitcast [[subA2]] : vector<8xf16> to vector<8xi16>
    // CHECK-NEXT: [[castDotB2:%.*]] = llvm.bitcast [[subB2]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f([[castDotA2]], [[castDotB2]], {{.*}} -> vector<8xf32>
    %63 = tt.load %57 {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16>, 1>
    %64 = tt.load %58 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16>, 1>
    %65 = tt.load %59 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16>, 1>
    %66 = triton_intel_gpu.extract %63[0] : tensor<32x32xf16> -> tensor<8x16xf16>
    %67 = triton_intel_gpu.extract %64[0] : tensor<32x32xf16> -> tensor<16x16xf16>
    %68 = tt.dot %66, %67, %41 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
    %69 = triton_intel_gpu.extract %63[4] : tensor<32x32xf16> -> tensor<8x16xf16>
    %70 = triton_intel_gpu.extract %64[1] : tensor<32x32xf16> -> tensor<16x16xf16>
    %71 = tt.dot %69, %70, %68 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
    // CHECK: [[oldOffset:%.*]] = llvm.extractelement {{.*}} : vector<2xi32>
    // CHECK-NEXT: [[newOffset:%.*]] = llvm.add [[oldOffset]], {{.*}}  : i32
    // CHECK-NEXT: llvm.insertelement [[newOffset]], {{.*}} : vector<2xi32>
    %115 = tt.advance %57, [%c0_i32, %c32_i32] : <tensor<32x32xf16>, 1>
    %117 = tt.advance %58, [%c32_i32, %c0_i32] : <tensor<32x32xf16>, 1>
    %118 = tt.advance %59, [%c32_i32, %c0_i32] : <tensor<32x32xf16>, 1>
    %119 = arith.addi %40, %c32_i32 : i32
    cf.br ^bb1(%119, %71, %115, %117, %118 : i32, tensor<8x16xf32>, !tt.ptr<tensor<32x32xf16>, 1>, !tt.ptr<tensor<32x32xf16>, 1>, !tt.ptr<tensor<32x32xf16>, 1>)
  ^bb3:
    %120 = tt.make_tensor_ptr %arg2, [%c4096_i64, %c4096_i64], [%c4096_i64, %c1_i64], [%21, %36] {order = array<i32: 1, 0>} : <tensor<8x16xf32>, 1>
    // CHECK: [[RES_PTR:%.*]] = llvm.ptrtoint %arg2 : !llvm.ptr<1> to i64
    // CHECK: llvm.call @llvm.genx.GenISA.LSC2DBlockWrite.v8i32([[RES_PTR]], {{.*}}
    tt.store %120, %41 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<8x16xf32>, 1>
    tt.return
  }
}
