// RUN: env TRITON_INTEL_ADVANCED_PATH=1 triton-opt %s --convert-triton-intel-gpu-to-llvm --split-input-file | FileCheck %s

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 1 : i32} {
  // CHECK-DAG: llvm.func spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi(i32, vector<8xi16>, vector<8xi32>, vector<8xf32>, i32) -> vector<8xf32> attributes {convergent, memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv(i32, i32, i32, i32, !llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) attributes {no_unwind, will_return}
  // CHECK-DAG: llvm.func spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {memory_effects = #llvm.memory_effects<other = none, argMem = read, inaccessibleMem = none>, no_unwind}

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

<<<<<<< HEAD
    // CHECK: llvm.call spir_funccc @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i({{.*}}, %arg0, {{.*}})
    ttig.prefetch %14 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xf16>, 1>
=======
    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%arg0, {{.*}})
    triton_intel_gpu.prefetch %14 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<8x32xf16>, 1>
>>>>>>> parent of e98561c932 (Change 2D block prefetch from OCL to SPV (#3767))
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
    // CHECK: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, %arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[A_PTR:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK: [[A:%.*]] = llvm.load [[A_PTR]] : !llvm.ptr -> vector<64xi16>
    // CHECK-NEXT: [[castA:%.*]] = llvm.bitcast [[A]] : vector<64xi16> to vector<64xf16>
    // CHECK: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, %arg1, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[B_PTR:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK: [[B0:%.*]] = llvm.load [[B_PTR]] : !llvm.ptr -> vector<32xi32>
    // CHECK-NEXT: [[castB:%.*]] = llvm.bitcast [[B0]] : vector<32xi32> to vector<64xf16>
    // CHECK: llvm.call spir_funccc @_Z41__spirv_Subgroup2DBlockLoadTransformINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, %arg1, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[B_PTR:%.*]]) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    // CHECK: [[B1:%.*]] = llvm.load [[B_PTR]] : !llvm.ptr -> vector<32xi32>
    // CHECK: [[subA1:%.*]] = llvm.shufflevector [[castA]], [[castA]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<64xf16>
    // CHECK: [[subB1:%.*]] = llvm.shufflevector [[castB]], [[castB]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xf16>
    // CHECK-NEXT: [[castDotA1:%.*]] = llvm.bitcast [[subA1]] : vector<8xf16> to vector<8xi16>
    // CHECK-NEXT: [[castDotB1:%.*]] = llvm.bitcast [[subB1]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi({{.*}}, [[castDotA1]], [[castDotB1]], {{.*}} -> vector<8xf32>
    // CHECK: [[subA2:%.*]] = llvm.shufflevector [[castA]], [[castA]] [32, 33, 34, 35, 36, 37, 38, 39] : vector<64xf16>
    // CHECK: [[subB2:%.*]] = llvm.shufflevector [[castB]], [[castB]] [16, 17, 18, 19, 20, 21, 22,  23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf16>
    // CHECK-NEXT: [[castDotA2:%.*]] = llvm.bitcast [[subA2]] : vector<8xf16> to vector<8xi16>
    // CHECK-NEXT: [[castDotB2:%.*]] = llvm.bitcast [[subB2]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call spir_funccc @_Z45__spirv_SubgroupMatrixMultiplyAccumulateINTELiDv8_sDv8_iDv8_fi({{.*}}, [[castDotA2]], [[castDotB2]], {{.*}} -> vector<8xf32>
    %63 = tt.load %57 {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16>, 1>
    %64 = tt.load %58 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16>, 1>
    %65 = tt.load %59 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<32x32xf16>, 1>
    %66 = ttig.extract %63[0] : tensor<32x32xf16> -> tensor<8x16xf16>
    %67 = ttig.extract %64[0] : tensor<32x32xf16> -> tensor<16x16xf16>
    %68 = tt.dot %66, %67, %41 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<8x16xf16> * tensor<16x16xf16> -> tensor<8x16xf32>
    %69 = ttig.extract %63[4] : tensor<32x32xf16> -> tensor<8x16xf16>
    %70 = ttig.extract %64[1] : tensor<32x32xf16> -> tensor<16x16xf16>
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
    // CHECK: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg2, {{.*}}
    tt.store %120, %41 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<8x16xf32>, 1>
    tt.return
  }
}

// -----

// COM: Checks the correct lowering of the A operand load for TF32, i.e. using 4xi32 and vnni=false.

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_kernel_with_block_pointers_tf32(
  // CHECK-SAME:                                                                  [[VAL_0:%.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 512, 1, 1>} {
  tt.func public @matmul_kernel_with_block_pointers_tf32(%arg0: !tt.ptr<f32>) {
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %1 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    // CHECK: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, %arg0, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    %2 = tt.load %0 {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    // CHECK: llvm.call spir_funccc @_Z32__spirv_Subgroup2DBlockLoadINTELiiiiPU3AS1viiiDv2_iPv({{.*}}, %arg0, {{.*}}) {{.*}} : (i32, i32, i32, i32, !llvm.ptr<1>{{.*}}, i32, i32, i32, vector<2xi32>, !llvm.ptr{{.*}}) -> ()
    %3 = tt.load %1 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    tt.return
  }
}

// -----

// COM: Checks the correct lowering of a 16-bit 2D-block-store.

module attributes {"ttig.support_sg_2d_block", "ttig.support_dpas", "ttg.num-warps" = 32 : i32, "ttg.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_kernel_with_block_pointers_f16accu(
  // CHECK-SAME:                                                                     [[VAL_0:%.*]]: !llvm.ptr<1>) attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 512, 1, 1>} {
  tt.func public @matmul_kernel_with_block_pointers_f16accu(%arg0: !tt.ptr<f16>) {
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<8x16xf16>
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf16>>
    // CHECK: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_16b_8r16x1cPU3AS1viiiDv2_iPt(%arg0, {{.*}})
    tt.store %0, %cst {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf16>>
    tt.return
  }
}

// -----

// COM: Checks the correct lowering of sub-group reductions.

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {

  // CHECK-LABEL: llvm.func spir_kernelcc @reduce_sum(
  // CHECK-SAME:                                      [[VAL_0:%.*]]: vector<8xf32>) -> f32 attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>}
  tt.func public @reduce_sum(%arg0: tensor<8x16xf32>) -> f32 {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_2:%.*]] = llvm.extractelement [[VAL_0]][[[VAL_1]] : i32] : vector<8xf32>
    // CHECK: [[VAL_3:%.*]] = llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFAddiif(%{{.*}}, %{{.*}}, [[VAL_2]]) {{.*}} : (i32, i32, f32) -> f32
    %0 = ttig.extract %arg0[0] : tensor<8x16xf32> -> tensor<16xf32>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.addf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16xf32>) -> f32
    tt.return %1: f32
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @reduce_max(
  // CHECK-SAME:                                        [[VAL_0:%.*]]: vector<8xf32>) -> f32 attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>}
  tt.func public @reduce_max(%arg0: tensor<8x16xf32>) -> f32 {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_2:%.*]] = llvm.extractelement [[VAL_0]][[[VAL_1]] : i32] : vector<8xf32>
    // CHECK: [[VAL_3:%.*]] = llvm.call spir_funccc @_Z27__spirv_GroupNonUniformFMaxiif(%{{.*}}, %{{.*}}, [[VAL_2]]) {{.*}} : (i32, i32, f32) -> f32
    %0 = ttig.extract %arg0[0] : tensor<8x16xf32> -> tensor<16xf32>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.maxnumf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16xf32>) -> f32
    tt.return %1: f32
  }
}

// -----

// COM: Checks the correct lowering of triton ops, including broadcast, splat, expand_dims, addptr.

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.min_sg_size = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {

  // CHECK: llvm.func spir_funccc @_Z12get_group_idj(i32) -> i64 attributes {memory_effects = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, no_unwind, will_return}
  // CHECK: llvm.func spir_funccc @_Z22get_sub_group_local_id() -> i32

  // CHECK-LABEL: llvm.func spir_kernelcc @broadcast(
  // CHECK-SAME:                                     [[VAL_0:%.*]]: f32) -> vector<16xf32>
  tt.func public @broadcast(%arg0: f32) -> tensor<16x16xf32> {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.poison : vector<1xf32>
    // CHECK: [[VAL_2:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_3:%.*]] = llvm.insertelement [[VAL_0]], [[VAL_1]][[[VAL_2]] : i32] : vector<1xf32>
    // CHECK: [[VAL_4:%.*]] = llvm.shufflevector [[VAL_3]], [[VAL_1]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<1xf32>
    %0 = tt.splat %arg0 : f32 -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xf32, #warp>
    %2 = ttig.broadcast %1 : tensor<16x1xf32, #warp> -> tensor<16x16xf32>
    tt.return %2 : tensor<16x16xf32>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @broadcast_range() -> vector<16xi32>
  tt.func public @broadcast_range() -> tensor<16x16xi32> {
    // CHECK: [[LAST_CONST:%.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK: [[RANGE:%.*]] = llvm.insertelement [[LAST_CONST]], {{%.*}}[[[LAST_CONST]] : i32] : vector<16xi32>
    // CHECK: [[LANE_ID_RAW:%.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id()
    // CHECK: [[LANE_ID_EXT:%.*]] = llvm.zext [[LANE_ID_RAW]] : i32 to i64
    // CHECK: [[LANE_ID:%.*]] = llvm.trunc [[LANE_ID_EXT]] : i64 to i32
    // CHECK: [[EXTRACT:%.*]] = llvm.extractelement [[RANGE]][[[LANE_ID]] : i32] : vector<16xi32>
    // CHECK: [[EMPTY:%.*]] = llvm.mlir.poison : vector<1xi32>
    // CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VEC:%.*]] = llvm.insertelement [[EXTRACT]], [[EMPTY]][[[ZERO]] : i32] : vector<1xi32>
    // CHECK: llvm.shufflevector [[VEC]], [[EMPTY]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<1xi32>
    %0 = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #warp}>> -> tensor<1x16xi32, #warp>
    %2 = ttig.broadcast %1 : tensor<1x16xi32, #warp> -> tensor<16x16xi32>
    tt.return %2 : tensor<16x16xi32>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @addptr(
  // CHECK-SAME:                                  [[VAL_0:%.*]]: !llvm.ptr<1>) -> !llvm.ptr<1> attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 128, 1, 1>}
  tt.func public @addptr(%arg0: !tt.ptr<f16>) -> !tt.ptr<f16> {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_2:%.*]] = llvm.call spir_funccc @_Z12get_group_idj([[VAL_1]]) {{.*}} : (i32) -> i64
    // CHECK: [[VAL_3:%.*]] = llvm.trunc [[VAL_2]] : i64 to i32
    // CHECK: [[VAL_4:%.*]] = llvm.sext [[VAL_3]] : i32 to i64
    // CHECK: [[VAL_5:%.*]] = llvm.getelementptr [[VAL_0]][[[VAL_4]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f16
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i64
    tt.return %2 : !tt.ptr<f16>
  }
}

// -----

// COM: Checks tt.load lowering for SLM

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [4, 2], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  // CHECK: llvm.func spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t(!llvm.ptr<3>) -> vector<8xi16>
  // CHECK-LABEL: @slm_load
  tt.func public @slm_load(%arg0: !tt.ptr<f16, 3>) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %ptr = tt.make_tensor_ptr %arg0, [%c0_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16, #dot0>, 3>
    // CHECK-COUNT-2: [[SUBGROUP_SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: [[READ1:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE1:%.*]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ2:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE2]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE1:%.*]] = llvm.getelementptr [[BASE2]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ3:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE1]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ4:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE2]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE1:%.*]] = llvm.getelementptr [[BASE2]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ5:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE1]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ6:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE2]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE1:%.*]] = llvm.getelementptr [[BASE2]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ7:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE1]]) {{.*}} -> vector<8xi16>
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[READ8:%.*]] = llvm.call spir_funccc @_Z30intel_sub_group_block_read_us8PU3AS3t([[BASE2]]) {{.*}} -> vector<8xi16>
    // CHECK: [[GLUE1:%.*]] = llvm.shufflevector [[READ1]], [[READ2]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi16>
    // CHECK: [[GLUE2:%.*]] = llvm.shufflevector [[READ3]], [[READ4]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi16>
    // CHECK: [[GLUE3:%.*]] = llvm.shufflevector [[READ5]], [[READ6]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi16>
    // CHECK: [[GLUE4:%.*]] = llvm.shufflevector [[READ7]], [[READ8]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<8xi16>
    // CHECK: [[GLUE5:%.*]] = llvm.shufflevector [[GLUE1]], [[GLUE2]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xi16>
    // CHECK: [[GLUE6:%.*]] = llvm.shufflevector [[GLUE3]], [[GLUE4]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<16xi16>
    // CHECK: [[READ:%.*]] = llvm.shufflevector [[GLUE5]], [[GLUE6]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] : vector<32xi16>
    // CHECK: llvm.bitcast [[READ]] : vector<64xi16> to vector<64xf16>
    %ld = tt.load %ptr {DotIdx = 0 : i32} : !tt.ptr<tensor<16x64xf16, #dot0>, 3>
    tt.return
  }
}

// -----

// COM: Checks tt.store lowering for SLM

#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [1, 1], repCluster = [1, 1], A = [8, 16], B = [16, 16], C = [8, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #dpas, kWidth=1}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  // CHECK: llvm.func spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t(!llvm.ptr<3>, vector<8xi16>)
  // CHECK-LABEL: @slm_store
  tt.func public @slm_store(%arg0: !tt.ptr<f16, 3>, %arg1: tensor<16x64xf16, #dot0>) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %ptr = tt.make_tensor_ptr %arg0, [%c0_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xf16, #dot0>, 3>
    // CHECK: [[CAST:%.*]] = llvm.bitcast {{.*}} : vector<64xf16> to vector<64xi16>
    // CHECK: [[SUBGROUP_SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE1:%.*]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE2]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE1:%.*]] = llvm.getelementptr [[BASE2]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [16, 17, 18, 19, 20, 21, 22, 23] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE1]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [24, 25, 26, 27, 28, 29, 30, 31] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE2]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE1:%.*]] = llvm.getelementptr [[BASE2]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [32, 33, 34, 35, 36, 37, 38, 39] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE1]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [40, 41, 42, 43, 44, 45, 46, 47] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE2]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE1:%.*]] = llvm.getelementptr [[BASE2]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [48, 49, 50, 51, 52, 53, 54, 55] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE1]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    // CHECK: [[BASE2:%.*]] = llvm.getelementptr [[BASE1]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi16>
    // CHECK: [[EXTRACT:%.*]] = llvm.shufflevector [[CAST]], [[CAST]] [56, 57, 58, 59, 60, 61, 62, 63] : vector<64xi16>
    // CHECK: llvm.call spir_funccc @_Z31intel_sub_group_block_write_us8PU3AS3tDv8_t([[BASE2]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi16>) -> ()
    tt.store %ptr, %arg1 {DotIdx = 0 : i32} : !tt.ptr<tensor<16x64xf16, #dot0>, 3>
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
// CHECK-LABEL:   llvm.func spir_kernelcc @test(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr<3>,
// CHECK-SAME:                                  %[[VAL_1:.*]]: vector<16xf32>) -> vector<16xf32>
// CHECK:           %[[VAL_2:.*]] = llvm.call spir_funccc @_Z16get_sub_group_id() {{{.*}}} : () -> i32
// CHECK:           %[[VAL_3:.*]] = llvm.zext %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = llvm.call spir_funccc @_Z22get_sub_group_local_id() {{{.*}}} : () -> i32
// CHECK:           %[[VAL_5:.*]] = llvm.zext %[[VAL_4]] : i32 to i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(16 : i64) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.mlir.constant(256 : i64) : i64
// CHECK:           %[[VAL_8:.*]] = llvm.mul %[[VAL_7]], %[[VAL_3]] : i64
// CHECK:           %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_10:.*]] = llvm.bitcast %[[VAL_1]] : vector<16xf32> to vector<16xi32>
// CHECK:           [[SUBGROUP_SIZE:%.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           [[EXTRACT:%.*]] = llvm.shufflevector %[[VAL_10]], %[[VAL_10]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<16xi32>
// CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j(%[[VAL_9]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi32>) -> ()
// CHECK:           [[BASE:%.*]] = llvm.getelementptr %[[VAL_9]][[[SUBGROUP_SIZE]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, vector<8xi32>
// CHECK:           [[EXTRACT:%.*]] = llvm.shufflevector %[[VAL_10]], %[[VAL_10]] [8, 9, 10, 11, 12, 13, 14, 15] : vector<16xi32>
// CHECK:           llvm.call spir_funccc @_Z31intel_sub_group_block_write_ui8PU3AS3jDv8_j([[BASE]], [[EXTRACT]]) {{.*}} : (!llvm.ptr<3>, vector<8xi32>) -> ()
// CHECK:           %[[VAL_11:.*]] = llvm.mul %[[VAL_6]], %[[VAL_5]] : i64
// CHECK:           %[[VAL_12:.*]] = llvm.getelementptr inbounds %[[VAL_9]]{{\[}}%[[VAL_11]]] : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, f32
// CHECK:           %[[VAL_13:.*]] = llvm.load %[[VAL_12]] : !llvm.ptr<3> -> vector<16xf32>
// CHECK:           llvm.return %[[VAL_13]] : vector<16xf32>
  tt.func @test(%arg0: !tt.ptr<f32, 3>, %arg1: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %0 = ttig.sub_group_transpose %arg0, %arg1 : tensor<16x16xf32>
    tt.return %0 : tensor<16x16xf32>
  }
}

// -----

#warp = #ttig.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>

// CHECK-LABEL:   llvm.func spir_kernelcc @test(
// CHECK-SAME:                                  %[[VAL_0:.*]]: f32) -> vector<16xf32> attributes {intel_reqd_sub_group_size = 16 : i32, reqd_work_group_size = array<i32: 64, 1, 1>} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.poison : vector<16xf32>
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_3]])
// CHECK:           %[[VAL_5:.*]] = llvm.insertelement %[[VAL_4]], %[[VAL_2]]{{\[}}%[[VAL_3]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_6]])
// CHECK:           %[[VAL_8:.*]] = llvm.insertelement %[[VAL_7]], %[[VAL_5]]{{\[}}%[[VAL_6]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_9]])
// CHECK:           %[[VAL_11:.*]] = llvm.insertelement %[[VAL_10]], %[[VAL_8]]{{\[}}%[[VAL_9]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_12]])
// CHECK:           %[[VAL_14:.*]] = llvm.insertelement %[[VAL_13]], %[[VAL_11]]{{\[}}%[[VAL_12]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_15]])
// CHECK:           %[[VAL_17:.*]] = llvm.insertelement %[[VAL_16]], %[[VAL_14]]{{\[}}%[[VAL_15]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_18:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_19:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_18]])
// CHECK:           %[[VAL_20:.*]] = llvm.insertelement %[[VAL_19]], %[[VAL_17]]{{\[}}%[[VAL_18]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_21]])
// CHECK:           %[[VAL_23:.*]] = llvm.insertelement %[[VAL_22]], %[[VAL_20]]{{\[}}%[[VAL_21]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_24]])
// CHECK:           %[[VAL_26:.*]] = llvm.insertelement %[[VAL_25]], %[[VAL_23]]{{\[}}%[[VAL_24]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_28:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_27]])
// CHECK:           %[[VAL_29:.*]] = llvm.insertelement %[[VAL_28]], %[[VAL_26]]{{\[}}%[[VAL_27]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:           %[[VAL_31:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_30]])
// CHECK:           %[[VAL_32:.*]] = llvm.insertelement %[[VAL_31]], %[[VAL_29]]{{\[}}%[[VAL_30]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_33:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           %[[VAL_34:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_33]])
// CHECK:           %[[VAL_35:.*]] = llvm.insertelement %[[VAL_34]], %[[VAL_32]]{{\[}}%[[VAL_33]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(11 : i32) : i32
// CHECK:           %[[VAL_37:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_36]])
// CHECK:           %[[VAL_38:.*]] = llvm.insertelement %[[VAL_37]], %[[VAL_35]]{{\[}}%[[VAL_36]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.constant(12 : i32) : i32
// CHECK:           %[[VAL_40:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_39]])
// CHECK:           %[[VAL_41:.*]] = llvm.insertelement %[[VAL_40]], %[[VAL_38]]{{\[}}%[[VAL_39]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_42:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_43:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_42]])
// CHECK:           %[[VAL_44:.*]] = llvm.insertelement %[[VAL_43]], %[[VAL_41]]{{\[}}%[[VAL_42]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_45:.*]] = llvm.mlir.constant(14 : i32) : i32
// CHECK:           %[[VAL_46:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_45]])
// CHECK:           %[[VAL_47:.*]] = llvm.insertelement %[[VAL_46]], %[[VAL_44]]{{\[}}%[[VAL_45]] : i32] : vector<16xf32>
// CHECK:           %[[VAL_48:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_49:.*]] = llvm.call spir_funccc @_Z17sub_group_shufflefj(%[[VAL_0]], %[[VAL_48]])
// CHECK:           %[[VAL_50:.*]] = llvm.insertelement %[[VAL_49]], %[[VAL_47]]{{\[}}%[[VAL_48]] : i32] : vector<16xf32>
// CHECK:           llvm.return %[[VAL_50]] : vector<16xf32>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 16 : i32, ttig.support_dpas, ttig.support_sg_2d_block} {
  tt.func @test(%arg0: tensor<16xf32>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>> {
    %0 = ttg.convert_layout %arg0 : tensor<16xf32> -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
    tt.return %0 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #warp}>>
  }
}
