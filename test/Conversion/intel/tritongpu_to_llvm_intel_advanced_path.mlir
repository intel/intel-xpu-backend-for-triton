// RUN: TRITON_INTEL_ADVANCED_PATH=1 triton-opt %s --convert-triton-intel-gpu-to-llvm --split-input-file | FileCheck %s

module attributes {"triton_intel_gpu.support_sg_2d_block", "triton_intel_gpu.support_dpas", "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  // CHECK-DAG: llvm.func spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f(vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32> attributes {passthrough = ["convergent"]}
  // CHECK-DAG: llvm.func spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {passthrough = ["nounwind"]}
  // CHECK-DAG: llvm.func spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.readonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.writeonly}) attributes {passthrough = ["nounwind"]}
  // CHECK-DAG: llvm.func spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1> {llvm.nonnull, llvm.writeonly}, i32, i32, i32, vector<2xi32>, !llvm.ptr {llvm.nonnull, llvm.readonly}) attributes {passthrough = ["nounwind"]}
  // CHECK-DAG: llvm.func spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(!llvm.ptr<1> {llvm.nonnull}, i32, i32, i32, vector<2xi32>) attributes {passthrough = ["nounwind", ["memory", "1"]]}

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

    // CHECK: llvm.call spir_funccc @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x2cPU3AS1viiiDv2_i(%arg0, {{.*}})
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
    // CHECK: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_read_16b_32r16x2cPU3AS1viiiDv2_iPt(%arg0, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[A_PTR:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK: [[A:%.*]] = llvm.load [[A_PTR]] : !llvm.ptr -> vector<64xi16>
    // CHECK-NEXT: [[castA:%.*]] = llvm.bitcast [[A]] : vector<64xi16> to vector<64xf16>
    // CHECK: llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(%arg1, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[B_PTR:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK: [[B0:%.*]] = llvm.load [[B_PTR]] : !llvm.ptr -> vector<32xi32>
    // CHECK-NEXT: [[castB:%.*]] = llvm.bitcast [[B0]] : vector<32xi32> to vector<64xf16>
    // CHECK: llvm.call spir_funccc @_Z52intel_sub_group_2d_block_read_transform_16b_32r16x2cPU3AS1viiiDv2_iPj(%arg1, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[B_PTR:%.*]]) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    // CHECK: [[B1:%.*]] = llvm.load [[B_PTR]] : !llvm.ptr -> vector<32xi32>
    // CHECK: [[subA1:%.*]] = llvm.shufflevector [[castA]], [[castA]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<64xf16>
    // CHECK: [[subB1:%.*]] = llvm.shufflevector [[castB]], [[castB]] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] : vector<64xf16>
    // CHECK-NEXT: [[castDotA1:%.*]] = llvm.bitcast [[subA1]] : vector<8xf16> to vector<8xi16>
    // CHECK-NEXT: [[castDotB1:%.*]] = llvm.bitcast [[subB1]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f([[castDotA1]], [[castDotB1]], {{.*}} -> vector<8xf32>
    // CHECK: [[subA2:%.*]] = llvm.shufflevector [[castA]], [[castA]] [32, 33, 34, 35, 36, 37, 38, 39] : vector<64xf16>
    // CHECK: [[subB2:%.*]] = llvm.shufflevector [[castB]], [[castB]] [16, 17, 18, 19, 20, 21, 22,  23, 24, 25, 26, 27, 28, 29, 30, 31] : vector<64xf16>
    // CHECK-NEXT: [[castDotA2:%.*]] = llvm.bitcast [[subA2]] : vector<8xf16> to vector<8xi16>
    // CHECK-NEXT: [[castDotB2:%.*]] = llvm.bitcast [[subB2]] : vector<16xf16> to vector<8xi32>
    // CHECK: llvm.call spir_funccc @_Z38intel_sub_group_f16_f16_matrix_mad_k16Dv8_sDv8_iDv8_f([[castDotA2]], [[castDotB2]], {{.*}} -> vector<8xf32>
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
    // CHECK: llvm.call spir_funccc @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg2, {{.*}}
    tt.store %120, %41 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<8x16xf32>, 1>
    tt.return
  }
}

// -----

// COM: Checks the correct lowering of the A operand load for TF32, i.e. using 4xi32 and vnni=false.

module attributes {"triton_intel_gpu.support_sg_2d_block", "triton_intel_gpu.support_dpas", "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_kernel_with_block_pointers_tf32(
  // CHECK-SAME:                                                                  [[VAL_0:%.*]]: !llvm.ptr<1>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
  tt.func public @matmul_kernel_with_block_pointers_tf32(%arg0: !tt.ptr<f32>) {
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x8xf32>>
    %1 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<8x16xf32>>
    // CHECK: llvm.call spir_funccc @_Z40intel_sub_group_2d_block_read_32b_8r8x1cPU3AS1viiiDv2_iPj(%arg0, {{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    %2 = tt.load %0 {DotIdx = 0 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x8xf32>>
    // CHECK: llvm.call spir_funccc @_Z41intel_sub_group_2d_block_read_32b_8r16x1cPU3AS1viiiDv2_iPj(%arg0, {{.*}}) {{.*}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    %3 = tt.load %1 {DotIdx = 1 : i32, boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<8x16xf32>>
    tt.return
  }
}

// -----

// COM: Checks the correct lowering of a 16-bit 2D-block-store.

module attributes {"triton_intel_gpu.support_sg_2d_block", "triton_intel_gpu.support_dpas", "triton_gpu.num-warps" = 32 : i32, "triton_gpu.threads-per-warp" = 16 : i32} {
  // CHECK-LABEL: llvm.func spir_kernelcc @matmul_kernel_with_block_pointers_f16accu(
  // CHECK-SAME:                                                                     [[VAL_0:%.*]]: !llvm.ptr<1>) attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [512 : i32, 1 : i32, 1 : i32]} {
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

#warp = #triton_intel_gpu.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

  // CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_maxf(f32) -> f32
  // CHECK-DAG: llvm.func spir_funccc @_Z32sub_group_non_uniform_reduce_addf(f32) -> f32

  // CHECK-LABEL: llvm.func spir_kernelcc @reduce_sum(
  // CHECK-SAME:                                      [[VAL_0:%.*]]: vector<8xf32>) -> f32 attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]}
  tt.func public @reduce_sum(%arg0: tensor<8x16xf32>) -> f32 {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_2:%.*]] = llvm.extractelement [[VAL_0]][[[VAL_1]] : i32] : vector<8xf32>
    // CHECK: [[VAL_3:%.*]] = llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_addf([[VAL_2]]) {{.*}} : (f32) -> f32
    %0 = triton_intel_gpu.extract %arg0[0] : tensor<8x16xf32> -> tensor<16xf32>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.addf %arg1, %arg2 fastmath<fast> : f32
      tt.reduce.return %2 : f32
    }) : (tensor<16xf32>) -> f32
    tt.return %1: f32
  }

  // CHECK-LABEL:   llvm.func spir_kernelcc @reduce_max(
  // CHECK-SAME:                                        [[VAL_0:%.*]]: vector<8xf32>) -> f32 attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]}
  tt.func public @reduce_max(%arg0: tensor<8x16xf32>) -> f32 {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_2:%.*]] = llvm.extractelement [[VAL_0]][[[VAL_1]] : i32] : vector<8xf32>
    // CHECK: [[VAL_3:%.*]] = llvm.call spir_funccc @_Z32sub_group_non_uniform_reduce_maxf([[VAL_2]]) {{.*}} : (f32) -> f32
    %0 = triton_intel_gpu.extract %arg0[0] : tensor<8x16xf32> -> tensor<16xf32>
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

#warp = #triton_intel_gpu.warp<{sizePerThread = [16, 64], threadsPerWarp = [1, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {

  // CHECK: llvm.func spir_funccc @_Z12get_group_idj(i32) -> i64 attributes {passthrough = ["nounwind", "willreturn", ["memory", "0"]]}

  // CHECK-LABEL: llvm.func spir_kernelcc @broadcast(
  // CHECK-SAME:                                     [[VAL_0:%.*]]: f32) -> vector<16xf32>
  tt.func public @broadcast(%arg0: f32) -> tensor<16x16xf32> {
    // CHECK: [[VAL_1:%.*]] = llvm.mlir.poison : vector<1xf32>
    // CHECK: [[VAL_2:%.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: [[VAL_3:%.*]] = llvm.insertelement [[VAL_0]], [[VAL_1]][[[VAL_2]] : i32] : vector<1xf32>
    // CHECK: [[VAL_4:%.*]] = llvm.shufflevector [[VAL_3]], [[VAL_1]] [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] : vector<1xf32>
    %0 = tt.splat %arg0 : f32 -> tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #warp}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xf32, #triton_gpu.slice<{dim = 1, parent = #warp}>> -> tensor<16x1xf32, #warp>
    %2 = tt.broadcast %1 : tensor<16x1xf32, #warp> -> tensor<16x16xf32>
    tt.return %2 : tensor<16x16xf32>
  }

  // CHECK-LABEL: llvm.func spir_kernelcc @addptr(
  // CHECK-SAME:                                  [[VAL_0:%.*]]: !llvm.ptr<1>) -> !llvm.ptr<1> attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]}
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

// COM: Checks the correct custom lowering of arithmetic operations.

module attributes {"triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 16 : i32, triton_intel_gpu.min_sg_size = 16 : i32, triton_intel_gpu.support_dpas, triton_intel_gpu.support_sg_2d_block} {
  // CHECK-LABEL: llvm.func spir_kernelcc @custom_arith_lowering(
  // CHECK-SAME:                                                 [[VAL_0:%.*]]: vector<8xf32>) -> vector<8xf32> attributes {triton_gen.intel_reqd_sub_group_size = [16 : i32], triton_gen.max_work_group_size = [128 : i32, 1 : i32, 1 : i32]} {
  tt.func public @custom_arith_lowering(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {

    // CHECK: [[VAL_1:%.*]] = llvm.mlir.constant(dense<2.000000e+00> : vector<8xf32>) : vector<8xf32>
    // CHECK: [[VAL_2:%.*]] = llvm.mlir.constant(dense<1.000000e+00> : vector<8xf32>) : vector<8xf32>
    // CHECK: [[VAL_3:%.*]] = llvm.fdiv [[VAL_2]], [[VAL_0]]  : vector<8xf32>
    // CHECK: [[VAL_4:%.*]] = llvm.fmul [[VAL_1]], [[VAL_3]]  : vector<8xf32>
    %cst = arith.constant dense<2.000000e+00> : tensor<8x16xf32>
    %0 = arith.divf %cst, %arg0 fastmath<fast> : tensor<8x16xf32>
    tt.return %0 : tensor<8x16xf32>
  }
}
