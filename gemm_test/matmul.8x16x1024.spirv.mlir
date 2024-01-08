
module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  spirv.func @llvm_genx_raw_send2_v64i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v8i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v64i32", linkage_type = <Import>>}
  spirv.func @test_kernel(%arg0: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg1: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg2: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}) "None" attributes {noinline = false, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst31_i32 = spirv.Constant 31 : i32
    %cst7_i32 = spirv.Constant 7 : i32
    %cst1023_i32 = spirv.Constant 1023 : i32
    %cst2047_i32 = spirv.Constant 2047 : i32
    %cst33686023_i32 = spirv.Constant 33686023 : i32
    %cst10_i32 = spirv.Constant 10 : i32
    %cst42074755_i32 = spirv.Constant 42074755 : i32
    %cst8_i8 = spirv.Constant 8 : i8
    %cst37880323_i32 = spirv.Constant 37880323 : i32
    %cst15_i8 = spirv.Constant 15 : i8
    %cst4_i8 = spirv.Constant 4 : i8
    %cst1_i8 = spirv.Constant 1 : i8
    %true = spirv.Constant true
    %cst0_i8 = spirv.Constant 0 : i8
    %cst3855_i32 = spirv.Constant 3855 : i32
    %cst1807_i32 = spirv.Constant 1807 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst5_i32 = spirv.Constant 5 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst3_i32 = spirv.Constant 3 : i32
    %cst2_i32 = spirv.Constant 2 : i32
    %cst_vec_128xf32 = spirv.Constant dense<0.000000e+00> : vector<128xf32>
    %cst1024_i32 = spirv.Constant 1024 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %cst0_i32 = spirv.Constant 0 : i32
    %0 = spirv.Undef : vector<4xi64>
    %1 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %2 = spirv.VectorInsertDynamic %1, %0[%cst0_i32] : vector<4xi64>, i32
    %3 = spirv.Bitcast %2 : vector<4xi64> to vector<8xi32>
    %4 = spirv.VectorInsertDynamic %cst2047_i32, %3[%cst2_i32] : vector<8xi32>, i32
    %5 = spirv.VectorInsertDynamic %cst7_i32, %4[%cst3_i32] : vector<8xi32>, i32
    %6 = spirv.VectorInsertDynamic %cst2047_i32, %5[%cst4_i32] : vector<8xi32>, i32
    %7 = spirv.VectorInsertDynamic %cst0_i32, %6[%cst5_i32] : vector<8xi32>, i32
    %8 = spirv.VectorInsertDynamic %cst0_i32, %7[%cst6_i32] : vector<8xi32>, i32
    %9 = spirv.VectorInsertDynamic %cst1807_i32, %8[%cst7_i32] : vector<8xi32>, i32
    %10 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %11 = spirv.VectorInsertDynamic %10, %0[%cst0_i32] : vector<4xi64>, i32
    %12 = spirv.Bitcast %11 : vector<4xi64> to vector<8xi32>
    %13 = spirv.VectorInsertDynamic %cst31_i32, %12[%cst2_i32] : vector<8xi32>, i32
    %14 = spirv.VectorInsertDynamic %cst1023_i32, %13[%cst3_i32] : vector<8xi32>, i32
    %15 = spirv.VectorInsertDynamic %cst31_i32, %14[%cst4_i32] : vector<8xi32>, i32
    %16 = spirv.VectorInsertDynamic %cst0_i32, %15[%cst5_i32] : vector<8xi32>, i32
    %17 = spirv.VectorInsertDynamic %cst0_i32, %16[%cst6_i32] : vector<8xi32>, i32
    %18 = spirv.VectorInsertDynamic %cst3855_i32, %17[%cst7_i32] : vector<8xi32>, i32
    %19 = spirv.Variable : !spirv.ptr<vector<128xf32>, Function>
    %20 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    %21 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%cst0_i32, %cst_vec_128xf32, %9, %18 : i32, vector<128xf32>, vector<8xi32>, vector<8xi32>)
    ^bb1(%35: i32, %36: vector<128xf32>, %37: vector<8xi32>, %38: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
      %39 = spirv.SLessThan %35, %cst1024_i32 : i32
      spirv.BranchConditional %39, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %40 = spirv.Undef : vector<64xi32>
      %41 = spirv.FunctionCall @llvm_genx_raw_send2_v64i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8, %cst0_i32, %cst37880323_i32, %37, %40) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32>
      %42 = spirv.Undef : vector<128xi32>
      %43 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %38, %42) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %44 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%36, %43, %41, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %45 = spirv.VectorExtractDynamic %37[%cst5_i32] : vector<8xi32>, i32
      %46 = spirv.IAdd %45, %cst16_i32 : i32
      %47 = spirv.VectorInsertDynamic %46, %37[%cst5_i32] : vector<8xi32>, i32
      %48 = spirv.VectorExtractDynamic %38[%cst6_i32] : vector<8xi32>, i32
      %49 = spirv.IAdd %48, %cst16_i32 : i32
      %50 = spirv.VectorInsertDynamic %49, %38[%cst6_i32] : vector<8xi32>, i32
      spirv.Store "Function" %19, %44 : vector<128xf32>
      spirv.Store "Function" %20, %47 : vector<8xi32>
      spirv.Store "Function" %21, %50 : vector<8xi32>
      %51 = spirv.IAdd %35, %cst16_i32 : i32
      spirv.Branch ^bb1(%51, %44, %47, %50 : i32, vector<128xf32>, vector<8xi32>, vector<8xi32>)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    %22 = spirv.Load "Function" %21 : vector<8xi32>
    %23 = spirv.Load "Function" %20 : vector<8xi32>
    %24 = spirv.Load "Function" %19 : vector<128xf32>
    %25 = spirv.FConvert %24 : vector<128xf32> to vector<128xf16>
    %26 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %27 = spirv.VectorInsertDynamic %26, %0[%cst0_i32] : vector<4xi64>, i32
    %28 = spirv.Bitcast %27 : vector<4xi64> to vector<8xi32>
    %29 = spirv.VectorInsertDynamic %cst31_i32, %28[%cst2_i32] : vector<8xi32>, i32
    %30 = spirv.VectorInsertDynamic %cst7_i32, %29[%cst3_i32] : vector<8xi32>, i32
    %31 = spirv.VectorInsertDynamic %cst31_i32, %30[%cst4_i32] : vector<8xi32>, i32
    %32 = spirv.VectorInsertDynamic %cst0_i32, %31[%cst5_i32] : vector<8xi32>, i32
    %33 = spirv.VectorInsertDynamic %cst0_i32, %32[%cst6_i32] : vector<8xi32>, i32
    %34 = spirv.VectorInsertDynamic %cst1807_i32, %33[%cst7_i32] : vector<8xi32>, i32
    spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %34, %25) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
    spirv.Return
  }
}

