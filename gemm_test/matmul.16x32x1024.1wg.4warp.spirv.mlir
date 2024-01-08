module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  spirv.GlobalVariable @__builtin__SubgroupId__ built_in("SubgroupId") : !spirv.ptr<i32, Input>
  spirv.func @llvm_genx_raw_send2_v64i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v8i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v64i32", linkage_type = <Import>>}
  spirv.func @test_kernel(%arg0: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg1: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg2: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}) "None" attributes {noinline = false, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst63_i32 = spirv.Constant 63 : i32
    %cst15_i32 = spirv.Constant 15 : i32
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
    %cst7_i32 = spirv.Constant 7 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst5_i32 = spirv.Constant 5 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst3_i32 = spirv.Constant 3 : i32
    %cst_vec_128xf32 = spirv.Constant dense<0.000000e+00> : vector<128xf32>
    %cst2_i32 = spirv.Constant 2 : i32
    %cst1024_i32 = spirv.Constant 1024 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %cst0_i32 = spirv.Constant 0 : i32
    %__builtin__SubgroupId___addr = spirv.mlir.addressof @__builtin__SubgroupId__ : !spirv.ptr<i32, Input>
    %0 = spirv.Load "Input" %__builtin__SubgroupId___addr : i32
    %1 = spirv.UConvert %0 : i32 to i64
    %2 = spirv.SConvert %1 : i64 to i32
    %3 = spirv.SDiv %2, %cst2_i32 : i32
    %4 = spirv.IMul %3, %cst8_i32 : i32
    %5 = spirv.Undef : vector<4xi64>
    %6 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %7 = spirv.VectorInsertDynamic %6, %5[%cst0_i32] : vector<4xi64>, i32
    %8 = spirv.Bitcast %7 : vector<4xi64> to vector<8xi32>
    %9 = spirv.VectorInsertDynamic %cst2047_i32, %8[%cst2_i32] : vector<8xi32>, i32
    %10 = spirv.VectorInsertDynamic %cst15_i32, %9[%cst3_i32] : vector<8xi32>, i32
    %11 = spirv.VectorInsertDynamic %cst2047_i32, %10[%cst4_i32] : vector<8xi32>, i32
    %12 = spirv.VectorInsertDynamic %cst0_i32, %11[%cst5_i32] : vector<8xi32>, i32
    %13 = spirv.VectorInsertDynamic %4, %12[%cst6_i32] : vector<8xi32>, i32
    %14 = spirv.VectorInsertDynamic %cst1807_i32, %13[%cst7_i32] : vector<8xi32>, i32
    %15 = spirv.CL.s_abs %2 : i32
    %16 = spirv.CL.s_abs %cst2_i32 : i32
    %17 = spirv.UMod %15, %16 : i32
    %18 = spirv.IEqual %2, %15 : i32
    %19 = spirv.SNegate %17 : i32
    %20 = spirv.Select %18, %17, %19 : i1, i32
    %21 = spirv.IMul %20, %cst16_i32 : i32
    %22 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %23 = spirv.VectorInsertDynamic %22, %5[%cst0_i32] : vector<4xi64>, i32
    %24 = spirv.Bitcast %23 : vector<4xi64> to vector<8xi32>
    %25 = spirv.VectorInsertDynamic %cst63_i32, %24[%cst2_i32] : vector<8xi32>, i32
    %26 = spirv.VectorInsertDynamic %cst1023_i32, %25[%cst3_i32] : vector<8xi32>, i32
    %27 = spirv.VectorInsertDynamic %cst63_i32, %26[%cst4_i32] : vector<8xi32>, i32
    %28 = spirv.VectorInsertDynamic %21, %27[%cst5_i32] : vector<8xi32>, i32
    %29 = spirv.VectorInsertDynamic %cst0_i32, %28[%cst6_i32] : vector<8xi32>, i32
    %30 = spirv.VectorInsertDynamic %cst3855_i32, %29[%cst7_i32] : vector<8xi32>, i32
    %31 = spirv.Variable : !spirv.ptr<vector<128xf32>, Function>
    %32 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    %33 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%cst0_i32, %cst_vec_128xf32, %14, %30 : i32, vector<128xf32>, vector<8xi32>, vector<8xi32>)
    ^bb1(%47: i32, %48: vector<128xf32>, %49: vector<8xi32>, %50: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
      %51 = spirv.SLessThan %47, %cst1024_i32 : i32
      spirv.BranchConditional %51, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %52 = spirv.Undef : vector<64xi32>
      %53 = spirv.FunctionCall @llvm_genx_raw_send2_v64i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8, %cst0_i32, %cst37880323_i32, %49, %52) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32>
      %54 = spirv.Undef : vector<128xi32>
      %55 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %50, %54) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %56 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%48, %55, %53, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %57 = spirv.VectorExtractDynamic %49[%cst5_i32] : vector<8xi32>, i32
      %58 = spirv.IAdd %57, %cst16_i32 : i32
      %59 = spirv.VectorInsertDynamic %58, %49[%cst5_i32] : vector<8xi32>, i32
      %60 = spirv.VectorExtractDynamic %50[%cst6_i32] : vector<8xi32>, i32
      %61 = spirv.IAdd %60, %cst16_i32 : i32
      %62 = spirv.VectorInsertDynamic %61, %50[%cst6_i32] : vector<8xi32>, i32
      spirv.Store "Function" %31, %56 : vector<128xf32>
      spirv.Store "Function" %32, %59 : vector<8xi32>
      spirv.Store "Function" %33, %62 : vector<8xi32>
      %63 = spirv.IAdd %47, %cst16_i32 : i32
      spirv.Branch ^bb1(%63, %56, %59, %62 : i32, vector<128xf32>, vector<8xi32>, vector<8xi32>)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    %34 = spirv.Load "Function" %33 : vector<8xi32>
    %35 = spirv.Load "Function" %32 : vector<8xi32>
    %36 = spirv.Load "Function" %31 : vector<128xf32>
    %37 = spirv.FConvert %36 : vector<128xf32> to vector<128xf16>
    %38 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %39 = spirv.VectorInsertDynamic %38, %5[%cst0_i32] : vector<4xi64>, i32
    %40 = spirv.Bitcast %39 : vector<4xi64> to vector<8xi32>
    %41 = spirv.VectorInsertDynamic %cst63_i32, %40[%cst2_i32] : vector<8xi32>, i32
    %42 = spirv.VectorInsertDynamic %cst15_i32, %41[%cst3_i32] : vector<8xi32>, i32
    %43 = spirv.VectorInsertDynamic %cst63_i32, %42[%cst4_i32] : vector<8xi32>, i32
    %44 = spirv.VectorInsertDynamic %21, %43[%cst5_i32] : vector<8xi32>, i32
    %45 = spirv.VectorInsertDynamic %4, %44[%cst6_i32] : vector<8xi32>, i32
    %46 = spirv.VectorInsertDynamic %cst1807_i32, %45[%cst7_i32] : vector<8xi32>, i32
    spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %46, %37) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
    spirv.Return
  }
}

