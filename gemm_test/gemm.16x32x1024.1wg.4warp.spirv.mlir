module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_16x1024xf16 : memref<16x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x32xf16_ : memref<1024x32xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_16x32xf16 : memref<16x32xf16> = dense<0.000000e+00>
  func.func @test(%arg0: memref<16x1024xf16>, %arg1: memref<1024x32xf16>) -> memref<16x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %memref = gpu.alloc  host_shared () : memref<16x1024xf16>
    memref.copy %arg0, %memref : memref<16x1024xf16> to memref<16x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x32xf16>
    memref.copy %arg1, %memref_0 : memref<1024x32xf16> to memref<1024x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)  args(%memref : memref<16x1024xf16>, %memref_0 : memref<1024x32xf16>, %memref_1 : memref<16x32xf16>)
    gpu.dealloc  %memref : memref<16x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x32xf16>
    return %memref_1 : memref<16x32xf16>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, VectorAnyINTEL, Int64, Addresses, Float16, Vector16, Kernel], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.GlobalVariable @__builtin__SubgroupId__ built_in("SubgroupId") : !spirv.ptr<i32, Input>
    spirv.func @llvm_genx_raw_send2_v64i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v8i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v64i32", linkage_type = <Import>>}
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<16384 x f16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<32768 x f16>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<512 x f16>, CrossWorkgroup>) "None" attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, workgroup_attributions = 0 : i64} {
      %cst33686023_i32 = spirv.Constant 33686023 : i32
      %cst16_i32 = spirv.Constant 16 : i32
      %cst8_i32 = spirv.Constant 8 : i32
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
      %cst1023_i32 = spirv.Constant 1023 : i32
      %cst63_i32 = spirv.Constant 63 : i32
      %cst1807_i32 = spirv.Constant 1807 : i32
      %cst15_i32 = spirv.Constant 15 : i32
      %cst2047_i32 = spirv.Constant 2047 : i32
      %cst7_i32 = spirv.Constant 7 : i32
      %cst6_i32 = spirv.Constant 6 : i32
      %cst5_i32 = spirv.Constant 5 : i32
      %cst4_i32 = spirv.Constant 4 : i32
      %cst3_i32 = spirv.Constant 3 : i32
      %cst2_i32 = spirv.Constant 2 : i32
      %cst0_i32 = spirv.Constant 0 : i32
      %cst_vec_128xf32 = spirv.Constant dense<0.000000e+00> : vector<128xf32>
      %cst2_i64 = spirv.Constant 2 : i64
      %cst0_i64 = spirv.Constant 0 : i64
      %cst16_i64 = spirv.Constant 16 : i64
      %cst8_i64 = spirv.Constant 8 : i64
      %cst1024_i64 = spirv.Constant 1024 : i64
      %__builtin__SubgroupId___addr = spirv.mlir.addressof @__builtin__SubgroupId__ : !spirv.ptr<i32, Input>
      %0 = spirv.Load "Input" %__builtin__SubgroupId___addr : i32
      %1 = spirv.UConvert %0 : i32 to i64
      %2 = spirv.SDiv %1, %cst2_i64 : i64
      %3 = spirv.IMul %2, %cst8_i64 : i64
      %4 = spirv.Undef : vector<4xi64>
      %5 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<!spirv.array<16384 x f16>, CrossWorkgroup> to i64
      %6 = spirv.VectorInsertDynamic %5, %4[%cst0_i32] : vector<4xi64>, i32
      %7 = spirv.Bitcast %6 : vector<4xi64> to vector<8xi32>
      %8 = spirv.VectorInsertDynamic %cst2047_i32, %7[%cst2_i32] : vector<8xi32>, i32
      %9 = spirv.VectorInsertDynamic %cst15_i32, %8[%cst3_i32] : vector<8xi32>, i32
      %10 = spirv.VectorInsertDynamic %cst2047_i32, %9[%cst4_i32] : vector<8xi32>, i32
      %11 = spirv.SConvert %cst0_i64 : i64 to i32
      %12 = spirv.SConvert %3 : i64 to i32
      %13 = spirv.VectorInsertDynamic %11, %10[%cst5_i32] : vector<8xi32>, i32
      %14 = spirv.VectorInsertDynamic %12, %13[%cst6_i32] : vector<8xi32>, i32
      %15 = spirv.VectorInsertDynamic %cst1807_i32, %14[%cst7_i32] : vector<8xi32>, i32
      %16 = spirv.CL.s_abs %1 : i64
      %17 = spirv.CL.s_abs %cst2_i64 : i64
      %18 = spirv.UMod %16, %17 : i64
      %19 = spirv.IEqual %1, %16 : i64
      %20 = spirv.SNegate %18 : i64
      %21 = spirv.Select %19, %18, %20 : i1, i64
      %22 = spirv.IMul %21, %cst16_i64 : i64
      %23 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<!spirv.array<32768 x f16>, CrossWorkgroup> to i64
      %24 = spirv.VectorInsertDynamic %23, %4[%cst0_i32] : vector<4xi64>, i32
      %25 = spirv.Bitcast %24 : vector<4xi64> to vector<8xi32>
      %26 = spirv.VectorInsertDynamic %cst63_i32, %25[%cst2_i32] : vector<8xi32>, i32
      %27 = spirv.VectorInsertDynamic %cst1023_i32, %26[%cst3_i32] : vector<8xi32>, i32
      %28 = spirv.VectorInsertDynamic %cst63_i32, %27[%cst4_i32] : vector<8xi32>, i32
      %29 = spirv.SConvert %22 : i64 to i32
      %30 = spirv.VectorInsertDynamic %29, %28[%cst5_i32] : vector<8xi32>, i32
      %31 = spirv.VectorInsertDynamic %11, %30[%cst6_i32] : vector<8xi32>, i32
      %32 = spirv.VectorInsertDynamic %cst3855_i32, %31[%cst7_i32] : vector<8xi32>, i32
      %33 = spirv.Variable : !spirv.ptr<vector<128xf32>, Function>
      %34 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
      %35 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%cst0_i64, %cst_vec_128xf32, %15, %32 : i64, vector<128xf32>, vector<8xi32>, vector<8xi32>)
      ^bb1(%49: i64, %50: vector<128xf32>, %51: vector<8xi32>, %52: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
        %53 = spirv.SLessThan %49, %cst1024_i64 : i64
        spirv.BranchConditional %53, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %54 = spirv.Undef : vector<64xi32>
        %55 = spirv.FunctionCall @llvm_genx_raw_send2_v64i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8, %cst0_i32, %cst37880323_i32, %51, %54) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32>
        %56 = spirv.Undef : vector<128xi32>
        %57 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %52, %56) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
        %58 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%50, %57, %55, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %59 = spirv.VectorExtractDynamic %51[%cst5_i32] : vector<8xi32>, i32
        %60 = spirv.IAdd %59, %cst16_i32 : i32
        %61 = spirv.VectorInsertDynamic %60, %51[%cst5_i32] : vector<8xi32>, i32
        %62 = spirv.VectorExtractDynamic %52[%cst6_i32] : vector<8xi32>, i32
        %63 = spirv.IAdd %62, %cst16_i32 : i32
        %64 = spirv.VectorInsertDynamic %63, %52[%cst6_i32] : vector<8xi32>, i32
        spirv.Store "Function" %33, %58 : vector<128xf32>
        spirv.Store "Function" %34, %61 : vector<8xi32>
        spirv.Store "Function" %35, %64 : vector<8xi32>
        %65 = spirv.IAdd %49, %cst16_i64 : i64
        spirv.Branch ^bb1(%65, %58, %61, %64 : i64, vector<128xf32>, vector<8xi32>, vector<8xi32>)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      %36 = spirv.Load "Function" %35 : vector<8xi32>
      %37 = spirv.Load "Function" %34 : vector<8xi32>
      %38 = spirv.Load "Function" %33 : vector<128xf32>
      %39 = spirv.FConvert %38 : vector<128xf32> to vector<128xf16>
      %40 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<!spirv.array<512 x f16>, CrossWorkgroup> to i64
      %41 = spirv.VectorInsertDynamic %40, %4[%cst0_i32] : vector<4xi64>, i32
      %42 = spirv.Bitcast %41 : vector<4xi64> to vector<8xi32>
      %43 = spirv.VectorInsertDynamic %cst63_i32, %42[%cst2_i32] : vector<8xi32>, i32
      %44 = spirv.VectorInsertDynamic %cst15_i32, %43[%cst3_i32] : vector<8xi32>, i32
      %45 = spirv.VectorInsertDynamic %cst63_i32, %44[%cst4_i32] : vector<8xi32>, i32
      %46 = spirv.VectorInsertDynamic %29, %45[%cst5_i32] : vector<8xi32>, i32
      %47 = spirv.VectorInsertDynamic %12, %46[%cst6_i32] : vector<8xi32>, i32
      %48 = spirv.VectorInsertDynamic %cst1807_i32, %47[%cst7_i32] : vector<8xi32>, i32
      spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst4_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %48, %39) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) -> ()
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel, @__builtin__SubgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<16x1024xf16>, %arg1: memref<1024x32xf16>, %arg2: memref<16x32xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %c0 = arith.constant 0 : index
      %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
      %c2 = arith.constant 2 : index
      %c16 = arith.constant 16 : index
      %c8 = arith.constant 8 : index
      %c1024 = arith.constant 1024 : index
      %0 = gpu.subgroup_id : index
      %1 = arith.divsi %0, %c2 : index
      %2 = arith.muli %1, %c8 : index
      %3 = xegpu.create_nd_tdesc %arg0[%2, %c0] {mode = vc} : memref<16x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %4 = arith.remsi %0, %c2 : index
      %5 = arith.muli %4, %c16 : index
      %6 = xegpu.create_nd_tdesc %arg1[%c0, %5] {mode = vc} : memref<1024x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      %7:3 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %cst, %arg5 = %3, %arg6 = %6) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>) {
        %10 = xegpu.load_nd %arg5 {mode = vc, vnni_axis = 1} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %11 = xegpu.load_nd %arg6 {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %12 = xegpu.dpas %10, %11, %arg4 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %13 = xegpu.update_nd_offset %arg5, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %14 = xegpu.update_nd_offset %arg6, [%c16, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        scf.yield %12, %13, %14 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>
      }
      %8 = arith.truncf %7#0 : vector<8x16xf32> to vector<8x16xf16>
      %9 = xegpu.create_nd_tdesc %arg2[%2, %5] {mode = vc} : memref<16x32xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %8, %9 {mode = vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 2.999880e-02 : f16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+03 : f16
    %c128_i16 = arith.constant 128 : i16
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_2 = arith.constant 0.000000e+00 : f16
    %0 = memref.get_global @__constant_16x1024xf16 : memref<16x1024xf16>
    %1 = memref.get_global @__constant_1024x32xf16_ : memref<1024x32xf16>
    %2 = memref.get_global @__constant_16x32xf16 : memref<16x32xf16>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %6 = arith.muli %4, %c128_i16 : i16
        %7 = arith.addi %5, %6 : i16
        %8 = arith.uitofp %7 : i16 to f16
        %9 = arith.divf %8, %cst_1 : f16
        memref.store %9, %0[%arg0, %arg1] : memref<16x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %6 = arith.addi %5, %4 : i16
        %7 = arith.uitofp %6 : i16 to f16
        %8 = arith.divf %7, %cst_1 : f16
        memref.store %8, %1[%arg0, %arg1] : memref<1024x32xf16>
      }
    }
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %4 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %cst_0) -> (f32) {
          %6 = memref.load %0[%arg0, %arg2] : memref<16x1024xf16>
          %7 = memref.load %1[%arg2, %arg1] : memref<1024x32xf16>
          %8 = arith.mulf %6, %7 : f16
          %9 = arith.extf %8 : f16 to f32
          %10 = arith.addf %9, %arg3 : f32
          scf.yield %10 : f32
        }
        %5 = arith.truncf %4 : f32 to f16
        memref.store %5, %2[%arg0, %arg1] : memref<16x32xf16>
      }
    }
    %3 = call @test(%0, %1) : (memref<16x1024xf16>, memref<1024x32xf16>) -> memref<16x32xf16>
    scf.for %arg0 = %c0 to %c16 step %c1 {
      scf.for %arg1 = %c0 to %c32 step %c1 {
        %4 = memref.load %3[%arg0, %arg1] : memref<16x32xf16>
        %5 = memref.load %2[%arg0, %arg1] : memref<16x32xf16>
        %6 = arith.subf %5, %4 : f16
        %7 = arith.divf %6, %5 : f16
        %8 = arith.cmpf olt, %7, %cst : f16
        %9 = arith.select %8, %cst_2, %7 : f16
        memref.store %9, %2[%arg0, %arg1] : memref<16x32xf16>
      }
    }
    %cast = memref.cast %2 : memref<16x32xf16> to memref<*xf16>
    call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
}

