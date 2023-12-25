module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_8x1024xf16 : memref<8x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x16xf16_ : memref<1024x16xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_8x16xf16 : memref<8x16xf16> = dense<0.000000e+00>
  func.func @test(%arg0: memref<8x1024xf16>, %arg1: memref<1024x16xf16>) -> memref<8x16xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<8x1024xf16>
    memref.copy %arg0, %memref : memref<8x1024xf16> to memref<8x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x16xf16>
    memref.copy %arg1, %memref_0 : memref<1024x16xf16> to memref<1024x16xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<8x16xf16>
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<8x1024xf16>, %memref_0 : memref<1024x16xf16>, %memref_1 : memref<8x16xf16>)
    gpu.dealloc  %memref : memref<8x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x16xf16>
    return %memref_1 : memref<8x16xf16>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int64, VectorAnyINTEL, Addresses, Float16, Vector16, Kernel], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @llvm_genx_raw_send2_v64i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<64xi32>) -> vector<64xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v64i32.i1.v8i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v64i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v64i32", linkage_type = <Import>>}
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<8192 x f16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<16384 x f16>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<128 x f16>, CrossWorkgroup>) "None" attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, workgroup_attributions = 0 : i64} {
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
      %cst31_i32 = spirv.Constant 31 : i32
      %cst1807_i32 = spirv.Constant 1807 : i32
      %cst2047_i32 = spirv.Constant 2047 : i32
      %cst7_i32 = spirv.Constant 7 : i32
      %cst6_i32 = spirv.Constant 6 : i32
      %cst5_i32 = spirv.Constant 5 : i32
      %cst4_i32 = spirv.Constant 4 : i32
      %cst3_i32 = spirv.Constant 3 : i32
      %cst2_i32 = spirv.Constant 2 : i32
      %cst0_i32 = spirv.Constant 0 : i32
      %cst0_i64 = spirv.Constant 0 : i64
      %cst16_i64 = spirv.Constant 16 : i64
      %cst1024_i64 = spirv.Constant 1024 : i64
      %cst_vec_128xf32 = spirv.Constant dense<0.000000e+00> : vector<128xf32>
      %0 = spirv.Undef : vector<4xi64>
      %1 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<!spirv.array<8192 x f16>, CrossWorkgroup> to i64
      %2 = spirv.VectorInsertDynamic %1, %0[%cst0_i32] : vector<4xi64>, i32
      %3 = spirv.Bitcast %2 : vector<4xi64> to vector<8xi32>
      %4 = spirv.VectorInsertDynamic %cst2047_i32, %3[%cst2_i32] : vector<8xi32>, i32
      %5 = spirv.VectorInsertDynamic %cst7_i32, %4[%cst3_i32] : vector<8xi32>, i32
      %6 = spirv.VectorInsertDynamic %cst2047_i32, %5[%cst4_i32] : vector<8xi32>, i32
      %7 = spirv.VectorInsertDynamic %cst0_i32, %6[%cst5_i32] : vector<8xi32>, i32
      %8 = spirv.VectorInsertDynamic %cst0_i32, %7[%cst6_i32] : vector<8xi32>, i32
      %9 = spirv.VectorInsertDynamic %cst1807_i32, %8[%cst7_i32] : vector<8xi32>, i32
      %10 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<!spirv.array<16384 x f16>, CrossWorkgroup> to i64
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
        spirv.Branch ^bb1(%cst0_i64, %cst_vec_128xf32, %9, %18 : i64, vector<128xf32>, vector<8xi32>, vector<8xi32>)
      ^bb1(%35: i64, %36: vector<128xf32>, %37: vector<8xi32>, %38: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
        %39 = spirv.SLessThan %35, %cst1024_i64 : i64
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
        %51 = spirv.IAdd %35, %cst16_i64 : i64
        spirv.Branch ^bb1(%51, %44, %47, %50 : i64, vector<128xf32>, vector<8xi32>, vector<8xi32>)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      %22 = spirv.Load "Function" %21 : vector<8xi32>
      %23 = spirv.Load "Function" %20 : vector<8xi32>
      %24 = spirv.Load "Function" %19 : vector<128xf32>
      %25 = spirv.FConvert %24 : vector<128xf32> to vector<128xf16>
      %26 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<!spirv.array<128 x f16>, CrossWorkgroup> to i64
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
    spirv.EntryPoint "Kernel" @test_kernel
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<8x1024xf16>, %arg1: memref<1024x16xf16>, %arg2: memref<8x16xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<0.000000e+00> : vector<8x16xf32>
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1024 = arith.constant 1024 : index
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] {mode = vc} : memref<8x1024xf16> -> !xegpu.tensor_desc<8x16xf16>
      %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] {mode = vc} : memref<1024x16xf16> -> !xegpu.tensor_desc<16x16xf16>
      %2:3 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %cst, %arg5 = %0, %arg6 = %1) -> (vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>) {
        %5 = xegpu.load_nd %arg5 {mode = vc, vnni_axis = 1} : !xegpu.tensor_desc<8x16xf16> -> vector<8x8x2xf16>
        %6 = xegpu.load_nd %arg6 {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %7 = xegpu.dpas %5, %6, %arg4 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %8 = xegpu.update_nd_offset %arg5, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
        %9 = xegpu.update_nd_offset %arg6, [%c16, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        scf.yield %7, %8, %9 : vector<8x16xf32>, !xegpu.tensor_desc<8x16xf16>, !xegpu.tensor_desc<16x16xf16>
      }
      %3 = arith.truncf %2#0 : vector<8x16xf32> to vector<8x16xf16>
      %4 = xegpu.create_nd_tdesc %arg2[%c0, %c0] {mode = vc} : memref<8x16xf16> -> !xegpu.tensor_desc<8x16xf16>
      xegpu.store_nd %3, %4 {mode = vc} : vector<8x16xf16>, !xegpu.tensor_desc<8x16xf16>
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.000000e+02 : f16
    %cst_1 = arith.constant 1.000000e+03 : f16
    %c128_i16 = arith.constant 128 : i16
    %c1024 = arith.constant 1024 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_8x1024xf16 : memref<8x1024xf16>
    %1 = memref.get_global @__constant_1024x16xf16_ : memref<1024x16xf16>
    %2 = memref.get_global @__constant_8x16xf16 : memref<8x16xf16>
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c1024 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %6 = arith.muli %4, %c128_i16 : i16
        %7 = arith.addi %5, %6 : i16
        %8 = arith.uitofp %7 : i16 to f16
        %9 = arith.divf %8, %cst_1 : f16
        memref.store %9, %0[%arg0, %arg1] : memref<8x1024xf16>
      }
    }
    scf.for %arg0 = %c0 to %c1024 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = arith.index_cast %arg0 : index to i16
        %5 = arith.index_cast %arg1 : index to i16
        %6 = arith.addi %5, %4 : i16
        %7 = arith.uitofp %6 : i16 to f16
        %8 = arith.divf %7, %cst_0 : f16
        memref.store %8, %1[%arg0, %arg1] : memref<1024x16xf16>
      }
    }
    scf.for %arg0 = %c0 to %c8 step %c1 {
      scf.for %arg1 = %c0 to %c16 step %c1 {
        %4 = scf.for %arg2 = %c0 to %c1024 step %c1 iter_args(%arg3 = %cst) -> (f32) {
          %6 = memref.load %0[%arg0, %arg2] : memref<8x1024xf16>
          %7 = memref.load %1[%arg2, %arg1] : memref<1024x16xf16>
          %8 = arith.mulf %6, %7 : f16
          %9 = arith.extf %8 : f16 to f32
          %10 = arith.addf %9, %arg3 : f32
          scf.yield %10 : f32
        }
        %5 = arith.truncf %4 : f32 to f16
        memref.store %5, %2[%arg0, %arg1] : memref<8x16xf16>
      }
    }
    %3 = call @test(%0, %1) : (memref<8x1024xf16>, memref<1024x16xf16>) -> memref<8x16xf16>
    %cast = memref.cast %3 : memref<8x16xf16> to memref<*xf16>
    call @printMemrefF16(%cast) : (memref<*xf16>) -> ()
    %cast_2 = memref.cast %2 : memref<8x16xf16> to memref<*xf16>
    call @printMemrefF16(%cast_2) : (memref<*xf16>) -> ()
    call @printAllcloseF16(%cast, %cast_2) : (memref<*xf16>, memref<*xf16>) -> ()
    return
  }
  func.func private @printMemrefF16(memref<*xf16>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF16(memref<*xf16>, memref<*xf16>) attributes {llvm.emit_c_interface}
}

