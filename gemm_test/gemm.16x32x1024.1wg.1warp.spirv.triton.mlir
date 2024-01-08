module @gemm attributes {gpu.container_module} {
  memref.global "private" @__constant_16x1024xf16 : memref<16x1024xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_1024x32xf16_ : memref<1024x32xf16> = dense<0.000000e+00>
  memref.global "private" @__constant_16x32xf16 : memref<16x32xf16> = dense<0.000000e+00>
  func.func @test(%arg0: memref<16x1024xf16>, %arg1: memref<1024x32xf16>) -> memref<16x32xf16> attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %memref = gpu.alloc  host_shared () : memref<16x1024xf16>
    memref.copy %arg0, %memref : memref<16x1024xf16> to memref<16x1024xf16>
    %memref_0 = gpu.alloc  host_shared () : memref<1024x32xf16>
    memref.copy %arg1, %memref_0 : memref<1024x32xf16> to memref<1024x32xf16>
    %memref_1 = gpu.alloc  host_shared () : memref<16x32xf16>
    %m = arith.constant 16 : i32
    %n = arith.constant 32 : i32
    %k = arith.constant 1024 : i32
    %stride_am = arith.constant 1024 : i32
    %stride_bk = arith.constant 32 : i32
    %stride_cm = arith.constant 32 : i32
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x1024xf16>, %memref_0 : memref<1024x32xf16>, %memref_1 : memref<16x32xf16>, %m : i32, %n : i32, %k : i32, %stride_am : i32, %stride_bk : i32, %stride_cm : i32)
    gpu.dealloc  %memref : memref<16x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x32xf16>
    return %memref_1 : memref<16x32xf16>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int64, VectorAnyINTEL, Addresses, Float16, Vector16, Kernel], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
  spirv.GlobalVariable @__builtin__WorkgroupId__ built_in("WorkgroupId") : !spirv.ptr<vector<3xi64>, Input>
  spirv.func @llvm.genx.smin.i32(i32, i32) -> i32 "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.smin.i32", linkage_type = <Import>>}
  spirv.func @llvm.genx.absi.i32(i32) -> i32 "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.absi.i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v128i32", linkage_type = <Import>>}
  spirv.func @test_kernel(%arg0: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg1: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg2: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) "None" attributes {VectorComputeFunctionINTEL, noinline = false, spirv.entry_point_abi = #spirv.entry_point_abi<>, sym_visibility = "public"} {
    %cst33686023_i32 = spirv.Constant 33686023 : i32
    %cst8_i32 = spirv.Constant 8 : i32
    %cst10_i32 = spirv.Constant 10 : i32
    %cst42074755_i32 = spirv.Constant 42074755 : i32
    %cst42074627_i32 = spirv.Constant 42074627 : i32
    %cst15_i8 = spirv.Constant 15 : i8
    %cst8_i8 = spirv.Constant 8 : i8
    %cst1_i8 = spirv.Constant 1 : i8
    %true = spirv.Constant true
    %cst0_i8 = spirv.Constant 0 : i8
    %cst3855_i32 = spirv.Constant 3855 : i32
    %cst7_i32 = spirv.Constant 7 : i32
    %cst6_i32 = spirv.Constant 6 : i32
    %cst5_i32 = spirv.Constant 5 : i32
    %cst4_i32 = spirv.Constant 4 : i32
    %cst3_i32 = spirv.Constant 3 : i32
    %cst2_i32 = spirv.Constant 2 : i32
    %cst_vec_256xf32 = spirv.Constant dense<0.000000e+00> : vector<256xf32>
    %cst32_i32 = spirv.Constant 32 : i32
    %cst0_i32 = spirv.Constant 0 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %cst15_i32 = spirv.Constant 15 : i32
    %cst31_i32 = spirv.Constant 31 : i32
    %cst1_i32 = spirv.Constant 1 : i32
    %__builtin__WorkgroupId___addr = spirv.mlir.addressof @__builtin__WorkgroupId__ : !spirv.ptr<vector<3xi64>, Input>
    %0 = spirv.Load "Input" %__builtin__WorkgroupId___addr : vector<3xi64>
    %1 = spirv.CompositeExtract %0[0 : i32] : vector<3xi64>
    %2 = spirv.SConvert %1 : i64 to i32
    %3 = spirv.IAdd %arg3, %cst15_i32 : i32
    %4 = spirv.SDiv %3, %cst16_i32 : i32
    %5 = spirv.IAdd %arg4, %cst31_i32 : i32
    %6 = spirv.SDiv %5, %cst32_i32 : i32
    %7 = spirv.SDiv %2, %6 : i32
    %8 = spirv.ISub %4, %7 : i32
    %9 = spirv.FunctionCall @llvm.genx.smin.i32(%8, %cst1_i32) : (i32, i32) -> i32
    %10 = spirv.FunctionCall @llvm.genx.absi.i32(%2) : (i32) -> i32
    %11 = spirv.FunctionCall @llvm.genx.absi.i32(%9) : (i32) -> i32
    %12 = spirv.UMod %10, %11 : i32
    %13 = spirv.IEqual %2, %10 : i32
    %14 = spirv.SNegate %12 : i32
    %15 = spirv.Select %13, %12, %14 : i1, i32
    %16 = spirv.IAdd %7, %15 : i32
    %17 = spirv.FunctionCall @llvm.genx.absi.i32(%2) : (i32) -> i32
    %18 = spirv.FunctionCall @llvm.genx.absi.i32(%6) : (i32) -> i32
    %19 = spirv.UMod %17, %18 : i32
    %20 = spirv.IEqual %2, %17 : i32
    %21 = spirv.SNegate %19 : i32
    %22 = spirv.Select %20, %19, %21 : i1, i32
    %23 = spirv.SDiv %22, %9 : i32
    %24 = spirv.IMul %16, %cst16_i32 : i32
    %25 = spirv.Undef : vector<4xi64>
    %26 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %27 = spirv.VectorInsertDynamic %26, %25[%cst0_i32] : vector<4xi64>, i32
    %28 = spirv.Bitcast %27 : vector<4xi64> to vector<8xi32>
    %29 = spirv.IMul %arg5, %cst2_i32 : i32
    %30 = spirv.ISub %29, %cst1_i32 : i32
    %31 = spirv.ISub %arg3, %cst1_i32 : i32
    %32 = spirv.IMul %arg6, %cst2_i32 : i32
    %33 = spirv.ISub %32, %cst1_i32 : i32
    %34 = spirv.VectorInsertDynamic %30, %28[%cst2_i32] : vector<8xi32>, i32
    %35 = spirv.VectorInsertDynamic %31, %34[%cst3_i32] : vector<8xi32>, i32
    %36 = spirv.VectorInsertDynamic %33, %35[%cst4_i32] : vector<8xi32>, i32
    %37 = spirv.VectorInsertDynamic %cst0_i32, %36[%cst5_i32] : vector<8xi32>, i32
    %38 = spirv.VectorInsertDynamic %24, %37[%cst6_i32] : vector<8xi32>, i32
    %39 = spirv.VectorInsertDynamic %cst3855_i32, %38[%cst7_i32] : vector<8xi32>, i32
    %40 = spirv.IMul %23, %cst32_i32 : i32
    %41 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %42 = spirv.VectorInsertDynamic %41, %25[%cst0_i32] : vector<4xi64>, i32
    %43 = spirv.Bitcast %42 : vector<4xi64> to vector<8xi32>
    %44 = spirv.IMul %arg4, %cst2_i32 : i32
    %45 = spirv.ISub %44, %cst1_i32 : i32
    %46 = spirv.ISub %arg5, %cst1_i32 : i32
    %47 = spirv.IMul %arg7, %cst2_i32 : i32
    %48 = spirv.ISub %47, %cst1_i32 : i32
    %49 = spirv.VectorInsertDynamic %45, %43[%cst2_i32] : vector<8xi32>, i32
    %50 = spirv.VectorInsertDynamic %46, %49[%cst3_i32] : vector<8xi32>, i32
    %51 = spirv.VectorInsertDynamic %48, %50[%cst4_i32] : vector<8xi32>, i32
    %52 = spirv.VectorInsertDynamic %40, %51[%cst5_i32] : vector<8xi32>, i32
    %53 = spirv.VectorInsertDynamic %cst0_i32, %52[%cst6_i32] : vector<8xi32>, i32
    %54 = spirv.VectorInsertDynamic %cst3855_i32, %53[%cst7_i32] : vector<8xi32>, i32
    %55 = spirv.IAdd %40, %cst16_i32 : i32
    %56 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %57 = spirv.VectorInsertDynamic %56, %25[%cst0_i32] : vector<4xi64>, i32
    %58 = spirv.Bitcast %57 : vector<4xi64> to vector<8xi32>
    %59 = spirv.VectorInsertDynamic %45, %58[%cst2_i32] : vector<8xi32>, i32
    %60 = spirv.VectorInsertDynamic %46, %59[%cst3_i32] : vector<8xi32>, i32
    %61 = spirv.VectorInsertDynamic %48, %60[%cst4_i32] : vector<8xi32>, i32
    %62 = spirv.VectorInsertDynamic %55, %61[%cst5_i32] : vector<8xi32>, i32
    %63 = spirv.VectorInsertDynamic %cst0_i32, %62[%cst6_i32] : vector<8xi32>, i32
    %64 = spirv.VectorInsertDynamic %cst3855_i32, %63[%cst7_i32] : vector<8xi32>, i32
    %65 = spirv.Variable : !spirv.ptr<vector<256xf32>, Function>
    %66 = spirv.Variable : !spirv.ptr<vector<256xf32>, Function>
    %67 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    %68 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    %69 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%cst0_i32, %cst_vec_256xf32, %cst_vec_256xf32, %39, %54, %64 : i32, vector<256xf32>, vector<256xf32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)
    ^bb1(%97: i32, %98: vector<256xf32>, %99: vector<256xf32>, %100: vector<8xi32>, %101: vector<8xi32>, %102: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
      %103 = spirv.SLessThan %97, %arg5 : i32
      spirv.BranchConditional %103, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %104 = spirv.Undef : vector<128xi32>
      %105 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074627_i32, %100, %104) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %106 = spirv.Bitcast %105 : vector<128xi32> to vector<256xf16>
      %107 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %101, %104) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %108 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %102, %104) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %109 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %98, %98 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %110 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %106, %106 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
      %111 = spirv.Bitcast %110 : vector<128xf16> to vector<64xi32>
      %112 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%109, %107, %111, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %113 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %98, %98 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %114 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %106, %106 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
      %115 = spirv.Bitcast %114 : vector<128xf16> to vector<64xi32>
      %116 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%113, %107, %115, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %117 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %112, %116 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
      %118 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %99, %99 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %119 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%118, %108, %111, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %120 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %99, %99 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %121 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%120, %108, %115, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %122 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %119, %121 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
      %123 = spirv.VectorExtractDynamic %100[%cst5_i32] : vector<8xi32>, i32
      %124 = spirv.IAdd %123, %cst16_i32 : i32
      %125 = spirv.VectorInsertDynamic %124, %100[%cst5_i32] : vector<8xi32>, i32
      %126 = spirv.VectorExtractDynamic %101[%cst6_i32] : vector<8xi32>, i32
      %127 = spirv.IAdd %126, %cst16_i32 : i32
      %128 = spirv.VectorInsertDynamic %127, %101[%cst6_i32] : vector<8xi32>, i32
      %129 = spirv.VectorExtractDynamic %102[%cst6_i32] : vector<8xi32>, i32
      %130 = spirv.IAdd %129, %cst16_i32 : i32
      %131 = spirv.VectorInsertDynamic %130, %102[%cst6_i32] : vector<8xi32>, i32
      spirv.Store "Function" %65, %117 : vector<256xf32>
      spirv.Store "Function" %66, %122 : vector<256xf32>
      spirv.Store "Function" %67, %125 : vector<8xi32>
      spirv.Store "Function" %68, %128 : vector<8xi32>
      spirv.Store "Function" %69, %131 : vector<8xi32>
      %132 = spirv.IAdd %97, %cst16_i32 : i32
      spirv.Branch ^bb1(%132, %117, %122, %125, %128, %131 : i32, vector<256xf32>, vector<256xf32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    %70 = spirv.Load "Function" %69 : vector<8xi32>
    %71 = spirv.Load "Function" %68 : vector<8xi32>
    %72 = spirv.Load "Function" %67 : vector<8xi32>
    %73 = spirv.Load "Function" %66 : vector<256xf32>
    %74 = spirv.Load "Function" %65 : vector<256xf32>
    %75 = spirv.FConvert %74 : vector<256xf32> to vector<256xf16>
    %76 = spirv.FConvert %73 : vector<256xf32> to vector<256xf16>
    %77 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %78 = spirv.VectorInsertDynamic %77, %25[%cst0_i32] : vector<4xi64>, i32
    %79 = spirv.Bitcast %78 : vector<4xi64> to vector<8xi32>
    %80 = spirv.IMul %arg8, %cst2_i32 : i32
    %81 = spirv.ISub %80, %cst1_i32 : i32
    %82 = spirv.VectorInsertDynamic %45, %79[%cst2_i32] : vector<8xi32>, i32
    %83 = spirv.VectorInsertDynamic %31, %82[%cst3_i32] : vector<8xi32>, i32
    %84 = spirv.VectorInsertDynamic %81, %83[%cst4_i32] : vector<8xi32>, i32
    %85 = spirv.VectorInsertDynamic %40, %84[%cst5_i32] : vector<8xi32>, i32
    %86 = spirv.VectorInsertDynamic %24, %85[%cst6_i32] : vector<8xi32>, i32
    %87 = spirv.VectorInsertDynamic %cst3855_i32, %86[%cst7_i32] : vector<8xi32>, i32
    %88 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %89 = spirv.VectorInsertDynamic %88, %25[%cst0_i32] : vector<4xi64>, i32
    %90 = spirv.Bitcast %89 : vector<4xi64> to vector<8xi32>
    %91 = spirv.VectorInsertDynamic %45, %90[%cst2_i32] : vector<8xi32>, i32
    %92 = spirv.VectorInsertDynamic %31, %91[%cst3_i32] : vector<8xi32>, i32
    %93 = spirv.VectorInsertDynamic %81, %92[%cst4_i32] : vector<8xi32>, i32
    %94 = spirv.VectorInsertDynamic %55, %93[%cst5_i32] : vector<8xi32>, i32
    %95 = spirv.VectorInsertDynamic %24, %94[%cst6_i32] : vector<8xi32>, i32
    %96 = spirv.VectorInsertDynamic %cst3855_i32, %95[%cst7_i32] : vector<8xi32>, i32
    spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %87, %75) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) -> ()
    spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %96, %76) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) -> ()
    spirv.Return
  }


    spirv.EntryPoint "Kernel" @test_kernel, @__builtin__WorkgroupId__
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<16x1024xf16>, %arg1: memref<1024x32xf16>, %arg2: memref<16x32xf16>, %m : i32, %n : i32, %k : i32, %stride_am : i32, %stride_bk : i32, %stride_cm : i32) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      gpu.return
    }
  }
  func.func @main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant 1.0e-03 : f16
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
    %cast0 = memref.cast %3 : memref<16x32xf16> to memref<*xf16>
    call @printMemrefF16(%cast0) : (memref<*xf16>) -> ()
    %cast1 = memref.cast %2 : memref<16x32xf16> to memref<*xf16>
    call @printMemrefF16(%cast1) : (memref<*xf16>) -> ()
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

