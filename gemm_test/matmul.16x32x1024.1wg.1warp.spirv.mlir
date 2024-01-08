module attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, triton_gpu.shared = 0 : i32, "triton_gpu.threads-per-warp" = 1 : i32} {
  spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
  spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v128i32", linkage_type = <Import>>}
  spirv.func @test_kernel(%arg0: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg1: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}, %arg2: !spirv.ptr<f16, CrossWorkgroup> {tt.divisibility = 16 : i32}) "None" attributes {noinline = false, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
    %cst63_i32 = spirv.Constant 63 : i32
    %cst15_i32 = spirv.Constant 15 : i32
    %cst1023_i32 = spirv.Constant 1023 : i32
    %cst2047_i32 = spirv.Constant 2047 : i32
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
    %cst1024_i32 = spirv.Constant 1024 : i32
    %cst0_i32 = spirv.Constant 0 : i32
    %cst16_i32 = spirv.Constant 16 : i32
    %0 = spirv.Undef : vector<4xi64>
    %1 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %2 = spirv.VectorInsertDynamic %1, %0[%cst0_i32] : vector<4xi64>, i32
    %3 = spirv.Bitcast %2 : vector<4xi64> to vector<8xi32>
    %4 = spirv.VectorInsertDynamic %cst2047_i32, %3[%cst2_i32] : vector<8xi32>, i32
    %5 = spirv.VectorInsertDynamic %cst15_i32, %4[%cst3_i32] : vector<8xi32>, i32
    %6 = spirv.VectorInsertDynamic %cst2047_i32, %5[%cst4_i32] : vector<8xi32>, i32
    %7 = spirv.VectorInsertDynamic %cst0_i32, %6[%cst5_i32] : vector<8xi32>, i32
    %8 = spirv.VectorInsertDynamic %cst0_i32, %7[%cst6_i32] : vector<8xi32>, i32
    %9 = spirv.VectorInsertDynamic %cst3855_i32, %8[%cst7_i32] : vector<8xi32>, i32
    %10 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %11 = spirv.VectorInsertDynamic %10, %0[%cst0_i32] : vector<4xi64>, i32
    %12 = spirv.Bitcast %11 : vector<4xi64> to vector<8xi32>
    %13 = spirv.VectorInsertDynamic %cst63_i32, %12[%cst2_i32] : vector<8xi32>, i32
    %14 = spirv.VectorInsertDynamic %cst1023_i32, %13[%cst3_i32] : vector<8xi32>, i32
    %15 = spirv.VectorInsertDynamic %cst63_i32, %14[%cst4_i32] : vector<8xi32>, i32
    %16 = spirv.VectorInsertDynamic %cst0_i32, %15[%cst5_i32] : vector<8xi32>, i32
    %17 = spirv.VectorInsertDynamic %cst0_i32, %16[%cst6_i32] : vector<8xi32>, i32
    %18 = spirv.VectorInsertDynamic %cst3855_i32, %17[%cst7_i32] : vector<8xi32>, i32
    %19 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %20 = spirv.VectorInsertDynamic %19, %0[%cst0_i32] : vector<4xi64>, i32
    %21 = spirv.Bitcast %20 : vector<4xi64> to vector<8xi32>
    %22 = spirv.VectorInsertDynamic %cst63_i32, %21[%cst2_i32] : vector<8xi32>, i32
    %23 = spirv.VectorInsertDynamic %cst1023_i32, %22[%cst3_i32] : vector<8xi32>, i32
    %24 = spirv.VectorInsertDynamic %cst63_i32, %23[%cst4_i32] : vector<8xi32>, i32
    %25 = spirv.VectorInsertDynamic %cst16_i32, %24[%cst5_i32] : vector<8xi32>, i32
    %26 = spirv.VectorInsertDynamic %cst0_i32, %25[%cst6_i32] : vector<8xi32>, i32
    %27 = spirv.VectorInsertDynamic %cst3855_i32, %26[%cst7_i32] : vector<8xi32>, i32
    %28 = spirv.Variable : !spirv.ptr<vector<256xf32>, Function>
    %29 = spirv.Variable : !spirv.ptr<vector<256xf32>, Function>
    %30 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    %31 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    %32 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
    spirv.mlir.loop {
      spirv.Branch ^bb1(%cst0_i32, %cst_vec_256xf32, %cst_vec_256xf32, %9, %18, %27 : i32, vector<256xf32>, vector<256xf32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)
    ^bb1(%58: i32, %59: vector<256xf32>, %60: vector<256xf32>, %61: vector<8xi32>, %62: vector<8xi32>, %63: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
      %64 = spirv.SLessThan %58, %cst1024_i32 : i32
      spirv.BranchConditional %64, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %65 = spirv.Undef : vector<128xi32>
      %66 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074627_i32, %61, %65) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %67 = spirv.Bitcast %66 : vector<128xi32> to vector<256xf16>
      %68 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %62, %65) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %69 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %63, %65) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
      %70 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %59, %59 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %71 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %67, %67 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
      %72 = spirv.Bitcast %71 : vector<128xf16> to vector<64xi32>
      %73 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%70, %68, %72, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %74 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %59, %59 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %75 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %67, %67 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
      %76 = spirv.Bitcast %75 : vector<128xf16> to vector<64xi32>
      %77 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%74, %68, %76, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %78 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %73, %77 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
      %79 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %60, %60 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %80 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%79, %69, %72, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %81 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %60, %60 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
      %82 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%81, %69, %76, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
      %83 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %80, %82 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
      %84 = spirv.VectorExtractDynamic %61[%cst5_i32] : vector<8xi32>, i32
      %85 = spirv.IAdd %84, %cst16_i32 : i32
      %86 = spirv.VectorInsertDynamic %85, %61[%cst5_i32] : vector<8xi32>, i32
      %87 = spirv.VectorExtractDynamic %62[%cst6_i32] : vector<8xi32>, i32
      %88 = spirv.IAdd %87, %cst16_i32 : i32
      %89 = spirv.VectorInsertDynamic %88, %62[%cst6_i32] : vector<8xi32>, i32
      %90 = spirv.VectorExtractDynamic %63[%cst6_i32] : vector<8xi32>, i32
      %91 = spirv.IAdd %90, %cst16_i32 : i32
      %92 = spirv.VectorInsertDynamic %91, %63[%cst6_i32] : vector<8xi32>, i32
      spirv.Store "Function" %28, %78 : vector<256xf32>
      spirv.Store "Function" %29, %83 : vector<256xf32>
      spirv.Store "Function" %30, %86 : vector<8xi32>
      spirv.Store "Function" %31, %89 : vector<8xi32>
      spirv.Store "Function" %32, %92 : vector<8xi32>
      %93 = spirv.IAdd %58, %cst16_i32 : i32
      spirv.Branch ^bb1(%93, %78, %83, %86, %89, %92 : i32, vector<256xf32>, vector<256xf32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)
    ^bb3:  // pred: ^bb1
      spirv.mlir.merge
    }
    %33 = spirv.Load "Function" %32 : vector<8xi32>
    %34 = spirv.Load "Function" %31 : vector<8xi32>
    %35 = spirv.Load "Function" %30 : vector<8xi32>
    %36 = spirv.Load "Function" %29 : vector<256xf32>
    %37 = spirv.Load "Function" %28 : vector<256xf32>
    %38 = spirv.FConvert %37 : vector<256xf32> to vector<256xf16>
    %39 = spirv.FConvert %36 : vector<256xf32> to vector<256xf16>
    %40 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %41 = spirv.VectorInsertDynamic %40, %0[%cst0_i32] : vector<4xi64>, i32
    %42 = spirv.Bitcast %41 : vector<4xi64> to vector<8xi32>
    %43 = spirv.VectorInsertDynamic %cst63_i32, %42[%cst2_i32] : vector<8xi32>, i32
    %44 = spirv.VectorInsertDynamic %cst15_i32, %43[%cst3_i32] : vector<8xi32>, i32
    %45 = spirv.VectorInsertDynamic %cst63_i32, %44[%cst4_i32] : vector<8xi32>, i32
    %46 = spirv.VectorInsertDynamic %cst0_i32, %45[%cst5_i32] : vector<8xi32>, i32
    %47 = spirv.VectorInsertDynamic %cst0_i32, %46[%cst6_i32] : vector<8xi32>, i32
    %48 = spirv.VectorInsertDynamic %cst3855_i32, %47[%cst7_i32] : vector<8xi32>, i32
    %49 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<f16, CrossWorkgroup> to i64
    %50 = spirv.VectorInsertDynamic %49, %0[%cst0_i32] : vector<4xi64>, i32
    %51 = spirv.Bitcast %50 : vector<4xi64> to vector<8xi32>
    %52 = spirv.VectorInsertDynamic %cst63_i32, %51[%cst2_i32] : vector<8xi32>, i32
    %53 = spirv.VectorInsertDynamic %cst15_i32, %52[%cst3_i32] : vector<8xi32>, i32
    %54 = spirv.VectorInsertDynamic %cst63_i32, %53[%cst4_i32] : vector<8xi32>, i32
    %55 = spirv.VectorInsertDynamic %cst16_i32, %54[%cst5_i32] : vector<8xi32>, i32
    %56 = spirv.VectorInsertDynamic %cst0_i32, %55[%cst6_i32] : vector<8xi32>, i32
    %57 = spirv.VectorInsertDynamic %cst3855_i32, %56[%cst7_i32] : vector<8xi32>, i32
    spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %48, %38) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) -> ()
    spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %57, %39) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) -> ()
    spirv.Return
  }
}

