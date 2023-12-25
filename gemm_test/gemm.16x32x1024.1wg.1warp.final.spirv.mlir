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
    gpu.launch_func  @test_kernel::@test_kernel blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1)  args(%memref : memref<16x1024xf16>, %memref_0 : memref<1024x32xf16>, %memref_1 : memref<16x32xf16>)
    gpu.dealloc  %memref : memref<16x1024xf16>
    gpu.dealloc  %memref_0 : memref<1024x32xf16>
    return %memref_1 : memref<16x32xf16>
  }
  spirv.module @__spv__test_kernel Physical64 OpenCL requires #spirv.vce<v1.0, [Int8, Int64, VectorAnyINTEL, Addresses, Float16, Vector16, Kernel], []> attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    spirv.func @llvm_genx_raw_send2_v128i32_i1_v8i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.send2.v128i32.i1.v8i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_dpas2_v128f32_v128i32_v64i32(vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32> "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.dpas2.v128f32.v128i32.v64i32", linkage_type = <Import>>}
    spirv.func @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) "None" attributes {VectorComputeFunctionINTEL, linkage_attributes = #spirv.linkage_attributes<linkage_name = "llvm.genx.raw.sends2.noresult.i1.v8i32.v128i32", linkage_type = <Import>>}
    spirv.func @test_kernel(%arg0: !spirv.ptr<!spirv.array<16384 x f16>, CrossWorkgroup>, %arg1: !spirv.ptr<!spirv.array<32768 x f16>, CrossWorkgroup>, %arg2: !spirv.ptr<!spirv.array<512 x f16>, CrossWorkgroup>) "None" attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, workgroup_attributions = 0 : i64} {
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
      %cst16_i32 = spirv.Constant 16 : i32
      %cst1023_i32 = spirv.Constant 1023 : i32
      %cst63_i32 = spirv.Constant 63 : i32
      %cst3855_i32 = spirv.Constant 3855 : i32
      %cst15_i32 = spirv.Constant 15 : i32
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
      %cst_vec_256xf32 = spirv.Constant dense<0.000000e+00> : vector<256xf32>
      %0 = spirv.Undef : vector<4xi64>
      %1 = spirv.ConvertPtrToU %arg0 : !spirv.ptr<!spirv.array<16384 x f16>, CrossWorkgroup> to i64
      %2 = spirv.VectorInsertDynamic %1, %0[%cst0_i32] : vector<4xi64>, i32
      %3 = spirv.Bitcast %2 : vector<4xi64> to vector<8xi32>
      %4 = spirv.VectorInsertDynamic %cst2047_i32, %3[%cst2_i32] : vector<8xi32>, i32
      %5 = spirv.VectorInsertDynamic %cst15_i32, %4[%cst3_i32] : vector<8xi32>, i32
      %6 = spirv.VectorInsertDynamic %cst2047_i32, %5[%cst4_i32] : vector<8xi32>, i32
      %7 = spirv.VectorInsertDynamic %cst0_i32, %6[%cst5_i32] : vector<8xi32>, i32
      %8 = spirv.VectorInsertDynamic %cst0_i32, %7[%cst6_i32] : vector<8xi32>, i32
      %9 = spirv.VectorInsertDynamic %cst3855_i32, %8[%cst7_i32] : vector<8xi32>, i32
      %10 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<!spirv.array<32768 x f16>, CrossWorkgroup> to i64
      %11 = spirv.VectorInsertDynamic %10, %0[%cst0_i32] : vector<4xi64>, i32
      %12 = spirv.Bitcast %11 : vector<4xi64> to vector<8xi32>
      %13 = spirv.VectorInsertDynamic %cst63_i32, %12[%cst2_i32] : vector<8xi32>, i32
      %14 = spirv.VectorInsertDynamic %cst1023_i32, %13[%cst3_i32] : vector<8xi32>, i32
      %15 = spirv.VectorInsertDynamic %cst63_i32, %14[%cst4_i32] : vector<8xi32>, i32
      %16 = spirv.VectorInsertDynamic %cst0_i32, %15[%cst5_i32] : vector<8xi32>, i32
      %17 = spirv.VectorInsertDynamic %cst0_i32, %16[%cst6_i32] : vector<8xi32>, i32
      %18 = spirv.VectorInsertDynamic %cst3855_i32, %17[%cst7_i32] : vector<8xi32>, i32
      %19 = spirv.ConvertPtrToU %arg1 : !spirv.ptr<!spirv.array<32768 x f16>, CrossWorkgroup> to i64
      %20 = spirv.Variable : !spirv.ptr<vector<256xf32>, Function>
      %21 = spirv.Variable : !spirv.ptr<vector<256xf32>, Function>
      %22 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
      %23 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
      %24 = spirv.Variable : !spirv.ptr<vector<8xi32>, Function>
      spirv.mlir.loop {
        spirv.Branch ^bb1(%cst0_i64, %cst_vec_256xf32, %cst_vec_256xf32, %9, %18, %18 : i64, vector<256xf32>, vector<256xf32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)
      ^bb1(%50: i64, %51: vector<256xf32>, %52: vector<256xf32>, %53: vector<8xi32>, %54: vector<8xi32>, %55: vector<8xi32>):  // 2 preds: ^bb0, ^bb2
        %56 = spirv.SLessThan %50, %cst1024_i64 : i64
        spirv.BranchConditional %56, ^bb2, ^bb3
      ^bb2:  // pred: ^bb1
        %57 = spirv.Undef : vector<128xi32>
        %58 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074627_i32, %53, %57) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
        %59 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %54, %57) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
        %60 = spirv.FunctionCall @llvm_genx_raw_send2_v128i32_i1_v8i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst42074755_i32, %55, %57) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<128xi32>) -> vector<128xi32>
        %61 = spirv.Bitcast %58 : vector<128xi32> to vector<256xf16>
        %62 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %61, %61 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
        %63 = spirv.Bitcast %62 : vector<128xf16> to vector<64xi32>
        %64 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %61, %61 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
        %65 = spirv.Bitcast %64 : vector<128xf16> to vector<64xi32>
        %66 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %51, %51 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %67 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %51, %51 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %68 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %52, %52 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %69 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %52, %52 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %70 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%66, %59, %63, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %71 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%68, %60, %63, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %72 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%67, %59, %65, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %73 = spirv.FunctionCall @llvm_genx_dpas2_v128f32_v128i32_v64i32(%69, %60, %65, %cst10_i32, %cst10_i32, %cst8_i32, %cst8_i32, %cst0_i32, %cst0_i32) : (vector<128xf32>, vector<128xi32>, vector<64xi32>, i32, i32, i32, i32, i32, i32) -> vector<128xf32>
        %74 = spirv.VectorExtractDynamic %53[%cst5_i32] : vector<8xi32>, i32
        %75 = spirv.IAdd %74, %cst16_i32 : i32
        %76 = spirv.VectorInsertDynamic %75, %53[%cst5_i32] : vector<8xi32>, i32
        %77 = spirv.VectorExtractDynamic %54[%cst6_i32] : vector<8xi32>, i32
        %78 = spirv.IAdd %77, %cst16_i32 : i32
        %79 = spirv.VectorInsertDynamic %78, %54[%cst6_i32] : vector<8xi32>, i32
        %80 = spirv.VectorExtractDynamic %55[%cst6_i32] : vector<8xi32>, i32
        %81 = spirv.IAdd %80, %cst16_i32 : i32
        %82 = spirv.VectorInsertDynamic %81, %55[%cst6_i32] : vector<8xi32>, i32
        %83 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %70, %72 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
        %84 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %71, %73 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
        spirv.Store "Function" %20, %83 : vector<256xf32>
        spirv.Store "Function" %21, %84 : vector<256xf32>
        spirv.Store "Function" %22, %76 : vector<8xi32>
        spirv.Store "Function" %23, %79 : vector<8xi32>
        spirv.Store "Function" %24, %82 : vector<8xi32>
        %85 = spirv.IAdd %50, %cst16_i64 : i64
        spirv.Branch ^bb1(%85, %83, %84, %76, %79, %82 : i64, vector<256xf32>, vector<256xf32>, vector<8xi32>, vector<8xi32>, vector<8xi32>)
      ^bb3:  // pred: ^bb1
        spirv.mlir.merge
      }
      %25 = spirv.Load "Function" %24 : vector<8xi32>
      %26 = spirv.Load "Function" %23 : vector<8xi32>
      %27 = spirv.Load "Function" %22 : vector<8xi32>
      %28 = spirv.Load "Function" %21 : vector<256xf32>
      %29 = spirv.Load "Function" %20 : vector<256xf32>
      %30 = spirv.FConvert %29 : vector<256xf32> to vector<256xf16>
      %31 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<!spirv.array<512 x f16>, CrossWorkgroup> to i64
      %32 = spirv.VectorInsertDynamic %31, %0[%cst0_i32] : vector<4xi64>, i32
      %33 = spirv.Bitcast %32 : vector<4xi64> to vector<8xi32>
      %34 = spirv.VectorInsertDynamic %cst63_i32, %33[%cst2_i32] : vector<8xi32>, i32
      %35 = spirv.VectorInsertDynamic %cst15_i32, %34[%cst3_i32] : vector<8xi32>, i32
      %36 = spirv.VectorInsertDynamic %cst63_i32, %35[%cst4_i32] : vector<8xi32>, i32
      %37 = spirv.VectorInsertDynamic %cst0_i32, %36[%cst5_i32] : vector<8xi32>, i32
      %38 = spirv.VectorInsertDynamic %cst0_i32, %37[%cst6_i32] : vector<8xi32>, i32
      %39 = spirv.VectorInsertDynamic %cst3855_i32, %38[%cst7_i32] : vector<8xi32>, i32
      spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %39, %30) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) -> ()
      %40 = spirv.FConvert %28 : vector<256xf32> to vector<256xf16>
      %41 = spirv.ConvertPtrToU %arg2 : !spirv.ptr<!spirv.array<512 x f16>, CrossWorkgroup> to i64
      %42 = spirv.VectorInsertDynamic %41, %0[%cst0_i32] : vector<4xi64>, i32
      %43 = spirv.Bitcast %42 : vector<4xi64> to vector<8xi32>
      %44 = spirv.VectorInsertDynamic %cst63_i32, %43[%cst2_i32] : vector<8xi32>, i32
      %45 = spirv.VectorInsertDynamic %cst15_i32, %44[%cst3_i32] : vector<8xi32>, i32
      %46 = spirv.VectorInsertDynamic %cst63_i32, %45[%cst4_i32] : vector<8xi32>, i32
      %47 = spirv.VectorInsertDynamic %cst16_i32, %46[%cst5_i32] : vector<8xi32>, i32
      %48 = spirv.VectorInsertDynamic %cst0_i32, %47[%cst6_i32] : vector<8xi32>, i32
      %49 = spirv.VectorInsertDynamic %cst3855_i32, %48[%cst7_i32] : vector<8xi32>, i32
      spirv.FunctionCall @llvm_genx_raw_sends2_noresult_i1_v8i32_v128i32(%cst0_i8, %cst0_i8, %true, %cst1_i8, %cst8_i8, %cst15_i8, %cst0_i32, %cst33686023_i32, %49, %40) : (i8, i8, i1, i8, i8, i8, i32, i32, vector<8xi32>, vector<256xf16>) -> ()
      spirv.Return
    }
    spirv.EntryPoint "Kernel" @test_kernel
  }
  gpu.module @test_kernel attributes {spirv.target_env = #spirv.target_env<#spirv.vce<v1.4, [Addresses, Float16Buffer, Int64, Int16, Int8, Kernel, Linkage, Vector16, GenericPointer, Groups, Float16, Float64, AtomicFloat32AddEXT, ExpectAssumeKHR, SubgroupDispatch, VectorComputeINTEL, VectorAnyINTEL], [SPV_EXT_shader_atomic_float_add, SPV_KHR_expect_assume, SPV_INTEL_vector_compute]>, api=OpenCL, #spirv.resource_limits<>>} {
    gpu.func @test_kernel(%arg0: memref<16x1024xf16>, %arg1: memref<1024x32xf16>, %arg2: memref<16x32xf16>) kernel attributes {VectorComputeFunctionINTEL, gpu.known_block_size = array<i32: 1, 1, 1>, gpu.known_grid_size = array<i32: 128, 64, 1>, spirv.entry_point_abi = #spirv.entry_point_abi<>} {
      %cst = arith.constant dense<0.000000e+00> : vector<16x16xf32>
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1024 = arith.constant 1024 : index
      %0 = xegpu.create_nd_tdesc %arg0[%c0, %c0] {mode = vc} : memref<16x1024xf16> -> !xegpu.tensor_desc<16x16xf16>
      %1 = xegpu.create_nd_tdesc %arg1[%c0, %c0] {mode = vc} : memref<1024x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      %2:5 = scf.for %arg3 = %c0 to %c1024 step %c16 iter_args(%arg4 = %cst, %arg5 = %cst, %arg6 = %0, %arg7 = %1, %arg8 = %1) -> (vector<16x16xf32>, vector<16x16xf32>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>) {
        %7 = xegpu.load_nd %arg6 {mode = vc, vnni_axis = 1} : !xegpu.tensor_desc<16x16xf16> -> vector<16x8x2xf16>
        %8 = xegpu.load_nd %arg7 {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %9 = xegpu.load_nd %arg8 {mode = vc, vnni_axis = 0} : !xegpu.tensor_desc<16x16xf16> -> vector<8x16x2xf16>
        %10 = vector.shape_cast %7 : vector<16x8x2xf16> to vector<256xf16>
        %11 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %10, %10 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
        %12 = vector.shape_cast %11 : vector<128xf16> to vector<8x8x2xf16>
        %13 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %10, %10 : vector<256xf16>, vector<256xf16> -> vector<128xf16>
        %14 = vector.shape_cast %13 : vector<128xf16> to vector<8x8x2xf16>
        %15 = vector.shape_cast %arg4 : vector<16x16xf32> to vector<256xf32>
        %16 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %15, %15 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %17 = vector.shape_cast %16 : vector<128xf32> to vector<8x16xf32>
        %18 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %15, %15 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %19 = vector.shape_cast %18 : vector<128xf32> to vector<8x16xf32>
        %20 = vector.shape_cast %arg5 : vector<16x16xf32> to vector<256xf32>
        %21 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32] %20, %20 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %22 = vector.shape_cast %21 : vector<128xf32> to vector<8x16xf32>
        %23 = spirv.VectorShuffle [128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %20, %20 : vector<256xf32>, vector<256xf32> -> vector<128xf32>
        %24 = vector.shape_cast %23 : vector<128xf32> to vector<8x16xf32>
        %25 = xegpu.dpas %12, %8, %17 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %26 = xegpu.dpas %12, %9, %22 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %27 = xegpu.dpas %14, %8, %19 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %28 = xegpu.dpas %14, %9, %24 {mode = vc} : vector<8x8x2xf16>, vector<8x16x2xf16>, vector<8x16xf32> -> vector<8x16xf32>
        %29 = xegpu.update_nd_offset %arg6, [%c0, %c16] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %30 = xegpu.update_nd_offset %arg7, [%c16, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %31 = xegpu.update_nd_offset %arg8, [%c16, %c0] {mode = vc} : !xegpu.tensor_desc<16x16xf16> -> !xegpu.tensor_desc<16x16xf16>
        %32 = vector.shape_cast %25 : vector<8x16xf32> to vector<128xf32>
        %33 = vector.shape_cast %27 : vector<8x16xf32> to vector<128xf32>
        %34 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %32, %33 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
        %35 = vector.shape_cast %34 : vector<256xf32> to vector<16x16xf32>
        %36 = vector.shape_cast %26 : vector<8x16xf32> to vector<128xf32>
        %37 = vector.shape_cast %28 : vector<8x16xf32> to vector<128xf32>
        %38 = spirv.VectorShuffle [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32, 8 : i32, 9 : i32, 10 : i32, 11 : i32, 12 : i32, 13 : i32, 14 : i32, 15 : i32, 16 : i32, 17 : i32, 18 : i32, 19 : i32, 20 : i32, 21 : i32, 22 : i32, 23 : i32, 24 : i32, 25 : i32, 26 : i32, 27 : i32, 28 : i32, 29 : i32, 30 : i32, 31 : i32, 32 : i32, 33 : i32, 34 : i32, 35 : i32, 36 : i32, 37 : i32, 38 : i32, 39 : i32, 40 : i32, 41 : i32, 42 : i32, 43 : i32, 44 : i32, 45 : i32, 46 : i32, 47 : i32, 48 : i32, 49 : i32, 50 : i32, 51 : i32, 52 : i32, 53 : i32, 54 : i32, 55 : i32, 56 : i32, 57 : i32, 58 : i32, 59 : i32, 60 : i32, 61 : i32, 62 : i32, 63 : i32, 64 : i32, 65 : i32, 66 : i32, 67 : i32, 68 : i32, 69 : i32, 70 : i32, 71 : i32, 72 : i32, 73 : i32, 74 : i32, 75 : i32, 76 : i32, 77 : i32, 78 : i32, 79 : i32, 80 : i32, 81 : i32, 82 : i32, 83 : i32, 84 : i32, 85 : i32, 86 : i32, 87 : i32, 88 : i32, 89 : i32, 90 : i32, 91 : i32, 92 : i32, 93 : i32, 94 : i32, 95 : i32, 96 : i32, 97 : i32, 98 : i32, 99 : i32, 100 : i32, 101 : i32, 102 : i32, 103 : i32, 104 : i32, 105 : i32, 106 : i32, 107 : i32, 108 : i32, 109 : i32, 110 : i32, 111 : i32, 112 : i32, 113 : i32, 114 : i32, 115 : i32, 116 : i32, 117 : i32, 118 : i32, 119 : i32, 120 : i32, 121 : i32, 122 : i32, 123 : i32, 124 : i32, 125 : i32, 126 : i32, 127 : i32, 128 : i32, 129 : i32, 130 : i32, 131 : i32, 132 : i32, 133 : i32, 134 : i32, 135 : i32, 136 : i32, 137 : i32, 138 : i32, 139 : i32, 140 : i32, 141 : i32, 142 : i32, 143 : i32, 144 : i32, 145 : i32, 146 : i32, 147 : i32, 148 : i32, 149 : i32, 150 : i32, 151 : i32, 152 : i32, 153 : i32, 154 : i32, 155 : i32, 156 : i32, 157 : i32, 158 : i32, 159 : i32, 160 : i32, 161 : i32, 162 : i32, 163 : i32, 164 : i32, 165 : i32, 166 : i32, 167 : i32, 168 : i32, 169 : i32, 170 : i32, 171 : i32, 172 : i32, 173 : i32, 174 : i32, 175 : i32, 176 : i32, 177 : i32, 178 : i32, 179 : i32, 180 : i32, 181 : i32, 182 : i32, 183 : i32, 184 : i32, 185 : i32, 186 : i32, 187 : i32, 188 : i32, 189 : i32, 190 : i32, 191 : i32, 192 : i32, 193 : i32, 194 : i32, 195 : i32, 196 : i32, 197 : i32, 198 : i32, 199 : i32, 200 : i32, 201 : i32, 202 : i32, 203 : i32, 204 : i32, 205 : i32, 206 : i32, 207 : i32, 208 : i32, 209 : i32, 210 : i32, 211 : i32, 212 : i32, 213 : i32, 214 : i32, 215 : i32, 216 : i32, 217 : i32, 218 : i32, 219 : i32, 220 : i32, 221 : i32, 222 : i32, 223 : i32, 224 : i32, 225 : i32, 226 : i32, 227 : i32, 228 : i32, 229 : i32, 230 : i32, 231 : i32, 232 : i32, 233 : i32, 234 : i32, 235 : i32, 236 : i32, 237 : i32, 238 : i32, 239 : i32, 240 : i32, 241 : i32, 242 : i32, 243 : i32, 244 : i32, 245 : i32, 246 : i32, 247 : i32, 248 : i32, 249 : i32, 250 : i32, 251 : i32, 252 : i32, 253 : i32, 254 : i32, 255 : i32] %36, %37 : vector<128xf32>, vector<128xf32> -> vector<256xf32>
        %39 = vector.shape_cast %38 : vector<256xf32> to vector<16x16xf32>
        scf.yield %35, %39, %29, %30, %31 : vector<16x16xf32>, vector<16x16xf32>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
      }
      %3 = arith.truncf %2#0 : vector<16x16xf32> to vector<16x16xf16>
      %4 = xegpu.create_nd_tdesc %arg2[%c0, %c0] {mode = vc} : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      xegpu.store_nd %3, %4 {mode = vc} : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
      %5 = arith.truncf %2#1 : vector<16x16xf32> to vector<16x16xf16>
      %6 = xegpu.create_nd_tdesc %arg2[%c0, %c16] {mode = vc} : memref<16x32xf16> -> !xegpu.tensor_desc<16x16xf16>
      xegpu.store_nd %5, %6 {mode = vc} : vector<16x16xf16>, !xegpu.tensor_desc<16x16xf16>
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

