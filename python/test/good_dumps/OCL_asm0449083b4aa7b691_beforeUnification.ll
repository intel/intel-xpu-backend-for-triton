; ------------------------------------------------
; OCL_asm0449083b4aa7b691_beforeUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir64-unknown-unknown"

@global_smem = external addrspace(3) global [1 x i8], align 16, !spirv.Decorations !0

; Function Attrs: nounwind
define spir_kernel void @kernel(float addrspace(1)* %0, float addrspace(1)* %1, float addrspace(1)* %2, float addrspace(1)* %3, i8 addrspace(1)* %4, i8 addrspace(1)* %5) #0 !kernel_arg_addr_space !270 !kernel_arg_access_qual !271 !kernel_arg_type !272 !kernel_arg_type_qual !273 !kernel_arg_base_type !272 !kernel_arg_name !273 !reqd_work_group_size !274 !intel_reqd_sub_group_size !275 {
  %7 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #2
  %8 = insertelement <3 x i64> undef, i64 %7, i32 0
  %9 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #2
  %10 = insertelement <3 x i64> %8, i64 %9, i32 1
  %11 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #2
  %12 = insertelement <3 x i64> %10, i64 %11, i32 2
  %13 = extractelement <3 x i64> %12, i32 0
  %14 = select i1 true, i64 %13, i64 0
  %15 = trunc i64 %14 to i32
  %16 = and i32 %15, 127
  %17 = urem i32 %16, 32
  %18 = udiv i32 %16, 32
  %19 = shl i32 %17, 0
  %20 = or i32 0, %19
  %21 = shl i32 %18, 5
  %22 = or i32 %20, %21
  %23 = and i32 %22, 127
  %24 = lshr i32 %23, 0
  %25 = or i32 %24, 0
  %26 = xor i32 0, %25
  %27 = xor i32 %26, 0
  %28 = add i32 %27, 0
  %29 = getelementptr float, float addrspace(1)* %0, i32 %28
  %30 = bitcast float addrspace(1)* %29 to i32 addrspace(1)*
  %31 = load i32, i32 addrspace(1)* %30, align 4
  %32 = bitcast i32 %31 to <1 x float>
  %33 = extractelement <1 x float> %32, i32 0
  %34 = getelementptr float, float addrspace(1)* %1, i32 %28
  %35 = bitcast float addrspace(1)* %34 to i32 addrspace(1)*
  %36 = load i32, i32 addrspace(1)* %35, align 4
  %37 = bitcast i32 %36 to <1 x float>
  %38 = extractelement <1 x float> %37, i32 0
  %39 = call spir_func float @__imf_fdiv_rn(float %33, float %38) #1
  %40 = fpext float %33 to double
  %41 = fpext float %38 to double
  %42 = fdiv double %40, %41
  %43 = fptrunc double %42 to float
  %44 = getelementptr float, float addrspace(1)* %2, i32 %28
  %45 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #2
  %46 = insertelement <3 x i64> undef, i64 %45, i32 0
  %47 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #2
  %48 = insertelement <3 x i64> %46, i64 %47, i32 1
  %49 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #2
  %50 = insertelement <3 x i64> %48, i64 %49, i32 2
  %51 = extractelement <3 x i64> %50, i32 0
  %52 = select i1 true, i64 %51, i64 0
  %53 = insertelement <1 x float> undef, float %39, i32 0
  %54 = bitcast <1 x float> %53 to i32
  %55 = insertelement <1 x i32> undef, i32 %54, i32 0
  %56 = bitcast float addrspace(1)* %44 to <1 x i32> addrspace(1)*
  store <1 x i32> %55, <1 x i32> addrspace(1)* %56, align 4
  %57 = getelementptr float, float addrspace(1)* %3, i32 %28
  %58 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 0) #2
  %59 = insertelement <3 x i64> undef, i64 %58, i32 0
  %60 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 1) #2
  %61 = insertelement <3 x i64> %59, i64 %60, i32 1
  %62 = call spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32 2) #2
  %63 = insertelement <3 x i64> %61, i64 %62, i32 2
  %64 = extractelement <3 x i64> %63, i32 0
  %65 = select i1 true, i64 %64, i64 0
  %66 = insertelement <1 x float> undef, float %43, i32 0
  %67 = bitcast <1 x float> %66 to i32
  %68 = insertelement <1 x i32> undef, i32 %67, i32 0
  %69 = bitcast float addrspace(1)* %57 to <1 x i32> addrspace(1)*
  store <1 x i32> %68, <1 x i32> addrspace(1)* %69, align 4
  ret void
}

; Function Attrs: alwaysinline nounwind
define internal spir_func float @__imf_fdiv_rn(float %0, float %1) #1 {
  %3 = call spir_func float @__devicelib_imf_fdiv_rn(float %0, float %1) #1
  ret float %3
}

; Function Attrs: alwaysinline nounwind
define internal spir_func float @__devicelib_imf_fdiv_rn(float %0, float %1) #1 {
  %3 = call spir_func float @_Z8__fp_divIfET_S0_S0_i(float %0, float %1, i32 0) #0
  ret float %3
}

; Function Attrs: nounwind
define internal spir_func float @_Z8__fp_divIfET_S0_S0_i(float %0, float %1, i32 %2) #0 {
  %4 = bitcast float %0 to i32
  %5 = bitcast float %1 to i32
  %6 = lshr i32 %4, 23
  %7 = and i32 %6, 255
  %8 = lshr i32 %5, 23
  %9 = and i32 %8, 255
  %10 = and i32 %4, 8388607
  %11 = and i32 %5, 8388607
  %12 = xor i32 %4, %5
  %13 = lshr i32 %12, 31
  %14 = icmp eq i32 %7, 255
  %15 = icmp ne i32 %10, 0
  %16 = and i1 %14, %15
  br i1 %16, label %281, label %17

17:                                               ; preds = %3
  %18 = icmp eq i32 %9, 255
  %19 = icmp ne i32 %11, 0
  %20 = and i1 %18, %19
  %21 = fcmp oeq float %1, 0.000000e+00
  %22 = or i1 %20, %21
  br i1 %22, label %281, label %23

23:                                               ; preds = %17
  %24 = icmp eq i32 %10, 0
  %25 = and i1 %14, %24
  br i1 %25, label %26, label %33

26:                                               ; preds = %23
  %27 = icmp eq i32 %11, 0
  %28 = and i1 %18, %27
  %29 = and i32 %12, -2147483648
  %30 = or i32 %29, 2139095040
  %31 = bitcast i32 %30 to float
  %32 = select i1 %28, float 0x7FF8000000000000, float %31
  br label %281

33:                                               ; preds = %23
  %34 = fcmp oeq float %0, 0.000000e+00
  br i1 %34, label %35, label %38

35:                                               ; preds = %33
  %36 = and i32 %12, -2147483648
  %37 = bitcast i32 %36 to float
  br label %281

38:                                               ; preds = %33
  %39 = icmp eq i32 %11, 0
  %40 = and i1 %18, %39
  br i1 %40, label %41, label %44

41:                                               ; preds = %38
  %42 = and i32 %12, -2147483648
  %43 = bitcast i32 %42 to float
  br label %281

44:                                               ; preds = %38
  %45 = icmp eq i32 %7, 0
  %46 = add nsw i32 %7, -127, !spirv.Decorations !276
  %47 = select i1 %45, i32 -126, i32 %46
  %48 = icmp eq i32 %9, 0
  %49 = add nsw i32 %9, -127, !spirv.Decorations !276
  %50 = select i1 %48, i32 -126, i32 %49
  %51 = sub nsw i32 %47, %50, !spirv.Decorations !276
  %52 = or i32 %10, 8388608
  %53 = select i1 %45, i32 %10, i32 %52
  %54 = or i32 %11, 8388608
  %55 = select i1 %48, i32 %11, i32 %54
  %56 = icmp ult i32 %53, %55
  br i1 %56, label %.preheader, label %57

.preheader:                                       ; preds = %44
  br label %189

57:                                               ; preds = %44
  %58 = udiv i32 %53, %55
  %59 = mul i32 %55, %58
  %60 = sub i32 %53, %59
  br label %61

61:                                               ; preds = %68, %57
  %62 = phi i64 [ 0, %57 ], [ %70, %68 ]
  %63 = phi i32 [ -2147483648, %57 ], [ %69, %68 ]
  %64 = icmp ugt i64 %62, 31
  %65 = and i32 %58, %63
  %66 = icmp eq i32 %65, %63
  %67 = select i1 %64, i1 true, i1 %66
  br i1 %67, label %71, label %68

68:                                               ; preds = %61
  %69 = lshr i32 %63, 1
  %70 = add nuw nsw i64 %62, 1, !spirv.Decorations !278
  br label %61

71:                                               ; preds = %61
  %72 = trunc i64 %62 to i32
  %73 = sub nsw i32 31, %72, !spirv.Decorations !276
  %74 = add nsw i32 %51, %73, !spirv.Decorations !276
  %75 = icmp sgt i32 %74, 127
  br i1 %75, label %76, label %86

76:                                               ; preds = %71
  %77 = icmp sgt i32 %12, -1
  br i1 %77, label %78, label %82

78:                                               ; preds = %76
  %79 = and i32 %2, -3
  %80 = icmp eq i32 %79, 1
  %81 = select i1 %80, float 0x47EFFFFFE0000000, float 0x7FF0000000000000
  br label %281

82:                                               ; preds = %76
  %83 = add i32 %2, -1
  %84 = icmp ult i32 %83, 2
  %85 = select i1 %84, float 0xC7EFFFFFE0000000, float 0xFFF0000000000000
  br label %281

86:                                               ; preds = %71
  %87 = icmp sgt i32 %74, -127
  br i1 %87, label %88, label %124

88:                                               ; preds = %86
  %89 = add nsw i32 %74, 127, !spirv.Decorations !276
  %90 = add nsw i32 %72, -8, !spirv.Decorations !276
  %91 = shl i32 %58, %90
  %92 = and i32 %91, 8388607
  %93 = add nsw i32 %72, -5, !spirv.Decorations !276
  %94 = call spir_func i32 @_ZL12fra_uint_divIjET_S0_S0_j(i32 %60, i32 %55, i32 %93) #0
  %95 = lshr i32 %94, 3
  %96 = or i32 %92, %95
  %97 = and i32 %94, 7
  %98 = call spir_func i32 @_ZL19__handling_roundingIjET_S0_S0_ji(i32 %13, i32 %96, i32 %97, i32 %2) #0
  %99 = icmp eq i32 %98, 0
  br i1 %99, label %116, label %100

100:                                              ; preds = %88
  %101 = add nuw nsw i32 %96, 1, !spirv.Decorations !278
  %102 = icmp ugt i32 %96, 8388606
  br i1 %102, label %103, label %116

103:                                              ; preds = %100
  %104 = add nsw i32 %74, 128, !spirv.Decorations !276
  %105 = icmp eq i32 %104, 255
  br i1 %105, label %106, label %116

106:                                              ; preds = %103
  %107 = icmp sgt i32 %12, -1
  br i1 %107, label %108, label %112

108:                                              ; preds = %106
  %109 = and i32 %2, -3
  %110 = icmp eq i32 %109, 1
  %111 = select i1 %110, float 0x47EFFFFFE0000000, float 0x7FF0000000000000
  br label %281

112:                                              ; preds = %106
  %113 = add i32 %2, -1
  %114 = icmp ult i32 %113, 2
  %115 = select i1 %114, float 0xC7EFFFFFE0000000, float 0xFFF0000000000000
  br label %281

116:                                              ; preds = %103, %100, %88
  %117 = phi i32 [ %101, %103 ], [ %101, %100 ], [ %96, %88 ]
  %118 = phi i32 [ %104, %103 ], [ %89, %100 ], [ %89, %88 ]
  %119 = and i32 %12, -2147483648
  %120 = shl nuw nsw i32 %118, 23, !spirv.Decorations !278
  %121 = or i32 %119, %120
  %122 = or i32 %121, %117
  %123 = bitcast i32 %122 to float
  br label %281

124:                                              ; preds = %86
  %125 = xor i32 %74, -1
  %126 = icmp ult i32 %74, -149
  br i1 %126, label %127, label %143

127:                                              ; preds = %124
  %128 = icmp eq i32 %74, -150
  br i1 %128, label %129, label %134

129:                                              ; preds = %127
  %130 = icmp ne i32 %53, %59
  %131 = lshr i32 -2147483648, %72
  %132 = icmp ne i32 %58, %131
  %133 = select i1 %130, i1 true, i1 %132
  br label %134

134:                                              ; preds = %129, %127
  %135 = phi i1 [ %133, %129 ], [ false, %127 ]
  %136 = icmp sgt i32 %12, -1
  br i1 %136, label %137, label %140

137:                                              ; preds = %134
  switch i32 %2, label %139 [
    i32 2, label %281
    i32 0, label %138
  ]

138:                                              ; preds = %137
  br i1 %135, label %281, label %139

139:                                              ; preds = %138, %137
  br label %281

140:                                              ; preds = %134
  switch i32 %2, label %142 [
    i32 3, label %281
    i32 0, label %141
  ]

141:                                              ; preds = %140
  br i1 %135, label %281, label %142

142:                                              ; preds = %141, %140
  br label %281

143:                                              ; preds = %124
  %144 = add nsw i32 %74, 152, !spirv.Decorations !276
  %145 = icmp sgt i32 %144, %73
  br i1 %145, label %167, label %146

146:                                              ; preds = %143
  %147 = sub nsw i32 %125, %72, !spirv.Decorations !276
  %148 = add nsw i32 %147, -117, !spirv.Decorations !276
  %149 = lshr i32 %58, %148
  %150 = add nsw i32 %147, -120, !spirv.Decorations !276
  %151 = lshr i32 %58, %150
  %152 = and i32 %151, 7
  %153 = and i32 %151, 1
  %154 = icmp eq i32 %153, 0
  br i1 %154, label %155, label %164

155:                                              ; preds = %146
  %156 = shl nsw i32 -1, %150, !spirv.Decorations !276
  %157 = xor i32 %156, -1
  %158 = and i32 %58, %157
  %159 = icmp ne i32 %158, 0
  %160 = icmp ne i32 %53, %59
  %161 = or i1 %159, %160
  %162 = select i1 %161, i32 1, i32 0
  %163 = or i32 %152, %162
  br label %164

164:                                              ; preds = %155, %146
  %165 = phi i32 [ %152, %146 ], [ %163, %155 ]
  %166 = call spir_func i32 @_ZL19__handling_roundingIjET_S0_S0_ji(i32 %13, i32 %149, i32 %165, i32 %2) #0
  br label %175

167:                                              ; preds = %143
  %168 = sub nsw i32 %144, %73, !spirv.Decorations !276
  %169 = shl i32 %58, %168
  %170 = call spir_func i32 @_ZL12fra_uint_divIjET_S0_S0_j(i32 %60, i32 %55, i32 %168) #0
  %171 = or i32 %169, %170
  %172 = and i32 %171, 7
  %173 = lshr i32 %171, 3
  %174 = call spir_func i32 @_ZL19__handling_roundingIjET_S0_S0_ji(i32 %13, i32 %173, i32 %172, i32 %2) #0
  br label %175

175:                                              ; preds = %167, %164
  %176 = phi i32 [ %166, %164 ], [ %174, %167 ]
  %177 = phi i32 [ %149, %164 ], [ %173, %167 ]
  %178 = icmp eq i32 %176, 0
  %179 = add nuw nsw i32 %177, 1, !spirv.Decorations !278
  %180 = icmp ugt i32 %177, 8388606
  %181 = select i1 %180, i32 0, i32 %179
  %182 = select i1 %180, i32 8388608, i32 0
  %183 = select i1 %178, i32 %177, i32 %181
  %184 = select i1 %178, i32 0, i32 %182
  %185 = and i32 %12, -2147483648
  %186 = or i32 %185, %184
  %187 = or i32 %186, %183
  %188 = bitcast i32 %187 to float
  br label %281

189:                                              ; preds = %194, %.preheader
  %190 = phi i32 [ %195, %194 ], [ 0, %.preheader ]
  %191 = phi i32 [ %192, %194 ], [ %53, %.preheader ]
  %192 = shl nuw nsw i32 %191, 1, !spirv.Decorations !278
  %193 = icmp ult i32 %192, %55
  br i1 %193, label %194, label %196

194:                                              ; preds = %189
  %195 = add i32 %190, 1
  br label %189

196:                                              ; preds = %189
  %197 = xor i32 %190, -1
  %198 = add i32 %51, %197
  %199 = icmp sgt i32 %198, 127
  br i1 %199, label %200, label %210

200:                                              ; preds = %196
  %201 = icmp sgt i32 %12, -1
  br i1 %201, label %202, label %206

202:                                              ; preds = %200
  %203 = and i32 %2, -3
  %204 = icmp eq i32 %203, 1
  %205 = select i1 %204, float 0x47EFFFFFE0000000, float 0x7FF0000000000000
  br label %281

206:                                              ; preds = %200
  %207 = add i32 %2, -1
  %208 = icmp ult i32 %207, 2
  %209 = select i1 %208, float 0xC7EFFFFFE0000000, float 0xFFF0000000000000
  br label %281

210:                                              ; preds = %196
  %211 = icmp sgt i32 %198, -127
  br i1 %211, label %212, label %245

212:                                              ; preds = %210
  %213 = add nsw i32 %198, 127, !spirv.Decorations !276
  %214 = shl i32 %53, %190
  %215 = call spir_func i32 @_ZL12fra_uint_divIjET_S0_S0_j(i32 %214, i32 %55, i32 27) #0
  %216 = lshr i32 %215, 3
  %217 = and i32 %216, 8388607
  %218 = and i32 %215, 7
  %219 = call spir_func i32 @_ZL19__handling_roundingIjET_S0_S0_ji(i32 %13, i32 %217, i32 %218, i32 %2) #0
  %220 = icmp eq i32 %219, 0
  br i1 %220, label %237, label %221

221:                                              ; preds = %212
  %222 = add nuw nsw i32 %217, 1, !spirv.Decorations !278
  %223 = icmp eq i32 %217, 8388607
  br i1 %223, label %224, label %237

224:                                              ; preds = %221
  %225 = add nsw i32 %198, 128, !spirv.Decorations !276
  %226 = icmp eq i32 %225, 255
  br i1 %226, label %227, label %237

227:                                              ; preds = %224
  %228 = icmp sgt i32 %12, -1
  br i1 %228, label %229, label %233

229:                                              ; preds = %227
  %230 = and i32 %2, -3
  %231 = icmp eq i32 %230, 1
  %232 = select i1 %231, float 0x47EFFFFFE0000000, float 0x7FF0000000000000
  br label %281

233:                                              ; preds = %227
  %234 = add i32 %2, -1
  %235 = icmp ult i32 %234, 2
  %236 = select i1 %235, float 0xC7EFFFFFE0000000, float 0xFFF0000000000000
  br label %281

237:                                              ; preds = %224, %221, %212
  %238 = phi i32 [ 0, %224 ], [ %222, %221 ], [ %217, %212 ]
  %239 = phi i32 [ %225, %224 ], [ %213, %221 ], [ %213, %212 ]
  %240 = and i32 %12, -2147483648
  %241 = shl nuw nsw i32 %239, 23, !spirv.Decorations !278
  %242 = or i32 %240, %241
  %243 = or i32 %242, %238
  %244 = bitcast i32 %243 to float
  br label %281

245:                                              ; preds = %210
  %246 = add i32 %190, -127
  %247 = sub i32 %246, %51
  %248 = add i32 %247, 1
  %249 = icmp ugt i32 %248, 22
  br i1 %249, label %250, label %263

250:                                              ; preds = %245
  %251 = icmp eq i32 %248, 23
  %252 = add i32 %190, 1
  %253 = shl i32 %53, %252
  %254 = icmp ugt i32 %253, %55
  %255 = select i1 %251, i1 %254, i1 false
  %256 = icmp sgt i32 %12, -1
  br i1 %256, label %257, label %260

257:                                              ; preds = %250
  switch i32 %2, label %259 [
    i32 2, label %281
    i32 0, label %258
  ]

258:                                              ; preds = %257
  br i1 %255, label %281, label %259

259:                                              ; preds = %258, %257
  br label %281

260:                                              ; preds = %250
  switch i32 %2, label %262 [
    i32 3, label %281
    i32 0, label %261
  ]

261:                                              ; preds = %260
  br i1 %255, label %281, label %262

262:                                              ; preds = %261, %260
  br label %281

263:                                              ; preds = %245
  %264 = shl i32 %53, %190
  %265 = sub nsw i32 25, %247, !spirv.Decorations !276
  %266 = call spir_func i32 @_ZL12fra_uint_divIjET_S0_S0_j(i32 %264, i32 %55, i32 %265) #0
  %267 = lshr i32 %266, 3
  %268 = and i32 %266, 7
  %269 = call spir_func i32 @_ZL19__handling_roundingIjET_S0_S0_ji(i32 %13, i32 %267, i32 %268, i32 %2) #0
  %270 = icmp eq i32 %269, 0
  %271 = add nuw nsw i32 %267, 1, !spirv.Decorations !278
  %272 = icmp ugt i32 %266, 67108855
  %273 = select i1 %272, i32 0, i32 %271
  %274 = select i1 %272, i32 8388608, i32 0
  %275 = select i1 %270, i32 %267, i32 %273
  %276 = select i1 %270, i32 0, i32 %274
  %277 = and i32 %12, -2147483648
  %278 = or i32 %277, %276
  %279 = or i32 %278, %275
  %280 = bitcast i32 %279 to float
  br label %281

281:                                              ; preds = %263, %262, %261, %260, %259, %258, %257, %237, %233, %229, %206, %202, %175, %142, %141, %140, %139, %138, %137, %116, %112, %108, %82, %78, %41, %35, %26, %17, %3
  %282 = phi float [ %37, %35 ], [ %43, %41 ], [ 0x7FF8000000000000, %17 ], [ 0x7FF8000000000000, %3 ], [ %32, %26 ], [ %244, %237 ], [ %280, %263 ], [ %188, %175 ], [ %123, %116 ], [ %81, %78 ], [ %85, %82 ], [ %111, %108 ], [ %115, %112 ], [ 0.000000e+00, %139 ], [ -0.000000e+00, %142 ], [ 0x36A0000000000000, %137 ], [ 0x36A0000000000000, %138 ], [ 0xB6A0000000000000, %140 ], [ 0xB6A0000000000000, %141 ], [ %205, %202 ], [ %209, %206 ], [ %232, %229 ], [ %236, %233 ], [ 0.000000e+00, %259 ], [ -0.000000e+00, %262 ], [ 0x36A0000000000000, %257 ], [ 0x36A0000000000000, %258 ], [ 0xB6A0000000000000, %260 ], [ 0xB6A0000000000000, %261 ]
  ret float %282
}

; Function Attrs: nounwind
define internal spir_func i32 @_ZL12fra_uint_divIjET_S0_S0_j(i32 %0, i32 %1, i32 %2) #0 {
  %4 = icmp eq i32 %0, 0
  br i1 %4, label %30, label %.preheader

.preheader:                                       ; preds = %3
  br label %5

5:                                                ; preds = %24, %.preheader
  %6 = phi i32 [ %25, %24 ], [ %0, %.preheader ]
  %7 = phi i32 [ %26, %24 ], [ 0, %.preheader ]
  %8 = phi i32 [ %27, %24 ], [ 0, %.preheader ]
  %9 = icmp ult i32 %8, %2
  br i1 %9, label %10, label %28

10:                                               ; preds = %5
  %11 = shl i32 %7, 1
  %12 = shl i32 %6, 1
  %13 = icmp ugt i32 %12, %1
  br i1 %13, label %14, label %17

14:                                               ; preds = %10
  %15 = sub nuw i32 %12, %1, !spirv.Decorations !280
  %16 = or i32 %11, 1
  br label %24

17:                                               ; preds = %10
  %18 = icmp eq i32 %12, %1
  br i1 %18, label %19, label %24

19:                                               ; preds = %17
  %20 = or i32 %11, 1
  %21 = xor i32 %8, -1
  %22 = add i32 %2, %21
  %23 = shl i32 %20, %22
  br label %30

24:                                               ; preds = %17, %14
  %25 = phi i32 [ %15, %14 ], [ %12, %17 ]
  %26 = phi i32 [ %16, %14 ], [ %11, %17 ]
  %27 = add nuw i32 %8, 1, !spirv.Decorations !280
  br label %5

28:                                               ; preds = %5
  %29 = or i32 %7, 1
  br label %30

30:                                               ; preds = %28, %19, %3
  %31 = phi i32 [ %23, %19 ], [ %29, %28 ], [ 0, %3 ]
  ret i32 %31
}

; Function Attrs: nounwind
define internal spir_func i32 @_ZL19__handling_roundingIjET_S0_S0_ji(i32 %0, i32 %1, i32 %2, i32 %3) #0 {
  %5 = icmp eq i32 %2, 0
  br i1 %5, label %24, label %6

6:                                                ; preds = %4
  %7 = icmp eq i32 %3, 2
  %8 = icmp eq i32 %0, 0
  %9 = and i1 %7, %8
  br i1 %9, label %24, label %10

10:                                               ; preds = %6
  %11 = icmp eq i32 %3, 3
  %12 = icmp eq i32 %0, 1
  %13 = and i1 %11, %12
  br i1 %13, label %24, label %14

14:                                               ; preds = %10
  %15 = icmp eq i32 %3, 0
  br i1 %15, label %16, label %23

16:                                               ; preds = %14
  %17 = icmp ugt i32 %2, 4
  br i1 %17, label %24, label %18

18:                                               ; preds = %16
  %19 = icmp ne i32 %2, 4
  %20 = and i32 %1, 1
  %21 = icmp eq i32 %20, 0
  %22 = or i1 %19, %21
  br i1 %22, label %23, label %24

23:                                               ; preds = %18, %14
  br label %24

24:                                               ; preds = %23, %18, %16, %10, %6, %4
  %25 = phi i32 [ 0, %23 ], [ 0, %4 ], [ 1, %6 ], [ 1, %10 ], [ 1, %18 ], [ 1, %16 ]
  ret i32 %25
}

; Function Attrs: nounwind readnone willreturn
declare spir_func i64 @_Z32__spirv_BuiltInLocalInvocationIdi(i32) #2

attributes #0 = { nounwind }
attributes #1 = { alwaysinline nounwind }
attributes #2 = { nounwind readnone willreturn }

!spirv.MemoryModel = !{!3}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!4}
!opencl.spir.version = !{!5}
!opencl.ocl.version = !{!6}
!opencl.used.extensions = !{!7}
!opencl.used.optional.core.features = !{!8}
!spirv.Generator = !{!9}
!opencl.compiler.options = !{!7}
!igc.spirv.extensions = !{!10}
!igc.functions = !{}
!IGCMetadata = !{!11}

!0 = !{!1, !2}
!1 = !{i32 41, !"global_smem", i32 1}
!2 = !{i32 44, i32 16}
!3 = !{i32 2, i32 2}
!4 = !{i32 3, i32 100000}
!5 = !{i32 1, i32 2}
!6 = !{i32 1, i32 0}
!7 = !{}
!8 = !{!"cl_doubles"}
!9 = !{i16 6, i16 14}
!10 = !{!"SPV_INTEL_vector_compute"}
!11 = !{!"ModuleMD", !12, !13, !101, !102, !133, !150, !173, !183, !185, !186, !201, !202, !203, !204, !208, !209, !216, !217, !218, !219, !220, !221, !222, !223, !224, !225, !226, !228, !232, !233, !234, !235, !236, !237, !238, !239, !240, !241, !242, !243, !244, !245, !246, !247, !248, !250, !252, !255, !256, !257, !259, !260, !261, !266, !267, !268, !269}
!12 = !{!"isPrecise", i1 false}
!13 = !{!"compOpt", !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100}
!14 = !{!"DenormsAreZero", i1 false}
!15 = !{!"BFTFDenormsAreZero", i1 false}
!16 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!17 = !{!"OptDisable", i1 false}
!18 = !{!"MadEnable", i1 false}
!19 = !{!"NoSignedZeros", i1 false}
!20 = !{!"NoNaNs", i1 false}
!21 = !{!"FloatRoundingMode", i32 0}
!22 = !{!"FloatCvtIntRoundingMode", i32 3}
!23 = !{!"LoadCacheDefault", i32 -1}
!24 = !{!"StoreCacheDefault", i32 -1}
!25 = !{!"VISAPreSchedRPThreshold", i32 0}
!26 = !{!"SetLoopUnrollThreshold", i32 0}
!27 = !{!"UnsafeMathOptimizations", i1 false}
!28 = !{!"disableCustomUnsafeOpts", i1 false}
!29 = !{!"disableReducePow", i1 false}
!30 = !{!"disableSqrtOpt", i1 false}
!31 = !{!"FiniteMathOnly", i1 false}
!32 = !{!"FastRelaxedMath", i1 false}
!33 = !{!"DashGSpecified", i1 false}
!34 = !{!"FastCompilation", i1 false}
!35 = !{!"UseScratchSpacePrivateMemory", i1 true}
!36 = !{!"RelaxedBuiltins", i1 false}
!37 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!38 = !{!"GreaterThan2GBBufferRequired", i1 true}
!39 = !{!"GreaterThan4GBBufferRequired", i1 true}
!40 = !{!"DisableA64WA", i1 false}
!41 = !{!"ForceEnableA64WA", i1 false}
!42 = !{!"PushConstantsEnable", i1 true}
!43 = !{!"HasPositivePointerOffset", i1 false}
!44 = !{!"HasBufferOffsetArg", i1 false}
!45 = !{!"BufferOffsetArgOptional", i1 true}
!46 = !{!"replaceGlobalOffsetsByZero", i1 false}
!47 = !{!"forcePixelShaderSIMDMode", i32 0}
!48 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!49 = !{!"UniformWGS", i1 false}
!50 = !{!"disableVertexComponentPacking", i1 false}
!51 = !{!"disablePartialVertexComponentPacking", i1 false}
!52 = !{!"PreferBindlessImages", i1 false}
!53 = !{!"UseBindlessMode", i1 false}
!54 = !{!"UseLegacyBindlessMode", i1 true}
!55 = !{!"disableMathRefactoring", i1 false}
!56 = !{!"atomicBranch", i1 false}
!57 = !{!"spillCompression", i1 false}
!58 = !{!"DisableEarlyOut", i1 false}
!59 = !{!"ForceInt32DivRemEmu", i1 false}
!60 = !{!"ForceInt32DivRemEmuSP", i1 false}
!61 = !{!"DisableFastestSingleCSSIMD", i1 false}
!62 = !{!"DisableFastestLinearScan", i1 false}
!63 = !{!"UseStatelessforPrivateMemory", i1 false}
!64 = !{!"EnableTakeGlobalAddress", i1 false}
!65 = !{!"IsLibraryCompilation", i1 false}
!66 = !{!"LibraryCompileSIMDSize", i32 0}
!67 = !{!"FastVISACompile", i1 false}
!68 = !{!"MatchSinCosPi", i1 false}
!69 = !{!"ExcludeIRFromZEBinary", i1 false}
!70 = !{!"EmitZeBinVISASections", i1 false}
!71 = !{!"FP64GenEmulationEnabled", i1 false}
!72 = !{!"FP64GenConvEmulationEnabled", i1 false}
!73 = !{!"allowDisableRematforCS", i1 false}
!74 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!75 = !{!"DisableCPSOmaskWA", i1 false}
!76 = !{!"DisableFastestGopt", i1 false}
!77 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!78 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!79 = !{!"DisableConstantCoalescing", i1 false}
!80 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!81 = !{!"WaEnableALTModeVisaWA", i1 false}
!82 = !{!"EnableLdStCombineforLoad", i1 false}
!83 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!84 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!85 = !{!"NewSpillCostFunction", i1 false}
!86 = !{!"EnableVRT", i1 false}
!87 = !{!"ForceLargeGRFNum4RQ", i1 false}
!88 = !{!"DisableEUFusion", i1 false}
!89 = !{!"DisableFDivToFMulInvOpt", i1 false}
!90 = !{!"initializePhiSampleSourceWA", i1 false}
!91 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!92 = !{!"DisableLoosenSimd32Occu", i1 false}
!93 = !{!"FastestS1Options", i32 0}
!94 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!95 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!96 = !{!"DisableLscSamplerRouting", i1 false}
!97 = !{!"UseBarrierControlFlowOptimization", i1 false}
!98 = !{!"EnableDynamicRQManagement", i1 false}
!99 = !{!"Quad8InputThreshold", i32 0}
!100 = !{!"UseResourceLoopUnrollNested", i1 false}
!101 = !{!"FuncMD"}
!102 = !{!"pushInfo", !103, !104, !105, !109, !110, !111, !112, !113, !114, !115, !116, !129, !130, !131, !132}
!103 = !{!"pushableAddresses"}
!104 = !{!"bindlessPushInfo"}
!105 = !{!"dynamicBufferInfo", !106, !107, !108}
!106 = !{!"firstIndex", i32 0}
!107 = !{!"numOffsets", i32 0}
!108 = !{!"forceDisabled", i1 false}
!109 = !{!"MaxNumberOfPushedBuffers", i32 0}
!110 = !{!"inlineConstantBufferSlot", i32 -1}
!111 = !{!"inlineConstantBufferOffset", i32 -1}
!112 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!113 = !{!"constants"}
!114 = !{!"inputs"}
!115 = !{!"constantReg"}
!116 = !{!"simplePushInfoArr", !117, !126, !127, !128}
!117 = !{!"simplePushInfoArrVec[0]", !118, !119, !120, !121, !122, !123, !124, !125}
!118 = !{!"cbIdx", i32 0}
!119 = !{!"pushableAddressGrfOffset", i32 -1}
!120 = !{!"pushableOffsetGrfOffset", i32 -1}
!121 = !{!"offset", i32 0}
!122 = !{!"size", i32 0}
!123 = !{!"isStateless", i1 false}
!124 = !{!"isBindless", i1 false}
!125 = !{!"simplePushLoads"}
!126 = !{!"simplePushInfoArrVec[1]", !118, !119, !120, !121, !122, !123, !124, !125}
!127 = !{!"simplePushInfoArrVec[2]", !118, !119, !120, !121, !122, !123, !124, !125}
!128 = !{!"simplePushInfoArrVec[3]", !118, !119, !120, !121, !122, !123, !124, !125}
!129 = !{!"simplePushBufferUsed", i32 0}
!130 = !{!"pushAnalysisWIInfos"}
!131 = !{!"inlineRTGlobalPtrOffset", i32 0}
!132 = !{!"rtSyncSurfPtrOffset", i32 0}
!133 = !{!"psInfo", !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149}
!134 = !{!"BlendStateDisabledMask", i8 0}
!135 = !{!"SkipSrc0Alpha", i1 false}
!136 = !{!"DualSourceBlendingDisabled", i1 false}
!137 = !{!"ForceEnableSimd32", i1 false}
!138 = !{!"DisableSimd32WithDiscard", i1 false}
!139 = !{!"outputDepth", i1 false}
!140 = !{!"outputStencil", i1 false}
!141 = !{!"outputMask", i1 false}
!142 = !{!"blendToFillEnabled", i1 false}
!143 = !{!"forceEarlyZ", i1 false}
!144 = !{!"hasVersionedLoop", i1 false}
!145 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!146 = !{!"NumSamples", i8 0}
!147 = !{!"blendOptimizationMode"}
!148 = !{!"colorOutputMask"}
!149 = !{!"WaDisableVRS", i1 false}
!150 = !{!"csInfo", !151, !152, !153, !154, !155, !25, !26, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !57, !169, !170, !171, !172}
!151 = !{!"maxWorkGroupSize", i32 0}
!152 = !{!"waveSize", i32 0}
!153 = !{!"ComputeShaderSecondCompile"}
!154 = !{!"forcedSIMDSize", i8 0}
!155 = !{!"forceTotalGRFNum", i32 0}
!156 = !{!"forceSpillCompression", i1 false}
!157 = !{!"allowLowerSimd", i1 false}
!158 = !{!"disableSimd32Slicing", i1 false}
!159 = !{!"disableSplitOnSpill", i1 false}
!160 = !{!"enableNewSpillCostFunction", i1 false}
!161 = !{!"forceVISAPreSched", i1 false}
!162 = !{!"forceUniformBuffer", i1 false}
!163 = !{!"forceUniformSurfaceSampler", i1 false}
!164 = !{!"disableLocalIdOrderOptimizations", i1 false}
!165 = !{!"disableDispatchAlongY", i1 false}
!166 = !{!"neededThreadIdLayout", i1* null}
!167 = !{!"forceTileYWalk", i1 false}
!168 = !{!"atomicBranch", i32 0}
!169 = !{!"disableEarlyOut", i1 false}
!170 = !{!"walkOrderEnabled", i1 false}
!171 = !{!"walkOrderOverride", i32 0}
!172 = !{!"ResForHfPacking"}
!173 = !{!"msInfo", !174, !175, !176, !177, !178, !179, !180, !181, !182}
!174 = !{!"PrimitiveTopology", i32 3}
!175 = !{!"MaxNumOfPrimitives", i32 0}
!176 = !{!"MaxNumOfVertices", i32 0}
!177 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!178 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!179 = !{!"WorkGroupSize", i32 0}
!180 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!181 = !{!"IndexFormat", i32 6}
!182 = !{!"SubgroupSize", i32 0}
!183 = !{!"taskInfo", !184, !179, !180, !182}
!184 = !{!"MaxNumOfOutputs", i32 0}
!185 = !{!"NBarrierCnt", i32 0}
!186 = !{!"rtInfo", !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200}
!187 = !{!"RayQueryAllocSizeInBytes", i32 0}
!188 = !{!"NumContinuations", i32 0}
!189 = !{!"RTAsyncStackAddrspace", i32 -1}
!190 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!191 = !{!"SWHotZoneAddrspace", i32 -1}
!192 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!193 = !{!"SWStackAddrspace", i32 -1}
!194 = !{!"SWStackSurfaceStateOffset", i1* null}
!195 = !{!"RTSyncStackAddrspace", i32 -1}
!196 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!197 = !{!"doSyncDispatchRays", i1 false}
!198 = !{!"MemStyle", !"Xe"}
!199 = !{!"GlobalDataStyle", !"Xe"}
!200 = !{!"uberTileDimensions", i1* null}
!201 = !{!"CurUniqueIndirectIdx", i32 0}
!202 = !{!"inlineDynTextures"}
!203 = !{!"inlineResInfoData"}
!204 = !{!"immConstant", !205, !206, !207}
!205 = !{!"data"}
!206 = !{!"sizes"}
!207 = !{!"zeroIdxs"}
!208 = !{!"stringConstants"}
!209 = !{!"inlineBuffers", !210, !214, !215}
!210 = !{!"inlineBuffersVec[0]", !211, !212, !213}
!211 = !{!"alignment", i32 0}
!212 = !{!"allocSize", i64 0}
!213 = !{!"Buffer"}
!214 = !{!"inlineBuffersVec[1]", !211, !212, !213}
!215 = !{!"inlineBuffersVec[2]", !211, !212, !213}
!216 = !{!"GlobalPointerProgramBinaryInfos"}
!217 = !{!"ConstantPointerProgramBinaryInfos"}
!218 = !{!"GlobalBufferAddressRelocInfo"}
!219 = !{!"ConstantBufferAddressRelocInfo"}
!220 = !{!"forceLscCacheList"}
!221 = !{!"SrvMap"}
!222 = !{!"RasterizerOrderedByteAddressBuffer"}
!223 = !{!"RasterizerOrderedViews"}
!224 = !{!"MinNOSPushConstantSize", i32 0}
!225 = !{!"inlineProgramScopeOffsets"}
!226 = !{!"shaderData", !227}
!227 = !{!"numReplicas", i32 0}
!228 = !{!"URBInfo", !229, !230, !231}
!229 = !{!"has64BVertexHeaderInput", i1 false}
!230 = !{!"has64BVertexHeaderOutput", i1 false}
!231 = !{!"hasVertexHeader", i1 true}
!232 = !{!"UseBindlessImage", i1 false}
!233 = !{!"enableRangeReduce", i1 false}
!234 = !{!"allowMatchMadOptimizationforVS", i1 false}
!235 = !{!"disableMatchMadOptimizationForCS", i1 false}
!236 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!237 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!238 = !{!"statefulResourcesNotAliased", i1 false}
!239 = !{!"disableMixMode", i1 false}
!240 = !{!"genericAccessesResolved", i1 false}
!241 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!242 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!243 = !{!"disableSeparateScratchWA", i1 false}
!244 = !{!"enableRemoveUnusedTGMFence", i1 false}
!245 = !{!"privateMemoryPerWI", i32 0}
!246 = !{!"PrivateMemoryPerFG"}
!247 = !{!"m_OptsToDisable"}
!248 = !{!"capabilities", !249}
!249 = !{!"globalVariableDecorationsINTEL", i1 false}
!250 = !{!"extensions", !251}
!251 = !{!"spvINTELBindlessImages", i1 false}
!252 = !{!"m_ShaderResourceViewMcsMask", !253, !254}
!253 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!254 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!255 = !{!"computedDepthMode", i32 0}
!256 = !{!"isHDCFastClearShader", i1 false}
!257 = !{!"argRegisterReservations", !258}
!258 = !{!"argRegisterReservationsVec[0]", i32 0}
!259 = !{!"SIMD16_SpillThreshold", i8 0}
!260 = !{!"SIMD32_SpillThreshold", i8 0}
!261 = !{!"m_CacheControlOption", !262, !263, !264, !265}
!262 = !{!"LscLoadCacheControlOverride", i8 0}
!263 = !{!"LscStoreCacheControlOverride", i8 0}
!264 = !{!"TgmLoadCacheControlOverride", i8 0}
!265 = !{!"TgmStoreCacheControlOverride", i8 0}
!266 = !{!"ModuleUsesBindless", i1 false}
!267 = !{!"predicationMap"}
!268 = !{!"lifeTimeStartMap"}
!269 = !{!"HitGroups"}
!270 = !{i32 1, i32 1, i32 1, i32 1, i32 1, i32 1}
!271 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!272 = !{!"float*", !"float*", !"float*", !"float*", !"char*", !"char*"}
!273 = !{!"", !"", !"", !"", !"", !""}
!274 = !{i32 128, i32 1, i32 1}
!275 = !{i32 32}
!276 = !{!277}
!277 = !{i32 4469}
!278 = !{!277, !279}
!279 = !{i32 4470}
!280 = !{!279}
