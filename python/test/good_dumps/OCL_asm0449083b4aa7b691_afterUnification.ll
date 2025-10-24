; ------------------------------------------------
; OCL_asm0449083b4aa7b691_afterUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

@0 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"

; Function Attrs: convergent nounwind
define spir_kernel void @kernel(float addrspace(1)* align 4 %0, float addrspace(1)* align 4 %1, float addrspace(1)* align 4 %2, float addrspace(1)* align 4 %3, i8 addrspace(1)* align 1 %4, i8 addrspace(1)* align 1 %5, <8 x i32> %r0, <8 x i32> %payloadHeader, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bufferOffset5) #0 {
  %7 = and i16 %localIdX, 127
  %8 = zext i16 %7 to i64
  %9 = getelementptr float, float addrspace(1)* %0, i64 %8
  %10 = bitcast float addrspace(1)* %9 to <1 x float> addrspace(1)*
  %11 = load <1 x float>, <1 x float> addrspace(1)* %10, align 4
  %12 = extractelement <1 x float> %11, i32 0
  %13 = zext i16 %7 to i64
  %14 = getelementptr float, float addrspace(1)* %1, i64 %13
  %15 = bitcast float addrspace(1)* %14 to <1 x float> addrspace(1)*
  %16 = load <1 x float>, <1 x float> addrspace(1)* %15, align 4
  %17 = extractelement <1 x float> %16, i32 0
  %18 = bitcast float %12 to i32
  %19 = bitcast float %17 to i32
  %20 = lshr i32 %18, 23
  %21 = and i32 %20, 255
  %22 = lshr i32 %19, 23
  %23 = and i32 %22, 255
  %24 = and i32 %18, 8388607
  %25 = and i32 %19, 8388607
  %26 = xor i32 %18, %19
  %27 = icmp eq i32 %21, 255
  %28 = icmp ne i32 %24, 0
  %29 = and i1 %27, %28
  br i1 %29, label %__imf_fdiv_rn.exit, label %30

30:                                               ; preds = %6
  %31 = icmp eq i32 %23, 255
  %32 = icmp ne i32 %25, 0
  %33 = and i1 %31, %32
  %34 = fcmp oeq float %17, 0.000000e+00
  %35 = or i1 %33, %34
  br i1 %35, label %__imf_fdiv_rn.exit, label %36

36:                                               ; preds = %30
  %37 = icmp eq i32 %24, 0
  %38 = and i1 %27, %37
  br i1 %38, label %39, label %46

39:                                               ; preds = %36
  %40 = icmp eq i32 %25, 0
  %41 = and i1 %31, %40
  %42 = and i32 %26, -2147483648
  %43 = or i32 %42, 2139095040
  %44 = bitcast i32 %43 to float
  %45 = select i1 %41, float 0x7FF8000000000000, float %44
  br label %__imf_fdiv_rn.exit

46:                                               ; preds = %36
  %47 = fcmp oeq float %12, 0.000000e+00
  br i1 %47, label %48, label %51

48:                                               ; preds = %46
  %49 = and i32 %26, -2147483648
  %50 = bitcast i32 %49 to float
  br label %__imf_fdiv_rn.exit

51:                                               ; preds = %46
  %52 = icmp eq i32 %25, 0
  %53 = and i1 %31, %52
  br i1 %53, label %54, label %57

54:                                               ; preds = %51
  %55 = and i32 %26, -2147483648
  %56 = bitcast i32 %55 to float
  br label %__imf_fdiv_rn.exit

57:                                               ; preds = %51
  %58 = icmp eq i32 %21, 0
  %59 = add nsw i32 %21, -127, !spirv.Decorations !405
  %60 = select i1 %58, i32 -126, i32 %59
  %61 = icmp eq i32 %23, 0
  %62 = add nsw i32 %23, -127, !spirv.Decorations !405
  %63 = select i1 %61, i32 -126, i32 %62
  %64 = sub nsw i32 %60, %63, !spirv.Decorations !405
  %65 = or i32 %24, 8388608
  %66 = select i1 %58, i32 %24, i32 %65
  %67 = or i32 %25, 8388608
  %68 = select i1 %61, i32 %25, i32 %67
  %69 = icmp ult i32 %66, %68
  br i1 %69, label %.preheader.i.i.i, label %70

.preheader.i.i.i:                                 ; preds = %57
  br label %263

70:                                               ; preds = %57
  %71 = udiv i32 %66, %68
  %freeze = freeze i32 %71
  %72 = mul i32 %68, %freeze
  %73 = sub i32 %66, %72
  br label %74

74:                                               ; preds = %81, %70
  %75 = phi i64 [ 0, %70 ], [ %83, %81 ]
  %76 = phi i32 [ -2147483648, %70 ], [ %82, %81 ]
  %77 = icmp ugt i64 %75, 31
  %78 = and i32 %freeze, %76
  %79 = icmp eq i32 %78, %76
  %80 = select i1 %77, i1 true, i1 %79
  br i1 %80, label %84, label %81

81:                                               ; preds = %74
  %82 = lshr i32 %76, 1
  %83 = add nuw nsw i64 %75, 1, !spirv.Decorations !407
  br label %74

84:                                               ; preds = %74
  %85 = trunc i64 %75 to i32
  %86 = sub nsw i32 31, %85, !spirv.Decorations !405
  %87 = add nsw i32 %64, %86, !spirv.Decorations !405
  %88 = icmp sgt i32 %87, 127
  br i1 %88, label %89, label %93

89:                                               ; preds = %84
  %90 = icmp sgt i32 %26, -1
  br i1 %90, label %91, label %92

91:                                               ; preds = %89
  br label %__imf_fdiv_rn.exit

92:                                               ; preds = %89
  br label %__imf_fdiv_rn.exit

93:                                               ; preds = %84
  %94 = icmp sgt i32 %87, -127
  br i1 %94, label %95, label %159

95:                                               ; preds = %93
  %96 = add nsw i32 %87, 127, !spirv.Decorations !405
  %97 = add nsw i32 %85, -8, !spirv.Decorations !405
  %98 = shl i32 %freeze, %97
  %99 = and i32 %98, 8388607
  %100 = add nsw i32 %85, -5, !spirv.Decorations !405
  %101 = icmp eq i32 %73, 0
  br i1 %101, label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i, label %.preheader.i.i.i.i

.preheader.i.i.i.i:                               ; preds = %95
  br label %102

102:                                              ; preds = %121, %.preheader.i.i.i.i
  %103 = phi i32 [ %122, %121 ], [ %73, %.preheader.i.i.i.i ]
  %104 = phi i32 [ %123, %121 ], [ 0, %.preheader.i.i.i.i ]
  %105 = phi i32 [ %124, %121 ], [ 0, %.preheader.i.i.i.i ]
  %106 = icmp ult i32 %105, %100
  br i1 %106, label %107, label %125

107:                                              ; preds = %102
  %108 = shl i32 %104, 1
  %109 = shl i32 %103, 1
  %110 = icmp ugt i32 %109, %68
  br i1 %110, label %111, label %114

111:                                              ; preds = %107
  %112 = sub nuw i32 %109, %68, !spirv.Decorations !409
  %113 = or i32 %108, 1
  br label %121

114:                                              ; preds = %107
  %115 = icmp eq i32 %109, %68
  br i1 %115, label %116, label %121

116:                                              ; preds = %114
  %117 = or i32 %108, 1
  %118 = xor i32 %105, -1
  %119 = add i32 %100, %118
  %120 = shl i32 %117, %119
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i

121:                                              ; preds = %114, %111
  %122 = phi i32 [ %112, %111 ], [ %109, %114 ]
  %123 = phi i32 [ %113, %111 ], [ %108, %114 ]
  %124 = add nuw i32 %105, 1, !spirv.Decorations !409
  br label %102

125:                                              ; preds = %102
  %126 = or i32 %104, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i:         ; preds = %125, %116, %95
  %127 = phi i32 [ %120, %116 ], [ %126, %125 ], [ 0, %95 ]
  %128 = lshr i32 %127, 3
  %129 = or i32 %99, %128
  %130 = and i32 %127, 7
  %131 = icmp eq i32 %130, 0
  br i1 %131, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i, label %132

132:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  %133 = icmp ugt i32 %130, 4
  br i1 %133, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i, label %134

134:                                              ; preds = %132
  %135 = icmp ne i32 %130, 4
  %136 = and i32 %129, 1
  %137 = icmp eq i32 %136, 0
  %138 = or i1 %135, %137
  br i1 %138, label %139, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i

139:                                              ; preds = %134
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i: ; preds = %139, %134, %132, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  %140 = phi i1 [ true, %139 ], [ false, %134 ], [ false, %132 ], [ true, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i ]
  br i1 %140, label %151, label %141

141:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
  %142 = add nuw nsw i32 %129, 1, !spirv.Decorations !407
  %143 = icmp ugt i32 %129, 8388606
  br i1 %143, label %144, label %151

144:                                              ; preds = %141
  %145 = add nsw i32 %87, 128, !spirv.Decorations !405
  %146 = icmp eq i32 %145, 255
  br i1 %146, label %147, label %151

147:                                              ; preds = %144
  %148 = icmp sgt i32 %26, -1
  br i1 %148, label %149, label %150

149:                                              ; preds = %147
  br label %__imf_fdiv_rn.exit

150:                                              ; preds = %147
  br label %__imf_fdiv_rn.exit

151:                                              ; preds = %144, %141, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
  %152 = phi i32 [ %142, %144 ], [ %142, %141 ], [ %129, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i ]
  %153 = phi i32 [ %145, %144 ], [ %96, %141 ], [ %96, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i ]
  %154 = and i32 %26, -2147483648
  %155 = shl nuw nsw i32 %153, 23, !spirv.Decorations !407
  %156 = or i32 %154, %155
  %157 = or i32 %156, %152
  %158 = bitcast i32 %157 to float
  br label %__imf_fdiv_rn.exit

159:                                              ; preds = %93
  %160 = xor i32 %87, -1
  %161 = icmp ult i32 %87, -149
  br i1 %161, label %162, label %176

162:                                              ; preds = %159
  %163 = icmp eq i32 %87, -150
  br i1 %163, label %164, label %169

164:                                              ; preds = %162
  %165 = icmp ne i32 %66, %72
  %166 = lshr i32 -2147483648, %85
  %167 = icmp ne i32 %freeze, %166
  %168 = select i1 %165, i1 true, i1 %167
  br label %169

169:                                              ; preds = %164, %162
  %170 = phi i1 [ %168, %164 ], [ false, %162 ]
  %171 = icmp sgt i32 %26, -1
  br i1 %171, label %172, label %174

172:                                              ; preds = %169
  br i1 %170, label %__imf_fdiv_rn.exit, label %173

173:                                              ; preds = %172
  br label %__imf_fdiv_rn.exit

174:                                              ; preds = %169
  br i1 %170, label %__imf_fdiv_rn.exit, label %175

175:                                              ; preds = %174
  br label %__imf_fdiv_rn.exit

176:                                              ; preds = %159
  %177 = add nsw i32 %87, 152, !spirv.Decorations !405
  %178 = icmp sgt i32 %177, %86
  br i1 %178, label %209, label %179

179:                                              ; preds = %176
  %180 = sub nsw i32 %160, %85, !spirv.Decorations !405
  %181 = add nsw i32 %180, -117, !spirv.Decorations !405
  %182 = lshr i32 %freeze, %181
  %183 = add nsw i32 %180, -120, !spirv.Decorations !405
  %184 = lshr i32 %freeze, %183
  %185 = and i32 %184, 7
  %186 = and i32 %184, 1
  %187 = icmp eq i32 %186, 0
  br i1 %187, label %188, label %197

188:                                              ; preds = %179
  %189 = shl nsw i32 -1, %183, !spirv.Decorations !405
  %190 = xor i32 %189, -1
  %191 = and i32 %freeze, %190
  %192 = icmp ne i32 %191, 0
  %193 = icmp ne i32 %66, %72
  %194 = or i1 %192, %193
  %195 = zext i1 %194 to i32
  %196 = or i32 %185, %195
  br label %197

197:                                              ; preds = %188, %179
  %198 = phi i32 [ %185, %179 ], [ %196, %188 ]
  %199 = icmp eq i32 %198, 0
  br i1 %199, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i, label %200

200:                                              ; preds = %197
  %201 = icmp ugt i32 %198, 4
  br i1 %201, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i, label %202

202:                                              ; preds = %200
  %203 = icmp ne i32 %198, 4
  %204 = and i32 %182, 1
  %205 = icmp eq i32 %204, 0
  %206 = or i1 %203, %205
  br i1 %206, label %207, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i

207:                                              ; preds = %202
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i: ; preds = %207, %202, %200, %197
  %208 = phi i32 [ 0, %207 ], [ 0, %197 ], [ 1, %202 ], [ 1, %200 ]
  br label %249

209:                                              ; preds = %176
  %210 = sub nsw i32 %177, %86, !spirv.Decorations !405
  %211 = shl i32 %freeze, %210
  %212 = icmp eq i32 %73, 0
  br i1 %212, label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, label %.preheader.i7.i.i.i

.preheader.i7.i.i.i:                              ; preds = %209
  br label %213

213:                                              ; preds = %232, %.preheader.i7.i.i.i
  %214 = phi i32 [ %233, %232 ], [ %73, %.preheader.i7.i.i.i ]
  %215 = phi i32 [ %234, %232 ], [ 0, %.preheader.i7.i.i.i ]
  %216 = phi i32 [ %235, %232 ], [ 0, %.preheader.i7.i.i.i ]
  %217 = icmp ult i32 %216, %210
  br i1 %217, label %218, label %236

218:                                              ; preds = %213
  %219 = shl i32 %215, 1
  %220 = shl i32 %214, 1
  %221 = icmp ugt i32 %220, %68
  br i1 %221, label %222, label %225

222:                                              ; preds = %218
  %223 = sub nuw i32 %220, %68, !spirv.Decorations !409
  %224 = or i32 %219, 1
  br label %232

225:                                              ; preds = %218
  %226 = icmp eq i32 %220, %68
  br i1 %226, label %227, label %232

227:                                              ; preds = %225
  %228 = or i32 %219, 1
  %229 = xor i32 %216, -1
  %230 = add i32 %210, %229
  %231 = shl i32 %228, %230
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

232:                                              ; preds = %225, %222
  %233 = phi i32 [ %223, %222 ], [ %220, %225 ]
  %234 = phi i32 [ %224, %222 ], [ %219, %225 ]
  %235 = add nuw i32 %216, 1, !spirv.Decorations !409
  br label %213

236:                                              ; preds = %213
  %237 = or i32 %215, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i:        ; preds = %236, %227, %209
  %238 = phi i32 [ %231, %227 ], [ %237, %236 ], [ 0, %209 ]
  %239 = or i32 %211, %238
  %240 = and i32 %239, 7
  %241 = lshr i32 %239, 3
  %242 = icmp eq i32 %240, 0
  br i1 %242, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i, label %243

243:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  %244 = icmp ugt i32 %240, 4
  br i1 %244, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i, label %245

245:                                              ; preds = %243
  %246 = and i32 %239, 15
  %.not = icmp eq i32 %246, 12
  br i1 %.not, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i, label %247

247:                                              ; preds = %245
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i: ; preds = %247, %245, %243, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  %248 = phi i32 [ 0, %247 ], [ 0, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i ], [ 1, %245 ], [ 1, %243 ]
  br label %249

249:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
  %250 = phi i32 [ %208, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %248, %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i ]
  %251 = phi i32 [ %182, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %241, %_ZL19__handling_roundingIjET_S0_S0_ji.exit6.i.i.i ]
  %252 = icmp eq i32 %250, 0
  %253 = add nuw nsw i32 %251, 1, !spirv.Decorations !407
  %254 = icmp ugt i32 %251, 8388606
  %255 = select i1 %254, i32 0, i32 %253
  %256 = select i1 %254, i32 8388608, i32 0
  %257 = select i1 %252, i32 %251, i32 %255
  %258 = select i1 %252, i32 0, i32 %256
  %259 = and i32 %26, -2147483648
  %260 = or i32 %259, %258
  %261 = or i32 %260, %257
  %262 = bitcast i32 %261 to float
  br label %__imf_fdiv_rn.exit

263:                                              ; preds = %268, %.preheader.i.i.i
  %264 = phi i32 [ %269, %268 ], [ 0, %.preheader.i.i.i ]
  %265 = phi i32 [ %266, %268 ], [ %66, %.preheader.i.i.i ]
  %266 = shl nuw nsw i32 %265, 1, !spirv.Decorations !407
  %267 = icmp ult i32 %266, %68
  br i1 %267, label %268, label %270

268:                                              ; preds = %263
  %269 = add i32 %264, 1
  br label %263

270:                                              ; preds = %263
  %271 = xor i32 %264, -1
  %272 = add i32 %64, %271
  %273 = icmp sgt i32 %272, 127
  br i1 %273, label %274, label %278

274:                                              ; preds = %270
  %275 = icmp sgt i32 %26, -1
  br i1 %275, label %276, label %277

276:                                              ; preds = %274
  br label %__imf_fdiv_rn.exit

277:                                              ; preds = %274
  br label %__imf_fdiv_rn.exit

278:                                              ; preds = %270
  %279 = icmp sgt i32 %272, -127
  br i1 %279, label %280, label %337

280:                                              ; preds = %278
  %281 = add nsw i32 %272, 127, !spirv.Decorations !405
  %282 = shl i32 %66, %264
  %283 = icmp eq i32 %282, 0
  br i1 %283, label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i, label %.preheader.i4.i.i.i

.preheader.i4.i.i.i:                              ; preds = %280
  br label %284

284:                                              ; preds = %302, %.preheader.i4.i.i.i
  %285 = phi i32 [ %303, %302 ], [ %282, %.preheader.i4.i.i.i ]
  %286 = phi i32 [ %304, %302 ], [ 0, %.preheader.i4.i.i.i ]
  %287 = phi i32 [ %305, %302 ], [ 0, %.preheader.i4.i.i.i ]
  %288 = icmp ult i32 %287, 27
  br i1 %288, label %289, label %306

289:                                              ; preds = %284
  %290 = shl i32 %286, 1
  %291 = shl i32 %285, 1
  %292 = icmp ugt i32 %291, %68
  br i1 %292, label %293, label %296

293:                                              ; preds = %289
  %294 = sub nuw i32 %291, %68, !spirv.Decorations !409
  %295 = or i32 %290, 1
  br label %302

296:                                              ; preds = %289
  %297 = icmp eq i32 %291, %68
  br i1 %297, label %298, label %302

298:                                              ; preds = %296
  %299 = or i32 %290, 1
  %300 = sub i32 26, %287
  %301 = shl i32 %299, %300
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i

302:                                              ; preds = %296, %293
  %303 = phi i32 [ %294, %293 ], [ %291, %296 ]
  %304 = phi i32 [ %295, %293 ], [ %290, %296 ]
  %305 = add nuw i32 %287, 1, !spirv.Decorations !409
  br label %284

306:                                              ; preds = %284
  %307 = or i32 %286, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i:        ; preds = %306, %298, %280
  %308 = phi i32 [ %301, %298 ], [ %307, %306 ], [ 0, %280 ]
  %309 = lshr i32 %308, 3
  %310 = and i32 %309, 8388607
  %311 = and i32 %308, 7
  %312 = icmp eq i32 %311, 0
  br i1 %312, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %313

313:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  %314 = icmp ugt i32 %311, 4
  br i1 %314, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %315

315:                                              ; preds = %313
  %316 = and i32 %308, 15
  %.not10 = icmp eq i32 %316, 12
  br i1 %.not10, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %317

317:                                              ; preds = %315
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i: ; preds = %317, %315, %313, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  %318 = phi i1 [ true, %317 ], [ false, %315 ], [ false, %313 ], [ true, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i ]
  br i1 %318, label %329, label %319

319:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  %320 = add nuw nsw i32 %310, 1, !spirv.Decorations !407
  %321 = icmp eq i32 %310, 8388607
  br i1 %321, label %322, label %329

322:                                              ; preds = %319
  %323 = add nsw i32 %272, 128, !spirv.Decorations !405
  %324 = icmp eq i32 %323, 255
  br i1 %324, label %325, label %329

325:                                              ; preds = %322
  %326 = icmp sgt i32 %26, -1
  br i1 %326, label %327, label %328

327:                                              ; preds = %325
  br label %__imf_fdiv_rn.exit

328:                                              ; preds = %325
  br label %__imf_fdiv_rn.exit

329:                                              ; preds = %322, %319, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  %330 = phi i32 [ 0, %322 ], [ %320, %319 ], [ %310, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i ]
  %331 = phi i32 [ %323, %322 ], [ %281, %319 ], [ %281, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i ]
  %332 = and i32 %26, -2147483648
  %333 = shl nuw nsw i32 %331, 23, !spirv.Decorations !407
  %334 = or i32 %332, %333
  %335 = or i32 %334, %330
  %336 = bitcast i32 %335 to float
  br label %__imf_fdiv_rn.exit

337:                                              ; preds = %278
  %338 = add i32 %264, -127
  %339 = sub i32 %338, %64
  %340 = add i32 %339, 1
  %341 = icmp ugt i32 %340, 22
  br i1 %341, label %342, label %353

342:                                              ; preds = %337
  %343 = icmp eq i32 %340, 23
  %344 = add i32 %264, 1
  %345 = shl i32 %66, %344
  %346 = icmp ugt i32 %345, %68
  %347 = select i1 %343, i1 %346, i1 false
  %348 = icmp sgt i32 %26, -1
  br i1 %348, label %349, label %351

349:                                              ; preds = %342
  br i1 %347, label %__imf_fdiv_rn.exit, label %350

350:                                              ; preds = %349
  br label %__imf_fdiv_rn.exit

351:                                              ; preds = %342
  br i1 %347, label %__imf_fdiv_rn.exit, label %352

352:                                              ; preds = %351
  br label %__imf_fdiv_rn.exit

353:                                              ; preds = %337
  %354 = shl i32 %66, %264
  %355 = sub nsw i32 25, %339, !spirv.Decorations !405
  %356 = icmp eq i32 %354, 0
  br i1 %356, label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i, label %.preheader.i1.i.i.i

.preheader.i1.i.i.i:                              ; preds = %353
  br label %357

357:                                              ; preds = %376, %.preheader.i1.i.i.i
  %358 = phi i32 [ %377, %376 ], [ %354, %.preheader.i1.i.i.i ]
  %359 = phi i32 [ %378, %376 ], [ 0, %.preheader.i1.i.i.i ]
  %360 = phi i32 [ %379, %376 ], [ 0, %.preheader.i1.i.i.i ]
  %361 = icmp ult i32 %360, %355
  br i1 %361, label %362, label %380

362:                                              ; preds = %357
  %363 = shl i32 %359, 1
  %364 = shl i32 %358, 1
  %365 = icmp ugt i32 %364, %68
  br i1 %365, label %366, label %369

366:                                              ; preds = %362
  %367 = sub nuw i32 %364, %68, !spirv.Decorations !409
  %368 = or i32 %363, 1
  br label %376

369:                                              ; preds = %362
  %370 = icmp eq i32 %364, %68
  br i1 %370, label %371, label %376

371:                                              ; preds = %369
  %372 = or i32 %363, 1
  %373 = xor i32 %360, -1
  %374 = add i32 %355, %373
  %375 = shl i32 %372, %374
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i

376:                                              ; preds = %369, %366
  %377 = phi i32 [ %367, %366 ], [ %364, %369 ]
  %378 = phi i32 [ %368, %366 ], [ %363, %369 ]
  %379 = add nuw i32 %360, 1, !spirv.Decorations !409
  br label %357

380:                                              ; preds = %357
  %381 = or i32 %359, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i:        ; preds = %380, %371, %353
  %382 = phi i32 [ %375, %371 ], [ %381, %380 ], [ 0, %353 ]
  %383 = lshr i32 %382, 3
  %384 = and i32 %382, 7
  %385 = icmp eq i32 %384, 0
  br i1 %385, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i, label %386

386:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %387 = icmp ugt i32 %384, 4
  br i1 %387, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i, label %388

388:                                              ; preds = %386
  %389 = and i32 %382, 15
  %.not9 = icmp eq i32 %389, 12
  br i1 %.not9, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i, label %390

390:                                              ; preds = %388
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i: ; preds = %390, %388, %386, %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %391 = phi i1 [ true, %390 ], [ false, %388 ], [ false, %386 ], [ true, %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i ]
  %392 = add nuw nsw i32 %383, 1, !spirv.Decorations !407
  %393 = icmp ugt i32 %382, 67108855
  %394 = select i1 %393, i32 0, i32 %392
  %395 = select i1 %393, i32 8388608, i32 0
  %396 = select i1 %391, i32 %383, i32 %394
  %397 = select i1 %391, i32 0, i32 %395
  %398 = and i32 %26, -2147483648
  %399 = or i32 %398, %397
  %400 = or i32 %399, %396
  %401 = bitcast i32 %400 to float
  br label %__imf_fdiv_rn.exit

__imf_fdiv_rn.exit:                               ; preds = %6, %30, %39, %48, %54, %91, %92, %149, %150, %151, %172, %173, %174, %175, %249, %276, %277, %327, %328, %329, %349, %350, %351, %352, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
  %402 = phi float [ %50, %48 ], [ %56, %54 ], [ 0x7FF8000000000000, %30 ], [ 0x7FF8000000000000, %6 ], [ %45, %39 ], [ %336, %329 ], [ %401, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ], [ %262, %249 ], [ %158, %151 ], [ 0x7FF0000000000000, %91 ], [ 0xFFF0000000000000, %92 ], [ 0x7FF0000000000000, %149 ], [ 0xFFF0000000000000, %150 ], [ 0.000000e+00, %173 ], [ -0.000000e+00, %175 ], [ 0x36A0000000000000, %172 ], [ 0xB6A0000000000000, %174 ], [ 0x7FF0000000000000, %276 ], [ 0xFFF0000000000000, %277 ], [ 0x7FF0000000000000, %327 ], [ 0xFFF0000000000000, %328 ], [ 0.000000e+00, %350 ], [ -0.000000e+00, %352 ], [ 0x36A0000000000000, %349 ], [ 0xB6A0000000000000, %351 ]
  %403 = fdiv float %12, %17
  %404 = zext i16 %7 to i64
  %405 = getelementptr float, float addrspace(1)* %2, i64 %404
  store float %402, float addrspace(1)* %405, align 4
  %406 = zext i16 %7 to i64
  %407 = getelementptr float, float addrspace(1)* %3, i64 %406
  store float %403, float addrspace(1)* %407, align 4
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #1

; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

declare i32 @printf(i8 addrspace(2)*, ...)

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { inaccessiblememonly nocallback nofree nosync nounwind willreturn }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.spirv.extensions = !{!3}
!igc.functions = !{!4}
!IGCMetadata = !{!28}
!opencl.ocl.version = !{!402, !402, !402, !402, !402, !402, !402}
!opencl.spir.version = !{!402, !402, !402, !402, !402, !402, !402}
!llvm.ident = !{!403, !403, !403, !403, !403, !403, !403}
!llvm.module.flags = !{!404}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{!"SPV_INTEL_vector_compute"}
!4 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <8 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32)* @kernel, !5}
!5 = !{!6, !7, !26, !27}
!6 = !{!"function_type", i32 0}
!7 = !{!"implicit_arg_desc", !8, !9, !10, !11, !12, !13, !14, !16, !18, !20, !22, !24}
!8 = !{i32 0}
!9 = !{i32 1}
!10 = !{i32 8}
!11 = !{i32 9}
!12 = !{i32 10}
!13 = !{i32 13}
!14 = !{i32 15, !15}
!15 = !{!"explicit_arg_num", i32 0}
!16 = !{i32 15, !17}
!17 = !{!"explicit_arg_num", i32 1}
!18 = !{i32 15, !19}
!19 = !{!"explicit_arg_num", i32 2}
!20 = !{i32 15, !21}
!21 = !{!"explicit_arg_num", i32 3}
!22 = !{i32 15, !23}
!23 = !{!"explicit_arg_num", i32 4}
!24 = !{i32 15, !25}
!25 = !{!"explicit_arg_num", i32 5}
!26 = !{!"thread_group_size", i32 128, i32 1, i32 1}
!27 = !{!"sub_group_size", i32 32}
!28 = !{!"ModuleMD", !29, !30, !118, !235, !266, !283, !306, !316, !318, !319, !334, !335, !336, !337, !341, !342, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !361, !365, !366, !367, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !183, !378, !379, !380, !382, !384, !387, !388, !389, !391, !392, !393, !398, !399, !400, !401}
!29 = !{!"isPrecise", i1 false}
!30 = !{!"compOpt", !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117}
!31 = !{!"DenormsAreZero", i1 false}
!32 = !{!"BFTFDenormsAreZero", i1 false}
!33 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!34 = !{!"OptDisable", i1 false}
!35 = !{!"MadEnable", i1 true}
!36 = !{!"NoSignedZeros", i1 false}
!37 = !{!"NoNaNs", i1 false}
!38 = !{!"FloatRoundingMode", i32 0}
!39 = !{!"FloatCvtIntRoundingMode", i32 3}
!40 = !{!"LoadCacheDefault", i32 4}
!41 = !{!"StoreCacheDefault", i32 2}
!42 = !{!"VISAPreSchedRPThreshold", i32 0}
!43 = !{!"SetLoopUnrollThreshold", i32 0}
!44 = !{!"UnsafeMathOptimizations", i1 false}
!45 = !{!"disableCustomUnsafeOpts", i1 false}
!46 = !{!"disableReducePow", i1 false}
!47 = !{!"disableSqrtOpt", i1 false}
!48 = !{!"FiniteMathOnly", i1 false}
!49 = !{!"FastRelaxedMath", i1 false}
!50 = !{!"DashGSpecified", i1 false}
!51 = !{!"FastCompilation", i1 false}
!52 = !{!"UseScratchSpacePrivateMemory", i1 true}
!53 = !{!"RelaxedBuiltins", i1 false}
!54 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!55 = !{!"GreaterThan2GBBufferRequired", i1 true}
!56 = !{!"GreaterThan4GBBufferRequired", i1 true}
!57 = !{!"DisableA64WA", i1 false}
!58 = !{!"ForceEnableA64WA", i1 false}
!59 = !{!"PushConstantsEnable", i1 true}
!60 = !{!"HasPositivePointerOffset", i1 false}
!61 = !{!"HasBufferOffsetArg", i1 true}
!62 = !{!"BufferOffsetArgOptional", i1 true}
!63 = !{!"replaceGlobalOffsetsByZero", i1 false}
!64 = !{!"forcePixelShaderSIMDMode", i32 0}
!65 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!66 = !{!"UniformWGS", i1 false}
!67 = !{!"disableVertexComponentPacking", i1 false}
!68 = !{!"disablePartialVertexComponentPacking", i1 false}
!69 = !{!"PreferBindlessImages", i1 false}
!70 = !{!"UseBindlessMode", i1 false}
!71 = !{!"UseLegacyBindlessMode", i1 true}
!72 = !{!"disableMathRefactoring", i1 false}
!73 = !{!"atomicBranch", i1 false}
!74 = !{!"spillCompression", i1 false}
!75 = !{!"DisableEarlyOut", i1 false}
!76 = !{!"ForceInt32DivRemEmu", i1 false}
!77 = !{!"ForceInt32DivRemEmuSP", i1 false}
!78 = !{!"DisableFastestSingleCSSIMD", i1 false}
!79 = !{!"DisableFastestLinearScan", i1 false}
!80 = !{!"UseStatelessforPrivateMemory", i1 false}
!81 = !{!"EnableTakeGlobalAddress", i1 false}
!82 = !{!"IsLibraryCompilation", i1 false}
!83 = !{!"LibraryCompileSIMDSize", i32 0}
!84 = !{!"FastVISACompile", i1 false}
!85 = !{!"MatchSinCosPi", i1 false}
!86 = !{!"ExcludeIRFromZEBinary", i1 false}
!87 = !{!"EmitZeBinVISASections", i1 false}
!88 = !{!"FP64GenEmulationEnabled", i1 false}
!89 = !{!"FP64GenConvEmulationEnabled", i1 false}
!90 = !{!"allowDisableRematforCS", i1 false}
!91 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!92 = !{!"DisableCPSOmaskWA", i1 false}
!93 = !{!"DisableFastestGopt", i1 false}
!94 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!95 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!96 = !{!"DisableConstantCoalescing", i1 false}
!97 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!98 = !{!"WaEnableALTModeVisaWA", i1 false}
!99 = !{!"EnableLdStCombineforLoad", i1 false}
!100 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!101 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!102 = !{!"NewSpillCostFunction", i1 false}
!103 = !{!"EnableVRT", i1 false}
!104 = !{!"ForceLargeGRFNum4RQ", i1 false}
!105 = !{!"DisableEUFusion", i1 false}
!106 = !{!"DisableFDivToFMulInvOpt", i1 false}
!107 = !{!"initializePhiSampleSourceWA", i1 false}
!108 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!109 = !{!"DisableLoosenSimd32Occu", i1 false}
!110 = !{!"FastestS1Options", i32 0}
!111 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!112 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!113 = !{!"DisableLscSamplerRouting", i1 false}
!114 = !{!"UseBarrierControlFlowOptimization", i1 false}
!115 = !{!"EnableDynamicRQManagement", i1 false}
!116 = !{!"Quad8InputThreshold", i32 0}
!117 = !{!"UseResourceLoopUnrollNested", i1 false}
!118 = !{!"FuncMD", !119, !120}
!119 = !{!"FuncMDMap[0]", void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <8 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32)* @kernel}
!120 = !{!"FuncMDValue[0]", !121, !122, !126, !127, !128, !148, !175, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !198, !205, !212, !219, !226, !233, !234}
!121 = !{!"localOffsets"}
!122 = !{!"workGroupWalkOrder", !123, !124, !125}
!123 = !{!"dim0", i32 0}
!124 = !{!"dim1", i32 1}
!125 = !{!"dim2", i32 2}
!126 = !{!"funcArgs"}
!127 = !{!"functionType", !"KernelFunction"}
!128 = !{!"rtInfo", !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !146, !147}
!129 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!130 = !{!"isContinuation", i1 false}
!131 = !{!"hasTraceRayPayload", i1 false}
!132 = !{!"hasHitAttributes", i1 false}
!133 = !{!"hasCallableData", i1 false}
!134 = !{!"ShaderStackSize", i32 0}
!135 = !{!"ShaderHash", i64 0}
!136 = !{!"ShaderName", !""}
!137 = !{!"ParentName", !""}
!138 = !{!"SlotNum", i1* null}
!139 = !{!"NOSSize", i32 0}
!140 = !{!"globalRootSignatureSize", i32 0}
!141 = !{!"Entries"}
!142 = !{!"SpillUnions"}
!143 = !{!"CustomHitAttrSizeInBytes", i32 0}
!144 = !{!"Types", !145}
!145 = !{!"FullFrameTys"}
!146 = !{!"Aliases"}
!147 = !{!"NumCoherenceHintBits", i32 0}
!148 = !{!"resAllocMD", !149, !150, !151, !152, !174}
!149 = !{!"uavsNumType", i32 0}
!150 = !{!"srvsNumType", i32 0}
!151 = !{!"samplersNumType", i32 0}
!152 = !{!"argAllocMDList", !153, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173}
!153 = !{!"argAllocMDListVec[0]", !154, !155, !156}
!154 = !{!"type", i32 0}
!155 = !{!"extensionType", i32 -1}
!156 = !{!"indexType", i32 -1}
!157 = !{!"argAllocMDListVec[1]", !154, !155, !156}
!158 = !{!"argAllocMDListVec[2]", !154, !155, !156}
!159 = !{!"argAllocMDListVec[3]", !154, !155, !156}
!160 = !{!"argAllocMDListVec[4]", !154, !155, !156}
!161 = !{!"argAllocMDListVec[5]", !154, !155, !156}
!162 = !{!"argAllocMDListVec[6]", !154, !155, !156}
!163 = !{!"argAllocMDListVec[7]", !154, !155, !156}
!164 = !{!"argAllocMDListVec[8]", !154, !155, !156}
!165 = !{!"argAllocMDListVec[9]", !154, !155, !156}
!166 = !{!"argAllocMDListVec[10]", !154, !155, !156}
!167 = !{!"argAllocMDListVec[11]", !154, !155, !156}
!168 = !{!"argAllocMDListVec[12]", !154, !155, !156}
!169 = !{!"argAllocMDListVec[13]", !154, !155, !156}
!170 = !{!"argAllocMDListVec[14]", !154, !155, !156}
!171 = !{!"argAllocMDListVec[15]", !154, !155, !156}
!172 = !{!"argAllocMDListVec[16]", !154, !155, !156}
!173 = !{!"argAllocMDListVec[17]", !154, !155, !156}
!174 = !{!"inlineSamplersMD"}
!175 = !{!"maxByteOffsets"}
!176 = !{!"IsInitializer", i1 false}
!177 = !{!"IsFinalizer", i1 false}
!178 = !{!"CompiledSubGroupsNumber", i32 0}
!179 = !{!"hasInlineVmeSamplers", i1 false}
!180 = !{!"localSize", i32 0}
!181 = !{!"localIDPresent", i1 false}
!182 = !{!"groupIDPresent", i1 false}
!183 = !{!"privateMemoryPerWI", i32 0}
!184 = !{!"prevFPOffset", i32 0}
!185 = !{!"globalIDPresent", i1 false}
!186 = !{!"hasSyncRTCalls", i1 false}
!187 = !{!"hasNonKernelArgLoad", i1 false}
!188 = !{!"hasNonKernelArgStore", i1 false}
!189 = !{!"hasNonKernelArgAtomic", i1 false}
!190 = !{!"UserAnnotations"}
!191 = !{!"m_OpenCLArgAddressSpaces", !192, !193, !194, !195, !196, !197}
!192 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!193 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 1}
!194 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 1}
!195 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 1}
!196 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!197 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 1}
!198 = !{!"m_OpenCLArgAccessQualifiers", !199, !200, !201, !202, !203, !204}
!199 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!200 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!201 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!202 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!203 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!204 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!205 = !{!"m_OpenCLArgTypes", !206, !207, !208, !209, !210, !211}
!206 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!207 = !{!"m_OpenCLArgTypesVec[1]", !"float*"}
!208 = !{!"m_OpenCLArgTypesVec[2]", !"float*"}
!209 = !{!"m_OpenCLArgTypesVec[3]", !"float*"}
!210 = !{!"m_OpenCLArgTypesVec[4]", !"char*"}
!211 = !{!"m_OpenCLArgTypesVec[5]", !"char*"}
!212 = !{!"m_OpenCLArgBaseTypes", !213, !214, !215, !216, !217, !218}
!213 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!214 = !{!"m_OpenCLArgBaseTypesVec[1]", !"float*"}
!215 = !{!"m_OpenCLArgBaseTypesVec[2]", !"float*"}
!216 = !{!"m_OpenCLArgBaseTypesVec[3]", !"float*"}
!217 = !{!"m_OpenCLArgBaseTypesVec[4]", !"char*"}
!218 = !{!"m_OpenCLArgBaseTypesVec[5]", !"char*"}
!219 = !{!"m_OpenCLArgTypeQualifiers", !220, !221, !222, !223, !224, !225}
!220 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!221 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!222 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!223 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!224 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!225 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!226 = !{!"m_OpenCLArgNames", !227, !228, !229, !230, !231, !232}
!227 = !{!"m_OpenCLArgNamesVec[0]", !""}
!228 = !{!"m_OpenCLArgNamesVec[1]", !""}
!229 = !{!"m_OpenCLArgNamesVec[2]", !""}
!230 = !{!"m_OpenCLArgNamesVec[3]", !""}
!231 = !{!"m_OpenCLArgNamesVec[4]", !""}
!232 = !{!"m_OpenCLArgNamesVec[5]", !""}
!233 = !{!"m_OpenCLArgScalarAsPointers"}
!234 = !{!"m_OptsToDisablePerFunc"}
!235 = !{!"pushInfo", !236, !237, !238, !242, !243, !244, !245, !246, !247, !248, !249, !262, !263, !264, !265}
!236 = !{!"pushableAddresses"}
!237 = !{!"bindlessPushInfo"}
!238 = !{!"dynamicBufferInfo", !239, !240, !241}
!239 = !{!"firstIndex", i32 0}
!240 = !{!"numOffsets", i32 0}
!241 = !{!"forceDisabled", i1 false}
!242 = !{!"MaxNumberOfPushedBuffers", i32 0}
!243 = !{!"inlineConstantBufferSlot", i32 -1}
!244 = !{!"inlineConstantBufferOffset", i32 -1}
!245 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!246 = !{!"constants"}
!247 = !{!"inputs"}
!248 = !{!"constantReg"}
!249 = !{!"simplePushInfoArr", !250, !259, !260, !261}
!250 = !{!"simplePushInfoArrVec[0]", !251, !252, !253, !254, !255, !256, !257, !258}
!251 = !{!"cbIdx", i32 0}
!252 = !{!"pushableAddressGrfOffset", i32 -1}
!253 = !{!"pushableOffsetGrfOffset", i32 -1}
!254 = !{!"offset", i32 0}
!255 = !{!"size", i32 0}
!256 = !{!"isStateless", i1 false}
!257 = !{!"isBindless", i1 false}
!258 = !{!"simplePushLoads"}
!259 = !{!"simplePushInfoArrVec[1]", !251, !252, !253, !254, !255, !256, !257, !258}
!260 = !{!"simplePushInfoArrVec[2]", !251, !252, !253, !254, !255, !256, !257, !258}
!261 = !{!"simplePushInfoArrVec[3]", !251, !252, !253, !254, !255, !256, !257, !258}
!262 = !{!"simplePushBufferUsed", i32 0}
!263 = !{!"pushAnalysisWIInfos"}
!264 = !{!"inlineRTGlobalPtrOffset", i32 0}
!265 = !{!"rtSyncSurfPtrOffset", i32 0}
!266 = !{!"psInfo", !267, !268, !269, !270, !271, !272, !273, !274, !275, !276, !277, !278, !279, !280, !281, !282}
!267 = !{!"BlendStateDisabledMask", i8 0}
!268 = !{!"SkipSrc0Alpha", i1 false}
!269 = !{!"DualSourceBlendingDisabled", i1 false}
!270 = !{!"ForceEnableSimd32", i1 false}
!271 = !{!"DisableSimd32WithDiscard", i1 false}
!272 = !{!"outputDepth", i1 false}
!273 = !{!"outputStencil", i1 false}
!274 = !{!"outputMask", i1 false}
!275 = !{!"blendToFillEnabled", i1 false}
!276 = !{!"forceEarlyZ", i1 false}
!277 = !{!"hasVersionedLoop", i1 false}
!278 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!279 = !{!"NumSamples", i8 0}
!280 = !{!"blendOptimizationMode"}
!281 = !{!"colorOutputMask"}
!282 = !{!"WaDisableVRS", i1 false}
!283 = !{!"csInfo", !284, !285, !286, !287, !288, !42, !43, !289, !290, !291, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301, !74, !302, !303, !304, !305}
!284 = !{!"maxWorkGroupSize", i32 0}
!285 = !{!"waveSize", i32 0}
!286 = !{!"ComputeShaderSecondCompile"}
!287 = !{!"forcedSIMDSize", i8 0}
!288 = !{!"forceTotalGRFNum", i32 0}
!289 = !{!"forceSpillCompression", i1 false}
!290 = !{!"allowLowerSimd", i1 false}
!291 = !{!"disableSimd32Slicing", i1 false}
!292 = !{!"disableSplitOnSpill", i1 false}
!293 = !{!"enableNewSpillCostFunction", i1 false}
!294 = !{!"forceVISAPreSched", i1 false}
!295 = !{!"forceUniformBuffer", i1 false}
!296 = !{!"forceUniformSurfaceSampler", i1 false}
!297 = !{!"disableLocalIdOrderOptimizations", i1 false}
!298 = !{!"disableDispatchAlongY", i1 false}
!299 = !{!"neededThreadIdLayout", i1* null}
!300 = !{!"forceTileYWalk", i1 false}
!301 = !{!"atomicBranch", i32 0}
!302 = !{!"disableEarlyOut", i1 false}
!303 = !{!"walkOrderEnabled", i1 false}
!304 = !{!"walkOrderOverride", i32 0}
!305 = !{!"ResForHfPacking"}
!306 = !{!"msInfo", !307, !308, !309, !310, !311, !312, !313, !314, !315}
!307 = !{!"PrimitiveTopology", i32 3}
!308 = !{!"MaxNumOfPrimitives", i32 0}
!309 = !{!"MaxNumOfVertices", i32 0}
!310 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!311 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!312 = !{!"WorkGroupSize", i32 0}
!313 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!314 = !{!"IndexFormat", i32 6}
!315 = !{!"SubgroupSize", i32 0}
!316 = !{!"taskInfo", !317, !312, !313, !315}
!317 = !{!"MaxNumOfOutputs", i32 0}
!318 = !{!"NBarrierCnt", i32 0}
!319 = !{!"rtInfo", !320, !321, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333}
!320 = !{!"RayQueryAllocSizeInBytes", i32 0}
!321 = !{!"NumContinuations", i32 0}
!322 = !{!"RTAsyncStackAddrspace", i32 -1}
!323 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!324 = !{!"SWHotZoneAddrspace", i32 -1}
!325 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!326 = !{!"SWStackAddrspace", i32 -1}
!327 = !{!"SWStackSurfaceStateOffset", i1* null}
!328 = !{!"RTSyncStackAddrspace", i32 -1}
!329 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!330 = !{!"doSyncDispatchRays", i1 false}
!331 = !{!"MemStyle", !"Xe"}
!332 = !{!"GlobalDataStyle", !"Xe"}
!333 = !{!"uberTileDimensions", i1* null}
!334 = !{!"CurUniqueIndirectIdx", i32 0}
!335 = !{!"inlineDynTextures"}
!336 = !{!"inlineResInfoData"}
!337 = !{!"immConstant", !338, !339, !340}
!338 = !{!"data"}
!339 = !{!"sizes"}
!340 = !{!"zeroIdxs"}
!341 = !{!"stringConstants"}
!342 = !{!"inlineBuffers", !343, !347, !348}
!343 = !{!"inlineBuffersVec[0]", !344, !345, !346}
!344 = !{!"alignment", i32 0}
!345 = !{!"allocSize", i64 0}
!346 = !{!"Buffer"}
!347 = !{!"inlineBuffersVec[1]", !344, !345, !346}
!348 = !{!"inlineBuffersVec[2]", !344, !345, !346}
!349 = !{!"GlobalPointerProgramBinaryInfos"}
!350 = !{!"ConstantPointerProgramBinaryInfos"}
!351 = !{!"GlobalBufferAddressRelocInfo"}
!352 = !{!"ConstantBufferAddressRelocInfo"}
!353 = !{!"forceLscCacheList"}
!354 = !{!"SrvMap"}
!355 = !{!"RasterizerOrderedByteAddressBuffer"}
!356 = !{!"RasterizerOrderedViews"}
!357 = !{!"MinNOSPushConstantSize", i32 0}
!358 = !{!"inlineProgramScopeOffsets"}
!359 = !{!"shaderData", !360}
!360 = !{!"numReplicas", i32 0}
!361 = !{!"URBInfo", !362, !363, !364}
!362 = !{!"has64BVertexHeaderInput", i1 false}
!363 = !{!"has64BVertexHeaderOutput", i1 false}
!364 = !{!"hasVertexHeader", i1 true}
!365 = !{!"UseBindlessImage", i1 false}
!366 = !{!"enableRangeReduce", i1 false}
!367 = !{!"allowMatchMadOptimizationforVS", i1 false}
!368 = !{!"disableMatchMadOptimizationForCS", i1 false}
!369 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!370 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!371 = !{!"statefulResourcesNotAliased", i1 false}
!372 = !{!"disableMixMode", i1 false}
!373 = !{!"genericAccessesResolved", i1 false}
!374 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!375 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!376 = !{!"disableSeparateScratchWA", i1 false}
!377 = !{!"enableRemoveUnusedTGMFence", i1 false}
!378 = !{!"PrivateMemoryPerFG"}
!379 = !{!"m_OptsToDisable"}
!380 = !{!"capabilities", !381}
!381 = !{!"globalVariableDecorationsINTEL", i1 false}
!382 = !{!"extensions", !383}
!383 = !{!"spvINTELBindlessImages", i1 false}
!384 = !{!"m_ShaderResourceViewMcsMask", !385, !386}
!385 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!386 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!387 = !{!"computedDepthMode", i32 0}
!388 = !{!"isHDCFastClearShader", i1 false}
!389 = !{!"argRegisterReservations", !390}
!390 = !{!"argRegisterReservationsVec[0]", i32 0}
!391 = !{!"SIMD16_SpillThreshold", i8 0}
!392 = !{!"SIMD32_SpillThreshold", i8 0}
!393 = !{!"m_CacheControlOption", !394, !395, !396, !397}
!394 = !{!"LscLoadCacheControlOverride", i8 0}
!395 = !{!"LscStoreCacheControlOverride", i8 0}
!396 = !{!"TgmLoadCacheControlOverride", i8 0}
!397 = !{!"TgmStoreCacheControlOverride", i8 0}
!398 = !{!"ModuleUsesBindless", i1 false}
!399 = !{!"predicationMap"}
!400 = !{!"lifeTimeStartMap"}
!401 = !{!"HitGroups"}
!402 = !{i32 2, i32 0}
!403 = !{!"clang version 15.0.7"}
!404 = !{i32 1, !"wchar_size", i32 4}
!405 = !{!406}
!406 = !{i32 4469}
!407 = !{!406, !408}
!408 = !{i32 4470}
!409 = !{!408}
