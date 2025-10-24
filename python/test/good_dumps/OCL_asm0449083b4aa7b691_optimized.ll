; ------------------------------------------------
; OCL_asm0449083b4aa7b691_optimized.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

; Function Attrs: argmemonly nofree norecurse nosync nounwind
define spir_kernel void @kernel(float addrspace(1)* nocapture readonly align 4 %0, float addrspace(1)* nocapture readonly align 4 %1, float addrspace(1)* nocapture writeonly align 4 %2, float addrspace(1)* nocapture writeonly align 4 %3, i8 addrspace(1)* nocapture readnone align 1 %4, i8 addrspace(1)* nocapture readnone align 1 %5, <8 x i32> %r0, <8 x i32> %payloadHeader, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* nocapture readnone %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bufferOffset5) #0 {
  %7 = and i16 %localIdX, 127
  %8 = zext i16 %7 to i64
  %9 = getelementptr float, float addrspace(1)* %0, i64 %8
  %10 = bitcast float addrspace(1)* %9 to <1 x float> addrspace(1)*
  %11 = bitcast <1 x float> addrspace(1)* %10 to float addrspace(1)*
  %12 = load float, float addrspace(1)* %11, align 4
  %13 = getelementptr float, float addrspace(1)* %1, i64 %8
  %14 = bitcast float addrspace(1)* %13 to <1 x float> addrspace(1)*
  %15 = bitcast <1 x float> addrspace(1)* %14 to float addrspace(1)*
  %16 = load float, float addrspace(1)* %15, align 4
  %17 = bitcast float %12 to i32
  %18 = bitcast float %16 to i32
  %19 = lshr i32 %17, 23
  %20 = and i32 %19, 255
  %21 = lshr i32 %18, 23
  %22 = and i32 %21, 255
  %23 = and i32 %17, 8388607
  %24 = and i32 %18, 8388607
  %25 = xor i32 %17, %18
  %26 = icmp eq i32 %20, 255
  %27 = icmp ne i32 %23, 0
  %28 = and i1 %26, %27
  br i1 %28, label %__imf_fdiv_rn.exit, label %29

29:                                               ; preds = %6
  %30 = icmp eq i32 %22, 255
  %31 = icmp ne i32 %24, 0
  %32 = and i1 %30, %31
  %33 = fcmp oeq float %16, 0.000000e+00
  %34 = or i1 %32, %33
  br i1 %34, label %__imf_fdiv_rn.exit, label %35

35:                                               ; preds = %29
  %36 = icmp eq i32 %23, 0
  %37 = and i1 %26, %36
  br i1 %37, label %38, label %45

38:                                               ; preds = %35
  %39 = icmp eq i32 %24, 0
  %40 = and i1 %30, %39
  %41 = and i32 %25, -2147483648
  %42 = or i32 %41, 2139095040
  %43 = bitcast i32 %42 to float
  %44 = select i1 %40, float 0x7FF8000000000000, float %43
  br label %__imf_fdiv_rn.exit

45:                                               ; preds = %35
  %46 = fcmp oeq float %12, 0.000000e+00
  br i1 %46, label %47, label %50

47:                                               ; preds = %45
  %48 = and i32 %25, -2147483648
  %49 = bitcast i32 %48 to float
  br label %__imf_fdiv_rn.exit

50:                                               ; preds = %45
  %51 = icmp eq i32 %24, 0
  %52 = and i1 %30, %51
  br i1 %52, label %53, label %56

53:                                               ; preds = %50
  %54 = and i32 %25, -2147483648
  %55 = bitcast i32 %54 to float
  br label %__imf_fdiv_rn.exit

56:                                               ; preds = %50
  %57 = icmp eq i32 %20, 0
  %58 = add nsw i32 %20, -127, !spirv.Decorations !408
  %59 = select i1 %57, i32 -126, i32 %58
  %60 = icmp eq i32 %22, 0
  %61 = add nsw i32 %22, -127, !spirv.Decorations !408
  %62 = select i1 %60, i32 -126, i32 %61
  %63 = sub nsw i32 %59, %62, !spirv.Decorations !408
  %64 = or i32 %23, 8388608
  %65 = select i1 %57, i32 %23, i32 %64
  %66 = or i32 %24, 8388608
  %67 = select i1 %60, i32 %24, i32 %66
  %68 = icmp ult i32 %65, %67
  br i1 %68, label %.preheader.i.i.i, label %69

69:                                               ; preds = %56
  %70 = udiv i32 %65, %67
  %71 = mul i32 %67, %70
  br label %72

72:                                               ; preds = %69, %72
  %73 = phi i32 [ -2147483648, %69 ], [ %75, %72 ]
  %74 = phi i64 [ 0, %69 ], [ %76, %72 ]
  %75 = lshr i32 %73, 1
  %76 = add nuw nsw i64 %74, 1, !spirv.Decorations !410
  %77 = icmp ugt i64 %74, 30
  %78 = and i32 %70, %75
  %79 = icmp eq i32 %78, %75
  %80 = or i1 %77, %79
  br i1 %80, label %81, label %72

81:                                               ; preds = %72
  %82 = sub i32 %65, %71
  %83 = trunc i64 %76 to i32
  %84 = sub nsw i32 31, %83, !spirv.Decorations !408
  %85 = add nsw i32 %63, %84, !spirv.Decorations !408
  %86 = icmp sgt i32 %85, 127
  br i1 %86, label %87, label %89

87:                                               ; preds = %81
  %88 = icmp sgt i32 %25, -1
  %. = select i1 %88, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

89:                                               ; preds = %81
  %90 = icmp sgt i32 %85, -127
  br i1 %90, label %91, label %144

91:                                               ; preds = %89
  %92 = add nsw i32 %85, 127, !spirv.Decorations !408
  %93 = add nsw i32 %83, -8, !spirv.Decorations !408
  %94 = shl i32 %70, %93
  %95 = and i32 %94, 8388607
  %96 = add nsw i32 %83, -5, !spirv.Decorations !408
  %97 = icmp eq i32 %82, 0
  br i1 %97, label %.critedge46, label %.preheader.i.i.i.i.preheader

.preheader.i.i.i.i.preheader:                     ; preds = %91
  %.not92 = icmp eq i32 %96, 0
  br i1 %.not92, label %.preheader.i.i.i.i._crit_edge, label %.lr.ph87

.lr.ph87:                                         ; preds = %.preheader.i.i.i.i.preheader, %.preheader.i.i.i.i
  %98 = phi i32 [ %116, %.preheader.i.i.i.i ], [ 0, %.preheader.i.i.i.i.preheader ]
  %99 = phi i32 [ %115, %.preheader.i.i.i.i ], [ 0, %.preheader.i.i.i.i.preheader ]
  %100 = phi i32 [ %114, %.preheader.i.i.i.i ], [ %82, %.preheader.i.i.i.i.preheader ]
  %101 = shl i32 %99, 1
  %102 = shl i32 %100, 1
  %103 = icmp ugt i32 %102, %67
  br i1 %103, label %104, label %107

104:                                              ; preds = %.lr.ph87
  %105 = sub nuw i32 %102, %67, !spirv.Decorations !412
  %106 = or i32 %101, 1
  br label %.preheader.i.i.i.i

107:                                              ; preds = %.lr.ph87
  %108 = icmp eq i32 %102, %67
  br i1 %108, label %109, label %.preheader.i.i.i.i

109:                                              ; preds = %107
  %110 = or i32 %101, 1
  %111 = xor i32 %98, -1
  %112 = add i32 %96, %111
  %113 = shl i32 %110, %112
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i

.preheader.i.i.i.i:                               ; preds = %107, %104
  %114 = phi i32 [ %105, %104 ], [ %102, %107 ]
  %115 = phi i32 [ %106, %104 ], [ %101, %107 ]
  %116 = add nuw i32 %98, 1, !spirv.Decorations !412
  %117 = icmp ult i32 %116, %96
  br i1 %117, label %.lr.ph87, label %.preheader.i.i.i.i._crit_edge

.preheader.i.i.i.i._crit_edge:                    ; preds = %.preheader.i.i.i.i, %.preheader.i.i.i.i.preheader
  %.lcssa70 = phi i32 [ 0, %.preheader.i.i.i.i.preheader ], [ %115, %.preheader.i.i.i.i ]
  %118 = or i32 %.lcssa70, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i:         ; preds = %.preheader.i.i.i.i._crit_edge, %109
  %119 = phi i32 [ %113, %109 ], [ %118, %.preheader.i.i.i.i._crit_edge ]
  %120 = lshr i32 %119, 3
  %121 = or i32 %95, %120
  %122 = and i32 %119, 7
  %123 = icmp eq i32 %122, 0
  br i1 %123, label %.critedge46, label %124

124:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  %125 = icmp ugt i32 %122, 4
  br i1 %125, label %.critedge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i: ; preds = %124
  %126 = icmp ne i32 %122, 4
  %127 = and i32 %121, 1
  %128 = icmp eq i32 %127, 0
  %129 = or i1 %126, %128
  br i1 %129, label %.critedge46, label %.critedge

.critedge:                                        ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i, %124
  %130 = add nuw nsw i32 %121, 1, !spirv.Decorations !410
  %131 = icmp ugt i32 %121, 8388606
  br i1 %131, label %132, label %.critedge46

132:                                              ; preds = %.critedge
  %133 = add nsw i32 %85, 128, !spirv.Decorations !408
  %134 = icmp eq i32 %133, 255
  br i1 %134, label %135, label %.critedge46

135:                                              ; preds = %132
  %136 = icmp sgt i32 %25, -1
  %.47 = select i1 %136, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

.critedge46:                                      ; preds = %91, %132, %.critedge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  %137 = phi i32 [ %121, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i ], [ %121, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i ], [ %130, %.critedge ], [ %130, %132 ], [ %95, %91 ]
  %138 = phi i32 [ %92, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i ], [ %92, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i ], [ %92, %.critedge ], [ %133, %132 ], [ %92, %91 ]
  %139 = and i32 %25, -2147483648
  %140 = shl nuw nsw i32 %138, 23, !spirv.Decorations !410
  %141 = or i32 %139, %140
  %142 = or i32 %141, %137
  %143 = bitcast i32 %142 to float
  br label %__imf_fdiv_rn.exit

144:                                              ; preds = %89
  %145 = xor i32 %85, -1
  %146 = icmp ult i32 %85, -149
  br i1 %146, label %147, label %158

147:                                              ; preds = %144
  %148 = icmp eq i32 %85, -150
  br i1 %148, label %149, label %._crit_edge80

149:                                              ; preds = %147
  %150 = icmp ne i32 %65, %71
  %151 = lshr i32 -2147483648, %83
  %152 = icmp ne i32 %70, %151
  %153 = or i1 %150, %152
  br label %._crit_edge80

._crit_edge80:                                    ; preds = %147, %149
  %154 = phi i1 [ %153, %149 ], [ false, %147 ]
  %155 = icmp sgt i32 %25, -1
  br i1 %155, label %156, label %157

156:                                              ; preds = %._crit_edge80
  %spec.select48 = select i1 %154, float 0x36A0000000000000, float 0.000000e+00
  br label %__imf_fdiv_rn.exit

157:                                              ; preds = %._crit_edge80
  %spec.select49 = select i1 %154, float 0xB6A0000000000000, float -0.000000e+00
  br label %__imf_fdiv_rn.exit

158:                                              ; preds = %144
  %159 = add nsw i32 %85, 152, !spirv.Decorations !408
  %160 = icmp sgt i32 %159, %84
  br i1 %160, label %193, label %161

161:                                              ; preds = %158
  %162 = sub nsw i32 %145, %83, !spirv.Decorations !408
  %163 = add nsw i32 %162, -117, !spirv.Decorations !408
  %164 = lshr i32 %70, %163
  %165 = add nsw i32 %162, -120, !spirv.Decorations !408
  %166 = lshr i32 %70, %165
  %167 = and i32 %166, 7
  %168 = and i32 %166, 1
  %169 = icmp eq i32 %168, 0
  br i1 %169, label %170, label %._crit_edge81

170:                                              ; preds = %161
  %171 = shl nsw i32 -1, %165, !spirv.Decorations !408
  %172 = xor i32 %171, -1
  %173 = and i32 %70, %172
  %174 = icmp ne i32 %173, 0
  %175 = icmp ne i32 %65, %71
  %176 = or i1 %174, %175
  %177 = zext i1 %176 to i32
  %178 = or i32 %167, %177
  br label %._crit_edge81

._crit_edge81:                                    ; preds = %161, %170
  %179 = phi i32 [ %178, %170 ], [ %167, %161 ]
  %180 = icmp eq i32 %179, 0
  br i1 %180, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, label %181

181:                                              ; preds = %._crit_edge81
  %182 = icmp ugt i32 %179, 4
  br i1 %182, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118, label %186

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118: ; preds = %181
  %183 = add nuw nsw i32 %164, 1, !spirv.Decorations !410
  %184 = icmp ugt i32 %164, 8388606
  %185 = select i1 %184, i32 0, i32 %183
  br label %229

186:                                              ; preds = %181
  %187 = icmp eq i32 %179, 4
  %188 = and i32 %164, 1
  %189 = icmp ne i32 %188, 0
  %not. = and i1 %187, %189
  %cond.fr122 = freeze i1 %not.
  %190 = add nuw nsw i32 %164, 1, !spirv.Decorations !410
  %191 = icmp ugt i32 %164, 8388606
  %192 = select i1 %191, i32 0, i32 %190
  br i1 %cond.fr122, label %229, label %232

193:                                              ; preds = %158
  %194 = sub nsw i32 %159, %84, !spirv.Decorations !408
  %195 = shl i32 %70, %194
  %196 = icmp eq i32 %82, 0
  br i1 %196, label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, label %.lr.ph89

.lr.ph89:                                         ; preds = %193, %.preheader.i7.i.i.i
  %197 = phi i32 [ %215, %.preheader.i7.i.i.i ], [ 0, %193 ]
  %198 = phi i32 [ %214, %.preheader.i7.i.i.i ], [ 0, %193 ]
  %199 = phi i32 [ %213, %.preheader.i7.i.i.i ], [ %82, %193 ]
  %200 = shl i32 %198, 1
  %201 = shl i32 %199, 1
  %202 = icmp ugt i32 %201, %67
  br i1 %202, label %203, label %206

203:                                              ; preds = %.lr.ph89
  %204 = sub nuw i32 %201, %67, !spirv.Decorations !412
  %205 = or i32 %200, 1
  br label %.preheader.i7.i.i.i

206:                                              ; preds = %.lr.ph89
  %207 = icmp eq i32 %201, %67
  br i1 %207, label %208, label %.preheader.i7.i.i.i

208:                                              ; preds = %206
  %209 = or i32 %200, 1
  %210 = xor i32 %197, -1
  %211 = add i32 %194, %210
  %212 = shl i32 %209, %211
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

.preheader.i7.i.i.i:                              ; preds = %206, %203
  %213 = phi i32 [ %204, %203 ], [ %201, %206 ]
  %214 = phi i32 [ %205, %203 ], [ %200, %206 ]
  %215 = add nuw i32 %197, 1, !spirv.Decorations !412
  %216 = icmp ult i32 %215, %194
  br i1 %216, label %.lr.ph89, label %.preheader.i7.i.i.i._crit_edge

.preheader.i7.i.i.i._crit_edge:                   ; preds = %.preheader.i7.i.i.i
  %217 = or i32 %214, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i:        ; preds = %193, %.preheader.i7.i.i.i._crit_edge, %208
  %218 = phi i32 [ %212, %208 ], [ %217, %.preheader.i7.i.i.i._crit_edge ], [ 0, %193 ]
  %219 = or i32 %195, %218
  %220 = and i32 %219, 7
  %221 = lshr i32 %219, 3
  %222 = icmp eq i32 %220, 0
  br i1 %222, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread: ; preds = %._crit_edge81, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  %.ph = phi i32 [ %221, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i ], [ %164, %._crit_edge81 ]
  %223 = icmp ugt i32 %.ph, 8388606
  br label %232

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  %224 = icmp ugt i32 %220, 4
  %225 = and i32 %219, 15
  %.not = icmp eq i32 %225, 12
  %or.cond = or i1 %224, %.not
  %cond.fr = freeze i1 %or.cond
  %226 = add nuw nsw i32 %221, 1, !spirv.Decorations !410
  %227 = icmp ugt i32 %219, 67108855
  %228 = select i1 %227, i32 0, i32 %226
  br i1 %cond.fr, label %229, label %232

229:                                              ; preds = %186, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
  %230 = phi i32 [ %185, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %228, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %192, %186 ]
  %231 = phi i1 [ %184, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %227, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %191, %186 ]
  %.shrunk121 = phi i1 [ true, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %cond.fr, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %cond.fr122, %186 ]
  br label %232

232:                                              ; preds = %186, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i, %229
  %233 = phi i1 [ %231, %229 ], [ %227, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %223, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %191, %186 ]
  %.shrunk117 = phi i1 [ %.shrunk121, %229 ], [ %cond.fr, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ false, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %cond.fr122, %186 ]
  %234 = phi i32 [ %230, %229 ], [ %221, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i ], [ %.ph, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %164, %186 ]
  %235 = and i1 %.shrunk117, %233
  %236 = select i1 %235, i32 8388608, i32 0
  %237 = and i32 %25, -2147483648
  %238 = or i32 %237, %236
  %239 = or i32 %238, %234
  %240 = bitcast i32 %239 to float
  br label %__imf_fdiv_rn.exit

.preheader.i.i.i:                                 ; preds = %56, %.preheader.i.i.i
  %241 = phi i32 [ %245, %.preheader.i.i.i ], [ 0, %56 ]
  %242 = phi i32 [ %243, %.preheader.i.i.i ], [ %65, %56 ]
  %243 = shl nuw nsw i32 %242, 1, !spirv.Decorations !410
  %244 = icmp ult i32 %243, %67
  %245 = add i32 %241, 1
  br i1 %244, label %.preheader.i.i.i, label %246

246:                                              ; preds = %.preheader.i.i.i
  %247 = xor i32 %241, -1
  %248 = add i32 %63, %247
  %249 = icmp sgt i32 %248, 127
  br i1 %249, label %250, label %252

250:                                              ; preds = %246
  %251 = icmp sgt i32 %25, -1
  %.51 = select i1 %251, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

252:                                              ; preds = %246
  %253 = icmp sgt i32 %248, -127
  br i1 %253, label %254, label %549

254:                                              ; preds = %252
  %255 = add nsw i32 %248, 127, !spirv.Decorations !408
  %256 = shl i32 %65, %241
  %257 = icmp eq i32 %256, 0
  br i1 %257, label %.critedge53, label %.preheader.i4.i.i.i.preheader

.preheader.i4.i.i.i.preheader:                    ; preds = %254
  %258 = shl i32 %256, 1
  %259 = icmp ugt i32 %258, %67
  br i1 %259, label %260, label %262

260:                                              ; preds = %.preheader.i4.i.i.i.preheader
  %261 = sub nuw i32 %258, %67, !spirv.Decorations !412
  br label %.preheader.i4.i.i.i

262:                                              ; preds = %.preheader.i4.i.i.i.preheader
  %263 = icmp eq i32 %258, %67
  br i1 %263, label %264, label %.preheader.i4.i.i.i

264:                                              ; preds = %521, %511, %501, %491, %481, %471, %461, %451, %441, %431, %421, %411, %401, %391, %381, %371, %361, %351, %341, %331, %321, %311, %301, %291, %281, %271, %262
  %.lcssa95.neg = phi i32 [ 26, %262 ], [ 25, %271 ], [ 24, %281 ], [ 23, %291 ], [ 22, %301 ], [ 21, %311 ], [ 20, %321 ], [ 19, %331 ], [ 18, %341 ], [ 17, %351 ], [ 16, %361 ], [ 15, %371 ], [ 14, %381 ], [ 13, %391 ], [ 12, %401 ], [ 11, %411 ], [ 10, %421 ], [ 9, %431 ], [ 8, %441 ], [ 7, %451 ], [ 6, %461 ], [ 5, %471 ], [ 4, %481 ], [ 3, %491 ], [ 2, %501 ], [ 1, %511 ], [ 0, %521 ]
  %.lcssa = phi i32 [ 0, %262 ], [ %268, %271 ], [ %278, %281 ], [ %288, %291 ], [ %298, %301 ], [ %308, %311 ], [ %318, %321 ], [ %328, %331 ], [ %338, %341 ], [ %348, %351 ], [ %358, %361 ], [ %368, %371 ], [ %378, %381 ], [ %388, %391 ], [ %398, %401 ], [ %408, %411 ], [ %418, %421 ], [ %428, %431 ], [ %438, %441 ], [ %448, %451 ], [ %458, %461 ], [ %468, %471 ], [ %478, %481 ], [ %488, %491 ], [ %498, %501 ], [ %508, %511 ], [ %518, %521 ]
  %265 = or i32 %.lcssa, 1
  %266 = shl i32 %265, %.lcssa95.neg
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i

.preheader.i4.i.i.i:                              ; preds = %262, %260
  %267 = phi i32 [ %261, %260 ], [ %258, %262 ]
  %268 = phi i32 [ 2, %260 ], [ 0, %262 ]
  %269 = shl i32 %267, 1
  %270 = icmp ugt i32 %269, %67
  br i1 %270, label %273, label %271

271:                                              ; preds = %.preheader.i4.i.i.i
  %272 = icmp eq i32 %269, %67
  br i1 %272, label %264, label %.preheader.i4.i.i.i.1

273:                                              ; preds = %.preheader.i4.i.i.i
  %274 = sub nuw i32 %269, %67, !spirv.Decorations !412
  %275 = or i32 %268, 1
  br label %.preheader.i4.i.i.i.1

.preheader.i4.i.i.i.1:                            ; preds = %271, %273
  %276 = phi i32 [ %274, %273 ], [ %269, %271 ]
  %277 = phi i32 [ %275, %273 ], [ %268, %271 ]
  %278 = shl nsw i32 %277, 1
  %279 = shl i32 %276, 1
  %280 = icmp ugt i32 %279, %67
  br i1 %280, label %283, label %281

281:                                              ; preds = %.preheader.i4.i.i.i.1
  %282 = icmp eq i32 %279, %67
  br i1 %282, label %264, label %.preheader.i4.i.i.i.2

283:                                              ; preds = %.preheader.i4.i.i.i.1
  %284 = sub nuw i32 %279, %67, !spirv.Decorations !412
  %285 = or i32 %278, 1
  br label %.preheader.i4.i.i.i.2

.preheader.i4.i.i.i.2:                            ; preds = %281, %283
  %286 = phi i32 [ %284, %283 ], [ %279, %281 ]
  %287 = phi i32 [ %285, %283 ], [ %278, %281 ]
  %288 = shl i32 %287, 1
  %289 = shl i32 %286, 1
  %290 = icmp ugt i32 %289, %67
  br i1 %290, label %293, label %291

291:                                              ; preds = %.preheader.i4.i.i.i.2
  %292 = icmp eq i32 %289, %67
  br i1 %292, label %264, label %.preheader.i4.i.i.i.3

293:                                              ; preds = %.preheader.i4.i.i.i.2
  %294 = sub nuw i32 %289, %67, !spirv.Decorations !412
  %295 = or i32 %288, 1
  br label %.preheader.i4.i.i.i.3

.preheader.i4.i.i.i.3:                            ; preds = %291, %293
  %296 = phi i32 [ %294, %293 ], [ %289, %291 ]
  %297 = phi i32 [ %295, %293 ], [ %288, %291 ]
  %298 = shl i32 %297, 1
  %299 = shl i32 %296, 1
  %300 = icmp ugt i32 %299, %67
  br i1 %300, label %303, label %301

301:                                              ; preds = %.preheader.i4.i.i.i.3
  %302 = icmp eq i32 %299, %67
  br i1 %302, label %264, label %.preheader.i4.i.i.i.4

303:                                              ; preds = %.preheader.i4.i.i.i.3
  %304 = sub nuw i32 %299, %67, !spirv.Decorations !412
  %305 = or i32 %298, 1
  br label %.preheader.i4.i.i.i.4

.preheader.i4.i.i.i.4:                            ; preds = %301, %303
  %306 = phi i32 [ %304, %303 ], [ %299, %301 ]
  %307 = phi i32 [ %305, %303 ], [ %298, %301 ]
  %308 = shl i32 %307, 1
  %309 = shl i32 %306, 1
  %310 = icmp ugt i32 %309, %67
  br i1 %310, label %313, label %311

311:                                              ; preds = %.preheader.i4.i.i.i.4
  %312 = icmp eq i32 %309, %67
  br i1 %312, label %264, label %.preheader.i4.i.i.i.5

313:                                              ; preds = %.preheader.i4.i.i.i.4
  %314 = sub nuw i32 %309, %67, !spirv.Decorations !412
  %315 = or i32 %308, 1
  br label %.preheader.i4.i.i.i.5

.preheader.i4.i.i.i.5:                            ; preds = %311, %313
  %316 = phi i32 [ %314, %313 ], [ %309, %311 ]
  %317 = phi i32 [ %315, %313 ], [ %308, %311 ]
  %318 = shl i32 %317, 1
  %319 = shl i32 %316, 1
  %320 = icmp ugt i32 %319, %67
  br i1 %320, label %323, label %321

321:                                              ; preds = %.preheader.i4.i.i.i.5
  %322 = icmp eq i32 %319, %67
  br i1 %322, label %264, label %.preheader.i4.i.i.i.6

323:                                              ; preds = %.preheader.i4.i.i.i.5
  %324 = sub nuw i32 %319, %67, !spirv.Decorations !412
  %325 = or i32 %318, 1
  br label %.preheader.i4.i.i.i.6

.preheader.i4.i.i.i.6:                            ; preds = %321, %323
  %326 = phi i32 [ %324, %323 ], [ %319, %321 ]
  %327 = phi i32 [ %325, %323 ], [ %318, %321 ]
  %328 = shl i32 %327, 1
  %329 = shl i32 %326, 1
  %330 = icmp ugt i32 %329, %67
  br i1 %330, label %333, label %331

331:                                              ; preds = %.preheader.i4.i.i.i.6
  %332 = icmp eq i32 %329, %67
  br i1 %332, label %264, label %.preheader.i4.i.i.i.7

333:                                              ; preds = %.preheader.i4.i.i.i.6
  %334 = sub nuw i32 %329, %67, !spirv.Decorations !412
  %335 = or i32 %328, 1
  br label %.preheader.i4.i.i.i.7

.preheader.i4.i.i.i.7:                            ; preds = %331, %333
  %336 = phi i32 [ %334, %333 ], [ %329, %331 ]
  %337 = phi i32 [ %335, %333 ], [ %328, %331 ]
  %338 = shl i32 %337, 1
  %339 = shl i32 %336, 1
  %340 = icmp ugt i32 %339, %67
  br i1 %340, label %343, label %341

341:                                              ; preds = %.preheader.i4.i.i.i.7
  %342 = icmp eq i32 %339, %67
  br i1 %342, label %264, label %.preheader.i4.i.i.i.8

343:                                              ; preds = %.preheader.i4.i.i.i.7
  %344 = sub nuw i32 %339, %67, !spirv.Decorations !412
  %345 = or i32 %338, 1
  br label %.preheader.i4.i.i.i.8

.preheader.i4.i.i.i.8:                            ; preds = %341, %343
  %346 = phi i32 [ %344, %343 ], [ %339, %341 ]
  %347 = phi i32 [ %345, %343 ], [ %338, %341 ]
  %348 = shl i32 %347, 1
  %349 = shl i32 %346, 1
  %350 = icmp ugt i32 %349, %67
  br i1 %350, label %353, label %351

351:                                              ; preds = %.preheader.i4.i.i.i.8
  %352 = icmp eq i32 %349, %67
  br i1 %352, label %264, label %.preheader.i4.i.i.i.9

353:                                              ; preds = %.preheader.i4.i.i.i.8
  %354 = sub nuw i32 %349, %67, !spirv.Decorations !412
  %355 = or i32 %348, 1
  br label %.preheader.i4.i.i.i.9

.preheader.i4.i.i.i.9:                            ; preds = %351, %353
  %356 = phi i32 [ %354, %353 ], [ %349, %351 ]
  %357 = phi i32 [ %355, %353 ], [ %348, %351 ]
  %358 = shl i32 %357, 1
  %359 = shl i32 %356, 1
  %360 = icmp ugt i32 %359, %67
  br i1 %360, label %363, label %361

361:                                              ; preds = %.preheader.i4.i.i.i.9
  %362 = icmp eq i32 %359, %67
  br i1 %362, label %264, label %.preheader.i4.i.i.i.10

363:                                              ; preds = %.preheader.i4.i.i.i.9
  %364 = sub nuw i32 %359, %67, !spirv.Decorations !412
  %365 = or i32 %358, 1
  br label %.preheader.i4.i.i.i.10

.preheader.i4.i.i.i.10:                           ; preds = %361, %363
  %366 = phi i32 [ %364, %363 ], [ %359, %361 ]
  %367 = phi i32 [ %365, %363 ], [ %358, %361 ]
  %368 = shl i32 %367, 1
  %369 = shl i32 %366, 1
  %370 = icmp ugt i32 %369, %67
  br i1 %370, label %373, label %371

371:                                              ; preds = %.preheader.i4.i.i.i.10
  %372 = icmp eq i32 %369, %67
  br i1 %372, label %264, label %.preheader.i4.i.i.i.11

373:                                              ; preds = %.preheader.i4.i.i.i.10
  %374 = sub nuw i32 %369, %67, !spirv.Decorations !412
  %375 = or i32 %368, 1
  br label %.preheader.i4.i.i.i.11

.preheader.i4.i.i.i.11:                           ; preds = %371, %373
  %376 = phi i32 [ %374, %373 ], [ %369, %371 ]
  %377 = phi i32 [ %375, %373 ], [ %368, %371 ]
  %378 = shl i32 %377, 1
  %379 = shl i32 %376, 1
  %380 = icmp ugt i32 %379, %67
  br i1 %380, label %383, label %381

381:                                              ; preds = %.preheader.i4.i.i.i.11
  %382 = icmp eq i32 %379, %67
  br i1 %382, label %264, label %.preheader.i4.i.i.i.12

383:                                              ; preds = %.preheader.i4.i.i.i.11
  %384 = sub nuw i32 %379, %67, !spirv.Decorations !412
  %385 = or i32 %378, 1
  br label %.preheader.i4.i.i.i.12

.preheader.i4.i.i.i.12:                           ; preds = %381, %383
  %386 = phi i32 [ %384, %383 ], [ %379, %381 ]
  %387 = phi i32 [ %385, %383 ], [ %378, %381 ]
  %388 = shl i32 %387, 1
  %389 = shl i32 %386, 1
  %390 = icmp ugt i32 %389, %67
  br i1 %390, label %393, label %391

391:                                              ; preds = %.preheader.i4.i.i.i.12
  %392 = icmp eq i32 %389, %67
  br i1 %392, label %264, label %.preheader.i4.i.i.i.13

393:                                              ; preds = %.preheader.i4.i.i.i.12
  %394 = sub nuw i32 %389, %67, !spirv.Decorations !412
  %395 = or i32 %388, 1
  br label %.preheader.i4.i.i.i.13

.preheader.i4.i.i.i.13:                           ; preds = %391, %393
  %396 = phi i32 [ %394, %393 ], [ %389, %391 ]
  %397 = phi i32 [ %395, %393 ], [ %388, %391 ]
  %398 = shl i32 %397, 1
  %399 = shl i32 %396, 1
  %400 = icmp ugt i32 %399, %67
  br i1 %400, label %403, label %401

401:                                              ; preds = %.preheader.i4.i.i.i.13
  %402 = icmp eq i32 %399, %67
  br i1 %402, label %264, label %.preheader.i4.i.i.i.14

403:                                              ; preds = %.preheader.i4.i.i.i.13
  %404 = sub nuw i32 %399, %67, !spirv.Decorations !412
  %405 = or i32 %398, 1
  br label %.preheader.i4.i.i.i.14

.preheader.i4.i.i.i.14:                           ; preds = %401, %403
  %406 = phi i32 [ %404, %403 ], [ %399, %401 ]
  %407 = phi i32 [ %405, %403 ], [ %398, %401 ]
  %408 = shl i32 %407, 1
  %409 = shl i32 %406, 1
  %410 = icmp ugt i32 %409, %67
  br i1 %410, label %413, label %411

411:                                              ; preds = %.preheader.i4.i.i.i.14
  %412 = icmp eq i32 %409, %67
  br i1 %412, label %264, label %.preheader.i4.i.i.i.15

413:                                              ; preds = %.preheader.i4.i.i.i.14
  %414 = sub nuw i32 %409, %67, !spirv.Decorations !412
  %415 = or i32 %408, 1
  br label %.preheader.i4.i.i.i.15

.preheader.i4.i.i.i.15:                           ; preds = %411, %413
  %416 = phi i32 [ %414, %413 ], [ %409, %411 ]
  %417 = phi i32 [ %415, %413 ], [ %408, %411 ]
  %418 = shl i32 %417, 1
  %419 = shl i32 %416, 1
  %420 = icmp ugt i32 %419, %67
  br i1 %420, label %423, label %421

421:                                              ; preds = %.preheader.i4.i.i.i.15
  %422 = icmp eq i32 %419, %67
  br i1 %422, label %264, label %.preheader.i4.i.i.i.16

423:                                              ; preds = %.preheader.i4.i.i.i.15
  %424 = sub nuw i32 %419, %67, !spirv.Decorations !412
  %425 = or i32 %418, 1
  br label %.preheader.i4.i.i.i.16

.preheader.i4.i.i.i.16:                           ; preds = %421, %423
  %426 = phi i32 [ %424, %423 ], [ %419, %421 ]
  %427 = phi i32 [ %425, %423 ], [ %418, %421 ]
  %428 = shl i32 %427, 1
  %429 = shl i32 %426, 1
  %430 = icmp ugt i32 %429, %67
  br i1 %430, label %433, label %431

431:                                              ; preds = %.preheader.i4.i.i.i.16
  %432 = icmp eq i32 %429, %67
  br i1 %432, label %264, label %.preheader.i4.i.i.i.17

433:                                              ; preds = %.preheader.i4.i.i.i.16
  %434 = sub nuw i32 %429, %67, !spirv.Decorations !412
  %435 = or i32 %428, 1
  br label %.preheader.i4.i.i.i.17

.preheader.i4.i.i.i.17:                           ; preds = %431, %433
  %436 = phi i32 [ %434, %433 ], [ %429, %431 ]
  %437 = phi i32 [ %435, %433 ], [ %428, %431 ]
  %438 = shl i32 %437, 1
  %439 = shl i32 %436, 1
  %440 = icmp ugt i32 %439, %67
  br i1 %440, label %443, label %441

441:                                              ; preds = %.preheader.i4.i.i.i.17
  %442 = icmp eq i32 %439, %67
  br i1 %442, label %264, label %.preheader.i4.i.i.i.18

443:                                              ; preds = %.preheader.i4.i.i.i.17
  %444 = sub nuw i32 %439, %67, !spirv.Decorations !412
  %445 = or i32 %438, 1
  br label %.preheader.i4.i.i.i.18

.preheader.i4.i.i.i.18:                           ; preds = %441, %443
  %446 = phi i32 [ %444, %443 ], [ %439, %441 ]
  %447 = phi i32 [ %445, %443 ], [ %438, %441 ]
  %448 = shl i32 %447, 1
  %449 = shl i32 %446, 1
  %450 = icmp ugt i32 %449, %67
  br i1 %450, label %453, label %451

451:                                              ; preds = %.preheader.i4.i.i.i.18
  %452 = icmp eq i32 %449, %67
  br i1 %452, label %264, label %.preheader.i4.i.i.i.19

453:                                              ; preds = %.preheader.i4.i.i.i.18
  %454 = sub nuw i32 %449, %67, !spirv.Decorations !412
  %455 = or i32 %448, 1
  br label %.preheader.i4.i.i.i.19

.preheader.i4.i.i.i.19:                           ; preds = %451, %453
  %456 = phi i32 [ %454, %453 ], [ %449, %451 ]
  %457 = phi i32 [ %455, %453 ], [ %448, %451 ]
  %458 = shl i32 %457, 1
  %459 = shl i32 %456, 1
  %460 = icmp ugt i32 %459, %67
  br i1 %460, label %463, label %461

461:                                              ; preds = %.preheader.i4.i.i.i.19
  %462 = icmp eq i32 %459, %67
  br i1 %462, label %264, label %.preheader.i4.i.i.i.20

463:                                              ; preds = %.preheader.i4.i.i.i.19
  %464 = sub nuw i32 %459, %67, !spirv.Decorations !412
  %465 = or i32 %458, 1
  br label %.preheader.i4.i.i.i.20

.preheader.i4.i.i.i.20:                           ; preds = %461, %463
  %466 = phi i32 [ %464, %463 ], [ %459, %461 ]
  %467 = phi i32 [ %465, %463 ], [ %458, %461 ]
  %468 = shl i32 %467, 1
  %469 = shl i32 %466, 1
  %470 = icmp ugt i32 %469, %67
  br i1 %470, label %473, label %471

471:                                              ; preds = %.preheader.i4.i.i.i.20
  %472 = icmp eq i32 %469, %67
  br i1 %472, label %264, label %.preheader.i4.i.i.i.21

473:                                              ; preds = %.preheader.i4.i.i.i.20
  %474 = sub nuw i32 %469, %67, !spirv.Decorations !412
  %475 = or i32 %468, 1
  br label %.preheader.i4.i.i.i.21

.preheader.i4.i.i.i.21:                           ; preds = %471, %473
  %476 = phi i32 [ %474, %473 ], [ %469, %471 ]
  %477 = phi i32 [ %475, %473 ], [ %468, %471 ]
  %478 = shl i32 %477, 1
  %479 = shl i32 %476, 1
  %480 = icmp ugt i32 %479, %67
  br i1 %480, label %483, label %481

481:                                              ; preds = %.preheader.i4.i.i.i.21
  %482 = icmp eq i32 %479, %67
  br i1 %482, label %264, label %.preheader.i4.i.i.i.22

483:                                              ; preds = %.preheader.i4.i.i.i.21
  %484 = sub nuw i32 %479, %67, !spirv.Decorations !412
  %485 = or i32 %478, 1
  br label %.preheader.i4.i.i.i.22

.preheader.i4.i.i.i.22:                           ; preds = %481, %483
  %486 = phi i32 [ %484, %483 ], [ %479, %481 ]
  %487 = phi i32 [ %485, %483 ], [ %478, %481 ]
  %488 = shl i32 %487, 1
  %489 = shl i32 %486, 1
  %490 = icmp ugt i32 %489, %67
  br i1 %490, label %493, label %491

491:                                              ; preds = %.preheader.i4.i.i.i.22
  %492 = icmp eq i32 %489, %67
  br i1 %492, label %264, label %.preheader.i4.i.i.i.23

493:                                              ; preds = %.preheader.i4.i.i.i.22
  %494 = sub nuw i32 %489, %67, !spirv.Decorations !412
  %495 = or i32 %488, 1
  br label %.preheader.i4.i.i.i.23

.preheader.i4.i.i.i.23:                           ; preds = %491, %493
  %496 = phi i32 [ %494, %493 ], [ %489, %491 ]
  %497 = phi i32 [ %495, %493 ], [ %488, %491 ]
  %498 = shl i32 %497, 1
  %499 = shl i32 %496, 1
  %500 = icmp ugt i32 %499, %67
  br i1 %500, label %503, label %501

501:                                              ; preds = %.preheader.i4.i.i.i.23
  %502 = icmp eq i32 %499, %67
  br i1 %502, label %264, label %.preheader.i4.i.i.i.24

503:                                              ; preds = %.preheader.i4.i.i.i.23
  %504 = sub nuw i32 %499, %67, !spirv.Decorations !412
  %505 = or i32 %498, 1
  br label %.preheader.i4.i.i.i.24

.preheader.i4.i.i.i.24:                           ; preds = %501, %503
  %506 = phi i32 [ %504, %503 ], [ %499, %501 ]
  %507 = phi i32 [ %505, %503 ], [ %498, %501 ]
  %508 = shl i32 %507, 1
  %509 = shl i32 %506, 1
  %510 = icmp ugt i32 %509, %67
  br i1 %510, label %513, label %511

511:                                              ; preds = %.preheader.i4.i.i.i.24
  %512 = icmp eq i32 %509, %67
  br i1 %512, label %264, label %.preheader.i4.i.i.i.25

513:                                              ; preds = %.preheader.i4.i.i.i.24
  %514 = sub nuw i32 %509, %67, !spirv.Decorations !412
  %515 = or i32 %508, 1
  br label %.preheader.i4.i.i.i.25

.preheader.i4.i.i.i.25:                           ; preds = %511, %513
  %516 = phi i32 [ %514, %513 ], [ %509, %511 ]
  %517 = phi i32 [ %515, %513 ], [ %508, %511 ]
  %518 = shl i32 %517, 1
  %519 = shl i32 %516, 1
  %520 = icmp ugt i32 %519, %67
  br i1 %520, label %523, label %521

521:                                              ; preds = %.preheader.i4.i.i.i.25
  %522 = icmp eq i32 %519, %67
  br i1 %522, label %264, label %.preheader.i4.i.i.i.26

523:                                              ; preds = %.preheader.i4.i.i.i.25
  %524 = or i32 %518, 1
  br label %.preheader.i4.i.i.i.26

.preheader.i4.i.i.i.26:                           ; preds = %521, %523
  %525 = phi i32 [ %524, %523 ], [ %518, %521 ]
  %526 = or i32 %525, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i:        ; preds = %.preheader.i4.i.i.i.26, %264
  %527 = phi i32 [ %266, %264 ], [ %526, %.preheader.i4.i.i.i.26 ]
  %528 = lshr i32 %527, 3
  %529 = and i32 %528, 8388607
  %530 = and i32 %527, 7
  %531 = icmp eq i32 %530, 0
  br i1 %531, label %.critedge53, label %532

532:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  %533 = icmp ugt i32 %530, 4
  %534 = and i32 %527, 15
  %.not10 = icmp eq i32 %534, 12
  %or.cond52 = or i1 %533, %.not10
  br i1 %or.cond52, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %.critedge53

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i: ; preds = %532
  %535 = add nuw nsw i32 %529, 1, !spirv.Decorations !410
  %536 = icmp eq i32 %529, 8388607
  br i1 %536, label %537, label %.critedge53

537:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  %538 = add nsw i32 %248, 128, !spirv.Decorations !408
  %539 = icmp eq i32 %538, 255
  br i1 %539, label %540, label %.critedge53

540:                                              ; preds = %537
  %541 = icmp sgt i32 %25, -1
  %.54 = select i1 %541, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

.critedge53:                                      ; preds = %254, %537, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, %532, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  %542 = phi i32 [ %529, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i ], [ %529, %532 ], [ %535, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i ], [ 0, %537 ], [ 0, %254 ]
  %543 = phi i32 [ %255, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i ], [ %255, %532 ], [ %255, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i ], [ %538, %537 ], [ %255, %254 ]
  %544 = and i32 %25, -2147483648
  %545 = shl nuw nsw i32 %543, 23, !spirv.Decorations !410
  %546 = or i32 %544, %545
  %547 = or i32 %546, %542
  %548 = bitcast i32 %547 to float
  br label %__imf_fdiv_rn.exit

549:                                              ; preds = %252
  %550 = add i32 %241, -127
  %551 = sub i32 %550, %63
  %552 = add i32 %551, 1
  %553 = icmp ugt i32 %552, 22
  br i1 %553, label %554, label %562

554:                                              ; preds = %549
  %555 = icmp eq i32 %552, 23
  %556 = shl i32 %65, %245
  %557 = icmp ugt i32 %556, %67
  %558 = and i1 %555, %557
  %559 = icmp sgt i32 %25, -1
  br i1 %559, label %560, label %561

560:                                              ; preds = %554
  %spec.select55 = select i1 %558, float 0x36A0000000000000, float 0.000000e+00
  br label %__imf_fdiv_rn.exit

561:                                              ; preds = %554
  %spec.select56 = select i1 %558, float 0xB6A0000000000000, float -0.000000e+00
  br label %__imf_fdiv_rn.exit

562:                                              ; preds = %549
  %563 = shl i32 %65, %241
  %564 = sub nsw i32 25, %551, !spirv.Decorations !408
  %565 = icmp eq i32 %563, 0
  br i1 %565, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, label %.lr.ph

.lr.ph:                                           ; preds = %562, %.preheader.i1.i.i.i
  %566 = phi i32 [ %584, %.preheader.i1.i.i.i ], [ 0, %562 ]
  %567 = phi i32 [ %583, %.preheader.i1.i.i.i ], [ 0, %562 ]
  %568 = phi i32 [ %582, %.preheader.i1.i.i.i ], [ %563, %562 ]
  %569 = shl i32 %567, 1
  %570 = shl i32 %568, 1
  %571 = icmp ugt i32 %570, %67
  br i1 %571, label %572, label %575

572:                                              ; preds = %.lr.ph
  %573 = sub nuw i32 %570, %67, !spirv.Decorations !412
  %574 = or i32 %569, 1
  br label %.preheader.i1.i.i.i

575:                                              ; preds = %.lr.ph
  %576 = icmp eq i32 %570, %67
  br i1 %576, label %577, label %.preheader.i1.i.i.i

577:                                              ; preds = %575
  %578 = or i32 %569, 1
  %579 = xor i32 %566, -1
  %580 = add i32 %564, %579
  %581 = shl i32 %578, %580
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i

.preheader.i1.i.i.i:                              ; preds = %575, %572
  %582 = phi i32 [ %573, %572 ], [ %570, %575 ]
  %583 = phi i32 [ %574, %572 ], [ %569, %575 ]
  %584 = add nuw i32 %566, 1, !spirv.Decorations !412
  %585 = icmp ult i32 %584, %564
  br i1 %585, label %.lr.ph, label %.preheader.i1.i.i.i._crit_edge

.preheader.i1.i.i.i._crit_edge:                   ; preds = %.preheader.i1.i.i.i
  %586 = or i32 %583, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i:        ; preds = %.preheader.i1.i.i.i._crit_edge, %577
  %587 = phi i32 [ %581, %577 ], [ %586, %.preheader.i1.i.i.i._crit_edge ]
  %.fr = freeze i32 %587
  %588 = lshr i32 %.fr, 3
  %589 = and i32 %.fr, 7
  %590 = icmp eq i32 %589, 0
  br i1 %590, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %591 = icmp ugt i32 %.fr, 67108855
  %spec.select = select i1 %591, i32 8388608, i32 0
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %592 = icmp ult i32 %589, 5
  %593 = and i32 %.fr, 15
  %.not9 = icmp ne i32 %593, 12
  %not.or.cond57 = and i1 %592, %.not9
  %594 = add nuw nsw i32 %588, 1, !spirv.Decorations !410
  %595 = icmp ugt i32 %.fr, 67108855
  %596 = select i1 %595, i32 0, i32 %594
  %597 = select i1 %595, i32 8388608, i32 0
  br i1 %not.or.cond57, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, label %601

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread, %562, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
  %598 = phi i32 [ %597, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ], [ 0, %562 ], [ %spec.select, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %599 = phi i1 [ %not.or.cond57, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ], [ true, %562 ], [ true, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %600 = phi i32 [ %588, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ], [ 0, %562 ], [ %588, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  br label %601

601:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread
  %602 = phi i32 [ %598, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %597, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ]
  %603 = phi i1 [ %599, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %not.or.cond57, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ]
  %604 = phi i32 [ %600, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %596, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i ]
  %605 = select i1 %603, i32 0, i32 %602
  %606 = and i32 %25, -2147483648
  %607 = or i32 %606, %605
  %608 = or i32 %607, %604
  %609 = bitcast i32 %608 to float
  br label %__imf_fdiv_rn.exit

__imf_fdiv_rn.exit:                               ; preds = %29, %6, %561, %560, %157, %156, %540, %250, %135, %87, %38, %47, %53, %.critedge46, %232, %.critedge53, %601
  %610 = phi float [ %49, %47 ], [ %55, %53 ], [ %44, %38 ], [ %548, %.critedge53 ], [ %609, %601 ], [ %240, %232 ], [ %143, %.critedge46 ], [ %., %87 ], [ %.47, %135 ], [ %spec.select48, %156 ], [ %spec.select49, %157 ], [ %.51, %250 ], [ %.54, %540 ], [ %spec.select55, %560 ], [ %spec.select56, %561 ], [ 0x7FF8000000000000, %6 ], [ 0x7FF8000000000000, %29 ]
  %611 = fdiv float %12, %16
  %612 = getelementptr float, float addrspace(1)* %2, i64 %8
  store float %610, float addrspace(1)* %612, align 4
  %613 = getelementptr float, float addrspace(1)* %3, i64 %8
  store float %611, float addrspace(1)* %613, align 4
  ret void
}

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind readnone willreturn
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #1

; Function Attrs: inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #2

declare i32 @printf(i8 addrspace(2)*, ...)

attributes #0 = { argmemonly nofree norecurse nosync nounwind "less-precise-fpmad"="true" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.spirv.extensions = !{!3}
!igc.functions = !{!4}
!IGCMetadata = !{!28}
!opencl.ocl.version = !{!405, !405, !405, !405, !405, !405, !405}
!opencl.spir.version = !{!405, !405, !405, !405, !405, !405, !405}
!llvm.ident = !{!406, !406, !406, !406, !406, !406, !406}
!llvm.module.flags = !{!407}

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
!28 = !{!"ModuleMD", !29, !30, !118, !238, !269, !286, !309, !319, !321, !322, !337, !338, !339, !340, !344, !345, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !364, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !183, !381, !382, !383, !385, !387, !390, !391, !392, !394, !395, !396, !401, !402, !403, !404}
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
!234 = !{!"m_OptsToDisablePerFunc", !235, !236, !237}
!235 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!236 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!237 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!238 = !{!"pushInfo", !239, !240, !241, !245, !246, !247, !248, !249, !250, !251, !252, !265, !266, !267, !268}
!239 = !{!"pushableAddresses"}
!240 = !{!"bindlessPushInfo"}
!241 = !{!"dynamicBufferInfo", !242, !243, !244}
!242 = !{!"firstIndex", i32 0}
!243 = !{!"numOffsets", i32 0}
!244 = !{!"forceDisabled", i1 false}
!245 = !{!"MaxNumberOfPushedBuffers", i32 0}
!246 = !{!"inlineConstantBufferSlot", i32 -1}
!247 = !{!"inlineConstantBufferOffset", i32 -1}
!248 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!249 = !{!"constants"}
!250 = !{!"inputs"}
!251 = !{!"constantReg"}
!252 = !{!"simplePushInfoArr", !253, !262, !263, !264}
!253 = !{!"simplePushInfoArrVec[0]", !254, !255, !256, !257, !258, !259, !260, !261}
!254 = !{!"cbIdx", i32 0}
!255 = !{!"pushableAddressGrfOffset", i32 -1}
!256 = !{!"pushableOffsetGrfOffset", i32 -1}
!257 = !{!"offset", i32 0}
!258 = !{!"size", i32 0}
!259 = !{!"isStateless", i1 false}
!260 = !{!"isBindless", i1 false}
!261 = !{!"simplePushLoads"}
!262 = !{!"simplePushInfoArrVec[1]", !254, !255, !256, !257, !258, !259, !260, !261}
!263 = !{!"simplePushInfoArrVec[2]", !254, !255, !256, !257, !258, !259, !260, !261}
!264 = !{!"simplePushInfoArrVec[3]", !254, !255, !256, !257, !258, !259, !260, !261}
!265 = !{!"simplePushBufferUsed", i32 0}
!266 = !{!"pushAnalysisWIInfos"}
!267 = !{!"inlineRTGlobalPtrOffset", i32 0}
!268 = !{!"rtSyncSurfPtrOffset", i32 0}
!269 = !{!"psInfo", !270, !271, !272, !273, !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285}
!270 = !{!"BlendStateDisabledMask", i8 0}
!271 = !{!"SkipSrc0Alpha", i1 false}
!272 = !{!"DualSourceBlendingDisabled", i1 false}
!273 = !{!"ForceEnableSimd32", i1 false}
!274 = !{!"DisableSimd32WithDiscard", i1 false}
!275 = !{!"outputDepth", i1 false}
!276 = !{!"outputStencil", i1 false}
!277 = !{!"outputMask", i1 false}
!278 = !{!"blendToFillEnabled", i1 false}
!279 = !{!"forceEarlyZ", i1 false}
!280 = !{!"hasVersionedLoop", i1 false}
!281 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!282 = !{!"NumSamples", i8 0}
!283 = !{!"blendOptimizationMode"}
!284 = !{!"colorOutputMask"}
!285 = !{!"WaDisableVRS", i1 false}
!286 = !{!"csInfo", !287, !288, !289, !290, !291, !42, !43, !292, !293, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !74, !305, !306, !307, !308}
!287 = !{!"maxWorkGroupSize", i32 0}
!288 = !{!"waveSize", i32 0}
!289 = !{!"ComputeShaderSecondCompile"}
!290 = !{!"forcedSIMDSize", i8 0}
!291 = !{!"forceTotalGRFNum", i32 0}
!292 = !{!"forceSpillCompression", i1 false}
!293 = !{!"allowLowerSimd", i1 false}
!294 = !{!"disableSimd32Slicing", i1 false}
!295 = !{!"disableSplitOnSpill", i1 false}
!296 = !{!"enableNewSpillCostFunction", i1 false}
!297 = !{!"forceVISAPreSched", i1 false}
!298 = !{!"forceUniformBuffer", i1 false}
!299 = !{!"forceUniformSurfaceSampler", i1 false}
!300 = !{!"disableLocalIdOrderOptimizations", i1 false}
!301 = !{!"disableDispatchAlongY", i1 false}
!302 = !{!"neededThreadIdLayout", i1* null}
!303 = !{!"forceTileYWalk", i1 false}
!304 = !{!"atomicBranch", i32 0}
!305 = !{!"disableEarlyOut", i1 false}
!306 = !{!"walkOrderEnabled", i1 false}
!307 = !{!"walkOrderOverride", i32 0}
!308 = !{!"ResForHfPacking"}
!309 = !{!"msInfo", !310, !311, !312, !313, !314, !315, !316, !317, !318}
!310 = !{!"PrimitiveTopology", i32 3}
!311 = !{!"MaxNumOfPrimitives", i32 0}
!312 = !{!"MaxNumOfVertices", i32 0}
!313 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!314 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!315 = !{!"WorkGroupSize", i32 0}
!316 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!317 = !{!"IndexFormat", i32 6}
!318 = !{!"SubgroupSize", i32 0}
!319 = !{!"taskInfo", !320, !315, !316, !318}
!320 = !{!"MaxNumOfOutputs", i32 0}
!321 = !{!"NBarrierCnt", i32 0}
!322 = !{!"rtInfo", !323, !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336}
!323 = !{!"RayQueryAllocSizeInBytes", i32 0}
!324 = !{!"NumContinuations", i32 0}
!325 = !{!"RTAsyncStackAddrspace", i32 -1}
!326 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!327 = !{!"SWHotZoneAddrspace", i32 -1}
!328 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!329 = !{!"SWStackAddrspace", i32 -1}
!330 = !{!"SWStackSurfaceStateOffset", i1* null}
!331 = !{!"RTSyncStackAddrspace", i32 -1}
!332 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!333 = !{!"doSyncDispatchRays", i1 false}
!334 = !{!"MemStyle", !"Xe"}
!335 = !{!"GlobalDataStyle", !"Xe"}
!336 = !{!"uberTileDimensions", i1* null}
!337 = !{!"CurUniqueIndirectIdx", i32 0}
!338 = !{!"inlineDynTextures"}
!339 = !{!"inlineResInfoData"}
!340 = !{!"immConstant", !341, !342, !343}
!341 = !{!"data"}
!342 = !{!"sizes"}
!343 = !{!"zeroIdxs"}
!344 = !{!"stringConstants"}
!345 = !{!"inlineBuffers", !346, !350, !351}
!346 = !{!"inlineBuffersVec[0]", !347, !348, !349}
!347 = !{!"alignment", i32 0}
!348 = !{!"allocSize", i64 0}
!349 = !{!"Buffer"}
!350 = !{!"inlineBuffersVec[1]", !347, !348, !349}
!351 = !{!"inlineBuffersVec[2]", !347, !348, !349}
!352 = !{!"GlobalPointerProgramBinaryInfos"}
!353 = !{!"ConstantPointerProgramBinaryInfos"}
!354 = !{!"GlobalBufferAddressRelocInfo"}
!355 = !{!"ConstantBufferAddressRelocInfo"}
!356 = !{!"forceLscCacheList"}
!357 = !{!"SrvMap"}
!358 = !{!"RasterizerOrderedByteAddressBuffer"}
!359 = !{!"RasterizerOrderedViews"}
!360 = !{!"MinNOSPushConstantSize", i32 0}
!361 = !{!"inlineProgramScopeOffsets"}
!362 = !{!"shaderData", !363}
!363 = !{!"numReplicas", i32 0}
!364 = !{!"URBInfo", !365, !366, !367}
!365 = !{!"has64BVertexHeaderInput", i1 false}
!366 = !{!"has64BVertexHeaderOutput", i1 false}
!367 = !{!"hasVertexHeader", i1 true}
!368 = !{!"UseBindlessImage", i1 false}
!369 = !{!"enableRangeReduce", i1 false}
!370 = !{!"allowMatchMadOptimizationforVS", i1 false}
!371 = !{!"disableMatchMadOptimizationForCS", i1 false}
!372 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!373 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!374 = !{!"statefulResourcesNotAliased", i1 false}
!375 = !{!"disableMixMode", i1 false}
!376 = !{!"genericAccessesResolved", i1 false}
!377 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!378 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!379 = !{!"disableSeparateScratchWA", i1 false}
!380 = !{!"enableRemoveUnusedTGMFence", i1 false}
!381 = !{!"PrivateMemoryPerFG"}
!382 = !{!"m_OptsToDisable"}
!383 = !{!"capabilities", !384}
!384 = !{!"globalVariableDecorationsINTEL", i1 false}
!385 = !{!"extensions", !386}
!386 = !{!"spvINTELBindlessImages", i1 false}
!387 = !{!"m_ShaderResourceViewMcsMask", !388, !389}
!388 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!389 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!390 = !{!"computedDepthMode", i32 0}
!391 = !{!"isHDCFastClearShader", i1 false}
!392 = !{!"argRegisterReservations", !393}
!393 = !{!"argRegisterReservationsVec[0]", i32 0}
!394 = !{!"SIMD16_SpillThreshold", i8 0}
!395 = !{!"SIMD32_SpillThreshold", i8 0}
!396 = !{!"m_CacheControlOption", !397, !398, !399, !400}
!397 = !{!"LscLoadCacheControlOverride", i8 0}
!398 = !{!"LscStoreCacheControlOverride", i8 0}
!399 = !{!"TgmLoadCacheControlOverride", i8 0}
!400 = !{!"TgmStoreCacheControlOverride", i8 0}
!401 = !{!"ModuleUsesBindless", i1 false}
!402 = !{!"predicationMap"}
!403 = !{!"lifeTimeStartMap"}
!404 = !{!"HitGroups"}
!405 = !{i32 2, i32 0}
!406 = !{!"clang version 15.0.7"}
!407 = !{i32 1, !"wchar_size", i32 4}
!408 = !{!409}
!409 = !{i32 4469}
!410 = !{!409, !411}
!411 = !{i32 4470}
!412 = !{!411}
