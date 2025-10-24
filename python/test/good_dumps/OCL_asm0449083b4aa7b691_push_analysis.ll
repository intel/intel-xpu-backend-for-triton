; ------------------------------------------------
; OCL_asm0449083b4aa7b691_push_analysis.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

; Function Attrs: argmemonly nofree norecurse nosync nounwind null_pointer_is_valid
define spir_kernel void @kernel(float addrspace(1)* nocapture readonly align 4 %0, float addrspace(1)* nocapture readonly align 4 %1, float addrspace(1)* nocapture writeonly align 4 %2, float addrspace(1)* nocapture writeonly align 4 %3, i8 addrspace(1)* nocapture readnone align 1 %4, i8 addrspace(1)* nocapture readnone align 1 %5, <8 x i32> %r0, <8 x i32> %payloadHeader, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* nocapture readnone %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bufferOffset5) #0 {
  %7 = and i16 %localIdX, 127
  %8 = zext i16 %7 to i64
  %9 = ptrtoint float addrspace(1)* %0 to i64
  %10 = shl nuw nsw i64 %8, 2
  %11 = add i64 %10, %9
  %12 = inttoptr i64 %11 to float addrspace(1)*
  %13 = load float, float addrspace(1)* %12, align 4
  %14 = ptrtoint float addrspace(1)* %1 to i64
  %15 = add i64 %10, %14
  %16 = inttoptr i64 %15 to float addrspace(1)*
  %17 = load float, float addrspace(1)* %16, align 4
  %18 = bitcast float %13 to i32
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
  br i1 %29, label %.__imf_fdiv_rn.exit_crit_edge, label %30

.__imf_fdiv_rn.exit_crit_edge:                    ; preds = %6
  br label %__imf_fdiv_rn.exit

30:                                               ; preds = %6
  %31 = icmp eq i32 %23, 255
  %32 = icmp ne i32 %25, 0
  %33 = and i1 %31, %32
  %34 = fcmp oeq float %17, 0.000000e+00
  %35 = or i1 %33, %34
  br i1 %35, label %.__imf_fdiv_rn.exit_crit_edge168, label %36

.__imf_fdiv_rn.exit_crit_edge168:                 ; preds = %30
  br label %__imf_fdiv_rn.exit

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
  %47 = fcmp oeq float %13, 0.000000e+00
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
  %59 = add nsw i32 %21, -127, !spirv.Decorations !411
  %60 = select i1 %58, i32 -126, i32 %59
  %61 = icmp eq i32 %23, 0
  %62 = add nsw i32 %23, -127, !spirv.Decorations !411
  %63 = select i1 %61, i32 -126, i32 %62
  %64 = sub nsw i32 %60, %63, !spirv.Decorations !411
  %65 = or i32 %24, 8388608
  %66 = select i1 %58, i32 %24, i32 %65
  %67 = or i32 %25, 8388608
  %68 = select i1 %61, i32 %25, i32 %67
  %69 = icmp ult i32 %66, %68
  br i1 %69, label %.preheader.i.i.i.preheader, label %70

.preheader.i.i.i.preheader:                       ; preds = %57
  br label %.preheader.i.i.i

70:                                               ; preds = %57
  %tobool.i = icmp eq i32 %68, 0
  br i1 %tobool.i, label %.precompiled_u32divrem.exit_crit_edge, label %if.end.i

.precompiled_u32divrem.exit_crit_edge:            ; preds = %70
  br label %precompiled_u32divrem.exit

if.end.i:                                         ; preds = %70
  %conv.i = uitofp i32 %68 to float
  %conv1.i = uitofp i32 %68 to double
  %conv2.i = uitofp i32 %66 to double
  %div.i = fdiv float 1.000000e+00, %conv.i, !fpmath !413
  %conv3.i = fpext float %div.i to double
  %71 = fsub double 0.000000e+00, %conv1.i
  %72 = call double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double %71, double %conv3.i, double 0x3FF0000000004000)
  %73 = call double @llvm.genx.GenISA.mul.rtz.f64.f64.f64(double %conv3.i, double %conv2.i)
  %74 = call double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double %73, double %72, double %73)
  %conv6.i = fptoui double %74 to i32
  br label %precompiled_u32divrem.exit

precompiled_u32divrem.exit:                       ; preds = %.precompiled_u32divrem.exit_crit_edge, %if.end.i
  %retval.0.i = phi i32 [ %conv6.i, %if.end.i ], [ -1, %.precompiled_u32divrem.exit_crit_edge ]
  %75 = mul i32 %68, %retval.0.i
  br label %76

76:                                               ; preds = %._crit_edge, %precompiled_u32divrem.exit
  %77 = phi i32 [ -2147483648, %precompiled_u32divrem.exit ], [ %82, %._crit_edge ]
  %78 = phi i64 [ 0, %precompiled_u32divrem.exit ], [ %83, %._crit_edge ]
  %79 = bitcast i64 %78 to <2 x i32>
  %80 = extractelement <2 x i32> %79, i32 0
  %81 = extractelement <2 x i32> %79, i32 1
  %82 = lshr i32 %77, 1
  %83 = add nuw nsw i64 %78, 1, !spirv.Decorations !414
  %84 = icmp ugt i32 %80, 30
  %85 = icmp eq i32 %81, 0
  %86 = and i1 %85, %84
  %87 = icmp ugt i32 %81, 0
  %88 = or i1 %86, %87
  %89 = and i32 %retval.0.i, %82
  %90 = icmp eq i32 %89, %82
  %91 = or i1 %88, %90
  br i1 %91, label %92, label %._crit_edge

._crit_edge:                                      ; preds = %76
  br label %76

92:                                               ; preds = %76
  %93 = sub i32 %66, %75
  %94 = trunc i64 %83 to i32
  %95 = sub nsw i32 31, %94, !spirv.Decorations !411
  %96 = add nsw i32 %64, %95, !spirv.Decorations !411
  %97 = icmp sgt i32 %96, 127
  br i1 %97, label %98, label %100

98:                                               ; preds = %92
  %99 = icmp sgt i32 %26, -1
  %. = select i1 %99, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

100:                                              ; preds = %92
  %101 = icmp sgt i32 %96, -127
  br i1 %101, label %102, label %155

102:                                              ; preds = %100
  %103 = add nsw i32 %96, 127, !spirv.Decorations !411
  %104 = add nsw i32 %94, -8, !spirv.Decorations !411
  %105 = shl i32 %retval.0.i, %104
  %106 = and i32 %105, 8388607
  %107 = add nsw i32 %94, -5, !spirv.Decorations !411
  %108 = icmp eq i32 %93, 0
  br i1 %108, label %..critedge46_crit_edge, label %.preheader.i.i.i.i.preheader

..critedge46_crit_edge:                           ; preds = %102
  br label %.critedge46

.preheader.i.i.i.i.preheader:                     ; preds = %102
  %.not92 = icmp eq i32 %107, 0
  br i1 %.not92, label %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge, label %.lr.ph87.preheader

.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge: ; preds = %.preheader.i.i.i.i.preheader
  br label %.preheader.i.i.i.i._crit_edge

.lr.ph87.preheader:                               ; preds = %.preheader.i.i.i.i.preheader
  br label %.lr.ph87

.lr.ph87:                                         ; preds = %.preheader.i.i.i.i..lr.ph87_crit_edge, %.lr.ph87.preheader
  %109 = phi i32 [ %127, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ 0, %.lr.ph87.preheader ]
  %110 = phi i32 [ %126, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ 0, %.lr.ph87.preheader ]
  %111 = phi i32 [ %125, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ %93, %.lr.ph87.preheader ]
  %112 = shl i32 %110, 1
  %113 = shl i32 %111, 1
  %114 = icmp ugt i32 %113, %68
  br i1 %114, label %115, label %118

115:                                              ; preds = %.lr.ph87
  %116 = sub nuw i32 %113, %68, !spirv.Decorations !416
  %117 = add i32 %112, 1
  br label %.preheader.i.i.i.i

118:                                              ; preds = %.lr.ph87
  %119 = icmp eq i32 %113, %68
  br i1 %119, label %120, label %..preheader.i.i.i.i_crit_edge

..preheader.i.i.i.i_crit_edge:                    ; preds = %118
  br label %.preheader.i.i.i.i

120:                                              ; preds = %118
  %121 = add i32 %112, 1
  %122 = xor i32 %109, -1
  %123 = add i32 %107, %122
  %124 = shl i32 %121, %123
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i

.preheader.i.i.i.i:                               ; preds = %..preheader.i.i.i.i_crit_edge, %115
  %125 = phi i32 [ %116, %115 ], [ %113, %..preheader.i.i.i.i_crit_edge ]
  %126 = phi i32 [ %117, %115 ], [ %112, %..preheader.i.i.i.i_crit_edge ]
  %127 = add nuw i32 %109, 1, !spirv.Decorations !416
  %128 = icmp ult i32 %127, %107
  br i1 %128, label %.preheader.i.i.i.i..lr.ph87_crit_edge, label %.preheader.i.i.i.i._crit_edge.loopexit

.preheader.i.i.i.i..lr.ph87_crit_edge:            ; preds = %.preheader.i.i.i.i
  br label %.lr.ph87

.preheader.i.i.i.i._crit_edge.loopexit:           ; preds = %.preheader.i.i.i.i
  br label %.preheader.i.i.i.i._crit_edge

.preheader.i.i.i.i._crit_edge:                    ; preds = %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge, %.preheader.i.i.i.i._crit_edge.loopexit
  %.lcssa70 = phi i32 [ 0, %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge ], [ %126, %.preheader.i.i.i.i._crit_edge.loopexit ]
  %129 = or i32 %.lcssa70, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i:         ; preds = %.preheader.i.i.i.i._crit_edge, %120
  %130 = phi i32 [ %124, %120 ], [ %129, %.preheader.i.i.i.i._crit_edge ]
  %131 = lshr i32 %130, 3
  %132 = or i32 %106, %131
  %133 = and i32 %130, 7
  %134 = icmp eq i32 %133, 0
  br i1 %134, label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge, label %135

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  br label %.critedge46

135:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  %136 = icmp ugt i32 %133, 4
  br i1 %136, label %..critedge_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i

..critedge_crit_edge:                             ; preds = %135
  br label %.critedge

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i: ; preds = %135
  %137 = icmp ne i32 %133, 4
  %138 = and i32 %132, 1
  %139 = icmp eq i32 %138, 0
  %140 = or i1 %137, %139
  br i1 %140, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
  br label %.critedge

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
  br label %.critedge46

.critedge:                                        ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge, %..critedge_crit_edge
  %141 = add nuw nsw i32 %132, 1, !spirv.Decorations !414
  %142 = icmp ugt i32 %132, 8388606
  br i1 %142, label %143, label %.critedge..critedge46_crit_edge

.critedge..critedge46_crit_edge:                  ; preds = %.critedge
  br label %.critedge46

143:                                              ; preds = %.critedge
  %144 = add nsw i32 %96, 128, !spirv.Decorations !411
  %145 = icmp eq i32 %144, 255
  br i1 %145, label %146, label %..critedge46_crit_edge169

..critedge46_crit_edge169:                        ; preds = %143
  br label %.critedge46

146:                                              ; preds = %143
  %147 = icmp sgt i32 %26, -1
  %.47 = select i1 %147, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

.critedge46:                                      ; preds = %..critedge46_crit_edge169, %.critedge..critedge46_crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge, %..critedge46_crit_edge
  %148 = phi i32 [ %132, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge ], [ %132, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge ], [ %141, %.critedge..critedge46_crit_edge ], [ %141, %..critedge46_crit_edge169 ], [ %106, %..critedge46_crit_edge ]
  %149 = phi i32 [ %103, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge ], [ %103, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge ], [ %103, %.critedge..critedge46_crit_edge ], [ %144, %..critedge46_crit_edge169 ], [ %103, %..critedge46_crit_edge ]
  %150 = and i32 %26, -2147483648
  %151 = shl nuw nsw i32 %149, 23, !spirv.Decorations !414
  %152 = or i32 %150, %151
  %153 = or i32 %152, %148
  %154 = bitcast i32 %153 to float
  br label %__imf_fdiv_rn.exit

155:                                              ; preds = %100
  %156 = xor i32 %96, -1
  %157 = icmp ult i32 %96, -149
  br i1 %157, label %158, label %171

158:                                              ; preds = %155
  %159 = icmp eq i32 %96, -150
  br i1 %159, label %160, label %.._crit_edge80_crit_edge

.._crit_edge80_crit_edge:                         ; preds = %158
  br label %._crit_edge80

160:                                              ; preds = %158
  %161 = icmp ne i32 %66, %75
  %162 = lshr i32 -2147483648, %94
  %163 = icmp ne i32 %retval.0.i, %162
  %164 = or i1 %161, %163
  %165 = sext i1 %164 to i32
  br label %._crit_edge80

._crit_edge80:                                    ; preds = %.._crit_edge80_crit_edge, %160
  %166 = phi i32 [ %165, %160 ], [ 0, %.._crit_edge80_crit_edge ]
  %167 = icmp ne i32 %166, 0
  %168 = icmp sgt i32 %26, -1
  br i1 %168, label %169, label %170

169:                                              ; preds = %._crit_edge80
  %spec.select48 = select i1 %167, float 0x36A0000000000000, float 0.000000e+00
  br label %__imf_fdiv_rn.exit

170:                                              ; preds = %._crit_edge80
  %spec.select49 = select i1 %167, float 0xB6A0000000000000, float -0.000000e+00
  br label %__imf_fdiv_rn.exit

171:                                              ; preds = %155
  %172 = add nsw i32 %96, 152, !spirv.Decorations !411
  %173 = icmp sgt i32 %172, %95
  br i1 %173, label %211, label %174

174:                                              ; preds = %171
  %175 = sub nsw i32 %156, %94, !spirv.Decorations !411
  %176 = add nsw i32 %175, -117, !spirv.Decorations !411
  %177 = lshr i32 %retval.0.i, %176
  %178 = add nsw i32 %175, -120, !spirv.Decorations !411
  %179 = lshr i32 %retval.0.i, %178
  %180 = and i32 %179, 7
  %181 = and i32 %179, 1
  %182 = icmp eq i32 %181, 0
  br i1 %182, label %183, label %.._crit_edge81_crit_edge

.._crit_edge81_crit_edge:                         ; preds = %174
  br label %._crit_edge81

183:                                              ; preds = %174
  %184 = shl nsw i32 -1, %178, !spirv.Decorations !411
  %185 = xor i32 %184, -1
  %186 = and i32 %retval.0.i, %185
  %187 = icmp ne i32 %186, 0
  %188 = icmp ne i32 %66, %75
  %189 = or i1 %187, %188
  %190 = zext i1 %189 to i32
  %191 = or i32 %180, %190
  br label %._crit_edge81

._crit_edge81:                                    ; preds = %.._crit_edge81_crit_edge, %183
  %192 = phi i32 [ %191, %183 ], [ %180, %.._crit_edge81_crit_edge ]
  %193 = icmp eq i32 %192, 0
  br i1 %193, label %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, label %194

._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge: ; preds = %._crit_edge81
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread

194:                                              ; preds = %._crit_edge81
  %195 = icmp ugt i32 %192, 4
  br i1 %195, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118, label %200

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118: ; preds = %194
  %196 = add nuw nsw i32 %177, 1, !spirv.Decorations !414
  %197 = icmp ugt i32 %177, 8388606
  %198 = select i1 %197, i32 0, i32 %196
  %199 = sext i1 %197 to i32
  br label %252

200:                                              ; preds = %194
  %201 = icmp eq i32 %192, 4
  %202 = and i32 %177, 1
  %203 = icmp ne i32 %202, 0
  %not. = and i1 %201, %203
  %cond.fr122 = freeze i1 %not.
  %204 = add nuw nsw i32 %177, 1, !spirv.Decorations !414
  %205 = icmp ugt i32 %177, 8388606
  %206 = select i1 %205, i32 0, i32 %204
  %207 = sext i1 %205 to i32
  %208 = sext i1 %cond.fr122 to i32
  %209 = sext i1 %205 to i32
  %210 = sext i1 %cond.fr122 to i32
  br i1 %cond.fr122, label %._crit_edge170, label %._crit_edge171

._crit_edge171:                                   ; preds = %200
  br label %260

._crit_edge170:                                   ; preds = %200
  br label %252

211:                                              ; preds = %171
  %212 = sub nsw i32 %172, %95, !spirv.Decorations !411
  %213 = shl i32 %retval.0.i, %212
  %214 = icmp eq i32 %93, 0
  br i1 %214, label %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge, label %.lr.ph89.preheader

._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge: ; preds = %211
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

.lr.ph89.preheader:                               ; preds = %211
  br label %.lr.ph89

.lr.ph89:                                         ; preds = %.preheader.i7.i.i.i..lr.ph89_crit_edge, %.lr.ph89.preheader
  %215 = phi i32 [ %233, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ 0, %.lr.ph89.preheader ]
  %216 = phi i32 [ %232, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ 0, %.lr.ph89.preheader ]
  %217 = phi i32 [ %231, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ %93, %.lr.ph89.preheader ]
  %218 = shl i32 %216, 1
  %219 = shl i32 %217, 1
  %220 = icmp ugt i32 %219, %68
  br i1 %220, label %221, label %224

221:                                              ; preds = %.lr.ph89
  %222 = sub nuw i32 %219, %68, !spirv.Decorations !416
  %223 = add i32 %218, 1
  br label %.preheader.i7.i.i.i

224:                                              ; preds = %.lr.ph89
  %225 = icmp eq i32 %219, %68
  br i1 %225, label %226, label %..preheader.i7.i.i.i_crit_edge

..preheader.i7.i.i.i_crit_edge:                   ; preds = %224
  br label %.preheader.i7.i.i.i

226:                                              ; preds = %224
  %227 = add i32 %218, 1
  %228 = xor i32 %215, -1
  %229 = add i32 %212, %228
  %230 = shl i32 %227, %229
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

.preheader.i7.i.i.i:                              ; preds = %..preheader.i7.i.i.i_crit_edge, %221
  %231 = phi i32 [ %222, %221 ], [ %219, %..preheader.i7.i.i.i_crit_edge ]
  %232 = phi i32 [ %223, %221 ], [ %218, %..preheader.i7.i.i.i_crit_edge ]
  %233 = add nuw i32 %215, 1, !spirv.Decorations !416
  %234 = icmp ult i32 %233, %212
  br i1 %234, label %.preheader.i7.i.i.i..lr.ph89_crit_edge, label %.preheader.i7.i.i.i._crit_edge

.preheader.i7.i.i.i..lr.ph89_crit_edge:           ; preds = %.preheader.i7.i.i.i
  br label %.lr.ph89

.preheader.i7.i.i.i._crit_edge:                   ; preds = %.preheader.i7.i.i.i
  %235 = or i32 %232, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i:        ; preds = %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge, %.preheader.i7.i.i.i._crit_edge, %226
  %236 = phi i32 [ %230, %226 ], [ %235, %.preheader.i7.i.i.i._crit_edge ], [ 0, %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge ]
  %237 = or i32 %213, %236
  %238 = and i32 %237, 7
  %239 = lshr i32 %237, 3
  %240 = icmp eq i32 %238, 0
  br i1 %240, label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge
  %.ph = phi i32 [ %239, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge ], [ %177, %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge ]
  %241 = icmp ugt i32 %.ph, 8388606
  %242 = sext i1 %241 to i32
  br label %260

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  %243 = icmp ugt i32 %238, 4
  %244 = and i32 %237, 15
  %.not = icmp eq i32 %244, 12
  %or.cond = or i1 %243, %.not
  %cond.fr = freeze i1 %or.cond
  %245 = add nuw nsw i32 %239, 1, !spirv.Decorations !414
  %246 = icmp ugt i32 %237, 67108855
  %247 = select i1 %246, i32 0, i32 %245
  %248 = sext i1 %246 to i32
  %249 = sext i1 %cond.fr to i32
  %250 = sext i1 %246 to i32
  %251 = sext i1 %cond.fr to i32
  br i1 %cond.fr, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
  br label %260

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
  br label %252

252:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge, %._crit_edge170, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118
  %253 = phi i32 [ %198, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %247, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %206, %._crit_edge170 ]
  %254 = phi i32 [ %199, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %248, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %207, %._crit_edge170 ]
  %255 = phi i32 [ -1, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %249, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %208, %._crit_edge170 ]
  %256 = icmp ne i32 %255, 0
  %257 = icmp ne i32 %254, 0
  %258 = sext i1 %257 to i32
  %259 = sext i1 %256 to i32
  br label %260

260:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172, %._crit_edge171, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, %252
  %261 = phi i32 [ %258, %252 ], [ %250, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ %242, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %209, %._crit_edge171 ]
  %262 = phi i32 [ %259, %252 ], [ %251, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ 0, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %210, %._crit_edge171 ]
  %263 = phi i32 [ %253, %252 ], [ %239, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ %.ph, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %177, %._crit_edge171 ]
  %264 = icmp ne i32 %262, 0
  %265 = icmp ne i32 %261, 0
  %266 = and i1 %264, %265
  %267 = select i1 %266, i32 8388608, i32 0
  %268 = and i32 %26, -2147483648
  %269 = or i32 %268, %267
  %270 = or i32 %269, %263
  %271 = bitcast i32 %270 to float
  br label %__imf_fdiv_rn.exit

.preheader.i.i.i:                                 ; preds = %.preheader.i.i.i..preheader.i.i.i_crit_edge, %.preheader.i.i.i.preheader
  %272 = phi i32 [ %276, %.preheader.i.i.i..preheader.i.i.i_crit_edge ], [ 0, %.preheader.i.i.i.preheader ]
  %273 = phi i32 [ %274, %.preheader.i.i.i..preheader.i.i.i_crit_edge ], [ %66, %.preheader.i.i.i.preheader ]
  %274 = shl nuw nsw i32 %273, 1, !spirv.Decorations !414
  %275 = icmp ult i32 %274, %68
  %276 = add i32 %272, 1
  br i1 %275, label %.preheader.i.i.i..preheader.i.i.i_crit_edge, label %277

.preheader.i.i.i..preheader.i.i.i_crit_edge:      ; preds = %.preheader.i.i.i
  br label %.preheader.i.i.i

277:                                              ; preds = %.preheader.i.i.i
  %278 = xor i32 %272, -1
  %279 = add i32 %64, %278
  %280 = icmp sgt i32 %279, 127
  br i1 %280, label %281, label %283

281:                                              ; preds = %277
  %282 = icmp sgt i32 %26, -1
  %.51 = select i1 %282, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

283:                                              ; preds = %277
  %284 = icmp sgt i32 %279, -127
  br i1 %284, label %285, label %580

285:                                              ; preds = %283
  %286 = add nsw i32 %279, 127, !spirv.Decorations !411
  %287 = shl i32 %66, %272
  %288 = icmp eq i32 %287, 0
  br i1 %288, label %..critedge53_crit_edge, label %.preheader.i4.i.i.i.preheader

..critedge53_crit_edge:                           ; preds = %285
  br label %.critedge53

.preheader.i4.i.i.i.preheader:                    ; preds = %285
  %289 = shl i32 %287, 1
  %290 = icmp ugt i32 %289, %68
  br i1 %290, label %291, label %293

291:                                              ; preds = %.preheader.i4.i.i.i.preheader
  %292 = sub nuw i32 %289, %68, !spirv.Decorations !416
  br label %.preheader.i4.i.i.i

293:                                              ; preds = %.preheader.i4.i.i.i.preheader
  %294 = icmp eq i32 %289, %68
  br i1 %294, label %._crit_edge173, label %..preheader.i4.i.i.i_crit_edge

..preheader.i4.i.i.i_crit_edge:                   ; preds = %293
  br label %.preheader.i4.i.i.i

._crit_edge173:                                   ; preds = %293
  br label %295

295:                                              ; preds = %._crit_edge199, %._crit_edge198, %._crit_edge197, %._crit_edge196, %._crit_edge195, %._crit_edge194, %._crit_edge193, %._crit_edge192, %._crit_edge191, %._crit_edge190, %._crit_edge189, %._crit_edge188, %._crit_edge187, %._crit_edge186, %._crit_edge185, %._crit_edge184, %._crit_edge183, %._crit_edge182, %._crit_edge181, %._crit_edge180, %._crit_edge179, %._crit_edge178, %._crit_edge177, %._crit_edge176, %._crit_edge175, %._crit_edge174, %._crit_edge173
  %.lcssa95.neg = phi i8 [ 26, %._crit_edge173 ], [ 25, %._crit_edge174 ], [ 24, %._crit_edge175 ], [ 23, %._crit_edge176 ], [ 22, %._crit_edge177 ], [ 21, %._crit_edge178 ], [ 20, %._crit_edge179 ], [ 19, %._crit_edge180 ], [ 18, %._crit_edge181 ], [ 17, %._crit_edge182 ], [ 16, %._crit_edge183 ], [ 15, %._crit_edge184 ], [ 14, %._crit_edge185 ], [ 13, %._crit_edge186 ], [ 12, %._crit_edge187 ], [ 11, %._crit_edge188 ], [ 10, %._crit_edge189 ], [ 9, %._crit_edge190 ], [ 8, %._crit_edge191 ], [ 7, %._crit_edge192 ], [ 6, %._crit_edge193 ], [ 5, %._crit_edge194 ], [ 4, %._crit_edge195 ], [ 3, %._crit_edge196 ], [ 2, %._crit_edge197 ], [ 1, %._crit_edge198 ], [ 0, %._crit_edge199 ]
  %.lcssa = phi i32 [ 0, %._crit_edge173 ], [ %.demoted.zext, %._crit_edge174 ], [ %309, %._crit_edge175 ], [ %319, %._crit_edge176 ], [ %329, %._crit_edge177 ], [ %339, %._crit_edge178 ], [ %349, %._crit_edge179 ], [ %359, %._crit_edge180 ], [ %369, %._crit_edge181 ], [ %379, %._crit_edge182 ], [ %389, %._crit_edge183 ], [ %399, %._crit_edge184 ], [ %409, %._crit_edge185 ], [ %419, %._crit_edge186 ], [ %429, %._crit_edge187 ], [ %439, %._crit_edge188 ], [ %449, %._crit_edge189 ], [ %459, %._crit_edge190 ], [ %469, %._crit_edge191 ], [ %479, %._crit_edge192 ], [ %489, %._crit_edge193 ], [ %499, %._crit_edge194 ], [ %509, %._crit_edge195 ], [ %519, %._crit_edge196 ], [ %529, %._crit_edge197 ], [ %539, %._crit_edge198 ], [ %549, %._crit_edge199 ]
  %.demoted.zext167 = zext i8 %.lcssa95.neg to i32
  %296 = or i32 %.lcssa, 1
  %297 = shl i32 %296, %.demoted.zext167
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i

.preheader.i4.i.i.i:                              ; preds = %..preheader.i4.i.i.i_crit_edge, %291
  %298 = phi i32 [ %292, %291 ], [ %289, %..preheader.i4.i.i.i_crit_edge ]
  %299 = phi i8 [ 2, %291 ], [ 0, %..preheader.i4.i.i.i_crit_edge ]
  %.demoted.zext = zext i8 %299 to i32
  %300 = shl i32 %298, 1
  %301 = icmp ugt i32 %300, %68
  br i1 %301, label %304, label %302

302:                                              ; preds = %.preheader.i4.i.i.i
  %303 = icmp eq i32 %300, %68
  br i1 %303, label %._crit_edge174, label %..preheader.i4.i.i.i.1_crit_edge

..preheader.i4.i.i.i.1_crit_edge:                 ; preds = %302
  br label %.preheader.i4.i.i.i.1

._crit_edge174:                                   ; preds = %302
  br label %295

304:                                              ; preds = %.preheader.i4.i.i.i
  %305 = sub nuw i32 %300, %68, !spirv.Decorations !416
  %306 = or i8 %299, 1
  br label %.preheader.i4.i.i.i.1

.preheader.i4.i.i.i.1:                            ; preds = %..preheader.i4.i.i.i.1_crit_edge, %304
  %307 = phi i32 [ %305, %304 ], [ %300, %..preheader.i4.i.i.i.1_crit_edge ]
  %308 = phi i8 [ %306, %304 ], [ %299, %..preheader.i4.i.i.i.1_crit_edge ]
  %.demoted.zext166 = zext i8 %308 to i32
  %309 = shl nsw i32 %.demoted.zext166, 1
  %310 = shl i32 %307, 1
  %311 = icmp ugt i32 %310, %68
  br i1 %311, label %314, label %312

312:                                              ; preds = %.preheader.i4.i.i.i.1
  %313 = icmp eq i32 %310, %68
  br i1 %313, label %._crit_edge175, label %..preheader.i4.i.i.i.2_crit_edge

..preheader.i4.i.i.i.2_crit_edge:                 ; preds = %312
  br label %.preheader.i4.i.i.i.2

._crit_edge175:                                   ; preds = %312
  br label %295

314:                                              ; preds = %.preheader.i4.i.i.i.1
  %315 = sub nuw i32 %310, %68, !spirv.Decorations !416
  %316 = add i32 %309, 1
  br label %.preheader.i4.i.i.i.2

.preheader.i4.i.i.i.2:                            ; preds = %..preheader.i4.i.i.i.2_crit_edge, %314
  %317 = phi i32 [ %315, %314 ], [ %310, %..preheader.i4.i.i.i.2_crit_edge ]
  %318 = phi i32 [ %316, %314 ], [ %309, %..preheader.i4.i.i.i.2_crit_edge ]
  %319 = shl i32 %318, 1
  %320 = shl i32 %317, 1
  %321 = icmp ugt i32 %320, %68
  br i1 %321, label %324, label %322

322:                                              ; preds = %.preheader.i4.i.i.i.2
  %323 = icmp eq i32 %320, %68
  br i1 %323, label %._crit_edge176, label %..preheader.i4.i.i.i.3_crit_edge

..preheader.i4.i.i.i.3_crit_edge:                 ; preds = %322
  br label %.preheader.i4.i.i.i.3

._crit_edge176:                                   ; preds = %322
  br label %295

324:                                              ; preds = %.preheader.i4.i.i.i.2
  %325 = sub nuw i32 %320, %68, !spirv.Decorations !416
  %326 = add i32 %319, 1
  br label %.preheader.i4.i.i.i.3

.preheader.i4.i.i.i.3:                            ; preds = %..preheader.i4.i.i.i.3_crit_edge, %324
  %327 = phi i32 [ %325, %324 ], [ %320, %..preheader.i4.i.i.i.3_crit_edge ]
  %328 = phi i32 [ %326, %324 ], [ %319, %..preheader.i4.i.i.i.3_crit_edge ]
  %329 = shl i32 %328, 1
  %330 = shl i32 %327, 1
  %331 = icmp ugt i32 %330, %68
  br i1 %331, label %334, label %332

332:                                              ; preds = %.preheader.i4.i.i.i.3
  %333 = icmp eq i32 %330, %68
  br i1 %333, label %._crit_edge177, label %..preheader.i4.i.i.i.4_crit_edge

..preheader.i4.i.i.i.4_crit_edge:                 ; preds = %332
  br label %.preheader.i4.i.i.i.4

._crit_edge177:                                   ; preds = %332
  br label %295

334:                                              ; preds = %.preheader.i4.i.i.i.3
  %335 = sub nuw i32 %330, %68, !spirv.Decorations !416
  %336 = add i32 %329, 1
  br label %.preheader.i4.i.i.i.4

.preheader.i4.i.i.i.4:                            ; preds = %..preheader.i4.i.i.i.4_crit_edge, %334
  %337 = phi i32 [ %335, %334 ], [ %330, %..preheader.i4.i.i.i.4_crit_edge ]
  %338 = phi i32 [ %336, %334 ], [ %329, %..preheader.i4.i.i.i.4_crit_edge ]
  %339 = shl i32 %338, 1
  %340 = shl i32 %337, 1
  %341 = icmp ugt i32 %340, %68
  br i1 %341, label %344, label %342

342:                                              ; preds = %.preheader.i4.i.i.i.4
  %343 = icmp eq i32 %340, %68
  br i1 %343, label %._crit_edge178, label %..preheader.i4.i.i.i.5_crit_edge

..preheader.i4.i.i.i.5_crit_edge:                 ; preds = %342
  br label %.preheader.i4.i.i.i.5

._crit_edge178:                                   ; preds = %342
  br label %295

344:                                              ; preds = %.preheader.i4.i.i.i.4
  %345 = sub nuw i32 %340, %68, !spirv.Decorations !416
  %346 = add i32 %339, 1
  br label %.preheader.i4.i.i.i.5

.preheader.i4.i.i.i.5:                            ; preds = %..preheader.i4.i.i.i.5_crit_edge, %344
  %347 = phi i32 [ %345, %344 ], [ %340, %..preheader.i4.i.i.i.5_crit_edge ]
  %348 = phi i32 [ %346, %344 ], [ %339, %..preheader.i4.i.i.i.5_crit_edge ]
  %349 = shl i32 %348, 1
  %350 = shl i32 %347, 1
  %351 = icmp ugt i32 %350, %68
  br i1 %351, label %354, label %352

352:                                              ; preds = %.preheader.i4.i.i.i.5
  %353 = icmp eq i32 %350, %68
  br i1 %353, label %._crit_edge179, label %..preheader.i4.i.i.i.6_crit_edge

..preheader.i4.i.i.i.6_crit_edge:                 ; preds = %352
  br label %.preheader.i4.i.i.i.6

._crit_edge179:                                   ; preds = %352
  br label %295

354:                                              ; preds = %.preheader.i4.i.i.i.5
  %355 = sub nuw i32 %350, %68, !spirv.Decorations !416
  %356 = add i32 %349, 1
  br label %.preheader.i4.i.i.i.6

.preheader.i4.i.i.i.6:                            ; preds = %..preheader.i4.i.i.i.6_crit_edge, %354
  %357 = phi i32 [ %355, %354 ], [ %350, %..preheader.i4.i.i.i.6_crit_edge ]
  %358 = phi i32 [ %356, %354 ], [ %349, %..preheader.i4.i.i.i.6_crit_edge ]
  %359 = shl i32 %358, 1
  %360 = shl i32 %357, 1
  %361 = icmp ugt i32 %360, %68
  br i1 %361, label %364, label %362

362:                                              ; preds = %.preheader.i4.i.i.i.6
  %363 = icmp eq i32 %360, %68
  br i1 %363, label %._crit_edge180, label %..preheader.i4.i.i.i.7_crit_edge

..preheader.i4.i.i.i.7_crit_edge:                 ; preds = %362
  br label %.preheader.i4.i.i.i.7

._crit_edge180:                                   ; preds = %362
  br label %295

364:                                              ; preds = %.preheader.i4.i.i.i.6
  %365 = sub nuw i32 %360, %68, !spirv.Decorations !416
  %366 = add i32 %359, 1
  br label %.preheader.i4.i.i.i.7

.preheader.i4.i.i.i.7:                            ; preds = %..preheader.i4.i.i.i.7_crit_edge, %364
  %367 = phi i32 [ %365, %364 ], [ %360, %..preheader.i4.i.i.i.7_crit_edge ]
  %368 = phi i32 [ %366, %364 ], [ %359, %..preheader.i4.i.i.i.7_crit_edge ]
  %369 = shl i32 %368, 1
  %370 = shl i32 %367, 1
  %371 = icmp ugt i32 %370, %68
  br i1 %371, label %374, label %372

372:                                              ; preds = %.preheader.i4.i.i.i.7
  %373 = icmp eq i32 %370, %68
  br i1 %373, label %._crit_edge181, label %..preheader.i4.i.i.i.8_crit_edge

..preheader.i4.i.i.i.8_crit_edge:                 ; preds = %372
  br label %.preheader.i4.i.i.i.8

._crit_edge181:                                   ; preds = %372
  br label %295

374:                                              ; preds = %.preheader.i4.i.i.i.7
  %375 = sub nuw i32 %370, %68, !spirv.Decorations !416
  %376 = add i32 %369, 1
  br label %.preheader.i4.i.i.i.8

.preheader.i4.i.i.i.8:                            ; preds = %..preheader.i4.i.i.i.8_crit_edge, %374
  %377 = phi i32 [ %375, %374 ], [ %370, %..preheader.i4.i.i.i.8_crit_edge ]
  %378 = phi i32 [ %376, %374 ], [ %369, %..preheader.i4.i.i.i.8_crit_edge ]
  %379 = shl i32 %378, 1
  %380 = shl i32 %377, 1
  %381 = icmp ugt i32 %380, %68
  br i1 %381, label %384, label %382

382:                                              ; preds = %.preheader.i4.i.i.i.8
  %383 = icmp eq i32 %380, %68
  br i1 %383, label %._crit_edge182, label %..preheader.i4.i.i.i.9_crit_edge

..preheader.i4.i.i.i.9_crit_edge:                 ; preds = %382
  br label %.preheader.i4.i.i.i.9

._crit_edge182:                                   ; preds = %382
  br label %295

384:                                              ; preds = %.preheader.i4.i.i.i.8
  %385 = sub nuw i32 %380, %68, !spirv.Decorations !416
  %386 = add i32 %379, 1
  br label %.preheader.i4.i.i.i.9

.preheader.i4.i.i.i.9:                            ; preds = %..preheader.i4.i.i.i.9_crit_edge, %384
  %387 = phi i32 [ %385, %384 ], [ %380, %..preheader.i4.i.i.i.9_crit_edge ]
  %388 = phi i32 [ %386, %384 ], [ %379, %..preheader.i4.i.i.i.9_crit_edge ]
  %389 = shl i32 %388, 1
  %390 = shl i32 %387, 1
  %391 = icmp ugt i32 %390, %68
  br i1 %391, label %394, label %392

392:                                              ; preds = %.preheader.i4.i.i.i.9
  %393 = icmp eq i32 %390, %68
  br i1 %393, label %._crit_edge183, label %..preheader.i4.i.i.i.10_crit_edge

..preheader.i4.i.i.i.10_crit_edge:                ; preds = %392
  br label %.preheader.i4.i.i.i.10

._crit_edge183:                                   ; preds = %392
  br label %295

394:                                              ; preds = %.preheader.i4.i.i.i.9
  %395 = sub nuw i32 %390, %68, !spirv.Decorations !416
  %396 = add i32 %389, 1
  br label %.preheader.i4.i.i.i.10

.preheader.i4.i.i.i.10:                           ; preds = %..preheader.i4.i.i.i.10_crit_edge, %394
  %397 = phi i32 [ %395, %394 ], [ %390, %..preheader.i4.i.i.i.10_crit_edge ]
  %398 = phi i32 [ %396, %394 ], [ %389, %..preheader.i4.i.i.i.10_crit_edge ]
  %399 = shl i32 %398, 1
  %400 = shl i32 %397, 1
  %401 = icmp ugt i32 %400, %68
  br i1 %401, label %404, label %402

402:                                              ; preds = %.preheader.i4.i.i.i.10
  %403 = icmp eq i32 %400, %68
  br i1 %403, label %._crit_edge184, label %..preheader.i4.i.i.i.11_crit_edge

..preheader.i4.i.i.i.11_crit_edge:                ; preds = %402
  br label %.preheader.i4.i.i.i.11

._crit_edge184:                                   ; preds = %402
  br label %295

404:                                              ; preds = %.preheader.i4.i.i.i.10
  %405 = sub nuw i32 %400, %68, !spirv.Decorations !416
  %406 = add i32 %399, 1
  br label %.preheader.i4.i.i.i.11

.preheader.i4.i.i.i.11:                           ; preds = %..preheader.i4.i.i.i.11_crit_edge, %404
  %407 = phi i32 [ %405, %404 ], [ %400, %..preheader.i4.i.i.i.11_crit_edge ]
  %408 = phi i32 [ %406, %404 ], [ %399, %..preheader.i4.i.i.i.11_crit_edge ]
  %409 = shl i32 %408, 1
  %410 = shl i32 %407, 1
  %411 = icmp ugt i32 %410, %68
  br i1 %411, label %414, label %412

412:                                              ; preds = %.preheader.i4.i.i.i.11
  %413 = icmp eq i32 %410, %68
  br i1 %413, label %._crit_edge185, label %..preheader.i4.i.i.i.12_crit_edge

..preheader.i4.i.i.i.12_crit_edge:                ; preds = %412
  br label %.preheader.i4.i.i.i.12

._crit_edge185:                                   ; preds = %412
  br label %295

414:                                              ; preds = %.preheader.i4.i.i.i.11
  %415 = sub nuw i32 %410, %68, !spirv.Decorations !416
  %416 = add i32 %409, 1
  br label %.preheader.i4.i.i.i.12

.preheader.i4.i.i.i.12:                           ; preds = %..preheader.i4.i.i.i.12_crit_edge, %414
  %417 = phi i32 [ %415, %414 ], [ %410, %..preheader.i4.i.i.i.12_crit_edge ]
  %418 = phi i32 [ %416, %414 ], [ %409, %..preheader.i4.i.i.i.12_crit_edge ]
  %419 = shl i32 %418, 1
  %420 = shl i32 %417, 1
  %421 = icmp ugt i32 %420, %68
  br i1 %421, label %424, label %422

422:                                              ; preds = %.preheader.i4.i.i.i.12
  %423 = icmp eq i32 %420, %68
  br i1 %423, label %._crit_edge186, label %..preheader.i4.i.i.i.13_crit_edge

..preheader.i4.i.i.i.13_crit_edge:                ; preds = %422
  br label %.preheader.i4.i.i.i.13

._crit_edge186:                                   ; preds = %422
  br label %295

424:                                              ; preds = %.preheader.i4.i.i.i.12
  %425 = sub nuw i32 %420, %68, !spirv.Decorations !416
  %426 = add i32 %419, 1
  br label %.preheader.i4.i.i.i.13

.preheader.i4.i.i.i.13:                           ; preds = %..preheader.i4.i.i.i.13_crit_edge, %424
  %427 = phi i32 [ %425, %424 ], [ %420, %..preheader.i4.i.i.i.13_crit_edge ]
  %428 = phi i32 [ %426, %424 ], [ %419, %..preheader.i4.i.i.i.13_crit_edge ]
  %429 = shl i32 %428, 1
  %430 = shl i32 %427, 1
  %431 = icmp ugt i32 %430, %68
  br i1 %431, label %434, label %432

432:                                              ; preds = %.preheader.i4.i.i.i.13
  %433 = icmp eq i32 %430, %68
  br i1 %433, label %._crit_edge187, label %..preheader.i4.i.i.i.14_crit_edge

..preheader.i4.i.i.i.14_crit_edge:                ; preds = %432
  br label %.preheader.i4.i.i.i.14

._crit_edge187:                                   ; preds = %432
  br label %295

434:                                              ; preds = %.preheader.i4.i.i.i.13
  %435 = sub nuw i32 %430, %68, !spirv.Decorations !416
  %436 = add i32 %429, 1
  br label %.preheader.i4.i.i.i.14

.preheader.i4.i.i.i.14:                           ; preds = %..preheader.i4.i.i.i.14_crit_edge, %434
  %437 = phi i32 [ %435, %434 ], [ %430, %..preheader.i4.i.i.i.14_crit_edge ]
  %438 = phi i32 [ %436, %434 ], [ %429, %..preheader.i4.i.i.i.14_crit_edge ]
  %439 = shl i32 %438, 1
  %440 = shl i32 %437, 1
  %441 = icmp ugt i32 %440, %68
  br i1 %441, label %444, label %442

442:                                              ; preds = %.preheader.i4.i.i.i.14
  %443 = icmp eq i32 %440, %68
  br i1 %443, label %._crit_edge188, label %..preheader.i4.i.i.i.15_crit_edge

..preheader.i4.i.i.i.15_crit_edge:                ; preds = %442
  br label %.preheader.i4.i.i.i.15

._crit_edge188:                                   ; preds = %442
  br label %295

444:                                              ; preds = %.preheader.i4.i.i.i.14
  %445 = sub nuw i32 %440, %68, !spirv.Decorations !416
  %446 = add i32 %439, 1
  br label %.preheader.i4.i.i.i.15

.preheader.i4.i.i.i.15:                           ; preds = %..preheader.i4.i.i.i.15_crit_edge, %444
  %447 = phi i32 [ %445, %444 ], [ %440, %..preheader.i4.i.i.i.15_crit_edge ]
  %448 = phi i32 [ %446, %444 ], [ %439, %..preheader.i4.i.i.i.15_crit_edge ]
  %449 = shl i32 %448, 1
  %450 = shl i32 %447, 1
  %451 = icmp ugt i32 %450, %68
  br i1 %451, label %454, label %452

452:                                              ; preds = %.preheader.i4.i.i.i.15
  %453 = icmp eq i32 %450, %68
  br i1 %453, label %._crit_edge189, label %..preheader.i4.i.i.i.16_crit_edge

..preheader.i4.i.i.i.16_crit_edge:                ; preds = %452
  br label %.preheader.i4.i.i.i.16

._crit_edge189:                                   ; preds = %452
  br label %295

454:                                              ; preds = %.preheader.i4.i.i.i.15
  %455 = sub nuw i32 %450, %68, !spirv.Decorations !416
  %456 = add i32 %449, 1
  br label %.preheader.i4.i.i.i.16

.preheader.i4.i.i.i.16:                           ; preds = %..preheader.i4.i.i.i.16_crit_edge, %454
  %457 = phi i32 [ %455, %454 ], [ %450, %..preheader.i4.i.i.i.16_crit_edge ]
  %458 = phi i32 [ %456, %454 ], [ %449, %..preheader.i4.i.i.i.16_crit_edge ]
  %459 = shl i32 %458, 1
  %460 = shl i32 %457, 1
  %461 = icmp ugt i32 %460, %68
  br i1 %461, label %464, label %462

462:                                              ; preds = %.preheader.i4.i.i.i.16
  %463 = icmp eq i32 %460, %68
  br i1 %463, label %._crit_edge190, label %..preheader.i4.i.i.i.17_crit_edge

..preheader.i4.i.i.i.17_crit_edge:                ; preds = %462
  br label %.preheader.i4.i.i.i.17

._crit_edge190:                                   ; preds = %462
  br label %295

464:                                              ; preds = %.preheader.i4.i.i.i.16
  %465 = sub nuw i32 %460, %68, !spirv.Decorations !416
  %466 = add i32 %459, 1
  br label %.preheader.i4.i.i.i.17

.preheader.i4.i.i.i.17:                           ; preds = %..preheader.i4.i.i.i.17_crit_edge, %464
  %467 = phi i32 [ %465, %464 ], [ %460, %..preheader.i4.i.i.i.17_crit_edge ]
  %468 = phi i32 [ %466, %464 ], [ %459, %..preheader.i4.i.i.i.17_crit_edge ]
  %469 = shl i32 %468, 1
  %470 = shl i32 %467, 1
  %471 = icmp ugt i32 %470, %68
  br i1 %471, label %474, label %472

472:                                              ; preds = %.preheader.i4.i.i.i.17
  %473 = icmp eq i32 %470, %68
  br i1 %473, label %._crit_edge191, label %..preheader.i4.i.i.i.18_crit_edge

..preheader.i4.i.i.i.18_crit_edge:                ; preds = %472
  br label %.preheader.i4.i.i.i.18

._crit_edge191:                                   ; preds = %472
  br label %295

474:                                              ; preds = %.preheader.i4.i.i.i.17
  %475 = sub nuw i32 %470, %68, !spirv.Decorations !416
  %476 = add i32 %469, 1
  br label %.preheader.i4.i.i.i.18

.preheader.i4.i.i.i.18:                           ; preds = %..preheader.i4.i.i.i.18_crit_edge, %474
  %477 = phi i32 [ %475, %474 ], [ %470, %..preheader.i4.i.i.i.18_crit_edge ]
  %478 = phi i32 [ %476, %474 ], [ %469, %..preheader.i4.i.i.i.18_crit_edge ]
  %479 = shl i32 %478, 1
  %480 = shl i32 %477, 1
  %481 = icmp ugt i32 %480, %68
  br i1 %481, label %484, label %482

482:                                              ; preds = %.preheader.i4.i.i.i.18
  %483 = icmp eq i32 %480, %68
  br i1 %483, label %._crit_edge192, label %..preheader.i4.i.i.i.19_crit_edge

..preheader.i4.i.i.i.19_crit_edge:                ; preds = %482
  br label %.preheader.i4.i.i.i.19

._crit_edge192:                                   ; preds = %482
  br label %295

484:                                              ; preds = %.preheader.i4.i.i.i.18
  %485 = sub nuw i32 %480, %68, !spirv.Decorations !416
  %486 = add i32 %479, 1
  br label %.preheader.i4.i.i.i.19

.preheader.i4.i.i.i.19:                           ; preds = %..preheader.i4.i.i.i.19_crit_edge, %484
  %487 = phi i32 [ %485, %484 ], [ %480, %..preheader.i4.i.i.i.19_crit_edge ]
  %488 = phi i32 [ %486, %484 ], [ %479, %..preheader.i4.i.i.i.19_crit_edge ]
  %489 = shl i32 %488, 1
  %490 = shl i32 %487, 1
  %491 = icmp ugt i32 %490, %68
  br i1 %491, label %494, label %492

492:                                              ; preds = %.preheader.i4.i.i.i.19
  %493 = icmp eq i32 %490, %68
  br i1 %493, label %._crit_edge193, label %..preheader.i4.i.i.i.20_crit_edge

..preheader.i4.i.i.i.20_crit_edge:                ; preds = %492
  br label %.preheader.i4.i.i.i.20

._crit_edge193:                                   ; preds = %492
  br label %295

494:                                              ; preds = %.preheader.i4.i.i.i.19
  %495 = sub nuw i32 %490, %68, !spirv.Decorations !416
  %496 = add i32 %489, 1
  br label %.preheader.i4.i.i.i.20

.preheader.i4.i.i.i.20:                           ; preds = %..preheader.i4.i.i.i.20_crit_edge, %494
  %497 = phi i32 [ %495, %494 ], [ %490, %..preheader.i4.i.i.i.20_crit_edge ]
  %498 = phi i32 [ %496, %494 ], [ %489, %..preheader.i4.i.i.i.20_crit_edge ]
  %499 = shl i32 %498, 1
  %500 = shl i32 %497, 1
  %501 = icmp ugt i32 %500, %68
  br i1 %501, label %504, label %502

502:                                              ; preds = %.preheader.i4.i.i.i.20
  %503 = icmp eq i32 %500, %68
  br i1 %503, label %._crit_edge194, label %..preheader.i4.i.i.i.21_crit_edge

..preheader.i4.i.i.i.21_crit_edge:                ; preds = %502
  br label %.preheader.i4.i.i.i.21

._crit_edge194:                                   ; preds = %502
  br label %295

504:                                              ; preds = %.preheader.i4.i.i.i.20
  %505 = sub nuw i32 %500, %68, !spirv.Decorations !416
  %506 = add i32 %499, 1
  br label %.preheader.i4.i.i.i.21

.preheader.i4.i.i.i.21:                           ; preds = %..preheader.i4.i.i.i.21_crit_edge, %504
  %507 = phi i32 [ %505, %504 ], [ %500, %..preheader.i4.i.i.i.21_crit_edge ]
  %508 = phi i32 [ %506, %504 ], [ %499, %..preheader.i4.i.i.i.21_crit_edge ]
  %509 = shl i32 %508, 1
  %510 = shl i32 %507, 1
  %511 = icmp ugt i32 %510, %68
  br i1 %511, label %514, label %512

512:                                              ; preds = %.preheader.i4.i.i.i.21
  %513 = icmp eq i32 %510, %68
  br i1 %513, label %._crit_edge195, label %..preheader.i4.i.i.i.22_crit_edge

..preheader.i4.i.i.i.22_crit_edge:                ; preds = %512
  br label %.preheader.i4.i.i.i.22

._crit_edge195:                                   ; preds = %512
  br label %295

514:                                              ; preds = %.preheader.i4.i.i.i.21
  %515 = sub nuw i32 %510, %68, !spirv.Decorations !416
  %516 = add i32 %509, 1
  br label %.preheader.i4.i.i.i.22

.preheader.i4.i.i.i.22:                           ; preds = %..preheader.i4.i.i.i.22_crit_edge, %514
  %517 = phi i32 [ %515, %514 ], [ %510, %..preheader.i4.i.i.i.22_crit_edge ]
  %518 = phi i32 [ %516, %514 ], [ %509, %..preheader.i4.i.i.i.22_crit_edge ]
  %519 = shl i32 %518, 1
  %520 = shl i32 %517, 1
  %521 = icmp ugt i32 %520, %68
  br i1 %521, label %524, label %522

522:                                              ; preds = %.preheader.i4.i.i.i.22
  %523 = icmp eq i32 %520, %68
  br i1 %523, label %._crit_edge196, label %..preheader.i4.i.i.i.23_crit_edge

..preheader.i4.i.i.i.23_crit_edge:                ; preds = %522
  br label %.preheader.i4.i.i.i.23

._crit_edge196:                                   ; preds = %522
  br label %295

524:                                              ; preds = %.preheader.i4.i.i.i.22
  %525 = sub nuw i32 %520, %68, !spirv.Decorations !416
  %526 = add i32 %519, 1
  br label %.preheader.i4.i.i.i.23

.preheader.i4.i.i.i.23:                           ; preds = %..preheader.i4.i.i.i.23_crit_edge, %524
  %527 = phi i32 [ %525, %524 ], [ %520, %..preheader.i4.i.i.i.23_crit_edge ]
  %528 = phi i32 [ %526, %524 ], [ %519, %..preheader.i4.i.i.i.23_crit_edge ]
  %529 = shl i32 %528, 1
  %530 = shl i32 %527, 1
  %531 = icmp ugt i32 %530, %68
  br i1 %531, label %534, label %532

532:                                              ; preds = %.preheader.i4.i.i.i.23
  %533 = icmp eq i32 %530, %68
  br i1 %533, label %._crit_edge197, label %..preheader.i4.i.i.i.24_crit_edge

..preheader.i4.i.i.i.24_crit_edge:                ; preds = %532
  br label %.preheader.i4.i.i.i.24

._crit_edge197:                                   ; preds = %532
  br label %295

534:                                              ; preds = %.preheader.i4.i.i.i.23
  %535 = sub nuw i32 %530, %68, !spirv.Decorations !416
  %536 = add i32 %529, 1
  br label %.preheader.i4.i.i.i.24

.preheader.i4.i.i.i.24:                           ; preds = %..preheader.i4.i.i.i.24_crit_edge, %534
  %537 = phi i32 [ %535, %534 ], [ %530, %..preheader.i4.i.i.i.24_crit_edge ]
  %538 = phi i32 [ %536, %534 ], [ %529, %..preheader.i4.i.i.i.24_crit_edge ]
  %539 = shl i32 %538, 1
  %540 = shl i32 %537, 1
  %541 = icmp ugt i32 %540, %68
  br i1 %541, label %544, label %542

542:                                              ; preds = %.preheader.i4.i.i.i.24
  %543 = icmp eq i32 %540, %68
  br i1 %543, label %._crit_edge198, label %..preheader.i4.i.i.i.25_crit_edge

..preheader.i4.i.i.i.25_crit_edge:                ; preds = %542
  br label %.preheader.i4.i.i.i.25

._crit_edge198:                                   ; preds = %542
  br label %295

544:                                              ; preds = %.preheader.i4.i.i.i.24
  %545 = sub nuw i32 %540, %68, !spirv.Decorations !416
  %546 = add i32 %539, 1
  br label %.preheader.i4.i.i.i.25

.preheader.i4.i.i.i.25:                           ; preds = %..preheader.i4.i.i.i.25_crit_edge, %544
  %547 = phi i32 [ %545, %544 ], [ %540, %..preheader.i4.i.i.i.25_crit_edge ]
  %548 = phi i32 [ %546, %544 ], [ %539, %..preheader.i4.i.i.i.25_crit_edge ]
  %549 = shl i32 %548, 1
  %550 = shl i32 %547, 1
  %551 = icmp ugt i32 %550, %68
  br i1 %551, label %554, label %552

552:                                              ; preds = %.preheader.i4.i.i.i.25
  %553 = icmp eq i32 %550, %68
  br i1 %553, label %._crit_edge199, label %..preheader.i4.i.i.i.26_crit_edge

..preheader.i4.i.i.i.26_crit_edge:                ; preds = %552
  br label %.preheader.i4.i.i.i.26

._crit_edge199:                                   ; preds = %552
  br label %295

554:                                              ; preds = %.preheader.i4.i.i.i.25
  %555 = add i32 %549, 1
  br label %.preheader.i4.i.i.i.26

.preheader.i4.i.i.i.26:                           ; preds = %..preheader.i4.i.i.i.26_crit_edge, %554
  %556 = phi i32 [ %555, %554 ], [ %549, %..preheader.i4.i.i.i.26_crit_edge ]
  %557 = or i32 %556, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i:        ; preds = %.preheader.i4.i.i.i.26, %295
  %558 = phi i32 [ %297, %295 ], [ %557, %.preheader.i4.i.i.i.26 ]
  %559 = lshr i32 %558, 3
  %560 = and i32 %559, 8388607
  %561 = and i32 %558, 7
  %562 = icmp eq i32 %561, 0
  br i1 %562, label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge, label %563

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  br label %.critedge53

563:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  %564 = icmp ugt i32 %561, 4
  %565 = and i32 %558, 15
  %.not10 = icmp eq i32 %565, 12
  %or.cond52 = or i1 %564, %.not10
  br i1 %or.cond52, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %..critedge53_crit_edge200

..critedge53_crit_edge200:                        ; preds = %563
  br label %.critedge53

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i: ; preds = %563
  %566 = add nuw nsw i32 %560, 1, !spirv.Decorations !414
  %567 = icmp eq i32 %560, 8388607
  br i1 %567, label %568, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  br label %.critedge53

568:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  %569 = add nsw i32 %279, 128, !spirv.Decorations !411
  %570 = icmp eq i32 %569, 255
  br i1 %570, label %571, label %..critedge53_crit_edge201

..critedge53_crit_edge201:                        ; preds = %568
  br label %.critedge53

571:                                              ; preds = %568
  %572 = icmp sgt i32 %26, -1
  %.54 = select i1 %572, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit

.critedge53:                                      ; preds = %..critedge53_crit_edge201, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge, %..critedge53_crit_edge200, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge, %..critedge53_crit_edge
  %573 = phi i32 [ %560, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge ], [ %560, %..critedge53_crit_edge200 ], [ %566, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge ], [ 0, %..critedge53_crit_edge201 ], [ 0, %..critedge53_crit_edge ]
  %574 = phi i32 [ %286, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge ], [ %286, %..critedge53_crit_edge200 ], [ %286, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge ], [ %569, %..critedge53_crit_edge201 ], [ %286, %..critedge53_crit_edge ]
  %575 = and i32 %26, -2147483648
  %576 = shl nuw nsw i32 %574, 23, !spirv.Decorations !414
  %577 = or i32 %575, %576
  %578 = or i32 %577, %573
  %579 = bitcast i32 %578 to float
  br label %__imf_fdiv_rn.exit

580:                                              ; preds = %283
  %581 = add i32 %272, -127
  %582 = sub i32 %581, %64
  %583 = add i32 %582, 1
  %584 = icmp ugt i32 %583, 22
  br i1 %584, label %585, label %593

585:                                              ; preds = %580
  %586 = icmp eq i32 %583, 23
  %587 = shl i32 %66, %276
  %588 = icmp ugt i32 %587, %68
  %589 = and i1 %586, %588
  %590 = icmp sgt i32 %26, -1
  br i1 %590, label %591, label %592

591:                                              ; preds = %585
  %spec.select55 = select i1 %589, float 0x36A0000000000000, float 0.000000e+00
  br label %__imf_fdiv_rn.exit

592:                                              ; preds = %585
  %spec.select56 = select i1 %589, float 0xB6A0000000000000, float -0.000000e+00
  br label %__imf_fdiv_rn.exit

593:                                              ; preds = %580
  %594 = shl i32 %66, %272
  %595 = sub nsw i32 25, %582, !spirv.Decorations !411
  %596 = icmp eq i32 %594, 0
  br i1 %596, label %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, label %.lr.ph.preheader

._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge: ; preds = %593
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread

.lr.ph.preheader:                                 ; preds = %593
  br label %.lr.ph

.lr.ph:                                           ; preds = %.preheader.i1.i.i.i..lr.ph_crit_edge, %.lr.ph.preheader
  %597 = phi i32 [ %615, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ 0, %.lr.ph.preheader ]
  %598 = phi i32 [ %614, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ 0, %.lr.ph.preheader ]
  %599 = phi i32 [ %613, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ %594, %.lr.ph.preheader ]
  %600 = shl i32 %598, 1
  %601 = shl i32 %599, 1
  %602 = icmp ugt i32 %601, %68
  br i1 %602, label %603, label %606

603:                                              ; preds = %.lr.ph
  %604 = sub nuw i32 %601, %68, !spirv.Decorations !416
  %605 = add i32 %600, 1
  br label %.preheader.i1.i.i.i

606:                                              ; preds = %.lr.ph
  %607 = icmp eq i32 %601, %68
  br i1 %607, label %608, label %..preheader.i1.i.i.i_crit_edge

..preheader.i1.i.i.i_crit_edge:                   ; preds = %606
  br label %.preheader.i1.i.i.i

608:                                              ; preds = %606
  %609 = add i32 %600, 1
  %610 = xor i32 %597, -1
  %611 = add i32 %595, %610
  %612 = shl i32 %609, %611
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i

.preheader.i1.i.i.i:                              ; preds = %..preheader.i1.i.i.i_crit_edge, %603
  %613 = phi i32 [ %604, %603 ], [ %601, %..preheader.i1.i.i.i_crit_edge ]
  %614 = phi i32 [ %605, %603 ], [ %600, %..preheader.i1.i.i.i_crit_edge ]
  %615 = add nuw i32 %597, 1, !spirv.Decorations !416
  %616 = icmp ult i32 %615, %595
  br i1 %616, label %.preheader.i1.i.i.i..lr.ph_crit_edge, label %.preheader.i1.i.i.i._crit_edge

.preheader.i1.i.i.i..lr.ph_crit_edge:             ; preds = %.preheader.i1.i.i.i
  br label %.lr.ph

.preheader.i1.i.i.i._crit_edge:                   ; preds = %.preheader.i1.i.i.i
  %617 = or i32 %614, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i

_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i:        ; preds = %.preheader.i1.i.i.i._crit_edge, %608
  %618 = phi i32 [ %612, %608 ], [ %617, %.preheader.i1.i.i.i._crit_edge ]
  %.fr = freeze i32 %618
  %619 = lshr i32 %.fr, 3
  %620 = and i32 %.fr, 7
  %621 = icmp eq i32 %620, 0
  br i1 %621, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %622 = icmp ugt i32 %.fr, 67108855
  %623 = sext i1 %622 to i32
  %624 = and i32 8388608, %623
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %625 = icmp ult i32 %620, 5
  %626 = and i32 %.fr, 15
  %.not9 = icmp ne i32 %626, 12
  %not.or.cond57 = and i1 %625, %.not9
  %627 = add nuw nsw i32 %619, 1, !spirv.Decorations !414
  %628 = icmp ugt i32 %.fr, 67108855
  %629 = select i1 %628, i32 0, i32 %627
  %630 = select i1 %628, i32 8388608, i32 0
  %631 = sext i1 %not.or.cond57 to i32
  %632 = sext i1 %not.or.cond57 to i32
  br i1 %not.or.cond57, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
  br label %638

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread
  %633 = phi i32 [ %630, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ 0, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ %624, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %634 = phi i32 [ %631, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ -1, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ -1, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %635 = phi i32 [ %619, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ 0, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ %619, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %636 = icmp ne i32 %634, 0
  %637 = sext i1 %636 to i32
  br label %638

638:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread
  %639 = phi i32 [ %633, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %630, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %640 = phi i32 [ %637, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %632, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %641 = phi i32 [ %635, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %629, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %642 = icmp ne i32 %640, 0
  %643 = select i1 %642, i32 0, i32 %639
  %644 = and i32 %26, -2147483648
  %645 = or i32 %644, %643
  %646 = or i32 %645, %641
  %647 = bitcast i32 %646 to float
  br label %__imf_fdiv_rn.exit

__imf_fdiv_rn.exit:                               ; preds = %.__imf_fdiv_rn.exit_crit_edge168, %.__imf_fdiv_rn.exit_crit_edge, %592, %591, %170, %169, %571, %281, %146, %98, %39, %48, %54, %.critedge46, %260, %.critedge53, %638
  %648 = phi float [ %50, %48 ], [ %56, %54 ], [ %45, %39 ], [ %579, %.critedge53 ], [ %647, %638 ], [ %271, %260 ], [ %154, %.critedge46 ], [ %., %98 ], [ %.47, %146 ], [ %spec.select48, %169 ], [ %spec.select49, %170 ], [ %.51, %281 ], [ %.54, %571 ], [ %spec.select55, %591 ], [ %spec.select56, %592 ], [ 0x7FF8000000000000, %.__imf_fdiv_rn.exit_crit_edge ], [ 0x7FF8000000000000, %.__imf_fdiv_rn.exit_crit_edge168 ]
  %649 = bitcast float %17 to i32
  %650 = and i32 %649, 2139095040
  %651 = icmp eq i32 %650, 0
  %652 = select i1 %651, float 0x41F0000000000000, float 1.000000e+00
  %653 = icmp uge i32 %650, 1677721600
  %654 = select i1 %653, float 0x3DF0000000000000, float %652
  %655 = fmul float %17, %654
  %656 = fdiv float 1.000000e+00, %655
  %657 = fmul float %656, %13
  %658 = fmul float %657, %654
  %659 = fcmp oeq float %13, %17
  %660 = and i32 %649, 8388607
  %661 = icmp eq i32 %650, 0
  %662 = icmp eq i32 %660, 0
  %663 = or i1 %661, %662
  %664 = xor i1 %663, true
  %665 = and i1 %659, %664
  %666 = select i1 %665, float 1.000000e+00, float %658
  %667 = ptrtoint float addrspace(1)* %2 to i64
  %668 = add i64 %10, %667
  %669 = inttoptr i64 %668 to float addrspace(1)*
  store float %648, float addrspace(1)* %669, align 4
  %670 = ptrtoint float addrspace(1)* %3 to i64
  %671 = add i64 %10, %670
  %672 = inttoptr i64 %671 to float addrspace(1)*
  store float %666, float addrspace(1)* %672, align 4
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

; Function Attrs: convergent
declare dso_local double @GenISA_fma_rtz_f64(double, double, double) local_unnamed_addr #3

; Function Attrs: convergent
declare dso_local double @GenISA_mul_rtz_f64(double, double) local_unnamed_addr #3

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Function Attrs: nounwind readnone
declare double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double, double, double) #4

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind readnone
declare double @llvm.genx.GenISA.mul.rtz.f64.f64.f64(double, double) #4

attributes #0 = { argmemonly nofree norecurse nosync nounwind null_pointer_is_valid "less-precise-fpmad"="false" }
attributes #1 = { convergent mustprogress nofree nounwind readnone willreturn "frame-pointer"="none" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { inaccessiblememonly mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { convergent "correctly-rounded-divide-sqrt-fp-math"="false" "denorms-are-zero"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind readnone }

!spirv.MemoryModel = !{!0}
!spirv.Source = !{!1}
!spirv.Generator = !{!2}
!igc.spirv.extensions = !{!3}
!igc.functions = !{!4}
!IGCMetadata = !{!28}
!opencl.ocl.version = !{!407, !407, !407, !407, !407, !407, !407, !407}
!opencl.spir.version = !{!407, !407, !407, !407, !407, !407, !407}
!llvm.ident = !{!408, !408, !408, !408, !408, !408, !408, !409}
!llvm.module.flags = !{!410}

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
!28 = !{!"ModuleMD", !29, !30, !118, !238, !269, !286, !309, !319, !321, !322, !337, !338, !339, !340, !344, !345, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !364, !368, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !183, !381, !384, !385, !387, !389, !392, !393, !394, !396, !397, !398, !403, !404, !405, !406}
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
!381 = !{!"PrivateMemoryPerFG", !382, !383}
!382 = !{!"PrivateMemoryPerFGMap[0]", void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <8 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32)* @kernel}
!383 = !{!"PrivateMemoryPerFGValue[0]", i32 0}
!384 = !{!"m_OptsToDisable"}
!385 = !{!"capabilities", !386}
!386 = !{!"globalVariableDecorationsINTEL", i1 false}
!387 = !{!"extensions", !388}
!388 = !{!"spvINTELBindlessImages", i1 false}
!389 = !{!"m_ShaderResourceViewMcsMask", !390, !391}
!390 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!391 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!392 = !{!"computedDepthMode", i32 0}
!393 = !{!"isHDCFastClearShader", i1 false}
!394 = !{!"argRegisterReservations", !395}
!395 = !{!"argRegisterReservationsVec[0]", i32 0}
!396 = !{!"SIMD16_SpillThreshold", i8 0}
!397 = !{!"SIMD32_SpillThreshold", i8 0}
!398 = !{!"m_CacheControlOption", !399, !400, !401, !402}
!399 = !{!"LscLoadCacheControlOverride", i8 0}
!400 = !{!"LscStoreCacheControlOverride", i8 0}
!401 = !{!"TgmLoadCacheControlOverride", i8 0}
!402 = !{!"TgmStoreCacheControlOverride", i8 0}
!403 = !{!"ModuleUsesBindless", i1 false}
!404 = !{!"predicationMap"}
!405 = !{!"lifeTimeStartMap"}
!406 = !{!"HitGroups"}
!407 = !{i32 2, i32 0}
!408 = !{!"clang version 15.0.7"}
!409 = !{!"clang version 9.0.0 (c68f557a081b1b2339a42d7cd6af3c2ab18c6061)"}
!410 = !{i32 1, !"wchar_size", i32 4}
!411 = !{!412}
!412 = !{i32 4469}
!413 = !{float 2.500000e+00}
!414 = !{!412, !415}
!415 = !{i32 4470}
!416 = !{!415}
