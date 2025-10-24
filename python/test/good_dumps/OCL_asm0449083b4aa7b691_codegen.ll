; ------------------------------------------------
; OCL_asm0449083b4aa7b691_codegen.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

; Function Attrs: argmemonly nofree norecurse nosync nounwind null_pointer_is_valid
define spir_kernel void @kernel(float addrspace(1)* nocapture readonly align 4 %0, float addrspace(1)* nocapture readonly align 4 %1, float addrspace(1)* nocapture writeonly align 4 %2, float addrspace(1)* nocapture writeonly align 4 %3, i8 addrspace(1)* nocapture readnone align 1 %4, i8 addrspace(1)* nocapture readnone align 1 %5, <8 x i32> %r0, <8 x i32> %payloadHeader, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* nocapture readnone %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bufferOffset5) #0 {
  %7 = and i16 %localIdX, 127
  %8 = ptrtoint float addrspace(1)* %0 to i64
  %9 = zext i16 %7 to i64
  %10 = shl nuw nsw i64 %9, 2
  %11 = add i64 %10, %8
  %12 = inttoptr i64 %11 to float addrspace(1)*
  %13 = load float, float addrspace(1)* %12, align 4
  %14 = ptrtoint float addrspace(1)* %1 to i64
  %15 = add i64 %10, %14
  %16 = inttoptr i64 %15 to float addrspace(1)*
  %17 = load float, float addrspace(1)* %16, align 4
  %18 = bitcast float %13 to i32
  %19 = lshr i32 %18, 23
  %20 = and i32 %19, 255
  %21 = and i32 %18, 8388607
  %22 = icmp eq i32 %20, 255
  %23 = icmp ne i32 %21, 0
  %24 = and i1 %22, %23
  br i1 %24, label %.__imf_fdiv_rn.exit_crit_edge, label %25, !stats.blockFrequency.digits !412, !stats.blockFrequency.scale !413

.__imf_fdiv_rn.exit_crit_edge:                    ; preds = %6
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !414, !stats.blockFrequency.scale !415

25:                                               ; preds = %6
  %26 = bitcast float %17 to i32
  %27 = lshr i32 %26, 23
  %28 = and i32 %27, 255
  %29 = and i32 %26, 8388607
  %30 = icmp eq i32 %28, 255
  %31 = icmp ne i32 %29, 0
  %32 = and i1 %30, %31
  %33 = fcmp oeq float %17, 0.000000e+00
  %34 = or i1 %32, %33
  br i1 %34, label %.__imf_fdiv_rn.exit_crit_edge168, label %35, !stats.blockFrequency.digits !414, !stats.blockFrequency.scale !415

.__imf_fdiv_rn.exit_crit_edge168:                 ; preds = %25
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !416, !stats.blockFrequency.scale !417

35:                                               ; preds = %25
  %36 = xor i32 %18, %26
  %37 = icmp eq i32 %21, 0
  %38 = icmp eq i32 %20, 255
  %39 = and i1 %38, %37
  br i1 %39, label %646, label %40, !stats.blockFrequency.digits !416, !stats.blockFrequency.scale !417

40:                                               ; preds = %35
  %41 = fcmp oeq float %13, 0.000000e+00
  br i1 %41, label %643, label %42, !stats.blockFrequency.digits !418, !stats.blockFrequency.scale !419

42:                                               ; preds = %40
  %43 = icmp eq i32 %29, 0
  %44 = icmp eq i32 %28, 255
  %45 = and i1 %44, %43
  br i1 %45, label %640, label %46, !stats.blockFrequency.digits !420, !stats.blockFrequency.scale !421

46:                                               ; preds = %42
  %47 = add nsw i32 %20, -127, !spirv.Decorations !422
  %48 = icmp eq i32 %20, 0
  %49 = select i1 %48, i32 -126, i32 %47
  %50 = add nsw i32 %28, -127, !spirv.Decorations !422
  %51 = icmp eq i32 %28, 0
  %52 = select i1 %51, i32 -126, i32 %50
  %53 = sub nsw i32 %49, %52, !spirv.Decorations !422
  %54 = or i32 %21, 8388608
  %55 = select i1 %48, i32 %21, i32 %54
  %56 = or i32 %29, 8388608
  %57 = select i1 %51, i32 %29, i32 %56
  %58 = icmp ult i32 %55, %57
  br i1 %58, label %.preheader.i.i.i.preheader, label %438, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !425

.preheader.i.i.i.preheader:                       ; preds = %46
  br label %.preheader.i.i.i, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !426

.preheader.i.i.i:                                 ; preds = %.preheader.i.i.i..preheader.i.i.i_crit_edge, %.preheader.i.i.i.preheader
  %59 = phi i32 [ %62, %.preheader.i.i.i..preheader.i.i.i_crit_edge ], [ 0, %.preheader.i.i.i.preheader ]
  %60 = phi i32 [ %61, %.preheader.i.i.i..preheader.i.i.i_crit_edge ], [ %55, %.preheader.i.i.i.preheader ]
  %61 = shl nuw nsw i32 %60, 1, !spirv.Decorations !427
  %62 = add i32 %59, 1
  %63 = icmp ult i32 %61, %57
  br i1 %63, label %.preheader.i.i.i..preheader.i.i.i_crit_edge, label %64, !stats.blockFrequency.digits !429, !stats.blockFrequency.scale !417

.preheader.i.i.i..preheader.i.i.i_crit_edge:      ; preds = %.preheader.i.i.i
  br label %.preheader.i.i.i, !stats.blockFrequency.digits !430, !stats.blockFrequency.scale !417

64:                                               ; preds = %.preheader.i.i.i
  %.lcssa208 = phi i32 [ %59, %.preheader.i.i.i ]
  %.lcssa207 = phi i32 [ %62, %.preheader.i.i.i ]
  %65 = xor i32 %.lcssa208, -1
  %66 = add i32 %53, %65
  %67 = icmp sgt i32 %66, 127
  br i1 %67, label %436, label %68, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !426

68:                                               ; preds = %64
  %69 = icmp sgt i32 %66, -127
  br i1 %69, label %138, label %70, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !431

70:                                               ; preds = %68
  %71 = add i32 %.lcssa208, -127
  %72 = sub i32 %71, %53
  %73 = add i32 %72, 1
  %74 = icmp ugt i32 %73, 22
  br i1 %74, label %130, label %75, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !433

75:                                               ; preds = %70
  %76 = shl i32 %55, %.lcssa208
  %77 = icmp eq i32 %76, 0
  br i1 %77, label %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, label %.lr.ph.preheader, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge: ; preds = %75
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !436

.lr.ph.preheader:                                 ; preds = %75
  %78 = sub nsw i32 25, %72, !spirv.Decorations !422
  br label %.lr.ph, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !434

.lr.ph:                                           ; preds = %.preheader.i1.i.i.i..lr.ph_crit_edge, %.lr.ph.preheader
  %79 = phi i32 [ %92, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ 0, %.lr.ph.preheader ]
  %80 = phi i32 [ %91, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ 0, %.lr.ph.preheader ]
  %81 = phi i32 [ %90, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ %76, %.lr.ph.preheader ]
  %82 = shl i32 %80, 1
  %83 = shl i32 %81, 1
  %84 = icmp ugt i32 %83, %57
  br i1 %84, label %87, label %85, !stats.blockFrequency.digits !438, !stats.blockFrequency.scale !425

85:                                               ; preds = %.lr.ph
  %86 = icmp eq i32 %83, %57
  br i1 %86, label %94, label %..preheader.i1.i.i.i_crit_edge, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !426

..preheader.i1.i.i.i_crit_edge:                   ; preds = %85
  br label %.preheader.i1.i.i.i, !stats.blockFrequency.digits !440, !stats.blockFrequency.scale !426

87:                                               ; preds = %.lr.ph
  %88 = sub nuw i32 %83, %57, !spirv.Decorations !441
  %89 = add i32 %82, 1
  br label %.preheader.i1.i.i.i, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !426

.preheader.i1.i.i.i:                              ; preds = %..preheader.i1.i.i.i_crit_edge, %87
  %90 = phi i32 [ %88, %87 ], [ %83, %..preheader.i1.i.i.i_crit_edge ]
  %91 = phi i32 [ %89, %87 ], [ %82, %..preheader.i1.i.i.i_crit_edge ]
  %92 = add nuw i32 %79, 1, !spirv.Decorations !441
  %93 = icmp ult i32 %92, %78
  br i1 %93, label %.preheader.i1.i.i.i..lr.ph_crit_edge, label %.preheader.i1.i.i.i._crit_edge, !stats.blockFrequency.digits !442, !stats.blockFrequency.scale !425

.preheader.i1.i.i.i..lr.ph_crit_edge:             ; preds = %.preheader.i1.i.i.i
  br label %.lr.ph, !stats.blockFrequency.digits !443, !stats.blockFrequency.scale !425

94:                                               ; preds = %85
  %.lcssa204 = phi i32 [ %79, %85 ]
  %.lcssa202 = phi i32 [ %82, %85 ]
  %95 = add i32 %.lcssa202, 1
  %96 = xor i32 %.lcssa204, -1
  %97 = add i32 %78, %96
  %98 = shl i32 %95, %97
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i, !stats.blockFrequency.digits !444, !stats.blockFrequency.scale !445

.preheader.i1.i.i.i._crit_edge:                   ; preds = %.preheader.i1.i.i.i
  %.lcssa206 = phi i32 [ %91, %.preheader.i1.i.i.i ]
  %99 = or i32 %.lcssa206, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !436

_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i:        ; preds = %.preheader.i1.i.i.i._crit_edge, %94
  %100 = phi i32 [ %98, %94 ], [ %99, %.preheader.i1.i.i.i._crit_edge ]
  %101 = lshr i32 %100, 3
  %102 = and i32 %100, 7
  %103 = icmp eq i32 %102, 0
  br i1 %103, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !434

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %104 = and i32 %100, 15
  %105 = icmp ult i32 %102, 5
  %.not9 = icmp ne i32 %104, 12
  %not.or.cond57 = and i1 %105, %.not9
  %106 = icmp ugt i32 %100, 67108855
  %107 = select i1 %106, i32 8388608, i32 0
  br i1 %not.or.cond57, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
  %108 = add nuw nsw i32 %101, 1, !spirv.Decorations !427
  %109 = select i1 %106, i32 0, i32 %108
  %110 = sext i1 %not.or.cond57 to i32
  br label %120, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !445

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
  %111 = sext i1 %not.or.cond57 to i32
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !445

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
  %112 = icmp ugt i32 %100, 67108855
  %113 = sext i1 %112 to i32
  %114 = and i32 8388608, %113
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, !stats.blockFrequency.digits !449, !stats.blockFrequency.scale !445

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread
  %115 = phi i32 [ %107, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ 0, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ %114, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %116 = phi i32 [ %111, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ -1, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ -1, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %117 = phi i32 [ %101, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ 0, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ %101, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %118 = icmp ne i32 %116, 0
  %119 = sext i1 %118 to i32
  br label %120, !stats.blockFrequency.digits !450, !stats.blockFrequency.scale !434

120:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread
  %121 = phi i32 [ %115, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %107, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %122 = phi i32 [ %119, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %110, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %123 = phi i32 [ %117, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %109, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %124 = icmp ne i32 %122, 0
  %125 = select i1 %124, i32 0, i32 %121
  %126 = and i32 %36, -2147483648
  %127 = or i32 %126, %125
  %128 = or i32 %127, %123
  %129 = bitcast i32 %128 to float
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

130:                                              ; preds = %70
  %131 = shl i32 %55, %.lcssa207
  %132 = icmp eq i32 %73, 23
  %133 = icmp ugt i32 %131, %57
  %134 = and i1 %132, %133
  %135 = icmp sgt i32 %36, -1
  br i1 %135, label %137, label %136, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

136:                                              ; preds = %130
  %spec.select56 = select i1 %134, float 0xB6A0000000000000, float -0.000000e+00
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !436

137:                                              ; preds = %130
  %spec.select55 = select i1 %134, float 0x36A0000000000000, float 0.000000e+00
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !434

138:                                              ; preds = %68
  %139 = add nsw i32 %66, 127, !spirv.Decorations !422
  %140 = shl i32 %55, %.lcssa208
  %141 = icmp eq i32 %140, 0
  br i1 %141, label %..critedge53_crit_edge, label %.preheader.i4.i.i.i.preheader, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !433

..critedge53_crit_edge:                           ; preds = %138
  br label %.critedge53, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !434

.preheader.i4.i.i.i.preheader:                    ; preds = %138
  %142 = shl i32 %140, 1
  %143 = icmp ugt i32 %142, %57
  br i1 %143, label %146, label %144, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !433

144:                                              ; preds = %.preheader.i4.i.i.i.preheader
  %145 = icmp eq i32 %142, %57
  br i1 %145, label %._crit_edge173, label %..preheader.i4.i.i.i_crit_edge, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !434

..preheader.i4.i.i.i_crit_edge:                   ; preds = %144
  br label %.preheader.i4.i.i.i, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !436

._crit_edge173:                                   ; preds = %144
  br label %405, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !436

146:                                              ; preds = %.preheader.i4.i.i.i.preheader
  %147 = sub nuw i32 %142, %57, !spirv.Decorations !441
  br label %.preheader.i4.i.i.i, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !434

.preheader.i4.i.i.i:                              ; preds = %..preheader.i4.i.i.i_crit_edge, %146
  %148 = phi i32 [ %147, %146 ], [ %142, %..preheader.i4.i.i.i_crit_edge ]
  %149 = phi i16 [ 2, %146 ], [ 0, %..preheader.i4.i.i.i_crit_edge ]
  %150 = shl i32 %148, 1
  %151 = icmp ugt i32 %150, %57
  br i1 %151, label %155, label %152, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !434

152:                                              ; preds = %.preheader.i4.i.i.i
  %153 = icmp eq i32 %150, %57
  br i1 %153, label %._crit_edge174, label %..preheader.i4.i.i.i.1_crit_edge, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !436

..preheader.i4.i.i.i.1_crit_edge:                 ; preds = %152
  br label %.preheader.i4.i.i.i.1, !stats.blockFrequency.digits !449, !stats.blockFrequency.scale !445

._crit_edge174:                                   ; preds = %152
  %154 = trunc i16 %149 to i8
  %.demoted.zext = zext i8 %154 to i32
  br label %405, !stats.blockFrequency.digits !449, !stats.blockFrequency.scale !445

155:                                              ; preds = %.preheader.i4.i.i.i
  %156 = sub nuw i32 %150, %57, !spirv.Decorations !441
  %b2s = or i16 %149, 1
  br label %.preheader.i4.i.i.i.1, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !436

.preheader.i4.i.i.i.1:                            ; preds = %..preheader.i4.i.i.i.1_crit_edge, %155
  %157 = phi i32 [ %156, %155 ], [ %150, %..preheader.i4.i.i.i.1_crit_edge ]
  %158 = phi i16 [ %b2s, %155 ], [ %149, %..preheader.i4.i.i.i.1_crit_edge ]
  %159 = trunc i16 %158 to i8
  %.demoted.zext166 = zext i8 %159 to i32
  %160 = shl nsw i32 %.demoted.zext166, 1
  %161 = shl i32 %157, 1
  %162 = icmp ugt i32 %161, %57
  br i1 %162, label %165, label %163, !stats.blockFrequency.digits !453, !stats.blockFrequency.scale !434

163:                                              ; preds = %.preheader.i4.i.i.i.1
  %164 = icmp eq i32 %161, %57
  br i1 %164, label %._crit_edge175, label %..preheader.i4.i.i.i.2_crit_edge, !stats.blockFrequency.digits !454, !stats.blockFrequency.scale !436

..preheader.i4.i.i.i.2_crit_edge:                 ; preds = %163
  br label %.preheader.i4.i.i.i.2, !stats.blockFrequency.digits !455, !stats.blockFrequency.scale !445

._crit_edge175:                                   ; preds = %163
  br label %405, !stats.blockFrequency.digits !455, !stats.blockFrequency.scale !445

165:                                              ; preds = %.preheader.i4.i.i.i.1
  %166 = sub nuw i32 %161, %57, !spirv.Decorations !441
  %167 = add i32 %160, 1
  br label %.preheader.i4.i.i.i.2, !stats.blockFrequency.digits !454, !stats.blockFrequency.scale !436

.preheader.i4.i.i.i.2:                            ; preds = %..preheader.i4.i.i.i.2_crit_edge, %165
  %168 = phi i32 [ %166, %165 ], [ %161, %..preheader.i4.i.i.i.2_crit_edge ]
  %169 = phi i32 [ %167, %165 ], [ %160, %..preheader.i4.i.i.i.2_crit_edge ]
  %170 = shl i32 %169, 1
  %171 = shl i32 %168, 1
  %172 = icmp ugt i32 %171, %57
  br i1 %172, label %175, label %173, !stats.blockFrequency.digits !456, !stats.blockFrequency.scale !436

173:                                              ; preds = %.preheader.i4.i.i.i.2
  %174 = icmp eq i32 %171, %57
  br i1 %174, label %._crit_edge176, label %..preheader.i4.i.i.i.3_crit_edge, !stats.blockFrequency.digits !456, !stats.blockFrequency.scale !445

..preheader.i4.i.i.i.3_crit_edge:                 ; preds = %173
  br label %.preheader.i4.i.i.i.3, !stats.blockFrequency.digits !457, !stats.blockFrequency.scale !458

._crit_edge176:                                   ; preds = %173
  br label %405, !stats.blockFrequency.digits !457, !stats.blockFrequency.scale !458

175:                                              ; preds = %.preheader.i4.i.i.i.2
  %176 = sub nuw i32 %171, %57, !spirv.Decorations !441
  %177 = add i32 %170, 1
  br label %.preheader.i4.i.i.i.3, !stats.blockFrequency.digits !456, !stats.blockFrequency.scale !445

.preheader.i4.i.i.i.3:                            ; preds = %..preheader.i4.i.i.i.3_crit_edge, %175
  %178 = phi i32 [ %176, %175 ], [ %171, %..preheader.i4.i.i.i.3_crit_edge ]
  %179 = phi i32 [ %177, %175 ], [ %170, %..preheader.i4.i.i.i.3_crit_edge ]
  %180 = shl i32 %179, 1
  %181 = shl i32 %178, 1
  %182 = icmp ugt i32 %181, %57
  br i1 %182, label %185, label %183, !stats.blockFrequency.digits !459, !stats.blockFrequency.scale !436

183:                                              ; preds = %.preheader.i4.i.i.i.3
  %184 = icmp eq i32 %181, %57
  br i1 %184, label %._crit_edge177, label %..preheader.i4.i.i.i.4_crit_edge, !stats.blockFrequency.digits !460, !stats.blockFrequency.scale !445

..preheader.i4.i.i.i.4_crit_edge:                 ; preds = %183
  br label %.preheader.i4.i.i.i.4, !stats.blockFrequency.digits !461, !stats.blockFrequency.scale !458

._crit_edge177:                                   ; preds = %183
  br label %405, !stats.blockFrequency.digits !461, !stats.blockFrequency.scale !458

185:                                              ; preds = %.preheader.i4.i.i.i.3
  %186 = sub nuw i32 %181, %57, !spirv.Decorations !441
  %187 = add i32 %180, 1
  br label %.preheader.i4.i.i.i.4, !stats.blockFrequency.digits !460, !stats.blockFrequency.scale !445

.preheader.i4.i.i.i.4:                            ; preds = %..preheader.i4.i.i.i.4_crit_edge, %185
  %188 = phi i32 [ %186, %185 ], [ %181, %..preheader.i4.i.i.i.4_crit_edge ]
  %189 = phi i32 [ %187, %185 ], [ %180, %..preheader.i4.i.i.i.4_crit_edge ]
  %190 = shl i32 %189, 1
  %191 = shl i32 %188, 1
  %192 = icmp ugt i32 %191, %57
  br i1 %192, label %195, label %193, !stats.blockFrequency.digits !462, !stats.blockFrequency.scale !436

193:                                              ; preds = %.preheader.i4.i.i.i.4
  %194 = icmp eq i32 %191, %57
  br i1 %194, label %._crit_edge178, label %..preheader.i4.i.i.i.5_crit_edge, !stats.blockFrequency.digits !462, !stats.blockFrequency.scale !445

..preheader.i4.i.i.i.5_crit_edge:                 ; preds = %193
  br label %.preheader.i4.i.i.i.5, !stats.blockFrequency.digits !463, !stats.blockFrequency.scale !458

._crit_edge178:                                   ; preds = %193
  br label %405, !stats.blockFrequency.digits !463, !stats.blockFrequency.scale !458

195:                                              ; preds = %.preheader.i4.i.i.i.4
  %196 = sub nuw i32 %191, %57, !spirv.Decorations !441
  %197 = add i32 %190, 1
  br label %.preheader.i4.i.i.i.5, !stats.blockFrequency.digits !462, !stats.blockFrequency.scale !445

.preheader.i4.i.i.i.5:                            ; preds = %..preheader.i4.i.i.i.5_crit_edge, %195
  %198 = phi i32 [ %196, %195 ], [ %191, %..preheader.i4.i.i.i.5_crit_edge ]
  %199 = phi i32 [ %197, %195 ], [ %190, %..preheader.i4.i.i.i.5_crit_edge ]
  %200 = shl i32 %199, 1
  %201 = shl i32 %198, 1
  %202 = icmp ugt i32 %201, %57
  br i1 %202, label %205, label %203, !stats.blockFrequency.digits !464, !stats.blockFrequency.scale !445

203:                                              ; preds = %.preheader.i4.i.i.i.5
  %204 = icmp eq i32 %201, %57
  br i1 %204, label %._crit_edge179, label %..preheader.i4.i.i.i.6_crit_edge, !stats.blockFrequency.digits !464, !stats.blockFrequency.scale !458

..preheader.i4.i.i.i.6_crit_edge:                 ; preds = %203
  br label %.preheader.i4.i.i.i.6, !stats.blockFrequency.digits !465, !stats.blockFrequency.scale !466

._crit_edge179:                                   ; preds = %203
  br label %405, !stats.blockFrequency.digits !465, !stats.blockFrequency.scale !466

205:                                              ; preds = %.preheader.i4.i.i.i.5
  %206 = sub nuw i32 %201, %57, !spirv.Decorations !441
  %207 = add i32 %200, 1
  br label %.preheader.i4.i.i.i.6, !stats.blockFrequency.digits !464, !stats.blockFrequency.scale !458

.preheader.i4.i.i.i.6:                            ; preds = %..preheader.i4.i.i.i.6_crit_edge, %205
  %208 = phi i32 [ %206, %205 ], [ %201, %..preheader.i4.i.i.i.6_crit_edge ]
  %209 = phi i32 [ %207, %205 ], [ %200, %..preheader.i4.i.i.i.6_crit_edge ]
  %210 = shl i32 %209, 1
  %211 = shl i32 %208, 1
  %212 = icmp ugt i32 %211, %57
  br i1 %212, label %215, label %213, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !445

213:                                              ; preds = %.preheader.i4.i.i.i.6
  %214 = icmp eq i32 %211, %57
  br i1 %214, label %._crit_edge180, label %..preheader.i4.i.i.i.7_crit_edge, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !458

..preheader.i4.i.i.i.7_crit_edge:                 ; preds = %213
  br label %.preheader.i4.i.i.i.7, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !466

._crit_edge180:                                   ; preds = %213
  br label %405, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !466

215:                                              ; preds = %.preheader.i4.i.i.i.6
  %216 = sub nuw i32 %211, %57, !spirv.Decorations !441
  %217 = add i32 %210, 1
  br label %.preheader.i4.i.i.i.7, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !458

.preheader.i4.i.i.i.7:                            ; preds = %..preheader.i4.i.i.i.7_crit_edge, %215
  %218 = phi i32 [ %216, %215 ], [ %211, %..preheader.i4.i.i.i.7_crit_edge ]
  %219 = phi i32 [ %217, %215 ], [ %210, %..preheader.i4.i.i.i.7_crit_edge ]
  %220 = shl i32 %219, 1
  %221 = shl i32 %218, 1
  %222 = icmp ugt i32 %221, %57
  br i1 %222, label %225, label %223, !stats.blockFrequency.digits !468, !stats.blockFrequency.scale !458

223:                                              ; preds = %.preheader.i4.i.i.i.7
  %224 = icmp eq i32 %221, %57
  br i1 %224, label %._crit_edge181, label %..preheader.i4.i.i.i.8_crit_edge, !stats.blockFrequency.digits !468, !stats.blockFrequency.scale !466

..preheader.i4.i.i.i.8_crit_edge:                 ; preds = %223
  br label %.preheader.i4.i.i.i.8, !stats.blockFrequency.digits !468, !stats.blockFrequency.scale !469

._crit_edge181:                                   ; preds = %223
  br label %405, !stats.blockFrequency.digits !468, !stats.blockFrequency.scale !469

225:                                              ; preds = %.preheader.i4.i.i.i.7
  %226 = sub nuw i32 %221, %57, !spirv.Decorations !441
  %227 = add i32 %220, 1
  br label %.preheader.i4.i.i.i.8, !stats.blockFrequency.digits !468, !stats.blockFrequency.scale !466

.preheader.i4.i.i.i.8:                            ; preds = %..preheader.i4.i.i.i.8_crit_edge, %225
  %228 = phi i32 [ %226, %225 ], [ %221, %..preheader.i4.i.i.i.8_crit_edge ]
  %229 = phi i32 [ %227, %225 ], [ %220, %..preheader.i4.i.i.i.8_crit_edge ]
  %230 = shl i32 %229, 1
  %231 = shl i32 %228, 1
  %232 = icmp ugt i32 %231, %57
  br i1 %232, label %235, label %233, !stats.blockFrequency.digits !470, !stats.blockFrequency.scale !458

233:                                              ; preds = %.preheader.i4.i.i.i.8
  %234 = icmp eq i32 %231, %57
  br i1 %234, label %._crit_edge182, label %..preheader.i4.i.i.i.9_crit_edge, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !466

..preheader.i4.i.i.i.9_crit_edge:                 ; preds = %233
  br label %.preheader.i4.i.i.i.9, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !469

._crit_edge182:                                   ; preds = %233
  br label %405, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !469

235:                                              ; preds = %.preheader.i4.i.i.i.8
  %236 = sub nuw i32 %231, %57, !spirv.Decorations !441
  %237 = add i32 %230, 1
  br label %.preheader.i4.i.i.i.9, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !466

.preheader.i4.i.i.i.9:                            ; preds = %..preheader.i4.i.i.i.9_crit_edge, %235
  %238 = phi i32 [ %236, %235 ], [ %231, %..preheader.i4.i.i.i.9_crit_edge ]
  %239 = phi i32 [ %237, %235 ], [ %230, %..preheader.i4.i.i.i.9_crit_edge ]
  %240 = shl i32 %239, 1
  %241 = shl i32 %238, 1
  %242 = icmp ugt i32 %241, %57
  br i1 %242, label %245, label %243, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !458

243:                                              ; preds = %.preheader.i4.i.i.i.9
  %244 = icmp eq i32 %241, %57
  br i1 %244, label %._crit_edge183, label %..preheader.i4.i.i.i.10_crit_edge, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !466

..preheader.i4.i.i.i.10_crit_edge:                ; preds = %243
  br label %.preheader.i4.i.i.i.10, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !469

._crit_edge183:                                   ; preds = %243
  br label %405, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !469

245:                                              ; preds = %.preheader.i4.i.i.i.9
  %246 = sub nuw i32 %241, %57, !spirv.Decorations !441
  %247 = add i32 %240, 1
  br label %.preheader.i4.i.i.i.10, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !466

.preheader.i4.i.i.i.10:                           ; preds = %..preheader.i4.i.i.i.10_crit_edge, %245
  %248 = phi i32 [ %246, %245 ], [ %241, %..preheader.i4.i.i.i.10_crit_edge ]
  %249 = phi i32 [ %247, %245 ], [ %240, %..preheader.i4.i.i.i.10_crit_edge ]
  %250 = shl i32 %249, 1
  %251 = shl i32 %248, 1
  %252 = icmp ugt i32 %251, %57
  br i1 %252, label %255, label %253, !stats.blockFrequency.digits !473, !stats.blockFrequency.scale !466

253:                                              ; preds = %.preheader.i4.i.i.i.10
  %254 = icmp eq i32 %251, %57
  br i1 %254, label %._crit_edge184, label %..preheader.i4.i.i.i.11_crit_edge, !stats.blockFrequency.digits !473, !stats.blockFrequency.scale !469

..preheader.i4.i.i.i.11_crit_edge:                ; preds = %253
  br label %.preheader.i4.i.i.i.11, !stats.blockFrequency.digits !474, !stats.blockFrequency.scale !475

._crit_edge184:                                   ; preds = %253
  br label %405, !stats.blockFrequency.digits !474, !stats.blockFrequency.scale !475

255:                                              ; preds = %.preheader.i4.i.i.i.10
  %256 = sub nuw i32 %251, %57, !spirv.Decorations !441
  %257 = add i32 %250, 1
  br label %.preheader.i4.i.i.i.11, !stats.blockFrequency.digits !473, !stats.blockFrequency.scale !469

.preheader.i4.i.i.i.11:                           ; preds = %..preheader.i4.i.i.i.11_crit_edge, %255
  %258 = phi i32 [ %256, %255 ], [ %251, %..preheader.i4.i.i.i.11_crit_edge ]
  %259 = phi i32 [ %257, %255 ], [ %250, %..preheader.i4.i.i.i.11_crit_edge ]
  %260 = shl i32 %259, 1
  %261 = shl i32 %258, 1
  %262 = icmp ugt i32 %261, %57
  br i1 %262, label %265, label %263, !stats.blockFrequency.digits !476, !stats.blockFrequency.scale !466

263:                                              ; preds = %.preheader.i4.i.i.i.11
  %264 = icmp eq i32 %261, %57
  br i1 %264, label %._crit_edge185, label %..preheader.i4.i.i.i.12_crit_edge, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !469

..preheader.i4.i.i.i.12_crit_edge:                ; preds = %263
  br label %.preheader.i4.i.i.i.12, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !475

._crit_edge185:                                   ; preds = %263
  br label %405, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !475

265:                                              ; preds = %.preheader.i4.i.i.i.11
  %266 = sub nuw i32 %261, %57, !spirv.Decorations !441
  %267 = add i32 %260, 1
  br label %.preheader.i4.i.i.i.12, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !469

.preheader.i4.i.i.i.12:                           ; preds = %..preheader.i4.i.i.i.12_crit_edge, %265
  %268 = phi i32 [ %266, %265 ], [ %261, %..preheader.i4.i.i.i.12_crit_edge ]
  %269 = phi i32 [ %267, %265 ], [ %260, %..preheader.i4.i.i.i.12_crit_edge ]
  %270 = shl i32 %269, 1
  %271 = shl i32 %268, 1
  %272 = icmp ugt i32 %271, %57
  br i1 %272, label %275, label %273, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !469

273:                                              ; preds = %.preheader.i4.i.i.i.12
  %274 = icmp eq i32 %271, %57
  br i1 %274, label %._crit_edge186, label %..preheader.i4.i.i.i.13_crit_edge, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !475

..preheader.i4.i.i.i.13_crit_edge:                ; preds = %273
  br label %.preheader.i4.i.i.i.13, !stats.blockFrequency.digits !480, !stats.blockFrequency.scale !481

._crit_edge186:                                   ; preds = %273
  br label %405, !stats.blockFrequency.digits !480, !stats.blockFrequency.scale !481

275:                                              ; preds = %.preheader.i4.i.i.i.12
  %276 = sub nuw i32 %271, %57, !spirv.Decorations !441
  %277 = add i32 %270, 1
  br label %.preheader.i4.i.i.i.13, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !475

.preheader.i4.i.i.i.13:                           ; preds = %..preheader.i4.i.i.i.13_crit_edge, %275
  %278 = phi i32 [ %276, %275 ], [ %271, %..preheader.i4.i.i.i.13_crit_edge ]
  %279 = phi i32 [ %277, %275 ], [ %270, %..preheader.i4.i.i.i.13_crit_edge ]
  %280 = shl i32 %279, 1
  %281 = shl i32 %278, 1
  %282 = icmp ugt i32 %281, %57
  br i1 %282, label %285, label %283, !stats.blockFrequency.digits !482, !stats.blockFrequency.scale !469

283:                                              ; preds = %.preheader.i4.i.i.i.13
  %284 = icmp eq i32 %281, %57
  br i1 %284, label %._crit_edge187, label %..preheader.i4.i.i.i.14_crit_edge, !stats.blockFrequency.digits !482, !stats.blockFrequency.scale !475

..preheader.i4.i.i.i.14_crit_edge:                ; preds = %283
  br label %.preheader.i4.i.i.i.14, !stats.blockFrequency.digits !483, !stats.blockFrequency.scale !481

._crit_edge187:                                   ; preds = %283
  br label %405, !stats.blockFrequency.digits !483, !stats.blockFrequency.scale !481

285:                                              ; preds = %.preheader.i4.i.i.i.13
  %286 = sub nuw i32 %281, %57, !spirv.Decorations !441
  %287 = add i32 %280, 1
  br label %.preheader.i4.i.i.i.14, !stats.blockFrequency.digits !482, !stats.blockFrequency.scale !475

.preheader.i4.i.i.i.14:                           ; preds = %..preheader.i4.i.i.i.14_crit_edge, %285
  %288 = phi i32 [ %286, %285 ], [ %281, %..preheader.i4.i.i.i.14_crit_edge ]
  %289 = phi i32 [ %287, %285 ], [ %280, %..preheader.i4.i.i.i.14_crit_edge ]
  %290 = shl i32 %289, 1
  %291 = shl i32 %288, 1
  %292 = icmp ugt i32 %291, %57
  br i1 %292, label %295, label %293, !stats.blockFrequency.digits !484, !stats.blockFrequency.scale !469

293:                                              ; preds = %.preheader.i4.i.i.i.14
  %294 = icmp eq i32 %291, %57
  br i1 %294, label %._crit_edge188, label %..preheader.i4.i.i.i.15_crit_edge, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !475

..preheader.i4.i.i.i.15_crit_edge:                ; preds = %293
  br label %.preheader.i4.i.i.i.15, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !481

._crit_edge188:                                   ; preds = %293
  br label %405, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !481

295:                                              ; preds = %.preheader.i4.i.i.i.14
  %296 = sub nuw i32 %291, %57, !spirv.Decorations !441
  %297 = add i32 %290, 1
  br label %.preheader.i4.i.i.i.15, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !475

.preheader.i4.i.i.i.15:                           ; preds = %..preheader.i4.i.i.i.15_crit_edge, %295
  %298 = phi i32 [ %296, %295 ], [ %291, %..preheader.i4.i.i.i.15_crit_edge ]
  %299 = phi i32 [ %297, %295 ], [ %290, %..preheader.i4.i.i.i.15_crit_edge ]
  %300 = shl i32 %299, 1
  %301 = shl i32 %298, 1
  %302 = icmp ugt i32 %301, %57
  br i1 %302, label %305, label %303, !stats.blockFrequency.digits !486, !stats.blockFrequency.scale !475

303:                                              ; preds = %.preheader.i4.i.i.i.15
  %304 = icmp eq i32 %301, %57
  br i1 %304, label %._crit_edge189, label %..preheader.i4.i.i.i.16_crit_edge, !stats.blockFrequency.digits !486, !stats.blockFrequency.scale !481

..preheader.i4.i.i.i.16_crit_edge:                ; preds = %303
  br label %.preheader.i4.i.i.i.16, !stats.blockFrequency.digits !486, !stats.blockFrequency.scale !487

._crit_edge189:                                   ; preds = %303
  br label %405, !stats.blockFrequency.digits !486, !stats.blockFrequency.scale !487

305:                                              ; preds = %.preheader.i4.i.i.i.15
  %306 = sub nuw i32 %301, %57, !spirv.Decorations !441
  %307 = add i32 %300, 1
  br label %.preheader.i4.i.i.i.16, !stats.blockFrequency.digits !486, !stats.blockFrequency.scale !481

.preheader.i4.i.i.i.16:                           ; preds = %..preheader.i4.i.i.i.16_crit_edge, %305
  %308 = phi i32 [ %306, %305 ], [ %301, %..preheader.i4.i.i.i.16_crit_edge ]
  %309 = phi i32 [ %307, %305 ], [ %300, %..preheader.i4.i.i.i.16_crit_edge ]
  %310 = shl i32 %309, 1
  %311 = shl i32 %308, 1
  %312 = icmp ugt i32 %311, %57
  br i1 %312, label %315, label %313, !stats.blockFrequency.digits !488, !stats.blockFrequency.scale !475

313:                                              ; preds = %.preheader.i4.i.i.i.16
  %314 = icmp eq i32 %311, %57
  br i1 %314, label %._crit_edge190, label %..preheader.i4.i.i.i.17_crit_edge, !stats.blockFrequency.digits !488, !stats.blockFrequency.scale !481

..preheader.i4.i.i.i.17_crit_edge:                ; preds = %313
  br label %.preheader.i4.i.i.i.17, !stats.blockFrequency.digits !489, !stats.blockFrequency.scale !487

._crit_edge190:                                   ; preds = %313
  br label %405, !stats.blockFrequency.digits !489, !stats.blockFrequency.scale !487

315:                                              ; preds = %.preheader.i4.i.i.i.16
  %316 = sub nuw i32 %311, %57, !spirv.Decorations !441
  %317 = add i32 %310, 1
  br label %.preheader.i4.i.i.i.17, !stats.blockFrequency.digits !488, !stats.blockFrequency.scale !481

.preheader.i4.i.i.i.17:                           ; preds = %..preheader.i4.i.i.i.17_crit_edge, %315
  %318 = phi i32 [ %316, %315 ], [ %311, %..preheader.i4.i.i.i.17_crit_edge ]
  %319 = phi i32 [ %317, %315 ], [ %310, %..preheader.i4.i.i.i.17_crit_edge ]
  %320 = shl i32 %319, 1
  %321 = shl i32 %318, 1
  %322 = icmp ugt i32 %321, %57
  br i1 %322, label %325, label %323, !stats.blockFrequency.digits !490, !stats.blockFrequency.scale !481

323:                                              ; preds = %.preheader.i4.i.i.i.17
  %324 = icmp eq i32 %321, %57
  br i1 %324, label %._crit_edge191, label %..preheader.i4.i.i.i.18_crit_edge, !stats.blockFrequency.digits !491, !stats.blockFrequency.scale !487

..preheader.i4.i.i.i.18_crit_edge:                ; preds = %323
  br label %.preheader.i4.i.i.i.18, !stats.blockFrequency.digits !492, !stats.blockFrequency.scale !493

._crit_edge191:                                   ; preds = %323
  br label %405, !stats.blockFrequency.digits !492, !stats.blockFrequency.scale !493

325:                                              ; preds = %.preheader.i4.i.i.i.17
  %326 = sub nuw i32 %321, %57, !spirv.Decorations !441
  %327 = add i32 %320, 1
  br label %.preheader.i4.i.i.i.18, !stats.blockFrequency.digits !491, !stats.blockFrequency.scale !487

.preheader.i4.i.i.i.18:                           ; preds = %..preheader.i4.i.i.i.18_crit_edge, %325
  %328 = phi i32 [ %326, %325 ], [ %321, %..preheader.i4.i.i.i.18_crit_edge ]
  %329 = phi i32 [ %327, %325 ], [ %320, %..preheader.i4.i.i.i.18_crit_edge ]
  %330 = shl i32 %329, 1
  %331 = shl i32 %328, 1
  %332 = icmp ugt i32 %331, %57
  br i1 %332, label %335, label %333, !stats.blockFrequency.digits !494, !stats.blockFrequency.scale !481

333:                                              ; preds = %.preheader.i4.i.i.i.18
  %334 = icmp eq i32 %331, %57
  br i1 %334, label %._crit_edge192, label %..preheader.i4.i.i.i.19_crit_edge, !stats.blockFrequency.digits !495, !stats.blockFrequency.scale !487

..preheader.i4.i.i.i.19_crit_edge:                ; preds = %333
  br label %.preheader.i4.i.i.i.19, !stats.blockFrequency.digits !496, !stats.blockFrequency.scale !493

._crit_edge192:                                   ; preds = %333
  br label %405, !stats.blockFrequency.digits !496, !stats.blockFrequency.scale !493

335:                                              ; preds = %.preheader.i4.i.i.i.18
  %336 = sub nuw i32 %331, %57, !spirv.Decorations !441
  %337 = add i32 %330, 1
  br label %.preheader.i4.i.i.i.19, !stats.blockFrequency.digits !495, !stats.blockFrequency.scale !487

.preheader.i4.i.i.i.19:                           ; preds = %..preheader.i4.i.i.i.19_crit_edge, %335
  %338 = phi i32 [ %336, %335 ], [ %331, %..preheader.i4.i.i.i.19_crit_edge ]
  %339 = phi i32 [ %337, %335 ], [ %330, %..preheader.i4.i.i.i.19_crit_edge ]
  %340 = shl i32 %339, 1
  %341 = shl i32 %338, 1
  %342 = icmp ugt i32 %341, %57
  br i1 %342, label %345, label %343, !stats.blockFrequency.digits !497, !stats.blockFrequency.scale !487

343:                                              ; preds = %.preheader.i4.i.i.i.19
  %344 = icmp eq i32 %341, %57
  br i1 %344, label %._crit_edge193, label %..preheader.i4.i.i.i.20_crit_edge, !stats.blockFrequency.digits !498, !stats.blockFrequency.scale !493

..preheader.i4.i.i.i.20_crit_edge:                ; preds = %343
  br label %.preheader.i4.i.i.i.20, !stats.blockFrequency.digits !499, !stats.blockFrequency.scale !500

._crit_edge193:                                   ; preds = %343
  br label %405, !stats.blockFrequency.digits !499, !stats.blockFrequency.scale !500

345:                                              ; preds = %.preheader.i4.i.i.i.19
  %346 = sub nuw i32 %341, %57, !spirv.Decorations !441
  %347 = add i32 %340, 1
  br label %.preheader.i4.i.i.i.20, !stats.blockFrequency.digits !498, !stats.blockFrequency.scale !493

.preheader.i4.i.i.i.20:                           ; preds = %..preheader.i4.i.i.i.20_crit_edge, %345
  %348 = phi i32 [ %346, %345 ], [ %341, %..preheader.i4.i.i.i.20_crit_edge ]
  %349 = phi i32 [ %347, %345 ], [ %340, %..preheader.i4.i.i.i.20_crit_edge ]
  %350 = shl i32 %349, 1
  %351 = shl i32 %348, 1
  %352 = icmp ugt i32 %351, %57
  br i1 %352, label %355, label %353, !stats.blockFrequency.digits !501, !stats.blockFrequency.scale !487

353:                                              ; preds = %.preheader.i4.i.i.i.20
  %354 = icmp eq i32 %351, %57
  br i1 %354, label %._crit_edge194, label %..preheader.i4.i.i.i.21_crit_edge, !stats.blockFrequency.digits !501, !stats.blockFrequency.scale !493

..preheader.i4.i.i.i.21_crit_edge:                ; preds = %353
  br label %.preheader.i4.i.i.i.21, !stats.blockFrequency.digits !502, !stats.blockFrequency.scale !500

._crit_edge194:                                   ; preds = %353
  br label %405, !stats.blockFrequency.digits !502, !stats.blockFrequency.scale !500

355:                                              ; preds = %.preheader.i4.i.i.i.20
  %356 = sub nuw i32 %351, %57, !spirv.Decorations !441
  %357 = add i32 %350, 1
  br label %.preheader.i4.i.i.i.21, !stats.blockFrequency.digits !501, !stats.blockFrequency.scale !493

.preheader.i4.i.i.i.21:                           ; preds = %..preheader.i4.i.i.i.21_crit_edge, %355
  %358 = phi i32 [ %356, %355 ], [ %351, %..preheader.i4.i.i.i.21_crit_edge ]
  %359 = phi i32 [ %357, %355 ], [ %350, %..preheader.i4.i.i.i.21_crit_edge ]
  %360 = shl i32 %359, 1
  %361 = shl i32 %358, 1
  %362 = icmp ugt i32 %361, %57
  br i1 %362, label %365, label %363, !stats.blockFrequency.digits !503, !stats.blockFrequency.scale !487

363:                                              ; preds = %.preheader.i4.i.i.i.21
  %364 = icmp eq i32 %361, %57
  br i1 %364, label %._crit_edge195, label %..preheader.i4.i.i.i.22_crit_edge, !stats.blockFrequency.digits !504, !stats.blockFrequency.scale !493

..preheader.i4.i.i.i.22_crit_edge:                ; preds = %363
  br label %.preheader.i4.i.i.i.22, !stats.blockFrequency.digits !504, !stats.blockFrequency.scale !500

._crit_edge195:                                   ; preds = %363
  br label %405, !stats.blockFrequency.digits !504, !stats.blockFrequency.scale !500

365:                                              ; preds = %.preheader.i4.i.i.i.21
  %366 = sub nuw i32 %361, %57, !spirv.Decorations !441
  %367 = add i32 %360, 1
  br label %.preheader.i4.i.i.i.22, !stats.blockFrequency.digits !504, !stats.blockFrequency.scale !493

.preheader.i4.i.i.i.22:                           ; preds = %..preheader.i4.i.i.i.22_crit_edge, %365
  %368 = phi i32 [ %366, %365 ], [ %361, %..preheader.i4.i.i.i.22_crit_edge ]
  %369 = phi i32 [ %367, %365 ], [ %360, %..preheader.i4.i.i.i.22_crit_edge ]
  %370 = shl i32 %369, 1
  %371 = shl i32 %368, 1
  %372 = icmp ugt i32 %371, %57
  br i1 %372, label %375, label %373, !stats.blockFrequency.digits !505, !stats.blockFrequency.scale !493

373:                                              ; preds = %.preheader.i4.i.i.i.22
  %374 = icmp eq i32 %371, %57
  br i1 %374, label %._crit_edge196, label %..preheader.i4.i.i.i.23_crit_edge, !stats.blockFrequency.digits !506, !stats.blockFrequency.scale !500

..preheader.i4.i.i.i.23_crit_edge:                ; preds = %373
  br label %.preheader.i4.i.i.i.23, !stats.blockFrequency.digits !507, !stats.blockFrequency.scale !508

._crit_edge196:                                   ; preds = %373
  br label %405, !stats.blockFrequency.digits !507, !stats.blockFrequency.scale !508

375:                                              ; preds = %.preheader.i4.i.i.i.22
  %376 = sub nuw i32 %371, %57, !spirv.Decorations !441
  %377 = add i32 %370, 1
  br label %.preheader.i4.i.i.i.23, !stats.blockFrequency.digits !506, !stats.blockFrequency.scale !500

.preheader.i4.i.i.i.23:                           ; preds = %..preheader.i4.i.i.i.23_crit_edge, %375
  %378 = phi i32 [ %376, %375 ], [ %371, %..preheader.i4.i.i.i.23_crit_edge ]
  %379 = phi i32 [ %377, %375 ], [ %370, %..preheader.i4.i.i.i.23_crit_edge ]
  %380 = shl i32 %379, 1
  %381 = shl i32 %378, 1
  %382 = icmp ugt i32 %381, %57
  br i1 %382, label %385, label %383, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !493

383:                                              ; preds = %.preheader.i4.i.i.i.23
  %384 = icmp eq i32 %381, %57
  br i1 %384, label %._crit_edge197, label %..preheader.i4.i.i.i.24_crit_edge, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !500

..preheader.i4.i.i.i.24_crit_edge:                ; preds = %383
  br label %.preheader.i4.i.i.i.24, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !508

._crit_edge197:                                   ; preds = %383
  br label %405, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !508

385:                                              ; preds = %.preheader.i4.i.i.i.23
  %386 = sub nuw i32 %381, %57, !spirv.Decorations !441
  %387 = add i32 %380, 1
  br label %.preheader.i4.i.i.i.24, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !500

.preheader.i4.i.i.i.24:                           ; preds = %..preheader.i4.i.i.i.24_crit_edge, %385
  %388 = phi i32 [ %386, %385 ], [ %381, %..preheader.i4.i.i.i.24_crit_edge ]
  %389 = phi i32 [ %387, %385 ], [ %380, %..preheader.i4.i.i.i.24_crit_edge ]
  %390 = shl i32 %389, 1
  %391 = shl i32 %388, 1
  %392 = icmp ugt i32 %391, %57
  br i1 %392, label %395, label %393, !stats.blockFrequency.digits !480, !stats.blockFrequency.scale !500

393:                                              ; preds = %.preheader.i4.i.i.i.24
  %394 = icmp eq i32 %391, %57
  br i1 %394, label %._crit_edge198, label %..preheader.i4.i.i.i.25_crit_edge, !stats.blockFrequency.digits !480, !stats.blockFrequency.scale !508

..preheader.i4.i.i.i.25_crit_edge:                ; preds = %393
  br label %.preheader.i4.i.i.i.25, !stats.blockFrequency.digits !509, !stats.blockFrequency.scale !510

._crit_edge198:                                   ; preds = %393
  br label %405, !stats.blockFrequency.digits !509, !stats.blockFrequency.scale !510

395:                                              ; preds = %.preheader.i4.i.i.i.24
  %396 = sub nuw i32 %391, %57, !spirv.Decorations !441
  %397 = add i32 %390, 1
  br label %.preheader.i4.i.i.i.25, !stats.blockFrequency.digits !480, !stats.blockFrequency.scale !508

.preheader.i4.i.i.i.25:                           ; preds = %..preheader.i4.i.i.i.25_crit_edge, %395
  %398 = phi i32 [ %396, %395 ], [ %391, %..preheader.i4.i.i.i.25_crit_edge ]
  %399 = phi i32 [ %397, %395 ], [ %390, %..preheader.i4.i.i.i.25_crit_edge ]
  %400 = shl i32 %399, 1
  %401 = shl i32 %398, 1
  %402 = icmp ugt i32 %401, %57
  br i1 %402, label %410, label %403, !stats.blockFrequency.digits !511, !stats.blockFrequency.scale !500

403:                                              ; preds = %.preheader.i4.i.i.i.25
  %404 = icmp eq i32 %401, %57
  br i1 %404, label %._crit_edge199, label %..preheader.i4.i.i.i.26_crit_edge, !stats.blockFrequency.digits !511, !stats.blockFrequency.scale !508

..preheader.i4.i.i.i.26_crit_edge:                ; preds = %403
  br label %.preheader.i4.i.i.i.26, !stats.blockFrequency.digits !511, !stats.blockFrequency.scale !510

._crit_edge199:                                   ; preds = %403
  br label %405, !stats.blockFrequency.digits !511, !stats.blockFrequency.scale !510

405:                                              ; preds = %._crit_edge199, %._crit_edge198, %._crit_edge197, %._crit_edge196, %._crit_edge195, %._crit_edge194, %._crit_edge193, %._crit_edge192, %._crit_edge191, %._crit_edge190, %._crit_edge189, %._crit_edge188, %._crit_edge187, %._crit_edge186, %._crit_edge185, %._crit_edge184, %._crit_edge183, %._crit_edge182, %._crit_edge181, %._crit_edge180, %._crit_edge179, %._crit_edge178, %._crit_edge177, %._crit_edge176, %._crit_edge175, %._crit_edge174, %._crit_edge173
  %406 = phi i16 [ 26, %._crit_edge173 ], [ 25, %._crit_edge174 ], [ 24, %._crit_edge175 ], [ 23, %._crit_edge176 ], [ 22, %._crit_edge177 ], [ 21, %._crit_edge178 ], [ 20, %._crit_edge179 ], [ 19, %._crit_edge180 ], [ 18, %._crit_edge181 ], [ 17, %._crit_edge182 ], [ 16, %._crit_edge183 ], [ 15, %._crit_edge184 ], [ 14, %._crit_edge185 ], [ 13, %._crit_edge186 ], [ 12, %._crit_edge187 ], [ 11, %._crit_edge188 ], [ 10, %._crit_edge189 ], [ 9, %._crit_edge190 ], [ 8, %._crit_edge191 ], [ 7, %._crit_edge192 ], [ 6, %._crit_edge193 ], [ 5, %._crit_edge194 ], [ 4, %._crit_edge195 ], [ 3, %._crit_edge196 ], [ 2, %._crit_edge197 ], [ 1, %._crit_edge198 ], [ 0, %._crit_edge199 ]
  %.lcssa = phi i32 [ 0, %._crit_edge173 ], [ %.demoted.zext, %._crit_edge174 ], [ %160, %._crit_edge175 ], [ %170, %._crit_edge176 ], [ %180, %._crit_edge177 ], [ %190, %._crit_edge178 ], [ %200, %._crit_edge179 ], [ %210, %._crit_edge180 ], [ %220, %._crit_edge181 ], [ %230, %._crit_edge182 ], [ %240, %._crit_edge183 ], [ %250, %._crit_edge184 ], [ %260, %._crit_edge185 ], [ %270, %._crit_edge186 ], [ %280, %._crit_edge187 ], [ %290, %._crit_edge188 ], [ %300, %._crit_edge189 ], [ %310, %._crit_edge190 ], [ %320, %._crit_edge191 ], [ %330, %._crit_edge192 ], [ %340, %._crit_edge193 ], [ %350, %._crit_edge194 ], [ %360, %._crit_edge195 ], [ %370, %._crit_edge196 ], [ %380, %._crit_edge197 ], [ %390, %._crit_edge198 ], [ %400, %._crit_edge199 ]
  %407 = or i32 %.lcssa, 1
  %408 = trunc i16 %406 to i8
  %.demoted.zext167 = zext i8 %408 to i32
  %409 = shl i32 %407, %.demoted.zext167
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i, !stats.blockFrequency.digits !512, !stats.blockFrequency.scale !433

410:                                              ; preds = %.preheader.i4.i.i.i.25
  %411 = add i32 %400, 1
  br label %.preheader.i4.i.i.i.26, !stats.blockFrequency.digits !511, !stats.blockFrequency.scale !508

.preheader.i4.i.i.i.26:                           ; preds = %..preheader.i4.i.i.i.26_crit_edge, %410
  %412 = phi i32 [ %411, %410 ], [ %400, %..preheader.i4.i.i.i.26_crit_edge ]
  %413 = or i32 %412, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i, !stats.blockFrequency.digits !513, !stats.blockFrequency.scale !500

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i:        ; preds = %.preheader.i4.i.i.i.26, %405
  %414 = phi i32 [ %409, %405 ], [ %413, %.preheader.i4.i.i.i.26 ]
  %415 = lshr i32 %414, 3
  %416 = and i32 %415, 8388607
  %417 = and i32 %414, 7
  %418 = icmp eq i32 %417, 0
  br i1 %418, label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge, label %419, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !433

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  br label %.critedge53, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !436

419:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
  %420 = and i32 %414, 15
  %421 = icmp ugt i32 %417, 4
  %.not10 = icmp eq i32 %420, 12
  %or.cond52 = or i1 %421, %.not10
  br i1 %or.cond52, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %..critedge53_crit_edge200, !stats.blockFrequency.digits !514, !stats.blockFrequency.scale !434

..critedge53_crit_edge200:                        ; preds = %419
  br label %.critedge53, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i: ; preds = %419
  %422 = icmp eq i32 %416, 8388607
  br i1 %422, label %424, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  %423 = add nuw nsw i32 %416, 1, !spirv.Decorations !427
  br label %.critedge53, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !445

424:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
  %425 = add nsw i32 %66, 128, !spirv.Decorations !422
  %426 = icmp eq i32 %425, 255
  br i1 %426, label %434, label %..critedge53_crit_edge201, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !445

..critedge53_crit_edge201:                        ; preds = %424
  br label %.critedge53, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !458

.critedge53:                                      ; preds = %..critedge53_crit_edge201, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge, %..critedge53_crit_edge200, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge, %..critedge53_crit_edge
  %427 = phi i32 [ %416, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge ], [ %416, %..critedge53_crit_edge200 ], [ %423, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge ], [ 0, %..critedge53_crit_edge201 ], [ 0, %..critedge53_crit_edge ]
  %428 = phi i32 [ %139, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge ], [ %139, %..critedge53_crit_edge200 ], [ %139, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge ], [ %425, %..critedge53_crit_edge201 ], [ %139, %..critedge53_crit_edge ]
  %429 = and i32 %36, -2147483648
  %430 = shl nuw nsw i32 %428, 23, !spirv.Decorations !427
  %431 = or i32 %429, %430
  %432 = or i32 %431, %427
  %433 = bitcast i32 %432 to float
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !515, !stats.blockFrequency.scale !433

434:                                              ; preds = %424
  %435 = icmp sgt i32 %36, -1
  %.54 = select i1 %435, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !458

436:                                              ; preds = %64
  %437 = icmp sgt i32 %36, -1
  %.51 = select i1 %437, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !431

438:                                              ; preds = %46
  %tobool.i = icmp eq i32 %57, 0
  br i1 %tobool.i, label %.precompiled_u32divrem.exit_crit_edge, label %if.end.i, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !426

.precompiled_u32divrem.exit_crit_edge:            ; preds = %438
  br label %precompiled_u32divrem.exit, !stats.blockFrequency.digits !516, !stats.blockFrequency.scale !431

if.end.i:                                         ; preds = %438
  %conv.i = uitofp i32 %57 to float
  %div.i = fdiv float 1.000000e+00, %conv.i, !fpmath !517
  %conv1.i = uitofp i32 %57 to double
  %439 = fsub double 0.000000e+00, %conv1.i
  %conv3.i = fpext float %div.i to double
  %conv2.i = uitofp i32 %55 to double
  %440 = call double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double %439, double %conv3.i, double 0x3FF0000000004000)
  %441 = call double @llvm.genx.GenISA.mul.rtz.f64.f64.f64(double %conv3.i, double %conv2.i)
  %442 = call double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double %441, double %440, double %441)
  %conv6.i = fptoui double %442 to i32
  br label %precompiled_u32divrem.exit, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !426

precompiled_u32divrem.exit:                       ; preds = %.precompiled_u32divrem.exit_crit_edge, %if.end.i
  %retval.0.i = phi i32 [ %conv6.i, %if.end.i ], [ -1, %.precompiled_u32divrem.exit_crit_edge ]
  br label %443, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !426

443:                                              ; preds = %._crit_edge, %precompiled_u32divrem.exit
  %444 = phi i32 [ -2147483648, %precompiled_u32divrem.exit ], [ %449, %._crit_edge ]
  %445 = phi i64 [ 0, %precompiled_u32divrem.exit ], [ %450, %._crit_edge ]
  %446 = bitcast i64 %445 to <2 x i32>
  %447 = extractelement <2 x i32> %446, i32 0
  %448 = extractelement <2 x i32> %446, i32 1
  %449 = lshr i32 %444, 1
  %450 = add nuw nsw i64 %445, 1, !spirv.Decorations !427
  %451 = icmp eq i32 %448, 0
  %452 = icmp ugt i32 %447, 30
  %453 = and i1 %451, %452
  %454 = icmp ugt i32 %448, 0
  %455 = or i1 %453, %454
  %456 = and i32 %retval.0.i, %449
  %457 = icmp eq i32 %456, %449
  %458 = or i1 %455, %457
  br i1 %458, label %459, label %._crit_edge, !stats.blockFrequency.digits !429, !stats.blockFrequency.scale !417

._crit_edge:                                      ; preds = %443
  br label %443, !stats.blockFrequency.digits !430, !stats.blockFrequency.scale !417

459:                                              ; preds = %443
  %.lcssa219 = phi i64 [ %450, %443 ]
  %460 = trunc i64 %.lcssa219 to i32
  %461 = sub nsw i32 31, %460, !spirv.Decorations !422
  %462 = add nsw i32 %53, %461, !spirv.Decorations !422
  %463 = icmp sgt i32 %462, 127
  br i1 %463, label %638, label %464, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !426

464:                                              ; preds = %459
  %465 = mul i32 %57, %retval.0.i
  %466 = sub i32 %55, %465
  %467 = icmp sgt i32 %462, -127
  br i1 %467, label %585, label %468, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !431

468:                                              ; preds = %464
  %469 = icmp ult i32 %462, -149
  br i1 %469, label %572, label %470, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !433

470:                                              ; preds = %468
  %471 = add nsw i32 %462, 152, !spirv.Decorations !422
  %472 = icmp sgt i32 %471, %461
  br i1 %472, label %511, label %473, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

473:                                              ; preds = %470
  %474 = xor i32 %462, -1
  %475 = sub nsw i32 %474, %460, !spirv.Decorations !422
  %476 = add nsw i32 %475, -117, !spirv.Decorations !422
  %477 = lshr i32 %retval.0.i, %476
  %478 = add nsw i32 %475, -120, !spirv.Decorations !422
  %479 = lshr i32 %retval.0.i, %478
  %480 = and i32 %479, 7
  %481 = and i32 %479, 1
  %482 = icmp eq i32 %481, 0
  br i1 %482, label %483, label %.._crit_edge81_crit_edge, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !436

.._crit_edge81_crit_edge:                         ; preds = %473
  br label %._crit_edge81, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !445

483:                                              ; preds = %473
  %484 = shl nsw i32 -1, %478, !spirv.Decorations !422
  %485 = xor i32 %484, -1
  %486 = and i32 %retval.0.i, %485
  %487 = icmp ne i32 %486, 0
  %488 = icmp ne i32 %55, %465
  %489 = or i1 %487, %488
  %490 = zext i1 %489 to i32
  %491 = or i32 %480, %490
  br label %._crit_edge81, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !445

._crit_edge81:                                    ; preds = %.._crit_edge81_crit_edge, %483
  %492 = phi i32 [ %491, %483 ], [ %480, %.._crit_edge81_crit_edge ]
  %493 = icmp eq i32 %492, 0
  br i1 %493, label %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, label %494, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !436

._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge: ; preds = %._crit_edge81
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !445

494:                                              ; preds = %._crit_edge81
  %495 = icmp ugt i32 %492, 4
  br i1 %495, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118, label %496, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !436

496:                                              ; preds = %494
  %497 = and i32 %477, 1
  %498 = icmp eq i32 %492, 4
  %499 = icmp ne i32 %497, 0
  %not. = and i1 %498, %499
  %500 = icmp ugt i32 %477, 8388606
  br i1 %not., label %._crit_edge170, label %._crit_edge171, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !445

._crit_edge171:                                   ; preds = %496
  %501 = sext i1 %500 to i32
  %502 = sext i1 %not. to i32
  br label %560, !stats.blockFrequency.digits !520, !stats.blockFrequency.scale !458

._crit_edge170:                                   ; preds = %496
  %503 = add nuw nsw i32 %477, 1, !spirv.Decorations !427
  %504 = select i1 %500, i32 0, i32 %503
  %505 = sext i1 %500 to i32
  %506 = sext i1 %not. to i32
  br label %552, !stats.blockFrequency.digits !520, !stats.blockFrequency.scale !458

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118: ; preds = %494
  %507 = add nuw nsw i32 %477, 1, !spirv.Decorations !427
  %508 = icmp ugt i32 %477, 8388606
  %509 = select i1 %508, i32 0, i32 %507
  %510 = sext i1 %508 to i32
  br label %552, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !445

511:                                              ; preds = %470
  %512 = sub nsw i32 %471, %461, !spirv.Decorations !422
  %513 = shl i32 %retval.0.i, %512
  %514 = icmp eq i32 %466, 0
  br i1 %514, label %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge, label %.lr.ph89.preheader, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !436

._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge: ; preds = %511
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !445

.lr.ph89.preheader:                               ; preds = %511
  br label %.lr.ph89, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !436

.lr.ph89:                                         ; preds = %.preheader.i7.i.i.i..lr.ph89_crit_edge, %.lr.ph89.preheader
  %515 = phi i32 [ %528, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ 0, %.lr.ph89.preheader ]
  %516 = phi i32 [ %527, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ 0, %.lr.ph89.preheader ]
  %517 = phi i32 [ %526, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ %466, %.lr.ph89.preheader ]
  %518 = shl i32 %516, 1
  %519 = shl i32 %517, 1
  %520 = icmp ugt i32 %519, %57
  br i1 %520, label %523, label %521, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !426

521:                                              ; preds = %.lr.ph89
  %522 = icmp eq i32 %519, %57
  br i1 %522, label %530, label %..preheader.i7.i.i.i_crit_edge, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !431

..preheader.i7.i.i.i_crit_edge:                   ; preds = %521
  br label %.preheader.i7.i.i.i, !stats.blockFrequency.digits !440, !stats.blockFrequency.scale !431

523:                                              ; preds = %.lr.ph89
  %524 = sub nuw i32 %519, %57, !spirv.Decorations !441
  %525 = add i32 %518, 1
  br label %.preheader.i7.i.i.i, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !431

.preheader.i7.i.i.i:                              ; preds = %..preheader.i7.i.i.i_crit_edge, %523
  %526 = phi i32 [ %524, %523 ], [ %519, %..preheader.i7.i.i.i_crit_edge ]
  %527 = phi i32 [ %525, %523 ], [ %518, %..preheader.i7.i.i.i_crit_edge ]
  %528 = add nuw i32 %515, 1, !spirv.Decorations !441
  %529 = icmp ult i32 %528, %512
  br i1 %529, label %.preheader.i7.i.i.i..lr.ph89_crit_edge, label %.preheader.i7.i.i.i._crit_edge, !stats.blockFrequency.digits !442, !stats.blockFrequency.scale !426

.preheader.i7.i.i.i..lr.ph89_crit_edge:           ; preds = %.preheader.i7.i.i.i
  br label %.lr.ph89, !stats.blockFrequency.digits !521, !stats.blockFrequency.scale !426

530:                                              ; preds = %521
  %.lcssa216 = phi i32 [ %515, %521 ]
  %.lcssa214 = phi i32 [ %518, %521 ]
  %531 = add i32 %.lcssa214, 1
  %532 = xor i32 %.lcssa216, -1
  %533 = add i32 %512, %532
  %534 = shl i32 %531, %533
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, !stats.blockFrequency.digits !444, !stats.blockFrequency.scale !458

.preheader.i7.i.i.i._crit_edge:                   ; preds = %.preheader.i7.i.i.i
  %.lcssa218 = phi i32 [ %527, %.preheader.i7.i.i.i ]
  %535 = or i32 %.lcssa218, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, !stats.blockFrequency.digits !522, !stats.blockFrequency.scale !445

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i:        ; preds = %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge, %.preheader.i7.i.i.i._crit_edge, %530
  %536 = phi i32 [ %534, %530 ], [ %535, %.preheader.i7.i.i.i._crit_edge ], [ 0, %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge ]
  %537 = or i32 %513, %536
  %538 = and i32 %537, 7
  %539 = lshr i32 %537, 3
  %540 = icmp eq i32 %538, 0
  br i1 %540, label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !436

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !445

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge
  %.ph = phi i32 [ %539, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge ], [ %477, %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge ]
  %541 = icmp ugt i32 %.ph, 8388606
  %542 = sext i1 %541 to i32
  br label %560, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
  %543 = and i32 %537, 15
  %544 = icmp ugt i32 %538, 4
  %.not = icmp eq i32 %543, 12
  %or.cond = or i1 %544, %.not
  %545 = icmp ugt i32 %537, 67108855
  br i1 %or.cond, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
  %546 = sext i1 %545 to i32
  %547 = sext i1 %or.cond to i32
  br label %560, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !445

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
  %548 = add nuw nsw i32 %539, 1, !spirv.Decorations !427
  %549 = select i1 %545, i32 0, i32 %548
  %550 = sext i1 %545 to i32
  %551 = sext i1 %or.cond to i32
  br label %552, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !445

552:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge, %._crit_edge170, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118
  %553 = phi i32 [ %509, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %549, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %504, %._crit_edge170 ]
  %554 = phi i32 [ %510, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %550, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %505, %._crit_edge170 ]
  %555 = phi i32 [ -1, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %551, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %506, %._crit_edge170 ]
  %556 = icmp ne i32 %554, 0
  %557 = sext i1 %556 to i32
  %558 = icmp ne i32 %555, 0
  %559 = sext i1 %558 to i32
  br label %560, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !436

560:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172, %._crit_edge171, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, %552
  %561 = phi i32 [ %557, %552 ], [ %546, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ %542, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %501, %._crit_edge171 ]
  %562 = phi i32 [ %559, %552 ], [ %547, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ 0, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %502, %._crit_edge171 ]
  %563 = phi i32 [ %553, %552 ], [ %539, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ %.ph, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %477, %._crit_edge171 ]
  %564 = icmp ne i32 %562, 0
  %565 = icmp ne i32 %561, 0
  %566 = and i1 %564, %565
  %567 = select i1 %566, i32 8388608, i32 0
  %568 = and i32 %36, -2147483648
  %569 = or i32 %568, %567
  %570 = or i32 %569, %563
  %571 = bitcast i32 %570 to float
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

572:                                              ; preds = %468
  %573 = icmp eq i32 %462, -150
  br i1 %573, label %574, label %.._crit_edge80_crit_edge, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

.._crit_edge80_crit_edge:                         ; preds = %572
  br label %._crit_edge80, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !436

574:                                              ; preds = %572
  %575 = lshr i32 -2147483648, %460
  %576 = icmp ne i32 %55, %465
  %577 = icmp ne i32 %retval.0.i, %575
  %578 = or i1 %576, %577
  %579 = sext i1 %578 to i32
  br label %._crit_edge80, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !436

._crit_edge80:                                    ; preds = %.._crit_edge80_crit_edge, %574
  %580 = phi i32 [ %579, %574 ], [ 0, %.._crit_edge80_crit_edge ]
  %581 = icmp ne i32 %580, 0
  %582 = icmp sgt i32 %36, -1
  br i1 %582, label %584, label %583, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !434

583:                                              ; preds = %._crit_edge80
  %spec.select49 = select i1 %581, float 0xB6A0000000000000, float -0.000000e+00
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !435, !stats.blockFrequency.scale !436

584:                                              ; preds = %._crit_edge80
  %spec.select48 = select i1 %581, float 0x36A0000000000000, float 0.000000e+00
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !434

585:                                              ; preds = %464
  %586 = add nsw i32 %462, 127, !spirv.Decorations !422
  %587 = add nsw i32 %460, -8, !spirv.Decorations !422
  %588 = shl i32 %retval.0.i, %587
  %589 = and i32 %588, 8388607
  %590 = icmp eq i32 %466, 0
  br i1 %590, label %..critedge46_crit_edge, label %.preheader.i.i.i.i.preheader, !stats.blockFrequency.digits !432, !stats.blockFrequency.scale !433

..critedge46_crit_edge:                           ; preds = %585
  br label %.critedge46, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !434

.preheader.i.i.i.i.preheader:                     ; preds = %585
  %591 = add nsw i32 %460, -5, !spirv.Decorations !422
  %.not92 = icmp eq i32 %591, 0
  br i1 %.not92, label %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge, label %.lr.ph87.preheader, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !433

.lr.ph87.preheader:                               ; preds = %.preheader.i.i.i.i.preheader
  br label %.lr.ph87, !stats.blockFrequency.digits !514, !stats.blockFrequency.scale !434

.lr.ph87:                                         ; preds = %.preheader.i.i.i.i..lr.ph87_crit_edge, %.lr.ph87.preheader
  %592 = phi i32 [ %605, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ 0, %.lr.ph87.preheader ]
  %593 = phi i32 [ %604, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ 0, %.lr.ph87.preheader ]
  %594 = phi i32 [ %603, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ %466, %.lr.ph87.preheader ]
  %595 = shl i32 %593, 1
  %596 = shl i32 %594, 1
  %597 = icmp ugt i32 %596, %57
  br i1 %597, label %600, label %598, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !425

598:                                              ; preds = %.lr.ph87
  %599 = icmp eq i32 %596, %57
  br i1 %599, label %607, label %..preheader.i.i.i.i_crit_edge, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !426

..preheader.i.i.i.i_crit_edge:                    ; preds = %598
  br label %.preheader.i.i.i.i, !stats.blockFrequency.digits !524, !stats.blockFrequency.scale !426

600:                                              ; preds = %.lr.ph87
  %601 = sub nuw i32 %596, %57, !spirv.Decorations !441
  %602 = add i32 %595, 1
  br label %.preheader.i.i.i.i, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !426

.preheader.i.i.i.i:                               ; preds = %..preheader.i.i.i.i_crit_edge, %600
  %603 = phi i32 [ %601, %600 ], [ %596, %..preheader.i.i.i.i_crit_edge ]
  %604 = phi i32 [ %602, %600 ], [ %595, %..preheader.i.i.i.i_crit_edge ]
  %605 = add nuw i32 %592, 1, !spirv.Decorations !441
  %606 = icmp ult i32 %605, %591
  br i1 %606, label %.preheader.i.i.i.i..lr.ph87_crit_edge, label %.preheader.i.i.i.i._crit_edge.loopexit, !stats.blockFrequency.digits !525, !stats.blockFrequency.scale !425

.preheader.i.i.i.i._crit_edge.loopexit:           ; preds = %.preheader.i.i.i.i
  %.lcssa213 = phi i32 [ %604, %.preheader.i.i.i.i ]
  br label %.preheader.i.i.i.i._crit_edge, !stats.blockFrequency.digits !526, !stats.blockFrequency.scale !436

.preheader.i.i.i.i..lr.ph87_crit_edge:            ; preds = %.preheader.i.i.i.i
  br label %.lr.ph87, !stats.blockFrequency.digits !527, !stats.blockFrequency.scale !425

607:                                              ; preds = %598
  %.lcssa211 = phi i32 [ %592, %598 ]
  %.lcssa209 = phi i32 [ %595, %598 ]
  %608 = add i32 %.lcssa209, 1
  %609 = xor i32 %.lcssa211, -1
  %610 = add i32 %591, %609
  %611 = shl i32 %608, %610
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i, !stats.blockFrequency.digits !528, !stats.blockFrequency.scale !445

.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge: ; preds = %.preheader.i.i.i.i.preheader
  br label %.preheader.i.i.i.i._crit_edge, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !436

.preheader.i.i.i.i._crit_edge:                    ; preds = %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge, %.preheader.i.i.i.i._crit_edge.loopexit
  %.lcssa70 = phi i32 [ 0, %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge ], [ %.lcssa213, %.preheader.i.i.i.i._crit_edge.loopexit ]
  %612 = or i32 %.lcssa70, 1
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !434

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i:         ; preds = %.preheader.i.i.i.i._crit_edge, %607
  %613 = phi i32 [ %611, %607 ], [ %612, %.preheader.i.i.i.i._crit_edge ]
  %614 = lshr i32 %613, 3
  %615 = or i32 %614, %589
  %616 = and i32 %613, 7
  %617 = icmp eq i32 %616, 0
  br i1 %617, label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge, label %618, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !433

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  br label %.critedge46, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !436

618:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
  %619 = icmp ugt i32 %616, 4
  br i1 %619, label %..critedge_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i, !stats.blockFrequency.digits !514, !stats.blockFrequency.scale !434

..critedge_crit_edge:                             ; preds = %618
  br label %.critedge, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i: ; preds = %618
  %620 = and i32 %615, 1
  %621 = icmp ne i32 %616, 4
  %622 = icmp eq i32 %620, 0
  %623 = or i1 %621, %622
  br i1 %623, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !436

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
  br label %.critedge, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !445

.critedge:                                        ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge, %..critedge_crit_edge
  %624 = add nuw nsw i32 %615, 1, !spirv.Decorations !427
  %625 = icmp ugt i32 %615, 8388606
  br i1 %625, label %626, label %.critedge..critedge46_crit_edge, !stats.blockFrequency.digits !530, !stats.blockFrequency.scale !434

.critedge..critedge46_crit_edge:                  ; preds = %.critedge
  br label %.critedge46, !stats.blockFrequency.digits !530, !stats.blockFrequency.scale !436

626:                                              ; preds = %.critedge
  %627 = add nsw i32 %462, 128, !spirv.Decorations !422
  %628 = icmp eq i32 %627, 255
  br i1 %628, label %629, label %..critedge46_crit_edge169, !stats.blockFrequency.digits !530, !stats.blockFrequency.scale !436

..critedge46_crit_edge169:                        ; preds = %626
  br label %.critedge46, !stats.blockFrequency.digits !530, !stats.blockFrequency.scale !445

629:                                              ; preds = %626
  %630 = icmp sgt i32 %36, -1
  %.47 = select i1 %630, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !530, !stats.blockFrequency.scale !445

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
  br label %.critedge46, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !445

.critedge46:                                      ; preds = %..critedge46_crit_edge169, %.critedge..critedge46_crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge, %..critedge46_crit_edge
  %631 = phi i32 [ %615, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge ], [ %615, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge ], [ %624, %.critedge..critedge46_crit_edge ], [ %624, %..critedge46_crit_edge169 ], [ %589, %..critedge46_crit_edge ]
  %632 = phi i32 [ %586, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge ], [ %586, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge ], [ %586, %.critedge..critedge46_crit_edge ], [ %627, %..critedge46_crit_edge169 ], [ %586, %..critedge46_crit_edge ]
  %633 = and i32 %36, -2147483648
  %634 = shl nuw nsw i32 %632, 23, !spirv.Decorations !427
  %635 = or i32 %633, %634
  %636 = or i32 %635, %631
  %637 = bitcast i32 %636 to float
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !531, !stats.blockFrequency.scale !433

638:                                              ; preds = %459
  %639 = icmp sgt i32 %36, -1
  %. = select i1 %639, float 0x7FF0000000000000, float 0xFFF0000000000000
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !431

640:                                              ; preds = %42
  %641 = and i32 %36, -2147483648
  %642 = bitcast i32 %641 to float
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !424, !stats.blockFrequency.scale !425

643:                                              ; preds = %40
  %644 = and i32 %36, -2147483648
  %645 = bitcast i32 %644 to float
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !532, !stats.blockFrequency.scale !533

646:                                              ; preds = %35
  %647 = icmp eq i32 %29, 0
  %648 = icmp eq i32 %28, 255
  %649 = and i1 %648, %647
  %650 = and i32 %36, -2147483648
  %651 = or i32 %650, 2139095040
  %652 = bitcast i32 %651 to float
  %653 = select i1 %649, float 0x7FF8000000000000, float %652
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !418, !stats.blockFrequency.scale !419

__imf_fdiv_rn.exit:                               ; preds = %.__imf_fdiv_rn.exit_crit_edge168, %.__imf_fdiv_rn.exit_crit_edge, %136, %137, %583, %584, %434, %436, %629, %638, %646, %643, %640, %.critedge46, %560, %.critedge53, %120
  %654 = phi float [ %645, %643 ], [ %642, %640 ], [ %653, %646 ], [ %433, %.critedge53 ], [ %129, %120 ], [ %571, %560 ], [ %637, %.critedge46 ], [ %., %638 ], [ %.47, %629 ], [ %spec.select48, %584 ], [ %spec.select49, %583 ], [ %.51, %436 ], [ %.54, %434 ], [ %spec.select55, %137 ], [ %spec.select56, %136 ], [ 0x7FF8000000000000, %.__imf_fdiv_rn.exit_crit_edge ], [ 0x7FF8000000000000, %.__imf_fdiv_rn.exit_crit_edge168 ]
  %655 = bitcast float %17 to i32
  %656 = and i32 %655, 2139095040
  %657 = icmp eq i32 %656, 0
  %658 = select i1 %657, float 0x41F0000000000000, float 1.000000e+00
  %659 = icmp uge i32 %656, 1677721600
  %660 = select i1 %659, float 0x3DF0000000000000, float %658
  %661 = fmul float %17, %660
  %662 = fdiv float 1.000000e+00, %661
  %663 = fmul float %662, %13
  %664 = fmul float %663, %660
  %665 = and i32 %655, 8388607
  %666 = icmp eq i32 %656, 0
  %667 = icmp eq i32 %665, 0
  %668 = or i1 %666, %667
  %669 = xor i1 %668, true
  %670 = fcmp oeq float %13, %17
  %671 = and i1 %670, %669
  %672 = select i1 %671, float 1.000000e+00, float %664
  %673 = ptrtoint float addrspace(1)* %2 to i64
  %674 = add i64 %10, %673
  %675 = inttoptr i64 %674 to float addrspace(1)*
  store float %654, float addrspace(1)* %675, align 4
  %676 = ptrtoint float addrspace(1)* %3 to i64
  %677 = add i64 %10, %676
  %678 = inttoptr i64 %677 to float addrspace(1)*
  store float %672, float addrspace(1)* %678, align 4
  ret void, !stats.blockFrequency.digits !412, !stats.blockFrequency.scale !413
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
!IGCMetadata = !{!29}
!opencl.ocl.version = !{!408, !408, !408, !408, !408, !408, !408, !408}
!opencl.spir.version = !{!408, !408, !408, !408, !408, !408, !408}
!llvm.ident = !{!409, !409, !409, !409, !409, !409, !409, !410}
!llvm.module.flags = !{!411}
!printf.strings = !{}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 100000}
!2 = !{i16 6, i16 14}
!3 = !{!"SPV_INTEL_vector_compute"}
!4 = !{void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <8 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32)* @kernel, !5}
!5 = !{!6, !7, !26, !27, !28}
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
!28 = !{!"max_reg_pressure", i32 0}
!29 = !{!"ModuleMD", !30, !31, !119, !239, !270, !287, !310, !320, !322, !323, !338, !339, !340, !341, !345, !346, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !365, !369, !370, !371, !372, !373, !374, !375, !376, !377, !378, !379, !380, !381, !184, !382, !385, !386, !388, !390, !393, !394, !395, !397, !398, !399, !404, !405, !406, !407}
!30 = !{!"isPrecise", i1 false}
!31 = !{!"compOpt", !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118}
!32 = !{!"DenormsAreZero", i1 false}
!33 = !{!"BFTFDenormsAreZero", i1 false}
!34 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!35 = !{!"OptDisable", i1 false}
!36 = !{!"MadEnable", i1 true}
!37 = !{!"NoSignedZeros", i1 false}
!38 = !{!"NoNaNs", i1 false}
!39 = !{!"FloatRoundingMode", i32 0}
!40 = !{!"FloatCvtIntRoundingMode", i32 3}
!41 = !{!"LoadCacheDefault", i32 4}
!42 = !{!"StoreCacheDefault", i32 2}
!43 = !{!"VISAPreSchedRPThreshold", i32 0}
!44 = !{!"SetLoopUnrollThreshold", i32 0}
!45 = !{!"UnsafeMathOptimizations", i1 false}
!46 = !{!"disableCustomUnsafeOpts", i1 false}
!47 = !{!"disableReducePow", i1 false}
!48 = !{!"disableSqrtOpt", i1 false}
!49 = !{!"FiniteMathOnly", i1 false}
!50 = !{!"FastRelaxedMath", i1 false}
!51 = !{!"DashGSpecified", i1 false}
!52 = !{!"FastCompilation", i1 false}
!53 = !{!"UseScratchSpacePrivateMemory", i1 true}
!54 = !{!"RelaxedBuiltins", i1 false}
!55 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!56 = !{!"GreaterThan2GBBufferRequired", i1 true}
!57 = !{!"GreaterThan4GBBufferRequired", i1 true}
!58 = !{!"DisableA64WA", i1 false}
!59 = !{!"ForceEnableA64WA", i1 false}
!60 = !{!"PushConstantsEnable", i1 true}
!61 = !{!"HasPositivePointerOffset", i1 false}
!62 = !{!"HasBufferOffsetArg", i1 true}
!63 = !{!"BufferOffsetArgOptional", i1 true}
!64 = !{!"replaceGlobalOffsetsByZero", i1 false}
!65 = !{!"forcePixelShaderSIMDMode", i32 0}
!66 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!67 = !{!"UniformWGS", i1 false}
!68 = !{!"disableVertexComponentPacking", i1 false}
!69 = !{!"disablePartialVertexComponentPacking", i1 false}
!70 = !{!"PreferBindlessImages", i1 false}
!71 = !{!"UseBindlessMode", i1 false}
!72 = !{!"UseLegacyBindlessMode", i1 true}
!73 = !{!"disableMathRefactoring", i1 false}
!74 = !{!"atomicBranch", i1 false}
!75 = !{!"spillCompression", i1 false}
!76 = !{!"DisableEarlyOut", i1 false}
!77 = !{!"ForceInt32DivRemEmu", i1 false}
!78 = !{!"ForceInt32DivRemEmuSP", i1 false}
!79 = !{!"DisableFastestSingleCSSIMD", i1 false}
!80 = !{!"DisableFastestLinearScan", i1 false}
!81 = !{!"UseStatelessforPrivateMemory", i1 false}
!82 = !{!"EnableTakeGlobalAddress", i1 false}
!83 = !{!"IsLibraryCompilation", i1 false}
!84 = !{!"LibraryCompileSIMDSize", i32 0}
!85 = !{!"FastVISACompile", i1 false}
!86 = !{!"MatchSinCosPi", i1 false}
!87 = !{!"ExcludeIRFromZEBinary", i1 false}
!88 = !{!"EmitZeBinVISASections", i1 false}
!89 = !{!"FP64GenEmulationEnabled", i1 false}
!90 = !{!"FP64GenConvEmulationEnabled", i1 false}
!91 = !{!"allowDisableRematforCS", i1 false}
!92 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!93 = !{!"DisableCPSOmaskWA", i1 false}
!94 = !{!"DisableFastestGopt", i1 false}
!95 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!96 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!97 = !{!"DisableConstantCoalescing", i1 false}
!98 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!99 = !{!"WaEnableALTModeVisaWA", i1 false}
!100 = !{!"EnableLdStCombineforLoad", i1 false}
!101 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!102 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!103 = !{!"NewSpillCostFunction", i1 false}
!104 = !{!"EnableVRT", i1 false}
!105 = !{!"ForceLargeGRFNum4RQ", i1 false}
!106 = !{!"DisableEUFusion", i1 false}
!107 = !{!"DisableFDivToFMulInvOpt", i1 false}
!108 = !{!"initializePhiSampleSourceWA", i1 false}
!109 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!110 = !{!"DisableLoosenSimd32Occu", i1 false}
!111 = !{!"FastestS1Options", i32 0}
!112 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!113 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!114 = !{!"DisableLscSamplerRouting", i1 false}
!115 = !{!"UseBarrierControlFlowOptimization", i1 false}
!116 = !{!"EnableDynamicRQManagement", i1 false}
!117 = !{!"Quad8InputThreshold", i32 0}
!118 = !{!"UseResourceLoopUnrollNested", i1 false}
!119 = !{!"FuncMD", !120, !121}
!120 = !{!"FuncMDMap[0]", void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <8 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32)* @kernel}
!121 = !{!"FuncMDValue[0]", !122, !123, !127, !128, !129, !149, !176, !177, !178, !179, !180, !181, !182, !183, !184, !185, !186, !187, !188, !189, !190, !191, !192, !199, !206, !213, !220, !227, !234, !235}
!122 = !{!"localOffsets"}
!123 = !{!"workGroupWalkOrder", !124, !125, !126}
!124 = !{!"dim0", i32 0}
!125 = !{!"dim1", i32 1}
!126 = !{!"dim2", i32 2}
!127 = !{!"funcArgs"}
!128 = !{!"functionType", !"KernelFunction"}
!129 = !{!"rtInfo", !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !147, !148}
!130 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!131 = !{!"isContinuation", i1 false}
!132 = !{!"hasTraceRayPayload", i1 false}
!133 = !{!"hasHitAttributes", i1 false}
!134 = !{!"hasCallableData", i1 false}
!135 = !{!"ShaderStackSize", i32 0}
!136 = !{!"ShaderHash", i64 0}
!137 = !{!"ShaderName", !""}
!138 = !{!"ParentName", !""}
!139 = !{!"SlotNum", i1* null}
!140 = !{!"NOSSize", i32 0}
!141 = !{!"globalRootSignatureSize", i32 0}
!142 = !{!"Entries"}
!143 = !{!"SpillUnions"}
!144 = !{!"CustomHitAttrSizeInBytes", i32 0}
!145 = !{!"Types", !146}
!146 = !{!"FullFrameTys"}
!147 = !{!"Aliases"}
!148 = !{!"NumCoherenceHintBits", i32 0}
!149 = !{!"resAllocMD", !150, !151, !152, !153, !175}
!150 = !{!"uavsNumType", i32 0}
!151 = !{!"srvsNumType", i32 0}
!152 = !{!"samplersNumType", i32 0}
!153 = !{!"argAllocMDList", !154, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !170, !171, !172, !173, !174}
!154 = !{!"argAllocMDListVec[0]", !155, !156, !157}
!155 = !{!"type", i32 0}
!156 = !{!"extensionType", i32 -1}
!157 = !{!"indexType", i32 -1}
!158 = !{!"argAllocMDListVec[1]", !155, !156, !157}
!159 = !{!"argAllocMDListVec[2]", !155, !156, !157}
!160 = !{!"argAllocMDListVec[3]", !155, !156, !157}
!161 = !{!"argAllocMDListVec[4]", !155, !156, !157}
!162 = !{!"argAllocMDListVec[5]", !155, !156, !157}
!163 = !{!"argAllocMDListVec[6]", !155, !156, !157}
!164 = !{!"argAllocMDListVec[7]", !155, !156, !157}
!165 = !{!"argAllocMDListVec[8]", !155, !156, !157}
!166 = !{!"argAllocMDListVec[9]", !155, !156, !157}
!167 = !{!"argAllocMDListVec[10]", !155, !156, !157}
!168 = !{!"argAllocMDListVec[11]", !155, !156, !157}
!169 = !{!"argAllocMDListVec[12]", !155, !156, !157}
!170 = !{!"argAllocMDListVec[13]", !155, !156, !157}
!171 = !{!"argAllocMDListVec[14]", !155, !156, !157}
!172 = !{!"argAllocMDListVec[15]", !155, !156, !157}
!173 = !{!"argAllocMDListVec[16]", !155, !156, !157}
!174 = !{!"argAllocMDListVec[17]", !155, !156, !157}
!175 = !{!"inlineSamplersMD"}
!176 = !{!"maxByteOffsets"}
!177 = !{!"IsInitializer", i1 false}
!178 = !{!"IsFinalizer", i1 false}
!179 = !{!"CompiledSubGroupsNumber", i32 0}
!180 = !{!"hasInlineVmeSamplers", i1 false}
!181 = !{!"localSize", i32 0}
!182 = !{!"localIDPresent", i1 false}
!183 = !{!"groupIDPresent", i1 false}
!184 = !{!"privateMemoryPerWI", i32 0}
!185 = !{!"prevFPOffset", i32 0}
!186 = !{!"globalIDPresent", i1 false}
!187 = !{!"hasSyncRTCalls", i1 false}
!188 = !{!"hasNonKernelArgLoad", i1 false}
!189 = !{!"hasNonKernelArgStore", i1 false}
!190 = !{!"hasNonKernelArgAtomic", i1 false}
!191 = !{!"UserAnnotations"}
!192 = !{!"m_OpenCLArgAddressSpaces", !193, !194, !195, !196, !197, !198}
!193 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!194 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 1}
!195 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 1}
!196 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 1}
!197 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!198 = !{!"m_OpenCLArgAddressSpacesVec[5]", i32 1}
!199 = !{!"m_OpenCLArgAccessQualifiers", !200, !201, !202, !203, !204, !205}
!200 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!201 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!202 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!203 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!204 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!205 = !{!"m_OpenCLArgAccessQualifiersVec[5]", !"none"}
!206 = !{!"m_OpenCLArgTypes", !207, !208, !209, !210, !211, !212}
!207 = !{!"m_OpenCLArgTypesVec[0]", !"float*"}
!208 = !{!"m_OpenCLArgTypesVec[1]", !"float*"}
!209 = !{!"m_OpenCLArgTypesVec[2]", !"float*"}
!210 = !{!"m_OpenCLArgTypesVec[3]", !"float*"}
!211 = !{!"m_OpenCLArgTypesVec[4]", !"char*"}
!212 = !{!"m_OpenCLArgTypesVec[5]", !"char*"}
!213 = !{!"m_OpenCLArgBaseTypes", !214, !215, !216, !217, !218, !219}
!214 = !{!"m_OpenCLArgBaseTypesVec[0]", !"float*"}
!215 = !{!"m_OpenCLArgBaseTypesVec[1]", !"float*"}
!216 = !{!"m_OpenCLArgBaseTypesVec[2]", !"float*"}
!217 = !{!"m_OpenCLArgBaseTypesVec[3]", !"float*"}
!218 = !{!"m_OpenCLArgBaseTypesVec[4]", !"char*"}
!219 = !{!"m_OpenCLArgBaseTypesVec[5]", !"char*"}
!220 = !{!"m_OpenCLArgTypeQualifiers", !221, !222, !223, !224, !225, !226}
!221 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!222 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!223 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!224 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!225 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!226 = !{!"m_OpenCLArgTypeQualifiersVec[5]", !""}
!227 = !{!"m_OpenCLArgNames", !228, !229, !230, !231, !232, !233}
!228 = !{!"m_OpenCLArgNamesVec[0]", !""}
!229 = !{!"m_OpenCLArgNamesVec[1]", !""}
!230 = !{!"m_OpenCLArgNamesVec[2]", !""}
!231 = !{!"m_OpenCLArgNamesVec[3]", !""}
!232 = !{!"m_OpenCLArgNamesVec[4]", !""}
!233 = !{!"m_OpenCLArgNamesVec[5]", !""}
!234 = !{!"m_OpenCLArgScalarAsPointers"}
!235 = !{!"m_OptsToDisablePerFunc", !236, !237, !238}
!236 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!237 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!238 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!239 = !{!"pushInfo", !240, !241, !242, !246, !247, !248, !249, !250, !251, !252, !253, !266, !267, !268, !269}
!240 = !{!"pushableAddresses"}
!241 = !{!"bindlessPushInfo"}
!242 = !{!"dynamicBufferInfo", !243, !244, !245}
!243 = !{!"firstIndex", i32 0}
!244 = !{!"numOffsets", i32 0}
!245 = !{!"forceDisabled", i1 false}
!246 = !{!"MaxNumberOfPushedBuffers", i32 0}
!247 = !{!"inlineConstantBufferSlot", i32 -1}
!248 = !{!"inlineConstantBufferOffset", i32 -1}
!249 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!250 = !{!"constants"}
!251 = !{!"inputs"}
!252 = !{!"constantReg"}
!253 = !{!"simplePushInfoArr", !254, !263, !264, !265}
!254 = !{!"simplePushInfoArrVec[0]", !255, !256, !257, !258, !259, !260, !261, !262}
!255 = !{!"cbIdx", i32 0}
!256 = !{!"pushableAddressGrfOffset", i32 -1}
!257 = !{!"pushableOffsetGrfOffset", i32 -1}
!258 = !{!"offset", i32 0}
!259 = !{!"size", i32 0}
!260 = !{!"isStateless", i1 false}
!261 = !{!"isBindless", i1 false}
!262 = !{!"simplePushLoads"}
!263 = !{!"simplePushInfoArrVec[1]", !255, !256, !257, !258, !259, !260, !261, !262}
!264 = !{!"simplePushInfoArrVec[2]", !255, !256, !257, !258, !259, !260, !261, !262}
!265 = !{!"simplePushInfoArrVec[3]", !255, !256, !257, !258, !259, !260, !261, !262}
!266 = !{!"simplePushBufferUsed", i32 0}
!267 = !{!"pushAnalysisWIInfos"}
!268 = !{!"inlineRTGlobalPtrOffset", i32 0}
!269 = !{!"rtSyncSurfPtrOffset", i32 0}
!270 = !{!"psInfo", !271, !272, !273, !274, !275, !276, !277, !278, !279, !280, !281, !282, !283, !284, !285, !286}
!271 = !{!"BlendStateDisabledMask", i8 0}
!272 = !{!"SkipSrc0Alpha", i1 false}
!273 = !{!"DualSourceBlendingDisabled", i1 false}
!274 = !{!"ForceEnableSimd32", i1 false}
!275 = !{!"DisableSimd32WithDiscard", i1 false}
!276 = !{!"outputDepth", i1 false}
!277 = !{!"outputStencil", i1 false}
!278 = !{!"outputMask", i1 false}
!279 = !{!"blendToFillEnabled", i1 false}
!280 = !{!"forceEarlyZ", i1 false}
!281 = !{!"hasVersionedLoop", i1 false}
!282 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!283 = !{!"NumSamples", i8 0}
!284 = !{!"blendOptimizationMode"}
!285 = !{!"colorOutputMask"}
!286 = !{!"WaDisableVRS", i1 false}
!287 = !{!"csInfo", !288, !289, !290, !291, !292, !43, !44, !293, !294, !295, !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !75, !306, !307, !308, !309}
!288 = !{!"maxWorkGroupSize", i32 0}
!289 = !{!"waveSize", i32 0}
!290 = !{!"ComputeShaderSecondCompile"}
!291 = !{!"forcedSIMDSize", i8 0}
!292 = !{!"forceTotalGRFNum", i32 0}
!293 = !{!"forceSpillCompression", i1 false}
!294 = !{!"allowLowerSimd", i1 false}
!295 = !{!"disableSimd32Slicing", i1 false}
!296 = !{!"disableSplitOnSpill", i1 false}
!297 = !{!"enableNewSpillCostFunction", i1 false}
!298 = !{!"forceVISAPreSched", i1 false}
!299 = !{!"forceUniformBuffer", i1 false}
!300 = !{!"forceUniformSurfaceSampler", i1 false}
!301 = !{!"disableLocalIdOrderOptimizations", i1 false}
!302 = !{!"disableDispatchAlongY", i1 false}
!303 = !{!"neededThreadIdLayout", i1* null}
!304 = !{!"forceTileYWalk", i1 false}
!305 = !{!"atomicBranch", i32 0}
!306 = !{!"disableEarlyOut", i1 false}
!307 = !{!"walkOrderEnabled", i1 false}
!308 = !{!"walkOrderOverride", i32 0}
!309 = !{!"ResForHfPacking"}
!310 = !{!"msInfo", !311, !312, !313, !314, !315, !316, !317, !318, !319}
!311 = !{!"PrimitiveTopology", i32 3}
!312 = !{!"MaxNumOfPrimitives", i32 0}
!313 = !{!"MaxNumOfVertices", i32 0}
!314 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!315 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!316 = !{!"WorkGroupSize", i32 0}
!317 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!318 = !{!"IndexFormat", i32 6}
!319 = !{!"SubgroupSize", i32 0}
!320 = !{!"taskInfo", !321, !316, !317, !319}
!321 = !{!"MaxNumOfOutputs", i32 0}
!322 = !{!"NBarrierCnt", i32 0}
!323 = !{!"rtInfo", !324, !325, !326, !327, !328, !329, !330, !331, !332, !333, !334, !335, !336, !337}
!324 = !{!"RayQueryAllocSizeInBytes", i32 0}
!325 = !{!"NumContinuations", i32 0}
!326 = !{!"RTAsyncStackAddrspace", i32 -1}
!327 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!328 = !{!"SWHotZoneAddrspace", i32 -1}
!329 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!330 = !{!"SWStackAddrspace", i32 -1}
!331 = !{!"SWStackSurfaceStateOffset", i1* null}
!332 = !{!"RTSyncStackAddrspace", i32 -1}
!333 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!334 = !{!"doSyncDispatchRays", i1 false}
!335 = !{!"MemStyle", !"Xe"}
!336 = !{!"GlobalDataStyle", !"Xe"}
!337 = !{!"uberTileDimensions", i1* null}
!338 = !{!"CurUniqueIndirectIdx", i32 0}
!339 = !{!"inlineDynTextures"}
!340 = !{!"inlineResInfoData"}
!341 = !{!"immConstant", !342, !343, !344}
!342 = !{!"data"}
!343 = !{!"sizes"}
!344 = !{!"zeroIdxs"}
!345 = !{!"stringConstants"}
!346 = !{!"inlineBuffers", !347, !351, !352}
!347 = !{!"inlineBuffersVec[0]", !348, !349, !350}
!348 = !{!"alignment", i32 0}
!349 = !{!"allocSize", i64 0}
!350 = !{!"Buffer"}
!351 = !{!"inlineBuffersVec[1]", !348, !349, !350}
!352 = !{!"inlineBuffersVec[2]", !348, !349, !350}
!353 = !{!"GlobalPointerProgramBinaryInfos"}
!354 = !{!"ConstantPointerProgramBinaryInfos"}
!355 = !{!"GlobalBufferAddressRelocInfo"}
!356 = !{!"ConstantBufferAddressRelocInfo"}
!357 = !{!"forceLscCacheList"}
!358 = !{!"SrvMap"}
!359 = !{!"RasterizerOrderedByteAddressBuffer"}
!360 = !{!"RasterizerOrderedViews"}
!361 = !{!"MinNOSPushConstantSize", i32 0}
!362 = !{!"inlineProgramScopeOffsets"}
!363 = !{!"shaderData", !364}
!364 = !{!"numReplicas", i32 0}
!365 = !{!"URBInfo", !366, !367, !368}
!366 = !{!"has64BVertexHeaderInput", i1 false}
!367 = !{!"has64BVertexHeaderOutput", i1 false}
!368 = !{!"hasVertexHeader", i1 true}
!369 = !{!"UseBindlessImage", i1 false}
!370 = !{!"enableRangeReduce", i1 false}
!371 = !{!"allowMatchMadOptimizationforVS", i1 false}
!372 = !{!"disableMatchMadOptimizationForCS", i1 false}
!373 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!374 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!375 = !{!"statefulResourcesNotAliased", i1 false}
!376 = !{!"disableMixMode", i1 false}
!377 = !{!"genericAccessesResolved", i1 false}
!378 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!379 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!380 = !{!"disableSeparateScratchWA", i1 false}
!381 = !{!"enableRemoveUnusedTGMFence", i1 false}
!382 = !{!"PrivateMemoryPerFG", !383, !384}
!383 = !{!"PrivateMemoryPerFGMap[0]", void (float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, float addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <8 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32)* @kernel}
!384 = !{!"PrivateMemoryPerFGValue[0]", i32 0}
!385 = !{!"m_OptsToDisable"}
!386 = !{!"capabilities", !387}
!387 = !{!"globalVariableDecorationsINTEL", i1 false}
!388 = !{!"extensions", !389}
!389 = !{!"spvINTELBindlessImages", i1 false}
!390 = !{!"m_ShaderResourceViewMcsMask", !391, !392}
!391 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!392 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!393 = !{!"computedDepthMode", i32 0}
!394 = !{!"isHDCFastClearShader", i1 false}
!395 = !{!"argRegisterReservations", !396}
!396 = !{!"argRegisterReservationsVec[0]", i32 0}
!397 = !{!"SIMD16_SpillThreshold", i8 0}
!398 = !{!"SIMD32_SpillThreshold", i8 0}
!399 = !{!"m_CacheControlOption", !400, !401, !402, !403}
!400 = !{!"LscLoadCacheControlOverride", i8 0}
!401 = !{!"LscStoreCacheControlOverride", i8 0}
!402 = !{!"TgmLoadCacheControlOverride", i8 0}
!403 = !{!"TgmStoreCacheControlOverride", i8 0}
!404 = !{!"ModuleUsesBindless", i1 false}
!405 = !{!"predicationMap"}
!406 = !{!"lifeTimeStartMap"}
!407 = !{!"HitGroups"}
!408 = !{i32 2, i32 0}
!409 = !{!"clang version 15.0.7"}
!410 = !{!"clang version 9.0.0 (c68f557a081b1b2339a42d7cd6af3c2ab18c6061)"}
!411 = !{i32 1, !"wchar_size", i32 4}
!412 = !{!"5497558138880"}
!413 = !{!"-39"}
!414 = !{!"11529214673724838942"}
!415 = !{!"-61"}
!416 = !{!"11529213929037577306"}
!417 = !{!"-62"}
!418 = !{!"11529212439663054033"}
!419 = !{!"-63"}
!420 = !{!"17293818659494581050"}
!421 = !{!"-65"}
!422 = !{!423}
!423 = !{i32 4469}
!424 = !{!"17293806744498394871"}
!425 = !{!"-66"}
!426 = !{!"-67"}
!427 = !{!423, !428}
!428 = !{i32 4470}
!429 = !{!"17293821638243627595"}
!430 = !{!"16753390177478052755"}
!431 = !{!"-68"}
!432 = !{!"17293711424528905438"}
!433 = !{!"-69"}
!434 = !{!"-70"}
!435 = !{!"12969616328610253045"}
!436 = !{!"-71"}
!437 = !{!"10808521980345821182"}
!438 = !{!"14563199408526971376"}
!439 = !{!"14563175578534599018"}
!440 = !{!"14108118044192044425"}
!441 = !{!428}
!442 = !{!"14335646811363321721"}
!443 = !{!"13887666784755357553"}
!444 = !{!"14561841098961746951"}
!445 = !{!"-72"}
!446 = !{!"14335360851454853421"}
!447 = !{!"13510271195554318744"}
!448 = !{!"13509508635798403278"}
!449 = !{!"16212020410762816306"}
!450 = !{!"13915952985701346885"}
!451 = !{!"12969997608488210779"}
!452 = !{!"16212782970518731773"}
!453 = !{!"12159396587950069963"}
!454 = !{!"12159015308072112230"}
!455 = !{!"12158252748316196764"}
!456 = !{!"18238904241986126078"}
!457 = !{!"18237379122474295145"}
!458 = !{!"-73"}
!459 = !{!"13679559461367552291"}
!460 = !{!"13678796901611636825"}
!461 = !{!"13677271782099805893"}
!462 = !{!"10259478956086685352"}
!463 = !{!"10257953836574854419"}
!464 = !{!"15388455874374112561"}
!465 = !{!"15385405635350450696"}
!466 = !{!"-74"}
!467 = !{!"11542104465536499888"}
!468 = !{!"17313156698304749831"}
!469 = !{!"-75"}
!470 = !{!"12984867523728562374"}
!471 = !{!"12981817284704900508"}
!472 = !{!"9736362963528675381"}
!473 = !{!"14604544445293013071"}
!474 = !{!"14592343489198365609"}
!475 = !{!"-76"}
!476 = !{!"10950358094946097938"}
!477 = !{!"10944257616898774206"}
!478 = !{!"10932056660804126744"}
!479 = !{!"16422486903395485041"}
!480 = !{!"16398084991206190115"}
!481 = !{!"-77"}
!482 = !{!"12322965655593937513"}
!483 = !{!"12298563743404642586"}
!484 = !{!"9236123763648129403"}
!485 = !{!"9223922807553481940"}
!486 = !{!"13860286123519517835"}
!487 = !{!"-78"}
!488 = !{!"10395214592639638377"}
!489 = !{!"10346410768261048525"}
!490 = !{!"15568419976770162639"}
!491 = !{!"15519616152391572788"}
!492 = !{!"15422008503634393085"}
!493 = !{!"-79"}
!494 = !{!"11664114026482974516"}
!495 = !{!"11615310202104384665"}
!496 = !{!"11517702553347204962"}
!497 = !{!"17471769127535166849"}
!498 = !{!"17374161478777987146"}
!499 = !{!"17178946181263627740"}
!500 = !{!"-80"}
!501 = !{!"13079424933462080211"}
!502 = !{!"12884209635947720805"}
!503 = !{!"9858372524475150010"}
!504 = !{!"9760764875717970307"}
!505 = !{!"14641147313576955460"}
!506 = !{!"14445932016062596054"}
!507 = !{!"14055501421033877241"}
!508 = !{!"-81"}
!509 = !{!"15617223801148752491"}
!510 = !{!"-82"}
!511 = !{!"12493779040919001993"}
!512 = !{!"10803946621810328383"}
!513 = !{!"9370334280689251494"}
!514 = !{!"13510652475432276478"}
!515 = !{!"16449367134791505233"}
!516 = !{!"12970283568396679078"}
!517 = !{float 2.500000e+00}
!518 = !{!"10808617300315310615"}
!519 = !{!"17293330144650947704"}
!520 = !{!"10806996860833990249"}
!521 = !{!"13887642954762985194"}
!522 = !{!"14334598291698937955"}
!523 = !{!"18204017133152993489"}
!524 = !{!"17635147555240055531"}
!525 = !{!"17919582344196524510"}
!526 = !{!"17919391704257545644"}
!527 = !{!"17359577523446103850"}
!528 = !{!"18203826493214014623"}
!529 = !{!"17066087337388138708"}
!530 = !{!"10132894036604717924"}
!531 = !{!"16027099669953315698"}
!532 = !{!"14411515549578817541"}
!533 = !{!"-64"}
