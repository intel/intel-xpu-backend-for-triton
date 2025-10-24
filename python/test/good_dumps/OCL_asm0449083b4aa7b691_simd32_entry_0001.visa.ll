; ------------------------------------------------
; OCL_asm0449083b4aa7b691_simd32_entry_0001.visa.ll
; ------------------------------------------------
; Function Attrs: argmemonly nofree norecurse nosync nounwind null_pointer_is_valid
define spir_kernel void @kernel(float addrspace(1)* nocapture readonly align 4 %0, float addrspace(1)* nocapture readonly align 4 %1, float addrspace(1)* nocapture writeonly align 4 %2, float addrspace(1)* nocapture writeonly align 4 %3, i8 addrspace(1)* nocapture readnone align 1 %4, i8 addrspace(1)* nocapture readnone align 1 %5, <8 x i32> %r0, <8 x i32> %payloadHeader, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* nocapture readnone %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bufferOffset5) #0 {
; BB0 :
  %7 = and i16 %localIdX, 127		; visa id: 2
  %8 = ptrtoint float addrspace(1)* %0 to i64		; visa id: 3
  %9 = zext i16 %7 to i64		; visa id: 3
  %10 = shl nuw nsw i64 %9, 2		; visa id: 4
  %11 = add i64 %10, %8		; visa id: 5
  %12 = inttoptr i64 %11 to float addrspace(1)*		; visa id: 6
  %13 = load float, float addrspace(1)* %12, align 4		; visa id: 6
  %14 = ptrtoint float addrspace(1)* %1 to i64		; visa id: 7
  %15 = add i64 %10, %14		; visa id: 7
  %16 = inttoptr i64 %15 to float addrspace(1)*		; visa id: 8
  %17 = load float, float addrspace(1)* %16, align 4		; visa id: 8
  %18 = bitcast float %13 to i32
  %19 = lshr i32 %18, 23		; visa id: 9
  %20 = and i32 %19, 255		; visa id: 10
  %21 = and i32 %18, 8388607		; visa id: 11
  %22 = icmp eq i32 %20, 255
  %23 = icmp ne i32 %21, 0		; visa id: 12
  %24 = and i1 %22, %23		; visa id: 13
  br i1 %24, label %.__imf_fdiv_rn.exit_crit_edge, label %25, !stats.blockFrequency.digits !411, !stats.blockFrequency.scale !412		; visa id: 15

.__imf_fdiv_rn.exit_crit_edge:                    ; preds = %6
; BB1 :
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !413, !stats.blockFrequency.scale !414		; visa id: 18

25:                                               ; preds = %6
; BB2 :
  %26 = bitcast float %17 to i32
  %27 = lshr i32 %26, 23		; visa id: 20
  %28 = and i32 %27, 255		; visa id: 21
  %29 = and i32 %26, 8388607		; visa id: 22
  %30 = icmp eq i32 %28, 255
  %31 = icmp ne i32 %29, 0		; visa id: 23
  %32 = and i1 %30, %31		; visa id: 24
  %33 = fcmp oeq float %17, 0.000000e+00
  %34 = or i1 %32, %33		; visa id: 26
  br i1 %34, label %.__imf_fdiv_rn.exit_crit_edge168, label %35, !stats.blockFrequency.digits !413, !stats.blockFrequency.scale !414		; visa id: 28

.__imf_fdiv_rn.exit_crit_edge168:                 ; preds = %25
; BB3 :
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !415, !stats.blockFrequency.scale !416		; visa id: 31

35:                                               ; preds = %25
; BB4 :
  %36 = xor i32 %18, %26		; visa id: 33
  %37 = icmp eq i32 %21, 0		; visa id: 34
  %38 = icmp eq i32 %20, 255
  %39 = and i1 %38, %37		; visa id: 35
  br i1 %39, label %646, label %40, !stats.blockFrequency.digits !415, !stats.blockFrequency.scale !416		; visa id: 37

40:                                               ; preds = %35
; BB5 :
  %41 = fcmp oeq float %13, 0.000000e+00		; visa id: 39
  br i1 %41, label %643, label %42, !stats.blockFrequency.digits !417, !stats.blockFrequency.scale !418		; visa id: 40

42:                                               ; preds = %40
; BB6 :
  %43 = icmp eq i32 %29, 0		; visa id: 42
  %44 = icmp eq i32 %28, 255
  %45 = and i1 %44, %43		; visa id: 43
  br i1 %45, label %640, label %46, !stats.blockFrequency.digits !419, !stats.blockFrequency.scale !420		; visa id: 45

46:                                               ; preds = %42
; BB7 :
  %47 = add nsw i32 %20, -127, !spirv.Decorations !421		; visa id: 47
  %48 = icmp eq i32 %20, 0		; visa id: 48
  %49 = select i1 %48, i32 -126, i32 %47		; visa id: 49
  %50 = add nsw i32 %28, -127, !spirv.Decorations !421		; visa id: 50
  %51 = icmp eq i32 %28, 0		; visa id: 51
  %52 = select i1 %51, i32 -126, i32 %50		; visa id: 52
  %53 = sub nsw i32 %49, %52, !spirv.Decorations !421		; visa id: 53
  %54 = or i32 %21, 8388608		; visa id: 54
  %55 = select i1 %48, i32 %21, i32 %54		; visa id: 55
  %56 = or i32 %29, 8388608		; visa id: 56
  %57 = select i1 %51, i32 %29, i32 %56		; visa id: 57
  %58 = icmp ult i32 %55, %57		; visa id: 58
  br i1 %58, label %.preheader.i.i.i.preheader, label %438, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !424		; visa id: 59

.preheader.i.i.i.preheader:                       ; preds = %46
; BB8 :
  br label %.preheader.i.i.i, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !425		; visa id: 63

.preheader.i.i.i:                                 ; preds = %.preheader.i.i.i..preheader.i.i.i_crit_edge, %.preheader.i.i.i.preheader
; BB9 :
  %59 = phi i32 [ %62, %.preheader.i.i.i..preheader.i.i.i_crit_edge ], [ 0, %.preheader.i.i.i.preheader ]
  %60 = phi i32 [ %61, %.preheader.i.i.i..preheader.i.i.i_crit_edge ], [ %55, %.preheader.i.i.i.preheader ]
  %61 = shl nuw nsw i32 %60, 1, !spirv.Decorations !426		; visa id: 64
  %62 = add i32 %59, 1		; visa id: 65
  %63 = icmp ult i32 %61, %57		; visa id: 66
  br i1 %63, label %.preheader.i.i.i..preheader.i.i.i_crit_edge, label %64, !stats.blockFrequency.digits !428, !stats.blockFrequency.scale !416		; visa id: 69

.preheader.i.i.i..preheader.i.i.i_crit_edge:      ; preds = %.preheader.i.i.i
; BB10 :
  br label %.preheader.i.i.i, !stats.blockFrequency.digits !429, !stats.blockFrequency.scale !416		; visa id: 73

64:                                               ; preds = %.preheader.i.i.i
; BB11 :
  %.lcssa208 = phi i32 [ %59, %.preheader.i.i.i ]
  %.lcssa207 = phi i32 [ %62, %.preheader.i.i.i ]
  %65 = xor i32 %.lcssa208, -1		; visa id: 75
  %66 = add i32 %53, %65		; visa id: 76
  %67 = icmp sgt i32 %66, 127		; visa id: 77
  br i1 %67, label %436, label %68, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !425		; visa id: 78

68:                                               ; preds = %64
; BB12 :
  %69 = icmp sgt i32 %66, -127		; visa id: 80
  br i1 %69, label %138, label %70, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !430		; visa id: 81

70:                                               ; preds = %68
; BB13 :
  %71 = add i32 %.lcssa208, -127		; visa id: 83
  %72 = sub i32 %71, %53		; visa id: 84
  %73 = add i32 %72, 1		; visa id: 85
  %74 = icmp ugt i32 %73, 22		; visa id: 86
  br i1 %74, label %130, label %75, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !432		; visa id: 87

75:                                               ; preds = %70
; BB14 :
  %76 = shl i32 %55, %.lcssa208		; visa id: 89
  %77 = icmp eq i32 %76, 0		; visa id: 90
  br i1 %77, label %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, label %.lr.ph.preheader, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 91

._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge: ; preds = %75
; BB15 :
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !435		; visa id: 96

.lr.ph.preheader:                                 ; preds = %75
; BB16 :
  %78 = sub nsw i32 25, %72, !spirv.Decorations !421		; visa id: 98
  br label %.lr.ph, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !433		; visa id: 101

.lr.ph:                                           ; preds = %.preheader.i1.i.i.i..lr.ph_crit_edge, %.lr.ph.preheader
; BB17 :
  %79 = phi i32 [ %92, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ 0, %.lr.ph.preheader ]
  %80 = phi i32 [ %91, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ 0, %.lr.ph.preheader ]
  %81 = phi i32 [ %90, %.preheader.i1.i.i.i..lr.ph_crit_edge ], [ %76, %.lr.ph.preheader ]
  %82 = shl i32 %80, 1		; visa id: 102
  %83 = shl i32 %81, 1		; visa id: 103
  %84 = icmp ugt i32 %83, %57		; visa id: 104
  br i1 %84, label %87, label %85, !stats.blockFrequency.digits !437, !stats.blockFrequency.scale !424		; visa id: 105

85:                                               ; preds = %.lr.ph
; BB18 :
  %86 = icmp eq i32 %83, %57		; visa id: 107
  br i1 %86, label %94, label %..preheader.i1.i.i.i_crit_edge, !stats.blockFrequency.digits !438, !stats.blockFrequency.scale !425		; visa id: 109

..preheader.i1.i.i.i_crit_edge:                   ; preds = %85
; BB:
  br label %.preheader.i1.i.i.i, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !425

87:                                               ; preds = %.lr.ph
; BB20 :
  %88 = sub nuw i32 %83, %57, !spirv.Decorations !440		; visa id: 112
  %89 = add i32 %82, 1		; visa id: 113
  br label %.preheader.i1.i.i.i, !stats.blockFrequency.digits !438, !stats.blockFrequency.scale !425		; visa id: 114

.preheader.i1.i.i.i:                              ; preds = %..preheader.i1.i.i.i_crit_edge, %87
; BB21 :
  %90 = phi i32 [ %88, %87 ], [ %83, %..preheader.i1.i.i.i_crit_edge ]
  %91 = phi i32 [ %89, %87 ], [ %82, %..preheader.i1.i.i.i_crit_edge ]
  %92 = add nuw i32 %79, 1, !spirv.Decorations !440		; visa id: 115
  %93 = icmp ult i32 %92, %78		; visa id: 116
  br i1 %93, label %.preheader.i1.i.i.i..lr.ph_crit_edge, label %.preheader.i1.i.i.i._crit_edge, !stats.blockFrequency.digits !441, !stats.blockFrequency.scale !424		; visa id: 117

.preheader.i1.i.i.i..lr.ph_crit_edge:             ; preds = %.preheader.i1.i.i.i
; BB:
  br label %.lr.ph, !stats.blockFrequency.digits !442, !stats.blockFrequency.scale !424

94:                                               ; preds = %85
; BB23 :
  %.lcssa204 = phi i32 [ %79, %85 ]
  %.lcssa202 = phi i32 [ %82, %85 ]
  %95 = add i32 %.lcssa202, 1		; visa id: 120
  %96 = xor i32 %.lcssa204, -1		; visa id: 121
  %97 = add i32 %78, %96		; visa id: 122
  %98 = shl i32 %95, %97		; visa id: 123
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i, !stats.blockFrequency.digits !443, !stats.blockFrequency.scale !444		; visa id: 124

.preheader.i1.i.i.i._crit_edge:                   ; preds = %.preheader.i1.i.i.i
; BB24 :
  %.lcssa206 = phi i32 [ %91, %.preheader.i1.i.i.i ]
  %99 = or i32 %.lcssa206, 1		; visa id: 126
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i, !stats.blockFrequency.digits !445, !stats.blockFrequency.scale !435		; visa id: 127

_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i:        ; preds = %.preheader.i1.i.i.i._crit_edge, %94
; BB25 :
  %100 = phi i32 [ %98, %94 ], [ %99, %.preheader.i1.i.i.i._crit_edge ]
  %101 = lshr i32 %100, 3		; visa id: 128
  %102 = and i32 %100, 7		; visa id: 129
  %103 = icmp eq i32 %102, 0		; visa id: 130
  br i1 %103, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !433		; visa id: 131

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
; BB26 :
  %104 = and i32 %100, 15		; visa id: 133
  %105 = icmp ult i32 %102, 5
  %.not9 = icmp ne i32 %104, 12		; visa id: 134
  %not.or.cond57 = and i1 %105, %.not9		; visa id: 135
  %106 = icmp ugt i32 %100, 67108855		; visa id: 137
  %107 = select i1 %106, i32 8388608, i32 0		; visa id: 138
  br i1 %not.or.cond57, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !435		; visa id: 139

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
; BB27 :
  %108 = add nuw nsw i32 %101, 1, !spirv.Decorations !426		; visa id: 141
  %109 = select i1 %106, i32 0, i32 %108		; visa id: 142
  %110 = sext i1 %not.or.cond57 to i32		; visa id: 143
  br label %120, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !444		; visa id: 144

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i
; BB28 :
  %111 = sext i1 %not.or.cond57 to i32		; visa id: 146
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !444		; visa id: 147

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit2.i.i.i
; BB29 :
  %112 = icmp ugt i32 %100, 67108855
  %113 = sext i1 %112 to i32		; visa id: 149
  %114 = and i32 8388608, %113		; visa id: 150
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !444		; visa id: 152

_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread
; BB30 :
  %115 = phi i32 [ %107, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ 0, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ %114, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %116 = phi i32 [ %111, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ -1, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ -1, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %117 = phi i32 [ %101, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ 0, %._ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread_crit_edge ], [ %101, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread ]
  %118 = icmp ne i32 %116, 0
  %119 = sext i1 %118 to i32		; visa id: 153
  br label %120, !stats.blockFrequency.digits !449, !stats.blockFrequency.scale !433		; visa id: 154

120:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread
; BB31 :
  %121 = phi i32 [ %115, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %107, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %122 = phi i32 [ %119, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %110, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %123 = phi i32 [ %117, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i.thread.thread ], [ %109, %_ZL19__handling_roundingIjET_S0_S0_ji.exit.i.i.i._crit_edge ]
  %124 = icmp ne i32 %122, 0		; visa id: 155
  %125 = select i1 %124, i32 0, i32 %121		; visa id: 156
  %126 = and i32 %36, -2147483648		; visa id: 157
  %127 = or i32 %126, %125
  %128 = or i32 %127, %123		; visa id: 158
  %129 = bitcast i32 %128 to float		; visa id: 159
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 159

130:                                              ; preds = %70
; BB32 :
  %131 = shl i32 %55, %.lcssa207		; visa id: 161
  %132 = icmp eq i32 %73, 23
  %133 = icmp ugt i32 %131, %57		; visa id: 162
  %134 = and i1 %132, %133		; visa id: 163
  %135 = icmp sgt i32 %36, -1		; visa id: 165
  br i1 %135, label %137, label %136, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 166

136:                                              ; preds = %130
; BB33 :
  %spec.select56 = select i1 %134, float 0xB6A0000000000000, float -0.000000e+00		; visa id: 168
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !435		; visa id: 169

137:                                              ; preds = %130
; BB34 :
  %spec.select55 = select i1 %134, float 0x36A0000000000000, float 0.000000e+00		; visa id: 171
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !433		; visa id: 172

138:                                              ; preds = %68
; BB35 :
  %139 = add nsw i32 %66, 127, !spirv.Decorations !421		; visa id: 174
  %140 = shl i32 %55, %.lcssa208		; visa id: 175
  %141 = icmp eq i32 %140, 0		; visa id: 176
  br i1 %141, label %..critedge53_crit_edge, label %.preheader.i4.i.i.i.preheader, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !432		; visa id: 177

..critedge53_crit_edge:                           ; preds = %138
; BB36 :
  br label %.critedge53, !stats.blockFrequency.digits !450, !stats.blockFrequency.scale !433		; visa id: 180

.preheader.i4.i.i.i.preheader:                    ; preds = %138
; BB37 :
  %142 = shl i32 %140, 1		; visa id: 182
  %143 = icmp ugt i32 %142, %57		; visa id: 183
  br i1 %143, label %146, label %144, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !432		; visa id: 184

144:                                              ; preds = %.preheader.i4.i.i.i.preheader
; BB38 :
  %145 = icmp eq i32 %142, %57		; visa id: 186
  br i1 %145, label %._crit_edge173, label %..preheader.i4.i.i.i_crit_edge, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !433		; visa id: 187

..preheader.i4.i.i.i_crit_edge:                   ; preds = %144
; BB39 :
  br label %.preheader.i4.i.i.i, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !435		; visa id: 190

._crit_edge173:                                   ; preds = %144
; BB40 :
  br label %405, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !435		; visa id: 194

146:                                              ; preds = %.preheader.i4.i.i.i.preheader
; BB41 :
  %147 = sub nuw i32 %142, %57, !spirv.Decorations !440		; visa id: 196
  br label %.preheader.i4.i.i.i, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !433		; visa id: 198

.preheader.i4.i.i.i:                              ; preds = %..preheader.i4.i.i.i_crit_edge, %146
; BB42 :
  %148 = phi i32 [ %147, %146 ], [ %142, %..preheader.i4.i.i.i_crit_edge ]
  %149 = phi i16 [ 2, %146 ], [ 0, %..preheader.i4.i.i.i_crit_edge ]
  %150 = shl i32 %148, 1		; visa id: 199
  %151 = icmp ugt i32 %150, %57		; visa id: 200
  br i1 %151, label %155, label %152, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !433		; visa id: 201

152:                                              ; preds = %.preheader.i4.i.i.i
; BB43 :
  %153 = icmp eq i32 %150, %57		; visa id: 203
  br i1 %153, label %._crit_edge174, label %..preheader.i4.i.i.i.1_crit_edge, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !435		; visa id: 204

..preheader.i4.i.i.i.1_crit_edge:                 ; preds = %152
; BB:
  br label %.preheader.i4.i.i.i.1, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !444

._crit_edge174:                                   ; preds = %152
; BB45 :
  %154 = trunc i16 %149 to i8		; visa id: 206
  %.demoted.zext = zext i8 %154 to i32		; visa id: 207
  br label %405, !stats.blockFrequency.digits !448, !stats.blockFrequency.scale !444		; visa id: 209

155:                                              ; preds = %.preheader.i4.i.i.i
; BB46 :
  %156 = sub nuw i32 %150, %57, !spirv.Decorations !440		; visa id: 211
  %b2s = or i16 %149, 1		; visa id: 212
  br label %.preheader.i4.i.i.i.1, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !435		; visa id: 213

.preheader.i4.i.i.i.1:                            ; preds = %..preheader.i4.i.i.i.1_crit_edge, %155
; BB47 :
  %157 = phi i32 [ %156, %155 ], [ %150, %..preheader.i4.i.i.i.1_crit_edge ]
  %158 = phi i16 [ %b2s, %155 ], [ %149, %..preheader.i4.i.i.i.1_crit_edge ]
  %159 = trunc i16 %158 to i8		; visa id: 214
  %.demoted.zext166 = zext i8 %159 to i32		; visa id: 215
  %160 = shl nsw i32 %.demoted.zext166, 1		; visa id: 216
  %161 = shl i32 %157, 1		; visa id: 217
  %162 = icmp ugt i32 %161, %57		; visa id: 218
  br i1 %162, label %165, label %163, !stats.blockFrequency.digits !452, !stats.blockFrequency.scale !433		; visa id: 219

163:                                              ; preds = %.preheader.i4.i.i.i.1
; BB48 :
  %164 = icmp eq i32 %161, %57		; visa id: 221
  br i1 %164, label %._crit_edge175, label %..preheader.i4.i.i.i.2_crit_edge, !stats.blockFrequency.digits !453, !stats.blockFrequency.scale !435		; visa id: 222

..preheader.i4.i.i.i.2_crit_edge:                 ; preds = %163
; BB:
  br label %.preheader.i4.i.i.i.2, !stats.blockFrequency.digits !454, !stats.blockFrequency.scale !444

._crit_edge175:                                   ; preds = %163
; BB50 :
  br label %405, !stats.blockFrequency.digits !454, !stats.blockFrequency.scale !444		; visa id: 225

165:                                              ; preds = %.preheader.i4.i.i.i.1
; BB51 :
  %166 = sub nuw i32 %161, %57, !spirv.Decorations !440		; visa id: 227
  %167 = add i32 %160, 1		; visa id: 228
  br label %.preheader.i4.i.i.i.2, !stats.blockFrequency.digits !453, !stats.blockFrequency.scale !435		; visa id: 229

.preheader.i4.i.i.i.2:                            ; preds = %..preheader.i4.i.i.i.2_crit_edge, %165
; BB52 :
  %168 = phi i32 [ %166, %165 ], [ %161, %..preheader.i4.i.i.i.2_crit_edge ]
  %169 = phi i32 [ %167, %165 ], [ %160, %..preheader.i4.i.i.i.2_crit_edge ]
  %170 = shl i32 %169, 1		; visa id: 230
  %171 = shl i32 %168, 1		; visa id: 231
  %172 = icmp ugt i32 %171, %57		; visa id: 232
  br i1 %172, label %175, label %173, !stats.blockFrequency.digits !455, !stats.blockFrequency.scale !435		; visa id: 233

173:                                              ; preds = %.preheader.i4.i.i.i.2
; BB53 :
  %174 = icmp eq i32 %171, %57		; visa id: 235
  br i1 %174, label %._crit_edge176, label %..preheader.i4.i.i.i.3_crit_edge, !stats.blockFrequency.digits !455, !stats.blockFrequency.scale !444		; visa id: 236

..preheader.i4.i.i.i.3_crit_edge:                 ; preds = %173
; BB:
  br label %.preheader.i4.i.i.i.3, !stats.blockFrequency.digits !456, !stats.blockFrequency.scale !457

._crit_edge176:                                   ; preds = %173
; BB55 :
  br label %405, !stats.blockFrequency.digits !456, !stats.blockFrequency.scale !457		; visa id: 239

175:                                              ; preds = %.preheader.i4.i.i.i.2
; BB56 :
  %176 = sub nuw i32 %171, %57, !spirv.Decorations !440		; visa id: 241
  %177 = add i32 %170, 1		; visa id: 242
  br label %.preheader.i4.i.i.i.3, !stats.blockFrequency.digits !455, !stats.blockFrequency.scale !444		; visa id: 243

.preheader.i4.i.i.i.3:                            ; preds = %..preheader.i4.i.i.i.3_crit_edge, %175
; BB57 :
  %178 = phi i32 [ %176, %175 ], [ %171, %..preheader.i4.i.i.i.3_crit_edge ]
  %179 = phi i32 [ %177, %175 ], [ %170, %..preheader.i4.i.i.i.3_crit_edge ]
  %180 = shl i32 %179, 1		; visa id: 244
  %181 = shl i32 %178, 1		; visa id: 245
  %182 = icmp ugt i32 %181, %57		; visa id: 246
  br i1 %182, label %185, label %183, !stats.blockFrequency.digits !458, !stats.blockFrequency.scale !435		; visa id: 247

183:                                              ; preds = %.preheader.i4.i.i.i.3
; BB58 :
  %184 = icmp eq i32 %181, %57		; visa id: 249
  br i1 %184, label %._crit_edge177, label %..preheader.i4.i.i.i.4_crit_edge, !stats.blockFrequency.digits !459, !stats.blockFrequency.scale !444		; visa id: 250

..preheader.i4.i.i.i.4_crit_edge:                 ; preds = %183
; BB:
  br label %.preheader.i4.i.i.i.4, !stats.blockFrequency.digits !460, !stats.blockFrequency.scale !457

._crit_edge177:                                   ; preds = %183
; BB60 :
  br label %405, !stats.blockFrequency.digits !460, !stats.blockFrequency.scale !457		; visa id: 253

185:                                              ; preds = %.preheader.i4.i.i.i.3
; BB61 :
  %186 = sub nuw i32 %181, %57, !spirv.Decorations !440		; visa id: 255
  %187 = add i32 %180, 1		; visa id: 256
  br label %.preheader.i4.i.i.i.4, !stats.blockFrequency.digits !459, !stats.blockFrequency.scale !444		; visa id: 257

.preheader.i4.i.i.i.4:                            ; preds = %..preheader.i4.i.i.i.4_crit_edge, %185
; BB62 :
  %188 = phi i32 [ %186, %185 ], [ %181, %..preheader.i4.i.i.i.4_crit_edge ]
  %189 = phi i32 [ %187, %185 ], [ %180, %..preheader.i4.i.i.i.4_crit_edge ]
  %190 = shl i32 %189, 1		; visa id: 258
  %191 = shl i32 %188, 1		; visa id: 259
  %192 = icmp ugt i32 %191, %57		; visa id: 260
  br i1 %192, label %195, label %193, !stats.blockFrequency.digits !461, !stats.blockFrequency.scale !435		; visa id: 261

193:                                              ; preds = %.preheader.i4.i.i.i.4
; BB63 :
  %194 = icmp eq i32 %191, %57		; visa id: 263
  br i1 %194, label %._crit_edge178, label %..preheader.i4.i.i.i.5_crit_edge, !stats.blockFrequency.digits !461, !stats.blockFrequency.scale !444		; visa id: 264

..preheader.i4.i.i.i.5_crit_edge:                 ; preds = %193
; BB:
  br label %.preheader.i4.i.i.i.5, !stats.blockFrequency.digits !462, !stats.blockFrequency.scale !457

._crit_edge178:                                   ; preds = %193
; BB65 :
  br label %405, !stats.blockFrequency.digits !462, !stats.blockFrequency.scale !457		; visa id: 267

195:                                              ; preds = %.preheader.i4.i.i.i.4
; BB66 :
  %196 = sub nuw i32 %191, %57, !spirv.Decorations !440		; visa id: 269
  %197 = add i32 %190, 1		; visa id: 270
  br label %.preheader.i4.i.i.i.5, !stats.blockFrequency.digits !461, !stats.blockFrequency.scale !444		; visa id: 271

.preheader.i4.i.i.i.5:                            ; preds = %..preheader.i4.i.i.i.5_crit_edge, %195
; BB67 :
  %198 = phi i32 [ %196, %195 ], [ %191, %..preheader.i4.i.i.i.5_crit_edge ]
  %199 = phi i32 [ %197, %195 ], [ %190, %..preheader.i4.i.i.i.5_crit_edge ]
  %200 = shl i32 %199, 1		; visa id: 272
  %201 = shl i32 %198, 1		; visa id: 273
  %202 = icmp ugt i32 %201, %57		; visa id: 274
  br i1 %202, label %205, label %203, !stats.blockFrequency.digits !463, !stats.blockFrequency.scale !444		; visa id: 275

203:                                              ; preds = %.preheader.i4.i.i.i.5
; BB68 :
  %204 = icmp eq i32 %201, %57		; visa id: 277
  br i1 %204, label %._crit_edge179, label %..preheader.i4.i.i.i.6_crit_edge, !stats.blockFrequency.digits !463, !stats.blockFrequency.scale !457		; visa id: 278

..preheader.i4.i.i.i.6_crit_edge:                 ; preds = %203
; BB:
  br label %.preheader.i4.i.i.i.6, !stats.blockFrequency.digits !464, !stats.blockFrequency.scale !465

._crit_edge179:                                   ; preds = %203
; BB70 :
  br label %405, !stats.blockFrequency.digits !464, !stats.blockFrequency.scale !465		; visa id: 281

205:                                              ; preds = %.preheader.i4.i.i.i.5
; BB71 :
  %206 = sub nuw i32 %201, %57, !spirv.Decorations !440		; visa id: 283
  %207 = add i32 %200, 1		; visa id: 284
  br label %.preheader.i4.i.i.i.6, !stats.blockFrequency.digits !463, !stats.blockFrequency.scale !457		; visa id: 285

.preheader.i4.i.i.i.6:                            ; preds = %..preheader.i4.i.i.i.6_crit_edge, %205
; BB72 :
  %208 = phi i32 [ %206, %205 ], [ %201, %..preheader.i4.i.i.i.6_crit_edge ]
  %209 = phi i32 [ %207, %205 ], [ %200, %..preheader.i4.i.i.i.6_crit_edge ]
  %210 = shl i32 %209, 1		; visa id: 286
  %211 = shl i32 %208, 1		; visa id: 287
  %212 = icmp ugt i32 %211, %57		; visa id: 288
  br i1 %212, label %215, label %213, !stats.blockFrequency.digits !466, !stats.blockFrequency.scale !444		; visa id: 289

213:                                              ; preds = %.preheader.i4.i.i.i.6
; BB73 :
  %214 = icmp eq i32 %211, %57		; visa id: 291
  br i1 %214, label %._crit_edge180, label %..preheader.i4.i.i.i.7_crit_edge, !stats.blockFrequency.digits !466, !stats.blockFrequency.scale !457		; visa id: 292

..preheader.i4.i.i.i.7_crit_edge:                 ; preds = %213
; BB:
  br label %.preheader.i4.i.i.i.7, !stats.blockFrequency.digits !466, !stats.blockFrequency.scale !465

._crit_edge180:                                   ; preds = %213
; BB75 :
  br label %405, !stats.blockFrequency.digits !466, !stats.blockFrequency.scale !465		; visa id: 295

215:                                              ; preds = %.preheader.i4.i.i.i.6
; BB76 :
  %216 = sub nuw i32 %211, %57, !spirv.Decorations !440		; visa id: 297
  %217 = add i32 %210, 1		; visa id: 298
  br label %.preheader.i4.i.i.i.7, !stats.blockFrequency.digits !466, !stats.blockFrequency.scale !457		; visa id: 299

.preheader.i4.i.i.i.7:                            ; preds = %..preheader.i4.i.i.i.7_crit_edge, %215
; BB77 :
  %218 = phi i32 [ %216, %215 ], [ %211, %..preheader.i4.i.i.i.7_crit_edge ]
  %219 = phi i32 [ %217, %215 ], [ %210, %..preheader.i4.i.i.i.7_crit_edge ]
  %220 = shl i32 %219, 1		; visa id: 300
  %221 = shl i32 %218, 1		; visa id: 301
  %222 = icmp ugt i32 %221, %57		; visa id: 302
  br i1 %222, label %225, label %223, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !457		; visa id: 303

223:                                              ; preds = %.preheader.i4.i.i.i.7
; BB78 :
  %224 = icmp eq i32 %221, %57		; visa id: 305
  br i1 %224, label %._crit_edge181, label %..preheader.i4.i.i.i.8_crit_edge, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !465		; visa id: 306

..preheader.i4.i.i.i.8_crit_edge:                 ; preds = %223
; BB:
  br label %.preheader.i4.i.i.i.8, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !468

._crit_edge181:                                   ; preds = %223
; BB80 :
  br label %405, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !468		; visa id: 309

225:                                              ; preds = %.preheader.i4.i.i.i.7
; BB81 :
  %226 = sub nuw i32 %221, %57, !spirv.Decorations !440		; visa id: 311
  %227 = add i32 %220, 1		; visa id: 312
  br label %.preheader.i4.i.i.i.8, !stats.blockFrequency.digits !467, !stats.blockFrequency.scale !465		; visa id: 313

.preheader.i4.i.i.i.8:                            ; preds = %..preheader.i4.i.i.i.8_crit_edge, %225
; BB82 :
  %228 = phi i32 [ %226, %225 ], [ %221, %..preheader.i4.i.i.i.8_crit_edge ]
  %229 = phi i32 [ %227, %225 ], [ %220, %..preheader.i4.i.i.i.8_crit_edge ]
  %230 = shl i32 %229, 1		; visa id: 314
  %231 = shl i32 %228, 1		; visa id: 315
  %232 = icmp ugt i32 %231, %57		; visa id: 316
  br i1 %232, label %235, label %233, !stats.blockFrequency.digits !469, !stats.blockFrequency.scale !457		; visa id: 317

233:                                              ; preds = %.preheader.i4.i.i.i.8
; BB83 :
  %234 = icmp eq i32 %231, %57		; visa id: 319
  br i1 %234, label %._crit_edge182, label %..preheader.i4.i.i.i.9_crit_edge, !stats.blockFrequency.digits !470, !stats.blockFrequency.scale !465		; visa id: 320

..preheader.i4.i.i.i.9_crit_edge:                 ; preds = %233
; BB:
  br label %.preheader.i4.i.i.i.9, !stats.blockFrequency.digits !470, !stats.blockFrequency.scale !468

._crit_edge182:                                   ; preds = %233
; BB85 :
  br label %405, !stats.blockFrequency.digits !470, !stats.blockFrequency.scale !468		; visa id: 323

235:                                              ; preds = %.preheader.i4.i.i.i.8
; BB86 :
  %236 = sub nuw i32 %231, %57, !spirv.Decorations !440		; visa id: 325
  %237 = add i32 %230, 1		; visa id: 326
  br label %.preheader.i4.i.i.i.9, !stats.blockFrequency.digits !470, !stats.blockFrequency.scale !465		; visa id: 327

.preheader.i4.i.i.i.9:                            ; preds = %..preheader.i4.i.i.i.9_crit_edge, %235
; BB87 :
  %238 = phi i32 [ %236, %235 ], [ %231, %..preheader.i4.i.i.i.9_crit_edge ]
  %239 = phi i32 [ %237, %235 ], [ %230, %..preheader.i4.i.i.i.9_crit_edge ]
  %240 = shl i32 %239, 1		; visa id: 328
  %241 = shl i32 %238, 1		; visa id: 329
  %242 = icmp ugt i32 %241, %57		; visa id: 330
  br i1 %242, label %245, label %243, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !457		; visa id: 331

243:                                              ; preds = %.preheader.i4.i.i.i.9
; BB88 :
  %244 = icmp eq i32 %241, %57		; visa id: 333
  br i1 %244, label %._crit_edge183, label %..preheader.i4.i.i.i.10_crit_edge, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !465		; visa id: 334

..preheader.i4.i.i.i.10_crit_edge:                ; preds = %243
; BB:
  br label %.preheader.i4.i.i.i.10, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !468

._crit_edge183:                                   ; preds = %243
; BB90 :
  br label %405, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !468		; visa id: 337

245:                                              ; preds = %.preheader.i4.i.i.i.9
; BB91 :
  %246 = sub nuw i32 %241, %57, !spirv.Decorations !440		; visa id: 339
  %247 = add i32 %240, 1		; visa id: 340
  br label %.preheader.i4.i.i.i.10, !stats.blockFrequency.digits !471, !stats.blockFrequency.scale !465		; visa id: 341

.preheader.i4.i.i.i.10:                           ; preds = %..preheader.i4.i.i.i.10_crit_edge, %245
; BB92 :
  %248 = phi i32 [ %246, %245 ], [ %241, %..preheader.i4.i.i.i.10_crit_edge ]
  %249 = phi i32 [ %247, %245 ], [ %240, %..preheader.i4.i.i.i.10_crit_edge ]
  %250 = shl i32 %249, 1		; visa id: 342
  %251 = shl i32 %248, 1		; visa id: 343
  %252 = icmp ugt i32 %251, %57		; visa id: 344
  br i1 %252, label %255, label %253, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !465		; visa id: 345

253:                                              ; preds = %.preheader.i4.i.i.i.10
; BB93 :
  %254 = icmp eq i32 %251, %57		; visa id: 347
  br i1 %254, label %._crit_edge184, label %..preheader.i4.i.i.i.11_crit_edge, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !468		; visa id: 348

..preheader.i4.i.i.i.11_crit_edge:                ; preds = %253
; BB:
  br label %.preheader.i4.i.i.i.11, !stats.blockFrequency.digits !473, !stats.blockFrequency.scale !474

._crit_edge184:                                   ; preds = %253
; BB95 :
  br label %405, !stats.blockFrequency.digits !473, !stats.blockFrequency.scale !474		; visa id: 351

255:                                              ; preds = %.preheader.i4.i.i.i.10
; BB96 :
  %256 = sub nuw i32 %251, %57, !spirv.Decorations !440		; visa id: 353
  %257 = add i32 %250, 1		; visa id: 354
  br label %.preheader.i4.i.i.i.11, !stats.blockFrequency.digits !472, !stats.blockFrequency.scale !468		; visa id: 355

.preheader.i4.i.i.i.11:                           ; preds = %..preheader.i4.i.i.i.11_crit_edge, %255
; BB97 :
  %258 = phi i32 [ %256, %255 ], [ %251, %..preheader.i4.i.i.i.11_crit_edge ]
  %259 = phi i32 [ %257, %255 ], [ %250, %..preheader.i4.i.i.i.11_crit_edge ]
  %260 = shl i32 %259, 1		; visa id: 356
  %261 = shl i32 %258, 1		; visa id: 357
  %262 = icmp ugt i32 %261, %57		; visa id: 358
  br i1 %262, label %265, label %263, !stats.blockFrequency.digits !475, !stats.blockFrequency.scale !465		; visa id: 359

263:                                              ; preds = %.preheader.i4.i.i.i.11
; BB98 :
  %264 = icmp eq i32 %261, %57		; visa id: 361
  br i1 %264, label %._crit_edge185, label %..preheader.i4.i.i.i.12_crit_edge, !stats.blockFrequency.digits !476, !stats.blockFrequency.scale !468		; visa id: 362

..preheader.i4.i.i.i.12_crit_edge:                ; preds = %263
; BB:
  br label %.preheader.i4.i.i.i.12, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !474

._crit_edge185:                                   ; preds = %263
; BB100 :
  br label %405, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !474		; visa id: 365

265:                                              ; preds = %.preheader.i4.i.i.i.11
; BB101 :
  %266 = sub nuw i32 %261, %57, !spirv.Decorations !440		; visa id: 367
  %267 = add i32 %260, 1		; visa id: 368
  br label %.preheader.i4.i.i.i.12, !stats.blockFrequency.digits !476, !stats.blockFrequency.scale !468		; visa id: 369

.preheader.i4.i.i.i.12:                           ; preds = %..preheader.i4.i.i.i.12_crit_edge, %265
; BB102 :
  %268 = phi i32 [ %266, %265 ], [ %261, %..preheader.i4.i.i.i.12_crit_edge ]
  %269 = phi i32 [ %267, %265 ], [ %260, %..preheader.i4.i.i.i.12_crit_edge ]
  %270 = shl i32 %269, 1		; visa id: 370
  %271 = shl i32 %268, 1		; visa id: 371
  %272 = icmp ugt i32 %271, %57		; visa id: 372
  br i1 %272, label %275, label %273, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !468		; visa id: 373

273:                                              ; preds = %.preheader.i4.i.i.i.12
; BB103 :
  %274 = icmp eq i32 %271, %57		; visa id: 375
  br i1 %274, label %._crit_edge186, label %..preheader.i4.i.i.i.13_crit_edge, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !474		; visa id: 376

..preheader.i4.i.i.i.13_crit_edge:                ; preds = %273
; BB:
  br label %.preheader.i4.i.i.i.13, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !480

._crit_edge186:                                   ; preds = %273
; BB105 :
  br label %405, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !480		; visa id: 379

275:                                              ; preds = %.preheader.i4.i.i.i.12
; BB106 :
  %276 = sub nuw i32 %271, %57, !spirv.Decorations !440		; visa id: 381
  %277 = add i32 %270, 1		; visa id: 382
  br label %.preheader.i4.i.i.i.13, !stats.blockFrequency.digits !478, !stats.blockFrequency.scale !474		; visa id: 383

.preheader.i4.i.i.i.13:                           ; preds = %..preheader.i4.i.i.i.13_crit_edge, %275
; BB107 :
  %278 = phi i32 [ %276, %275 ], [ %271, %..preheader.i4.i.i.i.13_crit_edge ]
  %279 = phi i32 [ %277, %275 ], [ %270, %..preheader.i4.i.i.i.13_crit_edge ]
  %280 = shl i32 %279, 1		; visa id: 384
  %281 = shl i32 %278, 1		; visa id: 385
  %282 = icmp ugt i32 %281, %57		; visa id: 386
  br i1 %282, label %285, label %283, !stats.blockFrequency.digits !481, !stats.blockFrequency.scale !468		; visa id: 387

283:                                              ; preds = %.preheader.i4.i.i.i.13
; BB108 :
  %284 = icmp eq i32 %281, %57		; visa id: 389
  br i1 %284, label %._crit_edge187, label %..preheader.i4.i.i.i.14_crit_edge, !stats.blockFrequency.digits !481, !stats.blockFrequency.scale !474		; visa id: 390

..preheader.i4.i.i.i.14_crit_edge:                ; preds = %283
; BB:
  br label %.preheader.i4.i.i.i.14, !stats.blockFrequency.digits !482, !stats.blockFrequency.scale !480

._crit_edge187:                                   ; preds = %283
; BB110 :
  br label %405, !stats.blockFrequency.digits !482, !stats.blockFrequency.scale !480		; visa id: 393

285:                                              ; preds = %.preheader.i4.i.i.i.13
; BB111 :
  %286 = sub nuw i32 %281, %57, !spirv.Decorations !440		; visa id: 395
  %287 = add i32 %280, 1		; visa id: 396
  br label %.preheader.i4.i.i.i.14, !stats.blockFrequency.digits !481, !stats.blockFrequency.scale !474		; visa id: 397

.preheader.i4.i.i.i.14:                           ; preds = %..preheader.i4.i.i.i.14_crit_edge, %285
; BB112 :
  %288 = phi i32 [ %286, %285 ], [ %281, %..preheader.i4.i.i.i.14_crit_edge ]
  %289 = phi i32 [ %287, %285 ], [ %280, %..preheader.i4.i.i.i.14_crit_edge ]
  %290 = shl i32 %289, 1		; visa id: 398
  %291 = shl i32 %288, 1		; visa id: 399
  %292 = icmp ugt i32 %291, %57		; visa id: 400
  br i1 %292, label %295, label %293, !stats.blockFrequency.digits !483, !stats.blockFrequency.scale !468		; visa id: 401

293:                                              ; preds = %.preheader.i4.i.i.i.14
; BB113 :
  %294 = icmp eq i32 %291, %57		; visa id: 403
  br i1 %294, label %._crit_edge188, label %..preheader.i4.i.i.i.15_crit_edge, !stats.blockFrequency.digits !484, !stats.blockFrequency.scale !474		; visa id: 404

..preheader.i4.i.i.i.15_crit_edge:                ; preds = %293
; BB:
  br label %.preheader.i4.i.i.i.15, !stats.blockFrequency.digits !484, !stats.blockFrequency.scale !480

._crit_edge188:                                   ; preds = %293
; BB115 :
  br label %405, !stats.blockFrequency.digits !484, !stats.blockFrequency.scale !480		; visa id: 407

295:                                              ; preds = %.preheader.i4.i.i.i.14
; BB116 :
  %296 = sub nuw i32 %291, %57, !spirv.Decorations !440		; visa id: 409
  %297 = add i32 %290, 1		; visa id: 410
  br label %.preheader.i4.i.i.i.15, !stats.blockFrequency.digits !484, !stats.blockFrequency.scale !474		; visa id: 411

.preheader.i4.i.i.i.15:                           ; preds = %..preheader.i4.i.i.i.15_crit_edge, %295
; BB117 :
  %298 = phi i32 [ %296, %295 ], [ %291, %..preheader.i4.i.i.i.15_crit_edge ]
  %299 = phi i32 [ %297, %295 ], [ %290, %..preheader.i4.i.i.i.15_crit_edge ]
  %300 = shl i32 %299, 1		; visa id: 412
  %301 = shl i32 %298, 1		; visa id: 413
  %302 = icmp ugt i32 %301, %57		; visa id: 414
  br i1 %302, label %305, label %303, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !474		; visa id: 415

303:                                              ; preds = %.preheader.i4.i.i.i.15
; BB118 :
  %304 = icmp eq i32 %301, %57		; visa id: 417
  br i1 %304, label %._crit_edge189, label %..preheader.i4.i.i.i.16_crit_edge, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !480		; visa id: 418

..preheader.i4.i.i.i.16_crit_edge:                ; preds = %303
; BB:
  br label %.preheader.i4.i.i.i.16, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !486

._crit_edge189:                                   ; preds = %303
; BB120 :
  br label %405, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !486		; visa id: 421

305:                                              ; preds = %.preheader.i4.i.i.i.15
; BB121 :
  %306 = sub nuw i32 %301, %57, !spirv.Decorations !440		; visa id: 423
  %307 = add i32 %300, 1		; visa id: 424
  br label %.preheader.i4.i.i.i.16, !stats.blockFrequency.digits !485, !stats.blockFrequency.scale !480		; visa id: 425

.preheader.i4.i.i.i.16:                           ; preds = %..preheader.i4.i.i.i.16_crit_edge, %305
; BB122 :
  %308 = phi i32 [ %306, %305 ], [ %301, %..preheader.i4.i.i.i.16_crit_edge ]
  %309 = phi i32 [ %307, %305 ], [ %300, %..preheader.i4.i.i.i.16_crit_edge ]
  %310 = shl i32 %309, 1		; visa id: 426
  %311 = shl i32 %308, 1		; visa id: 427
  %312 = icmp ugt i32 %311, %57		; visa id: 428
  br i1 %312, label %315, label %313, !stats.blockFrequency.digits !487, !stats.blockFrequency.scale !474		; visa id: 429

313:                                              ; preds = %.preheader.i4.i.i.i.16
; BB123 :
  %314 = icmp eq i32 %311, %57		; visa id: 431
  br i1 %314, label %._crit_edge190, label %..preheader.i4.i.i.i.17_crit_edge, !stats.blockFrequency.digits !487, !stats.blockFrequency.scale !480		; visa id: 432

..preheader.i4.i.i.i.17_crit_edge:                ; preds = %313
; BB:
  br label %.preheader.i4.i.i.i.17, !stats.blockFrequency.digits !488, !stats.blockFrequency.scale !486

._crit_edge190:                                   ; preds = %313
; BB125 :
  br label %405, !stats.blockFrequency.digits !488, !stats.blockFrequency.scale !486		; visa id: 435

315:                                              ; preds = %.preheader.i4.i.i.i.16
; BB126 :
  %316 = sub nuw i32 %311, %57, !spirv.Decorations !440		; visa id: 437
  %317 = add i32 %310, 1		; visa id: 438
  br label %.preheader.i4.i.i.i.17, !stats.blockFrequency.digits !487, !stats.blockFrequency.scale !480		; visa id: 439

.preheader.i4.i.i.i.17:                           ; preds = %..preheader.i4.i.i.i.17_crit_edge, %315
; BB127 :
  %318 = phi i32 [ %316, %315 ], [ %311, %..preheader.i4.i.i.i.17_crit_edge ]
  %319 = phi i32 [ %317, %315 ], [ %310, %..preheader.i4.i.i.i.17_crit_edge ]
  %320 = shl i32 %319, 1		; visa id: 440
  %321 = shl i32 %318, 1		; visa id: 441
  %322 = icmp ugt i32 %321, %57		; visa id: 442
  br i1 %322, label %325, label %323, !stats.blockFrequency.digits !489, !stats.blockFrequency.scale !480		; visa id: 443

323:                                              ; preds = %.preheader.i4.i.i.i.17
; BB128 :
  %324 = icmp eq i32 %321, %57		; visa id: 445
  br i1 %324, label %._crit_edge191, label %..preheader.i4.i.i.i.18_crit_edge, !stats.blockFrequency.digits !490, !stats.blockFrequency.scale !486		; visa id: 446

..preheader.i4.i.i.i.18_crit_edge:                ; preds = %323
; BB:
  br label %.preheader.i4.i.i.i.18, !stats.blockFrequency.digits !491, !stats.blockFrequency.scale !492

._crit_edge191:                                   ; preds = %323
; BB130 :
  br label %405, !stats.blockFrequency.digits !491, !stats.blockFrequency.scale !492		; visa id: 449

325:                                              ; preds = %.preheader.i4.i.i.i.17
; BB131 :
  %326 = sub nuw i32 %321, %57, !spirv.Decorations !440		; visa id: 451
  %327 = add i32 %320, 1		; visa id: 452
  br label %.preheader.i4.i.i.i.18, !stats.blockFrequency.digits !490, !stats.blockFrequency.scale !486		; visa id: 453

.preheader.i4.i.i.i.18:                           ; preds = %..preheader.i4.i.i.i.18_crit_edge, %325
; BB132 :
  %328 = phi i32 [ %326, %325 ], [ %321, %..preheader.i4.i.i.i.18_crit_edge ]
  %329 = phi i32 [ %327, %325 ], [ %320, %..preheader.i4.i.i.i.18_crit_edge ]
  %330 = shl i32 %329, 1		; visa id: 454
  %331 = shl i32 %328, 1		; visa id: 455
  %332 = icmp ugt i32 %331, %57		; visa id: 456
  br i1 %332, label %335, label %333, !stats.blockFrequency.digits !493, !stats.blockFrequency.scale !480		; visa id: 457

333:                                              ; preds = %.preheader.i4.i.i.i.18
; BB133 :
  %334 = icmp eq i32 %331, %57		; visa id: 459
  br i1 %334, label %._crit_edge192, label %..preheader.i4.i.i.i.19_crit_edge, !stats.blockFrequency.digits !494, !stats.blockFrequency.scale !486		; visa id: 460

..preheader.i4.i.i.i.19_crit_edge:                ; preds = %333
; BB:
  br label %.preheader.i4.i.i.i.19, !stats.blockFrequency.digits !495, !stats.blockFrequency.scale !492

._crit_edge192:                                   ; preds = %333
; BB135 :
  br label %405, !stats.blockFrequency.digits !495, !stats.blockFrequency.scale !492		; visa id: 463

335:                                              ; preds = %.preheader.i4.i.i.i.18
; BB136 :
  %336 = sub nuw i32 %331, %57, !spirv.Decorations !440		; visa id: 465
  %337 = add i32 %330, 1		; visa id: 466
  br label %.preheader.i4.i.i.i.19, !stats.blockFrequency.digits !494, !stats.blockFrequency.scale !486		; visa id: 467

.preheader.i4.i.i.i.19:                           ; preds = %..preheader.i4.i.i.i.19_crit_edge, %335
; BB137 :
  %338 = phi i32 [ %336, %335 ], [ %331, %..preheader.i4.i.i.i.19_crit_edge ]
  %339 = phi i32 [ %337, %335 ], [ %330, %..preheader.i4.i.i.i.19_crit_edge ]
  %340 = shl i32 %339, 1		; visa id: 468
  %341 = shl i32 %338, 1		; visa id: 469
  %342 = icmp ugt i32 %341, %57		; visa id: 470
  br i1 %342, label %345, label %343, !stats.blockFrequency.digits !496, !stats.blockFrequency.scale !486		; visa id: 471

343:                                              ; preds = %.preheader.i4.i.i.i.19
; BB138 :
  %344 = icmp eq i32 %341, %57		; visa id: 473
  br i1 %344, label %._crit_edge193, label %..preheader.i4.i.i.i.20_crit_edge, !stats.blockFrequency.digits !497, !stats.blockFrequency.scale !492		; visa id: 474

..preheader.i4.i.i.i.20_crit_edge:                ; preds = %343
; BB:
  br label %.preheader.i4.i.i.i.20, !stats.blockFrequency.digits !498, !stats.blockFrequency.scale !499

._crit_edge193:                                   ; preds = %343
; BB140 :
  br label %405, !stats.blockFrequency.digits !498, !stats.blockFrequency.scale !499		; visa id: 477

345:                                              ; preds = %.preheader.i4.i.i.i.19
; BB141 :
  %346 = sub nuw i32 %341, %57, !spirv.Decorations !440		; visa id: 479
  %347 = add i32 %340, 1		; visa id: 480
  br label %.preheader.i4.i.i.i.20, !stats.blockFrequency.digits !497, !stats.blockFrequency.scale !492		; visa id: 481

.preheader.i4.i.i.i.20:                           ; preds = %..preheader.i4.i.i.i.20_crit_edge, %345
; BB142 :
  %348 = phi i32 [ %346, %345 ], [ %341, %..preheader.i4.i.i.i.20_crit_edge ]
  %349 = phi i32 [ %347, %345 ], [ %340, %..preheader.i4.i.i.i.20_crit_edge ]
  %350 = shl i32 %349, 1		; visa id: 482
  %351 = shl i32 %348, 1		; visa id: 483
  %352 = icmp ugt i32 %351, %57		; visa id: 484
  br i1 %352, label %355, label %353, !stats.blockFrequency.digits !500, !stats.blockFrequency.scale !486		; visa id: 485

353:                                              ; preds = %.preheader.i4.i.i.i.20
; BB143 :
  %354 = icmp eq i32 %351, %57		; visa id: 487
  br i1 %354, label %._crit_edge194, label %..preheader.i4.i.i.i.21_crit_edge, !stats.blockFrequency.digits !500, !stats.blockFrequency.scale !492		; visa id: 488

..preheader.i4.i.i.i.21_crit_edge:                ; preds = %353
; BB:
  br label %.preheader.i4.i.i.i.21, !stats.blockFrequency.digits !501, !stats.blockFrequency.scale !499

._crit_edge194:                                   ; preds = %353
; BB145 :
  br label %405, !stats.blockFrequency.digits !501, !stats.blockFrequency.scale !499		; visa id: 491

355:                                              ; preds = %.preheader.i4.i.i.i.20
; BB146 :
  %356 = sub nuw i32 %351, %57, !spirv.Decorations !440		; visa id: 493
  %357 = add i32 %350, 1		; visa id: 494
  br label %.preheader.i4.i.i.i.21, !stats.blockFrequency.digits !500, !stats.blockFrequency.scale !492		; visa id: 495

.preheader.i4.i.i.i.21:                           ; preds = %..preheader.i4.i.i.i.21_crit_edge, %355
; BB147 :
  %358 = phi i32 [ %356, %355 ], [ %351, %..preheader.i4.i.i.i.21_crit_edge ]
  %359 = phi i32 [ %357, %355 ], [ %350, %..preheader.i4.i.i.i.21_crit_edge ]
  %360 = shl i32 %359, 1		; visa id: 496
  %361 = shl i32 %358, 1		; visa id: 497
  %362 = icmp ugt i32 %361, %57		; visa id: 498
  br i1 %362, label %365, label %363, !stats.blockFrequency.digits !502, !stats.blockFrequency.scale !486		; visa id: 499

363:                                              ; preds = %.preheader.i4.i.i.i.21
; BB148 :
  %364 = icmp eq i32 %361, %57		; visa id: 501
  br i1 %364, label %._crit_edge195, label %..preheader.i4.i.i.i.22_crit_edge, !stats.blockFrequency.digits !503, !stats.blockFrequency.scale !492		; visa id: 502

..preheader.i4.i.i.i.22_crit_edge:                ; preds = %363
; BB:
  br label %.preheader.i4.i.i.i.22, !stats.blockFrequency.digits !503, !stats.blockFrequency.scale !499

._crit_edge195:                                   ; preds = %363
; BB150 :
  br label %405, !stats.blockFrequency.digits !503, !stats.blockFrequency.scale !499		; visa id: 505

365:                                              ; preds = %.preheader.i4.i.i.i.21
; BB151 :
  %366 = sub nuw i32 %361, %57, !spirv.Decorations !440		; visa id: 507
  %367 = add i32 %360, 1		; visa id: 508
  br label %.preheader.i4.i.i.i.22, !stats.blockFrequency.digits !503, !stats.blockFrequency.scale !492		; visa id: 509

.preheader.i4.i.i.i.22:                           ; preds = %..preheader.i4.i.i.i.22_crit_edge, %365
; BB152 :
  %368 = phi i32 [ %366, %365 ], [ %361, %..preheader.i4.i.i.i.22_crit_edge ]
  %369 = phi i32 [ %367, %365 ], [ %360, %..preheader.i4.i.i.i.22_crit_edge ]
  %370 = shl i32 %369, 1		; visa id: 510
  %371 = shl i32 %368, 1		; visa id: 511
  %372 = icmp ugt i32 %371, %57		; visa id: 512
  br i1 %372, label %375, label %373, !stats.blockFrequency.digits !504, !stats.blockFrequency.scale !492		; visa id: 513

373:                                              ; preds = %.preheader.i4.i.i.i.22
; BB153 :
  %374 = icmp eq i32 %371, %57		; visa id: 515
  br i1 %374, label %._crit_edge196, label %..preheader.i4.i.i.i.23_crit_edge, !stats.blockFrequency.digits !505, !stats.blockFrequency.scale !499		; visa id: 516

..preheader.i4.i.i.i.23_crit_edge:                ; preds = %373
; BB:
  br label %.preheader.i4.i.i.i.23, !stats.blockFrequency.digits !506, !stats.blockFrequency.scale !507

._crit_edge196:                                   ; preds = %373
; BB155 :
  br label %405, !stats.blockFrequency.digits !506, !stats.blockFrequency.scale !507		; visa id: 519

375:                                              ; preds = %.preheader.i4.i.i.i.22
; BB156 :
  %376 = sub nuw i32 %371, %57, !spirv.Decorations !440		; visa id: 521
  %377 = add i32 %370, 1		; visa id: 522
  br label %.preheader.i4.i.i.i.23, !stats.blockFrequency.digits !505, !stats.blockFrequency.scale !499		; visa id: 523

.preheader.i4.i.i.i.23:                           ; preds = %..preheader.i4.i.i.i.23_crit_edge, %375
; BB157 :
  %378 = phi i32 [ %376, %375 ], [ %371, %..preheader.i4.i.i.i.23_crit_edge ]
  %379 = phi i32 [ %377, %375 ], [ %370, %..preheader.i4.i.i.i.23_crit_edge ]
  %380 = shl i32 %379, 1		; visa id: 524
  %381 = shl i32 %378, 1		; visa id: 525
  %382 = icmp ugt i32 %381, %57		; visa id: 526
  br i1 %382, label %385, label %383, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !492		; visa id: 527

383:                                              ; preds = %.preheader.i4.i.i.i.23
; BB158 :
  %384 = icmp eq i32 %381, %57		; visa id: 529
  br i1 %384, label %._crit_edge197, label %..preheader.i4.i.i.i.24_crit_edge, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !499		; visa id: 530

..preheader.i4.i.i.i.24_crit_edge:                ; preds = %383
; BB:
  br label %.preheader.i4.i.i.i.24, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !507

._crit_edge197:                                   ; preds = %383
; BB160 :
  br label %405, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !507		; visa id: 533

385:                                              ; preds = %.preheader.i4.i.i.i.23
; BB161 :
  %386 = sub nuw i32 %381, %57, !spirv.Decorations !440		; visa id: 535
  %387 = add i32 %380, 1		; visa id: 536
  br label %.preheader.i4.i.i.i.24, !stats.blockFrequency.digits !477, !stats.blockFrequency.scale !499		; visa id: 537

.preheader.i4.i.i.i.24:                           ; preds = %..preheader.i4.i.i.i.24_crit_edge, %385
; BB162 :
  %388 = phi i32 [ %386, %385 ], [ %381, %..preheader.i4.i.i.i.24_crit_edge ]
  %389 = phi i32 [ %387, %385 ], [ %380, %..preheader.i4.i.i.i.24_crit_edge ]
  %390 = shl i32 %389, 1		; visa id: 538
  %391 = shl i32 %388, 1		; visa id: 539
  %392 = icmp ugt i32 %391, %57		; visa id: 540
  br i1 %392, label %395, label %393, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !499		; visa id: 541

393:                                              ; preds = %.preheader.i4.i.i.i.24
; BB163 :
  %394 = icmp eq i32 %391, %57		; visa id: 543
  br i1 %394, label %._crit_edge198, label %..preheader.i4.i.i.i.25_crit_edge, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !507		; visa id: 544

..preheader.i4.i.i.i.25_crit_edge:                ; preds = %393
; BB:
  br label %.preheader.i4.i.i.i.25, !stats.blockFrequency.digits !508, !stats.blockFrequency.scale !509

._crit_edge198:                                   ; preds = %393
; BB165 :
  br label %405, !stats.blockFrequency.digits !508, !stats.blockFrequency.scale !509		; visa id: 547

395:                                              ; preds = %.preheader.i4.i.i.i.24
; BB166 :
  %396 = sub nuw i32 %391, %57, !spirv.Decorations !440		; visa id: 549
  %397 = add i32 %390, 1		; visa id: 550
  br label %.preheader.i4.i.i.i.25, !stats.blockFrequency.digits !479, !stats.blockFrequency.scale !507		; visa id: 551

.preheader.i4.i.i.i.25:                           ; preds = %..preheader.i4.i.i.i.25_crit_edge, %395
; BB167 :
  %398 = phi i32 [ %396, %395 ], [ %391, %..preheader.i4.i.i.i.25_crit_edge ]
  %399 = phi i32 [ %397, %395 ], [ %390, %..preheader.i4.i.i.i.25_crit_edge ]
  %400 = shl i32 %399, 1		; visa id: 552
  %401 = shl i32 %398, 1		; visa id: 553
  %402 = icmp ugt i32 %401, %57		; visa id: 554
  br i1 %402, label %410, label %403, !stats.blockFrequency.digits !510, !stats.blockFrequency.scale !499		; visa id: 555

403:                                              ; preds = %.preheader.i4.i.i.i.25
; BB168 :
  %404 = icmp eq i32 %401, %57		; visa id: 557
  br i1 %404, label %._crit_edge199, label %..preheader.i4.i.i.i.26_crit_edge, !stats.blockFrequency.digits !510, !stats.blockFrequency.scale !507		; visa id: 558

..preheader.i4.i.i.i.26_crit_edge:                ; preds = %403
; BB:
  br label %.preheader.i4.i.i.i.26, !stats.blockFrequency.digits !510, !stats.blockFrequency.scale !509

._crit_edge199:                                   ; preds = %403
; BB170 :
  br label %405, !stats.blockFrequency.digits !510, !stats.blockFrequency.scale !509		; visa id: 561

405:                                              ; preds = %._crit_edge199, %._crit_edge198, %._crit_edge197, %._crit_edge196, %._crit_edge195, %._crit_edge194, %._crit_edge193, %._crit_edge192, %._crit_edge191, %._crit_edge190, %._crit_edge189, %._crit_edge188, %._crit_edge187, %._crit_edge186, %._crit_edge185, %._crit_edge184, %._crit_edge183, %._crit_edge182, %._crit_edge181, %._crit_edge180, %._crit_edge179, %._crit_edge178, %._crit_edge177, %._crit_edge176, %._crit_edge175, %._crit_edge174, %._crit_edge173
; BB171 :
  %406 = phi i16 [ 26, %._crit_edge173 ], [ 25, %._crit_edge174 ], [ 24, %._crit_edge175 ], [ 23, %._crit_edge176 ], [ 22, %._crit_edge177 ], [ 21, %._crit_edge178 ], [ 20, %._crit_edge179 ], [ 19, %._crit_edge180 ], [ 18, %._crit_edge181 ], [ 17, %._crit_edge182 ], [ 16, %._crit_edge183 ], [ 15, %._crit_edge184 ], [ 14, %._crit_edge185 ], [ 13, %._crit_edge186 ], [ 12, %._crit_edge187 ], [ 11, %._crit_edge188 ], [ 10, %._crit_edge189 ], [ 9, %._crit_edge190 ], [ 8, %._crit_edge191 ], [ 7, %._crit_edge192 ], [ 6, %._crit_edge193 ], [ 5, %._crit_edge194 ], [ 4, %._crit_edge195 ], [ 3, %._crit_edge196 ], [ 2, %._crit_edge197 ], [ 1, %._crit_edge198 ], [ 0, %._crit_edge199 ]
  %.lcssa = phi i32 [ 0, %._crit_edge173 ], [ %.demoted.zext, %._crit_edge174 ], [ %160, %._crit_edge175 ], [ %170, %._crit_edge176 ], [ %180, %._crit_edge177 ], [ %190, %._crit_edge178 ], [ %200, %._crit_edge179 ], [ %210, %._crit_edge180 ], [ %220, %._crit_edge181 ], [ %230, %._crit_edge182 ], [ %240, %._crit_edge183 ], [ %250, %._crit_edge184 ], [ %260, %._crit_edge185 ], [ %270, %._crit_edge186 ], [ %280, %._crit_edge187 ], [ %290, %._crit_edge188 ], [ %300, %._crit_edge189 ], [ %310, %._crit_edge190 ], [ %320, %._crit_edge191 ], [ %330, %._crit_edge192 ], [ %340, %._crit_edge193 ], [ %350, %._crit_edge194 ], [ %360, %._crit_edge195 ], [ %370, %._crit_edge196 ], [ %380, %._crit_edge197 ], [ %390, %._crit_edge198 ], [ %400, %._crit_edge199 ]
  %407 = or i32 %.lcssa, 1		; visa id: 562
  %408 = trunc i16 %406 to i8		; visa id: 563
  %.demoted.zext167 = zext i8 %408 to i32		; visa id: 564
  %409 = shl i32 %407, %.demoted.zext167		; visa id: 565
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i, !stats.blockFrequency.digits !511, !stats.blockFrequency.scale !432		; visa id: 566

410:                                              ; preds = %.preheader.i4.i.i.i.25
; BB172 :
  %411 = add i32 %400, 1		; visa id: 568
  br label %.preheader.i4.i.i.i.26, !stats.blockFrequency.digits !510, !stats.blockFrequency.scale !507		; visa id: 569

.preheader.i4.i.i.i.26:                           ; preds = %..preheader.i4.i.i.i.26_crit_edge, %410
; BB173 :
  %412 = phi i32 [ %411, %410 ], [ %400, %..preheader.i4.i.i.i.26_crit_edge ]
  %413 = or i32 %412, 1		; visa id: 570
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i, !stats.blockFrequency.digits !512, !stats.blockFrequency.scale !499		; visa id: 571

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i:        ; preds = %.preheader.i4.i.i.i.26, %405
; BB174 :
  %414 = phi i32 [ %409, %405 ], [ %413, %.preheader.i4.i.i.i.26 ]
  %415 = lshr i32 %414, 3		; visa id: 572
  %416 = and i32 %415, 8388607		; visa id: 573
  %417 = and i32 %414, 7		; visa id: 574
  %418 = icmp eq i32 %417, 0		; visa id: 575
  br i1 %418, label %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge, label %419, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !432		; visa id: 576

_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
; BB:
  br label %.critedge53, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !435

419:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i
; BB176 :
  %420 = and i32 %414, 15		; visa id: 578
  %421 = icmp ugt i32 %417, 4
  %.not10 = icmp eq i32 %420, 12		; visa id: 579
  %or.cond52 = or i1 %421, %.not10		; visa id: 580
  br i1 %or.cond52, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i, label %..critedge53_crit_edge200, !stats.blockFrequency.digits !513, !stats.blockFrequency.scale !433		; visa id: 582

..critedge53_crit_edge200:                        ; preds = %419
; BB:
  br label %.critedge53, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !435

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i: ; preds = %419
; BB178 :
  %422 = icmp eq i32 %416, 8388607		; visa id: 584
  br i1 %422, label %424, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !435		; visa id: 585

_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
; BB179 :
  %423 = add nuw nsw i32 %416, 1, !spirv.Decorations !426		; visa id: 587
  br label %.critedge53, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !444		; visa id: 588

424:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i
; BB180 :
  %425 = add nsw i32 %66, 128, !spirv.Decorations !421		; visa id: 590
  %426 = icmp eq i32 %425, 255		; visa id: 591
  br i1 %426, label %434, label %..critedge53_crit_edge201, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !444		; visa id: 592

..critedge53_crit_edge201:                        ; preds = %424
; BB181 :
  br label %.critedge53, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !457		; visa id: 595

.critedge53:                                      ; preds = %..critedge53_crit_edge201, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge, %..critedge53_crit_edge200, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge, %..critedge53_crit_edge
; BB182 :
  %427 = phi i32 [ %416, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge ], [ %416, %..critedge53_crit_edge200 ], [ %423, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge ], [ 0, %..critedge53_crit_edge201 ], [ 0, %..critedge53_crit_edge ]
  %428 = phi i32 [ %139, %_ZL12fra_uint_divIjET_S0_S0_j.exit5.i.i.i..critedge53_crit_edge ], [ %139, %..critedge53_crit_edge200 ], [ %139, %_ZL19__handling_roundingIjET_S0_S0_ji.exit3.i.i.i..critedge53_crit_edge ], [ %425, %..critedge53_crit_edge201 ], [ %139, %..critedge53_crit_edge ]
  %429 = and i32 %36, -2147483648		; visa id: 596
  %430 = shl nuw nsw i32 %428, 23, !spirv.Decorations !426		; visa id: 597
  %431 = or i32 %429, %430
  %432 = or i32 %431, %427		; visa id: 598
  %433 = bitcast i32 %432 to float		; visa id: 599
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !514, !stats.blockFrequency.scale !432		; visa id: 599

434:                                              ; preds = %424
; BB183 :
  %435 = icmp sgt i32 %36, -1		; visa id: 601
  %.54 = select i1 %435, float 0x7FF0000000000000, float 0xFFF0000000000000		; visa id: 602
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !457		; visa id: 603

436:                                              ; preds = %64
; BB184 :
  %437 = icmp sgt i32 %36, -1		; visa id: 605
  %.51 = select i1 %437, float 0x7FF0000000000000, float 0xFFF0000000000000		; visa id: 606
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !430		; visa id: 607

438:                                              ; preds = %46
; BB185 :
  %tobool.i = icmp eq i32 %57, 0		; visa id: 609
  br i1 %tobool.i, label %.precompiled_u32divrem.exit_crit_edge, label %if.end.i, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !425		; visa id: 610

.precompiled_u32divrem.exit_crit_edge:            ; preds = %438
; BB186 :
  br label %precompiled_u32divrem.exit, !stats.blockFrequency.digits !515, !stats.blockFrequency.scale !430		; visa id: 613

if.end.i:                                         ; preds = %438
; BB187 :
  %conv.i = uitofp i32 %57 to float		; visa id: 615
  %div.i = fdiv float 1.000000e+00, %conv.i, !fpmath !516		; visa id: 616
  %conv1.i = uitofp i32 %57 to double		; visa id: 617
  %439 = fsub double 0.000000e+00, %conv1.i		; visa id: 618
  %conv3.i = fpext float %div.i to double		; visa id: 619
  %conv2.i = uitofp i32 %55 to double		; visa id: 620
  %440 = call double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double %439, double %conv3.i, double 0x3FF0000000004000)		; visa id: 621
  %441 = call double @llvm.genx.GenISA.mul.rtz.f64.f64.f64(double %conv3.i, double %conv2.i)		; visa id: 623
  %442 = call double @llvm.genx.GenISA.fma.rtz.f64.f64.f64.f64(double %441, double %440, double %441)		; visa id: 624
  %conv6.i = fptoui double %442 to i32		; visa id: 626
  br label %precompiled_u32divrem.exit, !stats.blockFrequency.digits !517, !stats.blockFrequency.scale !425		; visa id: 627

precompiled_u32divrem.exit:                       ; preds = %.precompiled_u32divrem.exit_crit_edge, %if.end.i
; BB188 :
  %retval.0.i = phi i32 [ %conv6.i, %if.end.i ], [ -1, %.precompiled_u32divrem.exit_crit_edge ]
  br label %443, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !425		; visa id: 630

443:                                              ; preds = %._crit_edge, %precompiled_u32divrem.exit
; BB189 :
  %444 = phi i32 [ -2147483648, %precompiled_u32divrem.exit ], [ %449, %._crit_edge ]
  %445 = phi i64 [ 0, %precompiled_u32divrem.exit ], [ %450, %._crit_edge ]
  %446 = bitcast i64 %445 to <2 x i32>		; visa id: 631
  %447 = extractelement <2 x i32> %446, i32 0		; visa id: 632
  %448 = extractelement <2 x i32> %446, i32 1		; visa id: 632
  %449 = lshr i32 %444, 1		; visa id: 632
  %450 = add nuw nsw i64 %445, 1, !spirv.Decorations !426		; visa id: 633
  %451 = icmp eq i32 %448, 0
  %452 = icmp ugt i32 %447, 30		; visa id: 634
  %453 = and i1 %451, %452		; visa id: 635
  %454 = icmp ugt i32 %448, 0
  %455 = or i1 %453, %454		; visa id: 637
  %456 = and i32 %retval.0.i, %449		; visa id: 639
  %457 = icmp eq i32 %456, %449
  %458 = or i1 %455, %457		; visa id: 640
  br i1 %458, label %459, label %._crit_edge, !stats.blockFrequency.digits !428, !stats.blockFrequency.scale !416		; visa id: 643

._crit_edge:                                      ; preds = %443
; BB:
  br label %443, !stats.blockFrequency.digits !429, !stats.blockFrequency.scale !416

459:                                              ; preds = %443
; BB191 :
  %.lcssa219 = phi i64 [ %450, %443 ]
  %460 = trunc i64 %.lcssa219 to i32		; visa id: 645
  %461 = sub nsw i32 31, %460, !spirv.Decorations !421		; visa id: 646
  %462 = add nsw i32 %53, %461, !spirv.Decorations !421		; visa id: 647
  %463 = icmp sgt i32 %462, 127		; visa id: 648
  br i1 %463, label %638, label %464, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !425		; visa id: 649

464:                                              ; preds = %459
; BB192 :
  %465 = mul i32 %57, %retval.0.i		; visa id: 651
  %466 = sub i32 %55, %465		; visa id: 652
  %467 = icmp sgt i32 %462, -127		; visa id: 653
  br i1 %467, label %585, label %468, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !430		; visa id: 654

468:                                              ; preds = %464
; BB193 :
  %469 = icmp ult i32 %462, -149		; visa id: 656
  br i1 %469, label %572, label %470, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !432		; visa id: 657

470:                                              ; preds = %468
; BB194 :
  %471 = add nsw i32 %462, 152, !spirv.Decorations !421		; visa id: 659
  %472 = icmp sgt i32 %471, %461		; visa id: 660
  br i1 %472, label %511, label %473, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 661

473:                                              ; preds = %470
; BB195 :
  %474 = xor i32 %462, -1		; visa id: 663
  %475 = sub nsw i32 %474, %460, !spirv.Decorations !421
  %476 = add nsw i32 %475, -117, !spirv.Decorations !421		; visa id: 664
  %477 = lshr i32 %retval.0.i, %476		; visa id: 665
  %478 = add nsw i32 %475, -120, !spirv.Decorations !421		; visa id: 666
  %479 = lshr i32 %retval.0.i, %478		; visa id: 667
  %480 = and i32 %479, 7		; visa id: 668
  %481 = and i32 %479, 1
  %482 = icmp eq i32 %481, 0		; visa id: 669
  br i1 %482, label %483, label %.._crit_edge81_crit_edge, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !435		; visa id: 671

.._crit_edge81_crit_edge:                         ; preds = %473
; BB:
  br label %._crit_edge81, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !444

483:                                              ; preds = %473
; BB197 :
  %484 = shl nsw i32 -1, %478, !spirv.Decorations !421		; visa id: 673
  %485 = xor i32 %484, -1
  %486 = and i32 %retval.0.i, %485
  %487 = icmp ne i32 %486, 0
  %488 = icmp ne i32 %55, %465		; visa id: 674
  %489 = or i1 %487, %488		; visa id: 675
  %490 = zext i1 %489 to i32		; visa id: 679
  %491 = or i32 %480, %490		; visa id: 680
  br label %._crit_edge81, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !444		; visa id: 682

._crit_edge81:                                    ; preds = %.._crit_edge81_crit_edge, %483
; BB198 :
  %492 = phi i32 [ %491, %483 ], [ %480, %.._crit_edge81_crit_edge ]
  %493 = icmp eq i32 %492, 0		; visa id: 683
  br i1 %493, label %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, label %494, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !435		; visa id: 684

._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge: ; preds = %._crit_edge81
; BB:
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !444

494:                                              ; preds = %._crit_edge81
; BB200 :
  %495 = icmp ugt i32 %492, 4		; visa id: 686
  br i1 %495, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118, label %496, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !435		; visa id: 687

496:                                              ; preds = %494
; BB201 :
  %497 = and i32 %477, 1
  %498 = icmp eq i32 %492, 4
  %499 = icmp ne i32 %497, 0		; visa id: 689
  %not. = and i1 %498, %499		; visa id: 691
  %500 = icmp ugt i32 %477, 8388606		; visa id: 693
  br i1 %not., label %._crit_edge170, label %._crit_edge171, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !444		; visa id: 694

._crit_edge171:                                   ; preds = %496
; BB202 :
  %501 = sext i1 %500 to i32		; visa id: 696
  %502 = sext i1 %not. to i32		; visa id: 697
  br label %560, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !457		; visa id: 698

._crit_edge170:                                   ; preds = %496
; BB203 :
  %503 = add nuw nsw i32 %477, 1, !spirv.Decorations !426		; visa id: 700
  %504 = select i1 %500, i32 0, i32 %503		; visa id: 701
  %505 = sext i1 %500 to i32		; visa id: 702
  %506 = sext i1 %not. to i32		; visa id: 703
  br label %552, !stats.blockFrequency.digits !519, !stats.blockFrequency.scale !457		; visa id: 705

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118: ; preds = %494
; BB204 :
  %507 = add nuw nsw i32 %477, 1, !spirv.Decorations !426		; visa id: 707
  %508 = icmp ugt i32 %477, 8388606		; visa id: 708
  %509 = select i1 %508, i32 0, i32 %507		; visa id: 709
  %510 = sext i1 %508 to i32		; visa id: 710
  br label %552, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !444		; visa id: 713

511:                                              ; preds = %470
; BB205 :
  %512 = sub nsw i32 %471, %461, !spirv.Decorations !421		; visa id: 715
  %513 = shl i32 %retval.0.i, %512		; visa id: 716
  %514 = icmp eq i32 %466, 0		; visa id: 717
  br i1 %514, label %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge, label %.lr.ph89.preheader, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !435		; visa id: 718

._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge: ; preds = %511
; BB206 :
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !444		; visa id: 721

.lr.ph89.preheader:                               ; preds = %511
; BB207 :
  br label %.lr.ph89, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !435		; visa id: 725

.lr.ph89:                                         ; preds = %.preheader.i7.i.i.i..lr.ph89_crit_edge, %.lr.ph89.preheader
; BB208 :
  %515 = phi i32 [ %528, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ 0, %.lr.ph89.preheader ]
  %516 = phi i32 [ %527, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ 0, %.lr.ph89.preheader ]
  %517 = phi i32 [ %526, %.preheader.i7.i.i.i..lr.ph89_crit_edge ], [ %466, %.lr.ph89.preheader ]
  %518 = shl i32 %516, 1		; visa id: 726
  %519 = shl i32 %517, 1		; visa id: 727
  %520 = icmp ugt i32 %519, %57		; visa id: 728
  br i1 %520, label %523, label %521, !stats.blockFrequency.digits !438, !stats.blockFrequency.scale !425		; visa id: 729

521:                                              ; preds = %.lr.ph89
; BB209 :
  %522 = icmp eq i32 %519, %57		; visa id: 731
  br i1 %522, label %530, label %..preheader.i7.i.i.i_crit_edge, !stats.blockFrequency.digits !438, !stats.blockFrequency.scale !430		; visa id: 733

..preheader.i7.i.i.i_crit_edge:                   ; preds = %521
; BB:
  br label %.preheader.i7.i.i.i, !stats.blockFrequency.digits !439, !stats.blockFrequency.scale !430

523:                                              ; preds = %.lr.ph89
; BB211 :
  %524 = sub nuw i32 %519, %57, !spirv.Decorations !440		; visa id: 736
  %525 = add i32 %518, 1		; visa id: 737
  br label %.preheader.i7.i.i.i, !stats.blockFrequency.digits !438, !stats.blockFrequency.scale !430		; visa id: 738

.preheader.i7.i.i.i:                              ; preds = %..preheader.i7.i.i.i_crit_edge, %523
; BB212 :
  %526 = phi i32 [ %524, %523 ], [ %519, %..preheader.i7.i.i.i_crit_edge ]
  %527 = phi i32 [ %525, %523 ], [ %518, %..preheader.i7.i.i.i_crit_edge ]
  %528 = add nuw i32 %515, 1, !spirv.Decorations !440		; visa id: 739
  %529 = icmp ult i32 %528, %512		; visa id: 740
  br i1 %529, label %.preheader.i7.i.i.i..lr.ph89_crit_edge, label %.preheader.i7.i.i.i._crit_edge, !stats.blockFrequency.digits !441, !stats.blockFrequency.scale !425		; visa id: 741

.preheader.i7.i.i.i..lr.ph89_crit_edge:           ; preds = %.preheader.i7.i.i.i
; BB:
  br label %.lr.ph89, !stats.blockFrequency.digits !520, !stats.blockFrequency.scale !425

530:                                              ; preds = %521
; BB214 :
  %.lcssa216 = phi i32 [ %515, %521 ]
  %.lcssa214 = phi i32 [ %518, %521 ]
  %531 = add i32 %.lcssa214, 1		; visa id: 744
  %532 = xor i32 %.lcssa216, -1		; visa id: 745
  %533 = add i32 %512, %532		; visa id: 746
  %534 = shl i32 %531, %533		; visa id: 747
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, !stats.blockFrequency.digits !443, !stats.blockFrequency.scale !457		; visa id: 748

.preheader.i7.i.i.i._crit_edge:                   ; preds = %.preheader.i7.i.i.i
; BB215 :
  %.lcssa218 = phi i32 [ %527, %.preheader.i7.i.i.i ]
  %535 = or i32 %.lcssa218, 1		; visa id: 750
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i, !stats.blockFrequency.digits !521, !stats.blockFrequency.scale !444		; visa id: 751

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i:        ; preds = %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge, %.preheader.i7.i.i.i._crit_edge, %530
; BB216 :
  %536 = phi i32 [ %534, %530 ], [ %535, %.preheader.i7.i.i.i._crit_edge ], [ 0, %._ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i_crit_edge ]
  %537 = or i32 %513, %536		; visa id: 752
  %538 = and i32 %537, 7		; visa id: 753
  %539 = lshr i32 %537, 3		; visa id: 754
  %540 = icmp eq i32 %538, 0		; visa id: 755
  br i1 %540, label %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !435		; visa id: 756

_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
; BB:
  br label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !444

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge, %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge
; BB218 :
  %.ph = phi i32 [ %539, %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge ], [ %477, %._crit_edge81._ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread_crit_edge ]
  %541 = icmp ugt i32 %.ph, 8388606
  %542 = sext i1 %541 to i32		; visa id: 758
  br label %560, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !435		; visa id: 760

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit8.i.i.i
; BB219 :
  %543 = and i32 %537, 15		; visa id: 762
  %544 = icmp ugt i32 %538, 4
  %.not = icmp eq i32 %543, 12		; visa id: 763
  %or.cond = or i1 %544, %.not		; visa id: 764
  %545 = icmp ugt i32 %537, 67108855		; visa id: 766
  br i1 %or.cond, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !435		; visa id: 767

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
; BB220 :
  %546 = sext i1 %545 to i32		; visa id: 769
  %547 = sext i1 %or.cond to i32		; visa id: 770
  br label %560, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !444		; visa id: 771

_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i
; BB221 :
  %548 = add nuw nsw i32 %539, 1, !spirv.Decorations !426		; visa id: 773
  %549 = select i1 %545, i32 0, i32 %548		; visa id: 774
  %550 = sext i1 %545 to i32		; visa id: 775
  %551 = sext i1 %or.cond to i32		; visa id: 776
  br label %552, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !444		; visa id: 777

552:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge, %._crit_edge170, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118
; BB222 :
  %553 = phi i32 [ %509, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %549, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %504, %._crit_edge170 ]
  %554 = phi i32 [ %510, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %550, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %505, %._crit_edge170 ]
  %555 = phi i32 [ -1, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread118 ], [ %551, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge ], [ %506, %._crit_edge170 ]
  %556 = icmp ne i32 %554, 0
  %557 = sext i1 %556 to i32		; visa id: 778
  %558 = icmp ne i32 %555, 0
  %559 = sext i1 %558 to i32		; visa id: 779
  br label %560, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !435		; visa id: 780

560:                                              ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172, %._crit_edge171, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread, %552
; BB223 :
  %561 = phi i32 [ %557, %552 ], [ %546, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ %542, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %501, %._crit_edge171 ]
  %562 = phi i32 [ %559, %552 ], [ %547, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ 0, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %502, %._crit_edge171 ]
  %563 = phi i32 [ %553, %552 ], [ %539, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i._crit_edge172 ], [ %.ph, %_ZL19__handling_roundingIjET_S0_S0_ji.exit9.i.i.i.thread ], [ %477, %._crit_edge171 ]
  %564 = icmp ne i32 %562, 0
  %565 = icmp ne i32 %561, 0		; visa id: 781
  %566 = and i1 %564, %565		; visa id: 782
  %567 = select i1 %566, i32 8388608, i32 0		; visa id: 784
  %568 = and i32 %36, -2147483648		; visa id: 785
  %569 = or i32 %568, %567
  %570 = or i32 %569, %563		; visa id: 786
  %571 = bitcast i32 %570 to float		; visa id: 787
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 787

572:                                              ; preds = %468
; BB224 :
  %573 = icmp eq i32 %462, -150		; visa id: 789
  br i1 %573, label %574, label %.._crit_edge80_crit_edge, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 790

.._crit_edge80_crit_edge:                         ; preds = %572
; BB225 :
  br label %._crit_edge80, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !435		; visa id: 793

574:                                              ; preds = %572
; BB226 :
  %575 = lshr i32 -2147483648, %460		; visa id: 795
  %576 = icmp ne i32 %55, %465
  %577 = icmp ne i32 %retval.0.i, %575		; visa id: 796
  %578 = or i1 %576, %577		; visa id: 797
  %579 = sext i1 %578 to i32		; visa id: 799
  br label %._crit_edge80, !stats.blockFrequency.digits !518, !stats.blockFrequency.scale !435		; visa id: 800

._crit_edge80:                                    ; preds = %.._crit_edge80_crit_edge, %574
; BB227 :
  %580 = phi i32 [ %579, %574 ], [ 0, %.._crit_edge80_crit_edge ]
  %581 = icmp ne i32 %580, 0		; visa id: 801
  %582 = icmp sgt i32 %36, -1		; visa id: 802
  br i1 %582, label %584, label %583, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !433		; visa id: 803

583:                                              ; preds = %._crit_edge80
; BB228 :
  %spec.select49 = select i1 %581, float 0xB6A0000000000000, float -0.000000e+00		; visa id: 805
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !434, !stats.blockFrequency.scale !435		; visa id: 806

584:                                              ; preds = %._crit_edge80
; BB229 :
  %spec.select48 = select i1 %581, float 0x36A0000000000000, float 0.000000e+00		; visa id: 808
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !433		; visa id: 809

585:                                              ; preds = %464
; BB230 :
  %586 = add nsw i32 %462, 127, !spirv.Decorations !421		; visa id: 811
  %587 = add nsw i32 %460, -8, !spirv.Decorations !421		; visa id: 812
  %588 = shl i32 %retval.0.i, %587		; visa id: 813
  %589 = and i32 %588, 8388607		; visa id: 814
  %590 = icmp eq i32 %466, 0		; visa id: 815
  br i1 %590, label %..critedge46_crit_edge, label %.preheader.i.i.i.i.preheader, !stats.blockFrequency.digits !431, !stats.blockFrequency.scale !432		; visa id: 816

..critedge46_crit_edge:                           ; preds = %585
; BB:
  br label %.critedge46, !stats.blockFrequency.digits !450, !stats.blockFrequency.scale !433

.preheader.i.i.i.i.preheader:                     ; preds = %585
; BB232 :
  %591 = add nsw i32 %460, -5, !spirv.Decorations !421		; visa id: 818
  %.not92 = icmp eq i32 %591, 0		; visa id: 819
  br i1 %.not92, label %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge, label %.lr.ph87.preheader, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !432		; visa id: 820

.lr.ph87.preheader:                               ; preds = %.preheader.i.i.i.i.preheader
; BB233 :
  br label %.lr.ph87, !stats.blockFrequency.digits !513, !stats.blockFrequency.scale !433		; visa id: 824

.lr.ph87:                                         ; preds = %.preheader.i.i.i.i..lr.ph87_crit_edge, %.lr.ph87.preheader
; BB234 :
  %592 = phi i32 [ %605, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ 0, %.lr.ph87.preheader ]
  %593 = phi i32 [ %604, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ 0, %.lr.ph87.preheader ]
  %594 = phi i32 [ %603, %.preheader.i.i.i.i..lr.ph87_crit_edge ], [ %466, %.lr.ph87.preheader ]
  %595 = shl i32 %593, 1		; visa id: 825
  %596 = shl i32 %594, 1		; visa id: 826
  %597 = icmp ugt i32 %596, %57		; visa id: 827
  br i1 %597, label %600, label %598, !stats.blockFrequency.digits !522, !stats.blockFrequency.scale !424		; visa id: 828

598:                                              ; preds = %.lr.ph87
; BB235 :
  %599 = icmp eq i32 %596, %57		; visa id: 830
  br i1 %599, label %607, label %..preheader.i.i.i.i_crit_edge, !stats.blockFrequency.digits !522, !stats.blockFrequency.scale !425		; visa id: 832

..preheader.i.i.i.i_crit_edge:                    ; preds = %598
; BB:
  br label %.preheader.i.i.i.i, !stats.blockFrequency.digits !523, !stats.blockFrequency.scale !425

600:                                              ; preds = %.lr.ph87
; BB237 :
  %601 = sub nuw i32 %596, %57, !spirv.Decorations !440		; visa id: 835
  %602 = add i32 %595, 1		; visa id: 836
  br label %.preheader.i.i.i.i, !stats.blockFrequency.digits !522, !stats.blockFrequency.scale !425		; visa id: 837

.preheader.i.i.i.i:                               ; preds = %..preheader.i.i.i.i_crit_edge, %600
; BB238 :
  %603 = phi i32 [ %601, %600 ], [ %596, %..preheader.i.i.i.i_crit_edge ]
  %604 = phi i32 [ %602, %600 ], [ %595, %..preheader.i.i.i.i_crit_edge ]
  %605 = add nuw i32 %592, 1, !spirv.Decorations !440		; visa id: 838
  %606 = icmp ult i32 %605, %591		; visa id: 839
  br i1 %606, label %.preheader.i.i.i.i..lr.ph87_crit_edge, label %.preheader.i.i.i.i._crit_edge.loopexit, !stats.blockFrequency.digits !524, !stats.blockFrequency.scale !424		; visa id: 840

.preheader.i.i.i.i._crit_edge.loopexit:           ; preds = %.preheader.i.i.i.i
; BB:
  %.lcssa213 = phi i32 [ %604, %.preheader.i.i.i.i ]
  br label %.preheader.i.i.i.i._crit_edge, !stats.blockFrequency.digits !525, !stats.blockFrequency.scale !435

.preheader.i.i.i.i..lr.ph87_crit_edge:            ; preds = %.preheader.i.i.i.i
; BB:
  br label %.lr.ph87, !stats.blockFrequency.digits !526, !stats.blockFrequency.scale !424

607:                                              ; preds = %598
; BB241 :
  %.lcssa211 = phi i32 [ %592, %598 ]
  %.lcssa209 = phi i32 [ %595, %598 ]
  %608 = add i32 %.lcssa209, 1		; visa id: 843
  %609 = xor i32 %.lcssa211, -1		; visa id: 844
  %610 = add i32 %591, %609		; visa id: 845
  %611 = shl i32 %608, %610		; visa id: 846
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i, !stats.blockFrequency.digits !527, !stats.blockFrequency.scale !444		; visa id: 847

.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge: ; preds = %.preheader.i.i.i.i.preheader
; BB242 :
  br label %.preheader.i.i.i.i._crit_edge, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !435		; visa id: 850

.preheader.i.i.i.i._crit_edge:                    ; preds = %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge, %.preheader.i.i.i.i._crit_edge.loopexit
; BB243 :
  %.lcssa70 = phi i32 [ 0, %.preheader.i.i.i.i.preheader..preheader.i.i.i.i._crit_edge_crit_edge ], [ %.lcssa213, %.preheader.i.i.i.i._crit_edge.loopexit ]
  %612 = or i32 %.lcssa70, 1		; visa id: 851
  br label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i, !stats.blockFrequency.digits !528, !stats.blockFrequency.scale !433		; visa id: 852

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i:         ; preds = %.preheader.i.i.i.i._crit_edge, %607
; BB244 :
  %613 = phi i32 [ %611, %607 ], [ %612, %.preheader.i.i.i.i._crit_edge ]
  %614 = lshr i32 %613, 3		; visa id: 853
  %615 = or i32 %614, %589		; visa id: 854
  %616 = and i32 %613, 7		; visa id: 856
  %617 = icmp eq i32 %616, 0		; visa id: 857
  br i1 %617, label %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge, label %618, !stats.blockFrequency.digits !436, !stats.blockFrequency.scale !432		; visa id: 858

_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge: ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
; BB:
  br label %.critedge46, !stats.blockFrequency.digits !451, !stats.blockFrequency.scale !435

618:                                              ; preds = %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i
; BB246 :
  %619 = icmp ugt i32 %616, 4		; visa id: 860
  br i1 %619, label %..critedge_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i, !stats.blockFrequency.digits !513, !stats.blockFrequency.scale !433		; visa id: 861

..critedge_crit_edge:                             ; preds = %618
; BB:
  br label %.critedge, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !435

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i: ; preds = %618
; BB248 :
  %620 = and i32 %615, 1
  %621 = icmp ne i32 %616, 4
  %622 = icmp eq i32 %620, 0		; visa id: 863
  %623 = or i1 %621, %622		; visa id: 865
  br i1 %623, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge, label %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge, !stats.blockFrequency.digits !446, !stats.blockFrequency.scale !435		; visa id: 867

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
; BB:
  br label %.critedge, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !444

.critedge:                                        ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge_crit_edge, %..critedge_crit_edge
; BB250 :
  %624 = add nuw nsw i32 %615, 1, !spirv.Decorations !426		; visa id: 869
  %625 = icmp ugt i32 %615, 8388606		; visa id: 870
  br i1 %625, label %626, label %.critedge..critedge46_crit_edge, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !433		; visa id: 871

.critedge..critedge46_crit_edge:                  ; preds = %.critedge
; BB251 :
  br label %.critedge46, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !435		; visa id: 874

626:                                              ; preds = %.critedge
; BB252 :
  %627 = add nsw i32 %462, 128, !spirv.Decorations !421		; visa id: 876
  %628 = icmp eq i32 %627, 255		; visa id: 877
  br i1 %628, label %629, label %..critedge46_crit_edge169, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !435		; visa id: 878

..critedge46_crit_edge169:                        ; preds = %626
; BB253 :
  br label %.critedge46, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !444		; visa id: 881

629:                                              ; preds = %626
; BB254 :
  %630 = icmp sgt i32 %36, -1		; visa id: 883
  %.47 = select i1 %630, float 0x7FF0000000000000, float 0xFFF0000000000000		; visa id: 884
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !529, !stats.blockFrequency.scale !444		; visa id: 885

_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge: ; preds = %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i
; BB:
  br label %.critedge46, !stats.blockFrequency.digits !447, !stats.blockFrequency.scale !444

.critedge46:                                      ; preds = %..critedge46_crit_edge169, %.critedge..critedge46_crit_edge, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge, %..critedge46_crit_edge
; BB256 :
  %631 = phi i32 [ %615, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge ], [ %615, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge ], [ %624, %.critedge..critedge46_crit_edge ], [ %624, %..critedge46_crit_edge169 ], [ %589, %..critedge46_crit_edge ]
  %632 = phi i32 [ %586, %_ZL12fra_uint_divIjET_S0_S0_j.exit.i.i.i..critedge46_crit_edge ], [ %586, %_ZL19__handling_roundingIjET_S0_S0_ji.exit10.i.i.i..critedge46_crit_edge ], [ %586, %.critedge..critedge46_crit_edge ], [ %627, %..critedge46_crit_edge169 ], [ %586, %..critedge46_crit_edge ]
  %633 = and i32 %36, -2147483648		; visa id: 887
  %634 = shl nuw nsw i32 %632, 23, !spirv.Decorations !426		; visa id: 888
  %635 = or i32 %633, %634
  %636 = or i32 %635, %631		; visa id: 889
  %637 = bitcast i32 %636 to float		; visa id: 890
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !530, !stats.blockFrequency.scale !432		; visa id: 890

638:                                              ; preds = %459
; BB257 :
  %639 = icmp sgt i32 %36, -1		; visa id: 892
  %. = select i1 %639, float 0x7FF0000000000000, float 0xFFF0000000000000		; visa id: 893
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !430		; visa id: 894

640:                                              ; preds = %42
; BB258 :
  %641 = and i32 %36, -2147483648		; visa id: 896
  %642 = bitcast i32 %641 to float		; visa id: 897
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !423, !stats.blockFrequency.scale !424		; visa id: 897

643:                                              ; preds = %40
; BB259 :
  %644 = and i32 %36, -2147483648		; visa id: 899
  %645 = bitcast i32 %644 to float		; visa id: 900
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !531, !stats.blockFrequency.scale !532		; visa id: 900

646:                                              ; preds = %35
; BB260 :
  %647 = icmp eq i32 %29, 0		; visa id: 902
  %648 = icmp eq i32 %28, 255
  %649 = and i1 %648, %647		; visa id: 903
  %650 = and i32 %36, -2147483648
  %651 = or i32 %650, 2139095040		; visa id: 905
  %652 = bitcast i32 %651 to float
  %653 = select i1 %649, float 0x7FF8000000000000, float %652		; visa id: 908
  br label %__imf_fdiv_rn.exit, !stats.blockFrequency.digits !417, !stats.blockFrequency.scale !418		; visa id: 909

__imf_fdiv_rn.exit:                               ; preds = %.__imf_fdiv_rn.exit_crit_edge168, %.__imf_fdiv_rn.exit_crit_edge, %136, %137, %583, %584, %434, %436, %629, %638, %646, %643, %640, %.critedge46, %560, %.critedge53, %120
; BB261 :
  %654 = phi float [ %645, %643 ], [ %642, %640 ], [ %653, %646 ], [ %433, %.critedge53 ], [ %129, %120 ], [ %571, %560 ], [ %637, %.critedge46 ], [ %., %638 ], [ %.47, %629 ], [ %spec.select48, %584 ], [ %spec.select49, %583 ], [ %.51, %436 ], [ %.54, %434 ], [ %spec.select55, %137 ], [ %spec.select56, %136 ], [ 0x7FF8000000000000, %.__imf_fdiv_rn.exit_crit_edge ], [ 0x7FF8000000000000, %.__imf_fdiv_rn.exit_crit_edge168 ]
  %655 = bitcast float %17 to i32
  %656 = and i32 %655, 2139095040		; visa id: 910
  %657 = icmp eq i32 %656, 0		; visa id: 911
  %658 = select i1 %657, float 0x41F0000000000000, float 1.000000e+00		; visa id: 912
  %659 = icmp uge i32 %656, 1677721600		; visa id: 913
  %660 = select i1 %659, float 0x3DF0000000000000, float %658		; visa id: 914
  %661 = fmul float %17, %660		; visa id: 915
  %662 = fdiv float 1.000000e+00, %661		; visa id: 916
  %663 = fmul float %662, %13		; visa id: 917
  %664 = fmul float %663, %660		; visa id: 918
  %665 = and i32 %655, 8388607
  %666 = icmp eq i32 %656, 0
  %667 = icmp eq i32 %665, 0		; visa id: 919
  %668 = or i1 %666, %667		; visa id: 921
  %669 = xor i1 %668, true		; visa id: 923
  %670 = fcmp oeq float %13, %17
  %671 = and i1 %670, %669		; visa id: 924
  %672 = select i1 %671, float 1.000000e+00, float %664		; visa id: 926
  %673 = ptrtoint float addrspace(1)* %2 to i64		; visa id: 927
  %674 = add i64 %10, %673		; visa id: 927
  %675 = inttoptr i64 %674 to float addrspace(1)*		; visa id: 928
  store float %654, float addrspace(1)* %675, align 4		; visa id: 928
  %676 = ptrtoint float addrspace(1)* %3 to i64		; visa id: 929
  %677 = add i64 %10, %676		; visa id: 929
  %678 = inttoptr i64 %677 to float addrspace(1)*		; visa id: 930
  store float %672, float addrspace(1)* %678, align 4		; visa id: 930
  ret void, !stats.blockFrequency.digits !411, !stats.blockFrequency.scale !412		; visa id: 931
}
