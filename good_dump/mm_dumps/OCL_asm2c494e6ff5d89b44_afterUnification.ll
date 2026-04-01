; ------------------------------------------------
; OCL_asm2c494e6ff5d89b44_afterUnification.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

@0 = internal unnamed_addr addrspace(2) constant [23 x i8] c"Pointer: %p, Type: %s\0A\00"

; Function Attrs: convergent nounwind
define spir_kernel void @matmul_kernel_with_tensor_descriptors(i8 addrspace(1)* align 1 %0, i8 addrspace(1)* align 1 %1, i8 addrspace(1)* align 1 %2, i8 addrspace(1)* nocapture readnone align 1 %3, i8 addrspace(1)* nocapture readnone align 1 %4, <8 x i32> %r0, <3 x i32> %globalOffset, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bindlessOffset, i32 %bindlessOffset5, i32 %bindlessOffset6, i32 %bindlessOffset7, i32 %bindlessOffset8) #0 !dbg !432 {
  call void @llvm.genx.GenISA.CatchAllDebugLine(), !dbg !438
  %6 = extractelement <8 x i32> %r0, i32 0
  %7 = extractelement <8 x i32> %r0, i32 1
  %8 = extractelement <8 x i32> %r0, i32 2
  %9 = extractelement <8 x i32> %r0, i32 3
  %10 = extractelement <8 x i32> %r0, i32 4
  %11 = extractelement <8 x i32> %r0, i32 5
  %12 = extractelement <8 x i32> %r0, i32 6
  %13 = extractelement <8 x i32> %r0, i32 7
  %14 = udiv i32 %7, 24, !dbg !439
  %15 = sub nsw i32 1, %14, !dbg !440, !spirv.Decorations !441
  %.neg = mul i32 %14, -24, !dbg !443
  %.decomposed = add i32 %.neg, %7, !dbg !443
  %16 = sdiv i32 %.decomposed, %15, !dbg !444
  %freeze = freeze i32 %16, !dbg !445
  %17 = mul i32 %freeze, %15, !dbg !445
  %.decomposed24 = sub i32 %.decomposed, %17, !dbg !445
  %18 = add nuw nsw i32 %14, %.decomposed24, !dbg !446, !spirv.Decorations !447
  %19 = shl nuw nsw i32 %18, 3, !dbg !449, !spirv.Decorations !447
  %20 = shl nsw i32 %freeze, 9, !dbg !450, !spirv.Decorations !441
  %21 = bitcast <8 x i32> %r0 to <32 x i8>, !dbg !451
  %22 = extractelement <32 x i8> %21, i32 0, !dbg !451
  %23 = extractelement <32 x i8> %21, i32 1, !dbg !451
  %24 = extractelement <32 x i8> %21, i32 2, !dbg !451
  %25 = extractelement <32 x i8> %21, i32 3, !dbg !451
  %26 = extractelement <32 x i8> %21, i32 4, !dbg !451
  %27 = extractelement <32 x i8> %21, i32 5, !dbg !451
  %28 = extractelement <32 x i8> %21, i32 6, !dbg !451
  %29 = extractelement <32 x i8> %21, i32 7, !dbg !451
  %30 = extractelement <32 x i8> %21, i32 8, !dbg !451
  %31 = extractelement <32 x i8> %21, i32 9, !dbg !451
  %32 = extractelement <32 x i8> %21, i32 10, !dbg !451
  %33 = extractelement <32 x i8> %21, i32 11, !dbg !451
  %34 = extractelement <32 x i8> %21, i32 12, !dbg !451
  %35 = extractelement <32 x i8> %21, i32 13, !dbg !451
  %36 = extractelement <32 x i8> %21, i32 14, !dbg !451
  %37 = extractelement <32 x i8> %21, i32 15, !dbg !451
  %38 = extractelement <32 x i8> %21, i32 16, !dbg !451
  %39 = extractelement <32 x i8> %21, i32 17, !dbg !451
  %40 = extractelement <32 x i8> %21, i32 18, !dbg !451
  %41 = extractelement <32 x i8> %21, i32 19, !dbg !451
  %42 = extractelement <32 x i8> %21, i32 20, !dbg !451
  %43 = extractelement <32 x i8> %21, i32 21, !dbg !451
  %44 = extractelement <32 x i8> %21, i32 22, !dbg !451
  %45 = extractelement <32 x i8> %21, i32 23, !dbg !451
  %46 = extractelement <32 x i8> %21, i32 24, !dbg !451
  %47 = extractelement <32 x i8> %21, i32 25, !dbg !451
  %48 = extractelement <32 x i8> %21, i32 26, !dbg !451
  %49 = extractelement <32 x i8> %21, i32 27, !dbg !451
  %50 = extractelement <32 x i8> %21, i32 28, !dbg !451
  %51 = extractelement <32 x i8> %21, i32 29, !dbg !451
  %52 = extractelement <32 x i8> %21, i32 30, !dbg !451
  %53 = extractelement <32 x i8> %21, i32 31, !dbg !451
  %localThreadId17 = zext i8 %30 to i32, !dbg !451
  %54 = and i32 %localThreadId17, 48, !dbg !451
  %55 = shl nuw nsw i32 %localThreadId17, 5, !dbg !451
  %56 = and i32 %55, 480, !dbg !451
  %57 = ptrtoint i8 addrspace(1)* %1 to i64, !dbg !451
  %58 = and i64 %57, -64, !dbg !451
  %59 = trunc i64 %57 to i32, !dbg !451
  %60 = and i32 %59, 63, !dbg !451
  %61 = lshr i32 %60, 1, !dbg !451
  %62 = or i32 %61, %56, !dbg !451
  %63 = or i32 %62, %20, !dbg !451
  %64 = add nuw nsw i32 %60, 24575
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %58, i32 %64, i32 4095, i32 24575, i32 %63, i32 %54, i32 16, i32 16, i32 16, i32 2, i1 false, i1 false, i32 4)
  %65 = ptrtoint i8 addrspace(1)* %0 to i64
  %66 = and i64 %65, -64
  %67 = trunc i64 %65 to i32
  %68 = and i32 %67, 63
  %69 = lshr i32 %68, 1
  %70 = or i32 %20, %61
  br label %71, !dbg !452

71:                                               ; preds = %71, %5
  %72 = phi i32 [ 0, %5 ], [ %81, %71 ]
  %73 = phi float [ 0.000000e+00, %5 ], [ %308, %71 ], !dbg !453
  %74 = phi float [ 0.000000e+00, %5 ], [ %309, %71 ], !dbg !453
  %75 = phi float [ 0.000000e+00, %5 ], [ %310, %71 ], !dbg !453
  %76 = phi float [ 0.000000e+00, %5 ], [ %311, %71 ], !dbg !453
  %77 = phi float [ 0.000000e+00, %5 ], [ %312, %71 ], !dbg !453
  %78 = phi float [ 0.000000e+00, %5 ], [ %313, %71 ], !dbg !453
  %79 = phi float [ 0.000000e+00, %5 ], [ %314, %71 ], !dbg !453
  %80 = phi float [ 0.000000e+00, %5 ], [ %315, %71 ], !dbg !453
  %81 = add nuw nsw i32 %72, 64, !dbg !454, !spirv.Decorations !447
  %82 = bitcast <8 x i32> %r0 to <32 x i8>, !dbg !451
  %83 = extractelement <32 x i8> %82, i32 0, !dbg !451
  %84 = extractelement <32 x i8> %82, i32 1, !dbg !451
  %85 = extractelement <32 x i8> %82, i32 2, !dbg !451
  %86 = extractelement <32 x i8> %82, i32 3, !dbg !451
  %87 = extractelement <32 x i8> %82, i32 4, !dbg !451
  %88 = extractelement <32 x i8> %82, i32 5, !dbg !451
  %89 = extractelement <32 x i8> %82, i32 6, !dbg !451
  %90 = extractelement <32 x i8> %82, i32 7, !dbg !451
  %91 = extractelement <32 x i8> %82, i32 8, !dbg !451
  %92 = extractelement <32 x i8> %82, i32 9, !dbg !451
  %93 = extractelement <32 x i8> %82, i32 10, !dbg !451
  %94 = extractelement <32 x i8> %82, i32 11, !dbg !451
  %95 = extractelement <32 x i8> %82, i32 12, !dbg !451
  %96 = extractelement <32 x i8> %82, i32 13, !dbg !451
  %97 = extractelement <32 x i8> %82, i32 14, !dbg !451
  %98 = extractelement <32 x i8> %82, i32 15, !dbg !451
  %99 = extractelement <32 x i8> %82, i32 16, !dbg !451
  %100 = extractelement <32 x i8> %82, i32 17, !dbg !451
  %101 = extractelement <32 x i8> %82, i32 18, !dbg !451
  %102 = extractelement <32 x i8> %82, i32 19, !dbg !451
  %103 = extractelement <32 x i8> %82, i32 20, !dbg !451
  %104 = extractelement <32 x i8> %82, i32 21, !dbg !451
  %105 = extractelement <32 x i8> %82, i32 22, !dbg !451
  %106 = extractelement <32 x i8> %82, i32 23, !dbg !451
  %107 = extractelement <32 x i8> %82, i32 24, !dbg !451
  %108 = extractelement <32 x i8> %82, i32 25, !dbg !451
  %109 = extractelement <32 x i8> %82, i32 26, !dbg !451
  %110 = extractelement <32 x i8> %82, i32 27, !dbg !451
  %111 = extractelement <32 x i8> %82, i32 28, !dbg !451
  %112 = extractelement <32 x i8> %82, i32 29, !dbg !451
  %113 = extractelement <32 x i8> %82, i32 30, !dbg !451
  %114 = extractelement <32 x i8> %82, i32 31, !dbg !451
  %localThreadId1920 = zext i8 %91 to i32, !dbg !451
  %115 = and i32 %localThreadId1920, 48, !dbg !451
  %116 = shl nuw nsw i32 %localThreadId1920, 5, !dbg !451
  %117 = and i32 %116, 480, !dbg !451
  %118 = or i32 %115, %81, !dbg !451
  %119 = or i32 %61, %117
  %.reass.reass = or i32 %119, %20
  %120 = add nuw nsw i32 %60, 24575
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %58, i32 %120, i32 4095, i32 24575, i32 %.reass.reass, i32 %118, i32 16, i32 16, i32 16, i32 2, i1 false, i1 false, i32 4)
  %121 = or i32 %72, %69, !dbg !455
  %122 = add nuw nsw i32 %68, 8191
  %123 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v16i16(i64 %66, i32 %122, i32 3, i32 8191, i32 %121, i32 %19, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %124 = extractelement <16 x i16> %123, i32 0, !dbg !455
  %125 = extractelement <16 x i16> %123, i32 1, !dbg !455
  %126 = extractelement <16 x i16> %123, i32 2, !dbg !455
  %127 = extractelement <16 x i16> %123, i32 3, !dbg !455
  %128 = extractelement <16 x i16> %123, i32 4, !dbg !455
  %129 = extractelement <16 x i16> %123, i32 5, !dbg !455
  %130 = extractelement <16 x i16> %123, i32 6, !dbg !455
  %131 = extractelement <16 x i16> %123, i32 7, !dbg !455
  %132 = extractelement <16 x i16> %123, i32 8, !dbg !455
  %133 = extractelement <16 x i16> %123, i32 9, !dbg !455
  %134 = extractelement <16 x i16> %123, i32 10, !dbg !455
  %135 = extractelement <16 x i16> %123, i32 11, !dbg !455
  %136 = extractelement <16 x i16> %123, i32 12, !dbg !455
  %137 = extractelement <16 x i16> %123, i32 13, !dbg !455
  %138 = extractelement <16 x i16> %123, i32 14, !dbg !455
  %139 = extractelement <16 x i16> %123, i32 15, !dbg !455
  %140 = insertelement <8 x i16> undef, i16 %124, i32 0, !dbg !455
  %141 = insertelement <8 x i16> %140, i16 %125, i32 1, !dbg !455
  %142 = insertelement <8 x i16> %141, i16 %126, i32 2, !dbg !455
  %143 = insertelement <8 x i16> %142, i16 %127, i32 3, !dbg !455
  %144 = insertelement <8 x i16> %143, i16 %128, i32 4, !dbg !455
  %145 = insertelement <8 x i16> %144, i16 %129, i32 5, !dbg !455
  %146 = insertelement <8 x i16> %145, i16 %130, i32 6, !dbg !455
  %147 = insertelement <8 x i16> %146, i16 %131, i32 7, !dbg !455
  %148 = insertelement <8 x i16> undef, i16 %132, i32 0, !dbg !455
  %149 = insertelement <8 x i16> %148, i16 %133, i32 1, !dbg !455
  %150 = insertelement <8 x i16> %149, i16 %134, i32 2, !dbg !455
  %151 = insertelement <8 x i16> %150, i16 %135, i32 3, !dbg !455
  %152 = insertelement <8 x i16> %151, i16 %136, i32 4, !dbg !455
  %153 = insertelement <8 x i16> %152, i16 %137, i32 5, !dbg !455
  %154 = insertelement <8 x i16> %153, i16 %138, i32 6, !dbg !455
  %155 = insertelement <8 x i16> %154, i16 %139, i32 7, !dbg !455
  %156 = or i32 %121, 32, !dbg !455
  %157 = add nuw nsw i32 %68, 8191
  %158 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v16i16(i64 %66, i32 %157, i32 3, i32 8191, i32 %156, i32 %19, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %159 = extractelement <16 x i16> %158, i32 0, !dbg !455
  %160 = extractelement <16 x i16> %158, i32 1, !dbg !455
  %161 = extractelement <16 x i16> %158, i32 2, !dbg !455
  %162 = extractelement <16 x i16> %158, i32 3, !dbg !455
  %163 = extractelement <16 x i16> %158, i32 4, !dbg !455
  %164 = extractelement <16 x i16> %158, i32 5, !dbg !455
  %165 = extractelement <16 x i16> %158, i32 6, !dbg !455
  %166 = extractelement <16 x i16> %158, i32 7, !dbg !455
  %167 = extractelement <16 x i16> %158, i32 8, !dbg !455
  %168 = extractelement <16 x i16> %158, i32 9, !dbg !455
  %169 = extractelement <16 x i16> %158, i32 10, !dbg !455
  %170 = extractelement <16 x i16> %158, i32 11, !dbg !455
  %171 = extractelement <16 x i16> %158, i32 12, !dbg !455
  %172 = extractelement <16 x i16> %158, i32 13, !dbg !455
  %173 = extractelement <16 x i16> %158, i32 14, !dbg !455
  %174 = extractelement <16 x i16> %158, i32 15, !dbg !455
  %175 = insertelement <8 x i16> undef, i16 %159, i32 0, !dbg !455
  %176 = insertelement <8 x i16> %175, i16 %160, i32 1, !dbg !455
  %177 = insertelement <8 x i16> %176, i16 %161, i32 2, !dbg !455
  %178 = insertelement <8 x i16> %177, i16 %162, i32 3, !dbg !455
  %179 = insertelement <8 x i16> %178, i16 %163, i32 4, !dbg !455
  %180 = insertelement <8 x i16> %179, i16 %164, i32 5, !dbg !455
  %181 = insertelement <8 x i16> %180, i16 %165, i32 6, !dbg !455
  %182 = insertelement <8 x i16> %181, i16 %166, i32 7, !dbg !455
  %183 = insertelement <8 x i16> undef, i16 %167, i32 0, !dbg !455
  %184 = insertelement <8 x i16> %183, i16 %168, i32 1, !dbg !455
  %185 = insertelement <8 x i16> %184, i16 %169, i32 2, !dbg !455
  %186 = insertelement <8 x i16> %185, i16 %170, i32 3, !dbg !455
  %187 = insertelement <8 x i16> %186, i16 %171, i32 4, !dbg !455
  %188 = insertelement <8 x i16> %187, i16 %172, i32 5, !dbg !455
  %189 = insertelement <8 x i16> %188, i16 %173, i32 6, !dbg !455
  %190 = insertelement <8 x i16> %189, i16 %174, i32 7, !dbg !455
  %191 = bitcast <8 x i32> %r0 to <32 x i8>, !dbg !451
  %192 = extractelement <32 x i8> %191, i32 0, !dbg !451
  %193 = extractelement <32 x i8> %191, i32 1, !dbg !451
  %194 = extractelement <32 x i8> %191, i32 2, !dbg !451
  %195 = extractelement <32 x i8> %191, i32 3, !dbg !451
  %196 = extractelement <32 x i8> %191, i32 4, !dbg !451
  %197 = extractelement <32 x i8> %191, i32 5, !dbg !451
  %198 = extractelement <32 x i8> %191, i32 6, !dbg !451
  %199 = extractelement <32 x i8> %191, i32 7, !dbg !451
  %200 = extractelement <32 x i8> %191, i32 8, !dbg !451
  %201 = extractelement <32 x i8> %191, i32 9, !dbg !451
  %202 = extractelement <32 x i8> %191, i32 10, !dbg !451
  %203 = extractelement <32 x i8> %191, i32 11, !dbg !451
  %204 = extractelement <32 x i8> %191, i32 12, !dbg !451
  %205 = extractelement <32 x i8> %191, i32 13, !dbg !451
  %206 = extractelement <32 x i8> %191, i32 14, !dbg !451
  %207 = extractelement <32 x i8> %191, i32 15, !dbg !451
  %208 = extractelement <32 x i8> %191, i32 16, !dbg !451
  %209 = extractelement <32 x i8> %191, i32 17, !dbg !451
  %210 = extractelement <32 x i8> %191, i32 18, !dbg !451
  %211 = extractelement <32 x i8> %191, i32 19, !dbg !451
  %212 = extractelement <32 x i8> %191, i32 20, !dbg !451
  %213 = extractelement <32 x i8> %191, i32 21, !dbg !451
  %214 = extractelement <32 x i8> %191, i32 22, !dbg !451
  %215 = extractelement <32 x i8> %191, i32 23, !dbg !451
  %216 = extractelement <32 x i8> %191, i32 24, !dbg !451
  %217 = extractelement <32 x i8> %191, i32 25, !dbg !451
  %218 = extractelement <32 x i8> %191, i32 26, !dbg !451
  %219 = extractelement <32 x i8> %191, i32 27, !dbg !451
  %220 = extractelement <32 x i8> %191, i32 28, !dbg !451
  %221 = extractelement <32 x i8> %191, i32 29, !dbg !451
  %222 = extractelement <32 x i8> %191, i32 30, !dbg !451
  %223 = extractelement <32 x i8> %191, i32 31, !dbg !451
  %localThreadId2223 = zext i8 %200 to i32, !dbg !451
  %224 = shl nuw nsw i32 %localThreadId2223, 4, !dbg !451
  %225 = and i32 %224, 496, !dbg !451
  %226 = add i32 %70, %225, !dbg !451
  %227 = add nuw nsw i32 %60, 24575
  %228 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v16i32(i64 %58, i32 %227, i32 4095, i32 24575, i32 %226, i32 %72, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %229 = extractelement <16 x i32> %228, i32 0, !dbg !451
  %230 = extractelement <16 x i32> %228, i32 1, !dbg !451
  %231 = extractelement <16 x i32> %228, i32 2, !dbg !451
  %232 = extractelement <16 x i32> %228, i32 3, !dbg !451
  %233 = extractelement <16 x i32> %228, i32 4, !dbg !451
  %234 = extractelement <16 x i32> %228, i32 5, !dbg !451
  %235 = extractelement <16 x i32> %228, i32 6, !dbg !451
  %236 = extractelement <16 x i32> %228, i32 7, !dbg !451
  %237 = extractelement <16 x i32> %228, i32 8, !dbg !451
  %238 = extractelement <16 x i32> %228, i32 9, !dbg !451
  %239 = extractelement <16 x i32> %228, i32 10, !dbg !451
  %240 = extractelement <16 x i32> %228, i32 11, !dbg !451
  %241 = extractelement <16 x i32> %228, i32 12, !dbg !451
  %242 = extractelement <16 x i32> %228, i32 13, !dbg !451
  %243 = extractelement <16 x i32> %228, i32 14, !dbg !451
  %244 = extractelement <16 x i32> %228, i32 15, !dbg !451
  %245 = insertelement <8 x i32> undef, i32 %229, i32 0, !dbg !451
  %246 = insertelement <8 x i32> %245, i32 %230, i32 1, !dbg !451
  %247 = insertelement <8 x i32> %246, i32 %231, i32 2, !dbg !451
  %248 = insertelement <8 x i32> %247, i32 %232, i32 3, !dbg !451
  %249 = insertelement <8 x i32> %248, i32 %233, i32 4, !dbg !451
  %250 = insertelement <8 x i32> %249, i32 %234, i32 5, !dbg !451
  %251 = insertelement <8 x i32> %250, i32 %235, i32 6, !dbg !451
  %252 = insertelement <8 x i32> %251, i32 %236, i32 7, !dbg !451
  %253 = insertelement <8 x i32> undef, i32 %237, i32 0, !dbg !451
  %254 = insertelement <8 x i32> %253, i32 %238, i32 1, !dbg !451
  %255 = insertelement <8 x i32> %254, i32 %239, i32 2, !dbg !451
  %256 = insertelement <8 x i32> %255, i32 %240, i32 3, !dbg !451
  %257 = insertelement <8 x i32> %256, i32 %241, i32 4, !dbg !451
  %258 = insertelement <8 x i32> %257, i32 %242, i32 5, !dbg !451
  %259 = insertelement <8 x i32> %258, i32 %243, i32 6, !dbg !451
  %260 = insertelement <8 x i32> %259, i32 %244, i32 7, !dbg !451
  %261 = or i32 %72, 32, !dbg !451
  %262 = add nuw nsw i32 %60, 24575
  %263 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v16i32(i64 %58, i32 %262, i32 4095, i32 24575, i32 %226, i32 %261, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %264 = extractelement <16 x i32> %263, i32 0, !dbg !451
  %265 = extractelement <16 x i32> %263, i32 1, !dbg !451
  %266 = extractelement <16 x i32> %263, i32 2, !dbg !451
  %267 = extractelement <16 x i32> %263, i32 3, !dbg !451
  %268 = extractelement <16 x i32> %263, i32 4, !dbg !451
  %269 = extractelement <16 x i32> %263, i32 5, !dbg !451
  %270 = extractelement <16 x i32> %263, i32 6, !dbg !451
  %271 = extractelement <16 x i32> %263, i32 7, !dbg !451
  %272 = extractelement <16 x i32> %263, i32 8, !dbg !451
  %273 = extractelement <16 x i32> %263, i32 9, !dbg !451
  %274 = extractelement <16 x i32> %263, i32 10, !dbg !451
  %275 = extractelement <16 x i32> %263, i32 11, !dbg !451
  %276 = extractelement <16 x i32> %263, i32 12, !dbg !451
  %277 = extractelement <16 x i32> %263, i32 13, !dbg !451
  %278 = extractelement <16 x i32> %263, i32 14, !dbg !451
  %279 = extractelement <16 x i32> %263, i32 15, !dbg !451
  %280 = insertelement <8 x i32> undef, i32 %264, i32 0, !dbg !451
  %281 = insertelement <8 x i32> %280, i32 %265, i32 1, !dbg !451
  %282 = insertelement <8 x i32> %281, i32 %266, i32 2, !dbg !451
  %283 = insertelement <8 x i32> %282, i32 %267, i32 3, !dbg !451
  %284 = insertelement <8 x i32> %283, i32 %268, i32 4, !dbg !451
  %285 = insertelement <8 x i32> %284, i32 %269, i32 5, !dbg !451
  %286 = insertelement <8 x i32> %285, i32 %270, i32 6, !dbg !451
  %287 = insertelement <8 x i32> %286, i32 %271, i32 7, !dbg !451
  %288 = insertelement <8 x i32> undef, i32 %272, i32 0, !dbg !451
  %289 = insertelement <8 x i32> %288, i32 %273, i32 1, !dbg !451
  %290 = insertelement <8 x i32> %289, i32 %274, i32 2, !dbg !451
  %291 = insertelement <8 x i32> %290, i32 %275, i32 3, !dbg !451
  %292 = insertelement <8 x i32> %291, i32 %276, i32 4, !dbg !451
  %293 = insertelement <8 x i32> %292, i32 %277, i32 5, !dbg !451
  %294 = insertelement <8 x i32> %293, i32 %278, i32 6, !dbg !451
  %295 = insertelement <8 x i32> %294, i32 %279, i32 7, !dbg !451
  %296 = insertelement <8 x float> undef, float %73, i32 0, !dbg !453
  %297 = insertelement <8 x float> %296, float %74, i32 1, !dbg !453
  %298 = insertelement <8 x float> %297, float %75, i32 2, !dbg !453
  %299 = insertelement <8 x float> %298, float %76, i32 3, !dbg !453
  %300 = insertelement <8 x float> %299, float %77, i32 4, !dbg !453
  %301 = insertelement <8 x float> %300, float %78, i32 5, !dbg !453
  %302 = insertelement <8 x float> %301, float %79, i32 6, !dbg !453
  %303 = insertelement <8 x float> %302, float %80, i32 7, !dbg !453
  %304 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %303, <8 x i16> %147, <8 x i32> %252, i32 11, i32 11, i32 8, i32 8, i1 false)
  %305 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %304, <8 x i16> %155, <8 x i32> %260, i32 11, i32 11, i32 8, i32 8, i1 false)
  %306 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %305, <8 x i16> %182, <8 x i32> %287, i32 11, i32 11, i32 8, i32 8, i1 false)
  %307 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %306, <8 x i16> %190, <8 x i32> %295, i32 11, i32 11, i32 8, i32 8, i1 false)
  %308 = extractelement <8 x float> %307, i32 0
  %309 = extractelement <8 x float> %307, i32 1
  %310 = extractelement <8 x float> %307, i32 2
  %311 = extractelement <8 x float> %307, i32 3
  %312 = extractelement <8 x float> %307, i32 4
  %313 = extractelement <8 x float> %307, i32 5
  %314 = extractelement <8 x float> %307, i32 6
  %315 = extractelement <8 x float> %307, i32 7
  %316 = icmp ult i32 %72, 4032, !dbg !452
  br i1 %316, label %71, label %317, !dbg !452

317:                                              ; preds = %71
  %318 = bitcast <8 x i32> %r0 to <32 x i8>, !dbg !456
  %319 = extractelement <32 x i8> %318, i32 0, !dbg !456
  %320 = extractelement <32 x i8> %318, i32 1, !dbg !456
  %321 = extractelement <32 x i8> %318, i32 2, !dbg !456
  %322 = extractelement <32 x i8> %318, i32 3, !dbg !456
  %323 = extractelement <32 x i8> %318, i32 4, !dbg !456
  %324 = extractelement <32 x i8> %318, i32 5, !dbg !456
  %325 = extractelement <32 x i8> %318, i32 6, !dbg !456
  %326 = extractelement <32 x i8> %318, i32 7, !dbg !456
  %327 = extractelement <32 x i8> %318, i32 8, !dbg !456
  %328 = extractelement <32 x i8> %318, i32 9, !dbg !456
  %329 = extractelement <32 x i8> %318, i32 10, !dbg !456
  %330 = extractelement <32 x i8> %318, i32 11, !dbg !456
  %331 = extractelement <32 x i8> %318, i32 12, !dbg !456
  %332 = extractelement <32 x i8> %318, i32 13, !dbg !456
  %333 = extractelement <32 x i8> %318, i32 14, !dbg !456
  %334 = extractelement <32 x i8> %318, i32 15, !dbg !456
  %335 = extractelement <32 x i8> %318, i32 16, !dbg !456
  %336 = extractelement <32 x i8> %318, i32 17, !dbg !456
  %337 = extractelement <32 x i8> %318, i32 18, !dbg !456
  %338 = extractelement <32 x i8> %318, i32 19, !dbg !456
  %339 = extractelement <32 x i8> %318, i32 20, !dbg !456
  %340 = extractelement <32 x i8> %318, i32 21, !dbg !456
  %341 = extractelement <32 x i8> %318, i32 22, !dbg !456
  %342 = extractelement <32 x i8> %318, i32 23, !dbg !456
  %343 = extractelement <32 x i8> %318, i32 24, !dbg !456
  %344 = extractelement <32 x i8> %318, i32 25, !dbg !456
  %345 = extractelement <32 x i8> %318, i32 26, !dbg !456
  %346 = extractelement <32 x i8> %318, i32 27, !dbg !456
  %347 = extractelement <32 x i8> %318, i32 28, !dbg !456
  %348 = extractelement <32 x i8> %318, i32 29, !dbg !456
  %349 = extractelement <32 x i8> %318, i32 30, !dbg !456
  %350 = extractelement <32 x i8> %318, i32 31, !dbg !456
  %localThreadId2829 = zext i8 %327 to i32, !dbg !456
  %351 = and i16 %localIdX, 512, !dbg !456
  %352 = icmp eq i16 %351, 0, !dbg !456
  %353 = shl nuw nsw i32 %localThreadId2829, 4, !dbg !456
  %354 = and i32 %353, 496, !dbg !456
  %355 = select i1 %352, i32 %19, i32 4, !dbg !456
  %356 = bitcast float %308 to i32, !dbg !456
  %357 = bitcast float %309 to i32, !dbg !456
  %358 = bitcast float %310 to i32, !dbg !456
  %359 = bitcast float %311 to i32, !dbg !456
  %360 = bitcast float %312 to i32, !dbg !456
  %361 = bitcast float %313 to i32, !dbg !456
  %362 = bitcast float %314 to i32, !dbg !456
  %363 = bitcast float %315 to i32, !dbg !456
  %364 = insertelement <8 x i32> undef, i32 %356, i32 0, !dbg !456
  %365 = insertelement <8 x i32> %364, i32 %357, i32 1, !dbg !456
  %366 = insertelement <8 x i32> %365, i32 %358, i32 2, !dbg !456
  %367 = insertelement <8 x i32> %366, i32 %359, i32 3, !dbg !456
  %368 = insertelement <8 x i32> %367, i32 %360, i32 4, !dbg !456
  %369 = insertelement <8 x i32> %368, i32 %361, i32 5, !dbg !456
  %370 = insertelement <8 x i32> %369, i32 %362, i32 6, !dbg !456
  %371 = insertelement <8 x i32> %370, i32 %363, i32 7, !dbg !456
  %372 = ptrtoint i8 addrspace(1)* %2 to i64, !dbg !456
  %373 = and i64 %372, -64, !dbg !456
  %374 = trunc i64 %372 to i32, !dbg !456
  %375 = and i32 %374, 63, !dbg !456
  %376 = lshr i32 %375, 2, !dbg !456
  %377 = or i32 %376, %354, !dbg !456
  %378 = or i32 %377, %20, !dbg !456
  %379 = add nuw nsw i32 %375, 49151
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %373, i32 %379, i32 3, i32 49151, i32 %378, i32 %355, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %371)
  ret void, !dbg !457
}

declare <8 x float> @__builtin_IB_sub_group16_fdpas_f_f_bf_bf_8_8(<8 x float>, <8 x i16>, <8 x i32>)

; Function Attrs: convergent nounwind
declare spir_func <16 x i16> @__builtin_IB_subgroup_block_read_cacheopts_u16_m8k16v2(i64 noundef, i32 noundef, i32 noundef, i32 noundef, <2 x i32> noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func <16 x i32> @__builtin_IB_subgroup_block_read_cacheopts_transform_u16_k32n16v1(i64 noundef, i32 noundef, i32 noundef, i32 noundef, <2 x i32> noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @__builtin_IB_subgroup_block_read_prefetch_u16_m16k16v2(i64 noundef, i32 noundef, i32 noundef, i32 noundef, <2 x i32> noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent nounwind
declare spir_func void @__builtin_IB_subgroup_block_write_cacheopts_u32_m8k16v1(i64 noundef, i32 noundef, i32 noundef, i32 noundef, <2 x i32> noundef, <8 x i32> noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_group_id(i32 noundef) local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_x() local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_y() local_unnamed_addr #2

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_id_z() local_unnamed_addr #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.assume(i1 noundef) #3

; Function Attrs: convergent mustprogress nofree nounwind willreturn memory(none)
declare spir_func i32 @__builtin_IB_get_local_thread_id() local_unnamed_addr #2

declare i32 @printf(i8 addrspace(2)*, ...)

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Arg 4: 
; Arg 5: 
; Arg 6: 
; Arg 7: 
; Function Attrs: convergent nounwind willreturn memory(none)
declare <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float>, <8 x i16>, <8 x i32>, i32, i32, i32, i32, i1) #4

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Arg 4: 
; Arg 5: 
; Arg 6: 
; Arg 7: 
; Arg 8: 
; Arg 9: 
; Arg 10: 
; Arg 11: 
; Arg 12: 
; Function Attrs: nounwind memory(readwrite)
declare void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #5

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Arg 4: 
; Arg 5: 
; Arg 6: 
; Arg 7: 
; Arg 8: 
; Arg 9: 
; Arg 10: 
; Arg 11: 
; Arg 12: 
; Function Attrs: nounwind memory(readwrite)
declare <16 x i16> @llvm.genx.GenISA.LSC2DBlockRead.v16i16(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #5

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Arg 4: 
; Arg 5: 
; Arg 6: 
; Arg 7: 
; Arg 8: 
; Arg 9: 
; Arg 10: 
; Arg 11: 
; Arg 12: 
; Function Attrs: nounwind memory(readwrite)
declare <16 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v16i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32) #5

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Arg 4: 
; Arg 5: 
; Arg 6: 
; Arg 7: 
; Arg 8: 
; Arg 9: 
; Arg 10: 
; Arg 11: 
; Arg 12: 
; Arg 13: 
; Function Attrs: nounwind memory(readwrite)
declare void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i1, i1, i32, <8 x i32>) #5

; Function Desc: 
; Output: 
; Function Attrs: nounwind willreturn memory(none)
declare void @llvm.genx.GenISA.CatchAllDebugLine() #6

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nounwind willreturn memory(none) }
attributes #5 = { nounwind memory(readwrite) }
attributes #6 = { nounwind willreturn memory(none) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}
!spirv.MemoryModel = !{!5}
!spirv.Source = !{!6}
!spirv.Generator = !{!7}
!igc.functions = !{!8}
!IGCMetadata = !{!35}
!opencl.ocl.version = !{!430, !430, !430, !430, !430, !430, !430, !430, !430, !430, !430, !430, !430}
!opencl.spir.version = !{!430, !430, !430, !430, !430, !430, !430, !430, !430, !430, !430, !430, !430}
!llvm.ident = !{!431, !431, !431, !431, !431, !431, !431, !431, !431, !431, !431, !431, !431}

!0 = !{i32 7, !"Dwarf Version", i32 0}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = distinct !DICompileUnit(language: DW_LANG_OpenCL, file: !4, producer: "triton", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!4 = !DIFile(filename: "gemm_benchmark.py", directory: "/home/gta/workspace/intel-xpu-backend-for-triton/benchmarks/triton_kernels_benchmark")
!5 = !{i32 2, i32 2}
!6 = !{i32 3, i32 100000}
!7 = !{i16 6, i16 14}
!8 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @matmul_kernel_with_tensor_descriptors, !9}
!9 = !{!10, !11, !33, !34}
!10 = !{!"function_type", i32 0}
!11 = !{!"implicit_arg_desc", !12, !13, !14, !15, !16, !17, !18, !20, !22, !24, !26, !28, !29, !30, !31, !32}
!12 = !{i32 0}
!13 = !{i32 2}
!14 = !{i32 8}
!15 = !{i32 9}
!16 = !{i32 10}
!17 = !{i32 13}
!18 = !{i32 15, !19}
!19 = !{!"explicit_arg_num", i32 0}
!20 = !{i32 15, !21}
!21 = !{!"explicit_arg_num", i32 1}
!22 = !{i32 15, !23}
!23 = !{!"explicit_arg_num", i32 2}
!24 = !{i32 15, !25}
!25 = !{!"explicit_arg_num", i32 3}
!26 = !{i32 15, !27}
!27 = !{!"explicit_arg_num", i32 4}
!28 = !{i32 59, !19}
!29 = !{i32 59, !21}
!30 = !{i32 59, !23}
!31 = !{i32 59, !25}
!32 = !{i32 59, !27}
!33 = !{!"thread_group_size", i32 1024, i32 1, i32 1}
!34 = !{!"sub_group_size", i32 16}
!35 = !{!"ModuleMD", !36, !37, !143, !264, !295, !312, !333, !343, !345, !346, !361, !362, !363, !364, !368, !369, !376, !377, !378, !379, !380, !381, !382, !383, !384, !385, !386, !388, !392, !393, !394, !395, !396, !397, !398, !399, !400, !401, !402, !403, !404, !405, !214, !406, !407, !408, !410, !412, !415, !416, !417, !419, !420, !421, !426, !427, !428, !429}
!36 = !{!"isPrecise", i1 false}
!37 = !{!"compOpt", !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142}
!38 = !{!"DenormsAreZero", i1 false}
!39 = !{!"BFTFDenormsAreZero", i1 false}
!40 = !{!"CorrectlyRoundedDivSqrt", i1 false}
!41 = !{!"OptDisable", i1 false}
!42 = !{!"MadEnable", i1 true}
!43 = !{!"NoSignedZeros", i1 false}
!44 = !{!"NoNaNs", i1 false}
!45 = !{!"FloatDenormMode16", !"FLOAT_DENORM_RETAIN"}
!46 = !{!"FloatDenormMode32", !"FLOAT_DENORM_RETAIN"}
!47 = !{!"FloatDenormMode64", !"FLOAT_DENORM_RETAIN"}
!48 = !{!"FloatDenormModeBFTF", !"FLOAT_DENORM_RETAIN"}
!49 = !{!"FloatRoundingMode", i32 0}
!50 = !{!"FloatCvtIntRoundingMode", i32 3}
!51 = !{!"LoadCacheDefault", i32 4}
!52 = !{!"StoreCacheDefault", i32 2}
!53 = !{!"VISAPreSchedRPThreshold", i32 0}
!54 = !{!"VISAPreSchedCtrl", i32 0}
!55 = !{!"SetLoopUnrollThreshold", i32 0}
!56 = !{!"UnsafeMathOptimizations", i1 false}
!57 = !{!"disableCustomUnsafeOpts", i1 false}
!58 = !{!"disableReducePow", i1 false}
!59 = !{!"disableSqrtOpt", i1 false}
!60 = !{!"FiniteMathOnly", i1 false}
!61 = !{!"FastRelaxedMath", i1 false}
!62 = !{!"DashGSpecified", i1 false}
!63 = !{!"FastCompilation", i1 false}
!64 = !{!"UseScratchSpacePrivateMemory", i1 true}
!65 = !{!"RelaxedBuiltins", i1 false}
!66 = !{!"SubgroupIndependentForwardProgressRequired", i1 true}
!67 = !{!"GreaterThan2GBBufferRequired", i1 true}
!68 = !{!"GreaterThan4GBBufferRequired", i1 true}
!69 = !{!"DisableA64WA", i1 false}
!70 = !{!"ForceEnableA64WA", i1 false}
!71 = !{!"PushConstantsEnable", i1 true}
!72 = !{!"HasPositivePointerOffset", i1 false}
!73 = !{!"HasBufferOffsetArg", i1 true}
!74 = !{!"BufferOffsetArgOptional", i1 true}
!75 = !{!"replaceGlobalOffsetsByZero", i1 false}
!76 = !{!"forcePixelShaderSIMDMode", i32 0}
!77 = !{!"forceTotalGRFNum", i32 0}
!78 = !{!"pixelShaderDoNotAbortOnSpill", i1 false}
!79 = !{!"UniformWGS", i1 false}
!80 = !{!"disableVertexComponentPacking", i1 false}
!81 = !{!"disablePartialVertexComponentPacking", i1 false}
!82 = !{!"PreferBindlessImages", i1 true}
!83 = !{!"UseBindlessMode", i1 true}
!84 = !{!"UseLegacyBindlessMode", i1 false}
!85 = !{!"disableMathRefactoring", i1 false}
!86 = !{!"atomicBranch", i1 false}
!87 = !{!"spillCompression", i1 false}
!88 = !{!"DisableEarlyOut", i1 false}
!89 = !{!"ForceInt32DivRemEmu", i1 false}
!90 = !{!"ForceInt32DivRemEmuSP", i1 false}
!91 = !{!"DisableIntDivRemIncrementReduction", i1 false}
!92 = !{!"DisableFastestSingleCSSIMD", i1 false}
!93 = !{!"DisableFastestLinearScan", i1 false}
!94 = !{!"UseStatelessforPrivateMemory", i1 false}
!95 = !{!"EnableTakeGlobalAddress", i1 false}
!96 = !{!"IsLibraryCompilation", i1 false}
!97 = !{!"LibraryCompileSIMDSize", i32 0}
!98 = !{!"FastVISACompile", i1 false}
!99 = !{!"MatchSinCosPi", i1 false}
!100 = !{!"ExcludeIRFromZEBinary", i1 false}
!101 = !{!"EmitZeBinVISASections", i1 false}
!102 = !{!"FP64GenEmulationEnabled", i1 false}
!103 = !{!"FP64GenConvEmulationEnabled", i1 false}
!104 = !{!"allowDisableRematforCS", i1 false}
!105 = !{!"DisableIncSpillCostAllAddrTaken", i1 false}
!106 = !{!"DisableCPSOmaskWA", i1 false}
!107 = !{!"DisableFastestGopt", i1 false}
!108 = !{!"WaForceHalfPromotionComputeShader", i1 false}
!109 = !{!"WaForceHalfPromotionPixelVertexShader", i1 false}
!110 = !{!"DisableConstantCoalescing", i1 false}
!111 = !{!"EnableUndefAlphaOutputAsRed", i1 true}
!112 = !{!"WaEnableALTModeVisaWA", i1 false}
!113 = !{!"EnableLdStCombineforLoad", i1 false}
!114 = !{!"EnableLdStCombinewithDummyLoad", i1 false}
!115 = !{!"ForceUniformBuffer", i1 false}
!116 = !{!"ForceUniformSurfaceSampler", i1 false}
!117 = !{!"EnableIndependentSharedMemoryFenceFunctionality", i1 false}
!118 = !{!"NewSpillCostFunction", i1 false}
!119 = !{!"EnableVRT", i1 false}
!120 = !{!"ForceLargeGRFNum4RQ", i1 false}
!121 = !{!"DisableEUFusion", i1 false}
!122 = !{!"DisableFDivToFMulInvOpt", i1 false}
!123 = !{!"initializePhiSampleSourceWA", i1 false}
!124 = !{!"WaDisableSubspanUseNoMaskForCB", i1 false}
!125 = !{!"DisableLoosenSimd32Occu", i1 false}
!126 = !{!"FastestS1Options", i32 0}
!127 = !{!"DisableFastestForWaveIntrinsicsCS", i1 false}
!128 = !{!"ForceLinearWalkOnLinearUAV", i1 false}
!129 = !{!"LscSamplerRouting", i32 0}
!130 = !{!"UseBarrierControlFlowOptimization", i1 false}
!131 = !{!"EnableDynamicRQManagement", i1 false}
!132 = !{!"WaDisablePayloadCoalescing", i1 false}
!133 = !{!"Quad8InputThreshold", i32 0}
!134 = !{!"UseResourceLoopUnrollNested", i1 false}
!135 = !{!"DisableLoopUnroll", i1 false}
!136 = !{!"ForcePushConstantMode", i32 0}
!137 = !{!"UseInstructionHoistingOptimization", i1 false}
!138 = !{!"DisableResourceLoopDestLifeTimeStart", i1 false}
!139 = !{!"ForceVRTGRFCeiling", i32 0}
!140 = !{!"DisableSamplerBackingByLSC", i32 0}
!141 = !{!"UseLinearScanRA", i1 false}
!142 = !{!"DisableConvertingAtomicIAddToIncDec", i1 false}
!143 = !{!"FuncMD", !144, !145}
!144 = !{!"FuncMDMap[0]", void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, <8 x i32>, <3 x i32>, i16, i16, i16, i8*, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)* @matmul_kernel_with_tensor_descriptors}
!145 = !{!"FuncMDValue[0]", !146, !147, !151, !152, !153, !176, !206, !207, !208, !209, !210, !211, !212, !213, !214, !215, !216, !217, !218, !219, !220, !221, !222, !223, !224, !225, !226, !232, !238, !244, !250, !256, !262, !263}
!146 = !{!"localOffsets"}
!147 = !{!"workGroupWalkOrder", !148, !149, !150}
!148 = !{!"dim0", i32 0}
!149 = !{!"dim1", i32 1}
!150 = !{!"dim2", i32 2}
!151 = !{!"funcArgs"}
!152 = !{!"functionType", !"KernelFunction"}
!153 = !{!"rtInfo", !154, !155, !156, !157, !158, !159, !160, !161, !162, !163, !164, !165, !166, !167, !168, !169, !171, !172, !173, !174, !175}
!154 = !{!"callableShaderType", !"NumberOfCallableShaderTypes"}
!155 = !{!"isContinuation", i1 false}
!156 = !{!"hasTraceRayPayload", i1 false}
!157 = !{!"hasHitAttributes", i1 false}
!158 = !{!"hasCallableData", i1 false}
!159 = !{!"ShaderStackSize", i32 0}
!160 = !{!"ShaderHash", i64 0}
!161 = !{!"ShaderName", !""}
!162 = !{!"ParentName", !""}
!163 = !{!"SlotNum", i1* null}
!164 = !{!"NOSSize", i32 0}
!165 = !{!"globalRootSignatureSize", i32 0}
!166 = !{!"Entries"}
!167 = !{!"SpillUnions"}
!168 = !{!"CustomHitAttrSizeInBytes", i32 0}
!169 = !{!"Types", !170}
!170 = !{!"FullFrameTys"}
!171 = !{!"Aliases"}
!172 = !{!"numSyncRTStacks", i32 0}
!173 = !{!"NumCoherenceHintBits", i32 0}
!174 = !{!"useSyncHWStack", i1 false}
!175 = !{!"OriginatingShaderName", !""}
!176 = !{!"resAllocMD", !177, !178, !179, !180, !205}
!177 = !{!"uavsNumType", i32 0}
!178 = !{!"srvsNumType", i32 0}
!179 = !{!"samplersNumType", i32 0}
!180 = !{!"argAllocMDList", !181, !185, !186, !187, !188, !189, !190, !191, !192, !193, !194, !195, !196, !197, !198, !199, !200, !201, !202, !203, !204}
!181 = !{!"argAllocMDListVec[0]", !182, !183, !184}
!182 = !{!"type", i32 0}
!183 = !{!"extensionType", i32 -1}
!184 = !{!"indexType", i32 -1}
!185 = !{!"argAllocMDListVec[1]", !182, !183, !184}
!186 = !{!"argAllocMDListVec[2]", !182, !183, !184}
!187 = !{!"argAllocMDListVec[3]", !182, !183, !184}
!188 = !{!"argAllocMDListVec[4]", !182, !183, !184}
!189 = !{!"argAllocMDListVec[5]", !182, !183, !184}
!190 = !{!"argAllocMDListVec[6]", !182, !183, !184}
!191 = !{!"argAllocMDListVec[7]", !182, !183, !184}
!192 = !{!"argAllocMDListVec[8]", !182, !183, !184}
!193 = !{!"argAllocMDListVec[9]", !182, !183, !184}
!194 = !{!"argAllocMDListVec[10]", !182, !183, !184}
!195 = !{!"argAllocMDListVec[11]", !182, !183, !184}
!196 = !{!"argAllocMDListVec[12]", !182, !183, !184}
!197 = !{!"argAllocMDListVec[13]", !182, !183, !184}
!198 = !{!"argAllocMDListVec[14]", !182, !183, !184}
!199 = !{!"argAllocMDListVec[15]", !182, !183, !184}
!200 = !{!"argAllocMDListVec[16]", !182, !183, !184}
!201 = !{!"argAllocMDListVec[17]", !182, !183, !184}
!202 = !{!"argAllocMDListVec[18]", !182, !183, !184}
!203 = !{!"argAllocMDListVec[19]", !182, !183, !184}
!204 = !{!"argAllocMDListVec[20]", !182, !183, !184}
!205 = !{!"inlineSamplersMD"}
!206 = !{!"maxByteOffsets"}
!207 = !{!"IsInitializer", i1 false}
!208 = !{!"IsFinalizer", i1 false}
!209 = !{!"CompiledSubGroupsNumber", i32 0}
!210 = !{!"hasInlineVmeSamplers", i1 false}
!211 = !{!"localSize", i32 0}
!212 = !{!"localIDPresent", i1 false}
!213 = !{!"groupIDPresent", i1 false}
!214 = !{!"privateMemoryPerWI", i32 0}
!215 = !{!"prevFPOffset", i32 0}
!216 = !{!"globalIDPresent", i1 false}
!217 = !{!"hasSyncRTCalls", i1 false}
!218 = !{!"hasPrintfCalls", i1 false}
!219 = !{!"requireAssertBuffer", i1 false}
!220 = !{!"requireSyncBuffer", i1 false}
!221 = !{!"hasIndirectCalls", i1 false}
!222 = !{!"hasNonKernelArgLoad", i1 false}
!223 = !{!"hasNonKernelArgStore", i1 false}
!224 = !{!"hasNonKernelArgAtomic", i1 false}
!225 = !{!"UserAnnotations"}
!226 = !{!"m_OpenCLArgAddressSpaces", !227, !228, !229, !230, !231}
!227 = !{!"m_OpenCLArgAddressSpacesVec[0]", i32 1}
!228 = !{!"m_OpenCLArgAddressSpacesVec[1]", i32 1}
!229 = !{!"m_OpenCLArgAddressSpacesVec[2]", i32 1}
!230 = !{!"m_OpenCLArgAddressSpacesVec[3]", i32 1}
!231 = !{!"m_OpenCLArgAddressSpacesVec[4]", i32 1}
!232 = !{!"m_OpenCLArgAccessQualifiers", !233, !234, !235, !236, !237}
!233 = !{!"m_OpenCLArgAccessQualifiersVec[0]", !"none"}
!234 = !{!"m_OpenCLArgAccessQualifiersVec[1]", !"none"}
!235 = !{!"m_OpenCLArgAccessQualifiersVec[2]", !"none"}
!236 = !{!"m_OpenCLArgAccessQualifiersVec[3]", !"none"}
!237 = !{!"m_OpenCLArgAccessQualifiersVec[4]", !"none"}
!238 = !{!"m_OpenCLArgTypes", !239, !240, !241, !242, !243}
!239 = !{!"m_OpenCLArgTypesVec[0]", !"char*"}
!240 = !{!"m_OpenCLArgTypesVec[1]", !"char*"}
!241 = !{!"m_OpenCLArgTypesVec[2]", !"char*"}
!242 = !{!"m_OpenCLArgTypesVec[3]", !"char*"}
!243 = !{!"m_OpenCLArgTypesVec[4]", !"char*"}
!244 = !{!"m_OpenCLArgBaseTypes", !245, !246, !247, !248, !249}
!245 = !{!"m_OpenCLArgBaseTypesVec[0]", !"char*"}
!246 = !{!"m_OpenCLArgBaseTypesVec[1]", !"char*"}
!247 = !{!"m_OpenCLArgBaseTypesVec[2]", !"char*"}
!248 = !{!"m_OpenCLArgBaseTypesVec[3]", !"char*"}
!249 = !{!"m_OpenCLArgBaseTypesVec[4]", !"char*"}
!250 = !{!"m_OpenCLArgTypeQualifiers", !251, !252, !253, !254, !255}
!251 = !{!"m_OpenCLArgTypeQualifiersVec[0]", !""}
!252 = !{!"m_OpenCLArgTypeQualifiersVec[1]", !""}
!253 = !{!"m_OpenCLArgTypeQualifiersVec[2]", !""}
!254 = !{!"m_OpenCLArgTypeQualifiersVec[3]", !""}
!255 = !{!"m_OpenCLArgTypeQualifiersVec[4]", !""}
!256 = !{!"m_OpenCLArgNames", !257, !258, !259, !260, !261}
!257 = !{!"m_OpenCLArgNamesVec[0]", !""}
!258 = !{!"m_OpenCLArgNamesVec[1]", !""}
!259 = !{!"m_OpenCLArgNamesVec[2]", !""}
!260 = !{!"m_OpenCLArgNamesVec[3]", !""}
!261 = !{!"m_OpenCLArgNamesVec[4]", !""}
!262 = !{!"m_OpenCLArgScalarAsPointers"}
!263 = !{!"m_OptsToDisablePerFunc"}
!264 = !{!"pushInfo", !265, !266, !267, !271, !272, !273, !274, !275, !276, !277, !278, !291, !292, !293, !294}
!265 = !{!"pushableAddresses"}
!266 = !{!"bindlessPushInfo"}
!267 = !{!"dynamicBufferInfo", !268, !269, !270}
!268 = !{!"firstIndex", i32 0}
!269 = !{!"numOffsets", i32 0}
!270 = !{!"forceDisabled", i1 false}
!271 = !{!"MaxNumberOfPushedBuffers", i32 0}
!272 = !{!"inlineConstantBufferSlot", i32 -1}
!273 = !{!"inlineConstantBufferOffset", i32 -1}
!274 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!275 = !{!"constants"}
!276 = !{!"inputs"}
!277 = !{!"constantReg"}
!278 = !{!"simplePushInfoArr", !279, !288, !289, !290}
!279 = !{!"simplePushInfoArrVec[0]", !280, !281, !282, !283, !284, !285, !286, !287}
!280 = !{!"cbIdx", i32 0}
!281 = !{!"pushableAddressGrfOffset", i32 -1}
!282 = !{!"pushableOffsetGrfOffset", i32 -1}
!283 = !{!"offset", i32 0}
!284 = !{!"size", i32 0}
!285 = !{!"isStateless", i1 false}
!286 = !{!"isBindless", i1 false}
!287 = !{!"simplePushLoads"}
!288 = !{!"simplePushInfoArrVec[1]", !280, !281, !282, !283, !284, !285, !286, !287}
!289 = !{!"simplePushInfoArrVec[2]", !280, !281, !282, !283, !284, !285, !286, !287}
!290 = !{!"simplePushInfoArrVec[3]", !280, !281, !282, !283, !284, !285, !286, !287}
!291 = !{!"simplePushBufferUsed", i32 0}
!292 = !{!"pushAnalysisWIInfos"}
!293 = !{!"inlineRTGlobalPtrOffset", i32 0}
!294 = !{!"rtSyncSurfPtrOffset", i32 0}
!295 = !{!"psInfo", !296, !297, !298, !299, !300, !301, !302, !303, !304, !305, !306, !307, !308, !309, !310, !311}
!296 = !{!"BlendStateDisabledMask", i8 0}
!297 = !{!"SkipSrc0Alpha", i1 false}
!298 = !{!"DualSourceBlendingDisabled", i1 false}
!299 = !{!"ForceEnableSimd32", i1 false}
!300 = !{!"DisableSimd32WithDiscard", i1 false}
!301 = !{!"outputDepth", i1 false}
!302 = !{!"outputStencil", i1 false}
!303 = !{!"outputMask", i1 false}
!304 = !{!"blendToFillEnabled", i1 false}
!305 = !{!"forceEarlyZ", i1 false}
!306 = !{!"hasVersionedLoop", i1 false}
!307 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!308 = !{!"NumSamples", i8 0}
!309 = !{!"blendOptimizationMode"}
!310 = !{!"colorOutputMask"}
!311 = !{!"WaDisableVRS", i1 false}
!312 = !{!"csInfo", !313, !314, !315, !316, !77, !53, !54, !317, !55, !318, !319, !320, !321, !322, !323, !324, !325, !326, !327, !328, !87, !329, !330, !331, !332}
!313 = !{!"maxWorkGroupSize", i32 0}
!314 = !{!"waveSize", i32 0}
!315 = !{!"ComputeShaderSecondCompile"}
!316 = !{!"forcedSIMDSize", i8 0}
!317 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!318 = !{!"forceSpillCompression", i1 false}
!319 = !{!"allowLowerSimd", i1 false}
!320 = !{!"disableSimd32Slicing", i1 false}
!321 = !{!"disableSplitOnSpill", i1 false}
!322 = !{!"enableNewSpillCostFunction", i1 false}
!323 = !{!"forceVISAPreSched", i1 false}
!324 = !{!"disableLocalIdOrderOptimizations", i1 false}
!325 = !{!"disableDispatchAlongY", i1 false}
!326 = !{!"neededThreadIdLayout", i1* null}
!327 = !{!"forceTileYWalk", i1 false}
!328 = !{!"atomicBranch", i32 0}
!329 = !{!"disableEarlyOut", i1 false}
!330 = !{!"walkOrderEnabled", i1 false}
!331 = !{!"walkOrderOverride", i32 0}
!332 = !{!"ResForHfPacking"}
!333 = !{!"msInfo", !334, !335, !336, !337, !338, !339, !340, !341, !342}
!334 = !{!"PrimitiveTopology", i32 3}
!335 = !{!"MaxNumOfPrimitives", i32 0}
!336 = !{!"MaxNumOfVertices", i32 0}
!337 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!338 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!339 = !{!"WorkGroupSize", i32 0}
!340 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!341 = !{!"IndexFormat", i32 6}
!342 = !{!"SubgroupSize", i32 0}
!343 = !{!"taskInfo", !344, !339, !340, !342}
!344 = !{!"MaxNumOfOutputs", i32 0}
!345 = !{!"NBarrierCnt", i32 0}
!346 = !{!"rtInfo", !347, !348, !349, !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !172}
!347 = !{!"RayQueryAllocSizeInBytes", i32 0}
!348 = !{!"NumContinuations", i32 0}
!349 = !{!"RTAsyncStackAddrspace", i32 -1}
!350 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!351 = !{!"SWHotZoneAddrspace", i32 -1}
!352 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!353 = !{!"SWStackAddrspace", i32 -1}
!354 = !{!"SWStackSurfaceStateOffset", i1* null}
!355 = !{!"RTSyncStackAddrspace", i32 -1}
!356 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!357 = !{!"doSyncDispatchRays", i1 false}
!358 = !{!"MemStyle", !"Xe"}
!359 = !{!"GlobalDataStyle", !"Xe"}
!360 = !{!"uberTileDimensions", i1* null}
!361 = !{!"CurUniqueIndirectIdx", i32 0}
!362 = !{!"inlineDynTextures"}
!363 = !{!"inlineResInfoData"}
!364 = !{!"immConstant", !365, !366, !367}
!365 = !{!"data"}
!366 = !{!"sizes"}
!367 = !{!"zeroIdxs"}
!368 = !{!"stringConstants"}
!369 = !{!"inlineBuffers", !370, !374, !375}
!370 = !{!"inlineBuffersVec[0]", !371, !372, !373}
!371 = !{!"alignment", i32 0}
!372 = !{!"allocSize", i64 0}
!373 = !{!"Buffer"}
!374 = !{!"inlineBuffersVec[1]", !371, !372, !373}
!375 = !{!"inlineBuffersVec[2]", !371, !372, !373}
!376 = !{!"GlobalPointerProgramBinaryInfos"}
!377 = !{!"ConstantPointerProgramBinaryInfos"}
!378 = !{!"GlobalBufferAddressRelocInfo"}
!379 = !{!"ConstantBufferAddressRelocInfo"}
!380 = !{!"forceLscCacheList"}
!381 = !{!"SrvMap"}
!382 = !{!"RasterizerOrderedByteAddressBuffer"}
!383 = !{!"RasterizerOrderedViews"}
!384 = !{!"MinNOSPushConstantSize", i32 0}
!385 = !{!"inlineProgramScopeOffsets"}
!386 = !{!"shaderData", !387}
!387 = !{!"numReplicas", i32 0}
!388 = !{!"URBInfo", !389, !390, !391}
!389 = !{!"has64BVertexHeaderInput", i1 false}
!390 = !{!"has64BVertexHeaderOutput", i1 false}
!391 = !{!"hasVertexHeader", i1 true}
!392 = !{!"UseBindlessImage", i1 true}
!393 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!394 = !{!"enableRangeReduce", i1 false}
!395 = !{!"allowMatchMadOptimizationforVS", i1 false}
!396 = !{!"disableMatchMadOptimizationForCS", i1 false}
!397 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!398 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!399 = !{!"statefulResourcesNotAliased", i1 false}
!400 = !{!"disableMixMode", i1 false}
!401 = !{!"genericAccessesResolved", i1 false}
!402 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!403 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!404 = !{!"disableSeparateScratchWA", i1 false}
!405 = !{!"enableRemoveUnusedTGMFence", i1 false}
!406 = !{!"PrivateMemoryPerFG"}
!407 = !{!"m_OptsToDisable"}
!408 = !{!"capabilities", !409}
!409 = !{!"globalVariableDecorationsINTEL", i1 false}
!410 = !{!"extensions", !411}
!411 = !{!"spvINTELBindlessImages", i1 false}
!412 = !{!"m_ShaderResourceViewMcsMask", !413, !414}
!413 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!414 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!415 = !{!"computedDepthMode", i32 0}
!416 = !{!"isHDCFastClearShader", i1 false}
!417 = !{!"argRegisterReservations", !418}
!418 = !{!"argRegisterReservationsVec[0]", i32 0}
!419 = !{!"SIMD16_SpillThreshold", i8 0}
!420 = !{!"SIMD32_SpillThreshold", i8 0}
!421 = !{!"m_CacheControlOption", !422, !423, !424, !425}
!422 = !{!"LscLoadCacheControlOverride", i8 0}
!423 = !{!"LscStoreCacheControlOverride", i8 0}
!424 = !{!"TgmLoadCacheControlOverride", i8 0}
!425 = !{!"TgmStoreCacheControlOverride", i8 0}
!426 = !{!"ModuleUsesBindless", i1 false}
!427 = !{!"predicationMap"}
!428 = !{!"lifeTimeStartMap"}
!429 = !{!"HitGroups"}
!430 = !{i32 2, i32 0}
!431 = !{!"clang version 16.0.6"}
!432 = distinct !DISubprogram(name: "matmul_kernel_with_tensor_descriptors", linkageName: "matmul_kernel_with_tensor_descriptors", scope: null, file: !4, line: 38, type: !433, scopeLine: 38, spFlags: DISPFlagDefinition | DISPFlagOptimized | DISPFlagMainSubprogram, unit: !3, templateParams: !437, retainedNodes: !437)
!433 = !DISubroutineType(types: !434)
!434 = !{null, !435, !435, !435, !435, !435}
!435 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !436, size: 64, dwarfAddressSpace: 1)
!436 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!437 = !{}
!438 = !DILocation(line: 38, scope: !432)
!439 = !DILocation(line: 54, column: 16, scope: !432)
!440 = !DILocation(line: 56, column: 24, scope: !432)
!441 = !{!442}
!442 = !{i32 4469}
!443 = !DILocation(line: 57, column: 29, scope: !432)
!444 = !DILocation(line: 58, column: 13, scope: !432)
!445 = !DILocation(line: 57, column: 28, scope: !432)
!446 = !DILocation(line: 57, column: 13, scope: !432)
!447 = !{!442, !448}
!448 = !{i32 4470}
!449 = !DILocation(line: 79, column: 30, scope: !432)
!450 = !DILocation(line: 83, column: 37, scope: !432)
!451 = !DILocation(line: 83, column: 17, scope: !432)
!452 = !DILocation(line: 75, column: 5, scope: !432)
!453 = !DILocation(line: 84, column: 24, scope: !432)
!454 = !DILocation(line: 85, column: 9, scope: !432)
!455 = !DILocation(line: 79, column: 17, scope: !432)
!456 = !DILocation(line: 90, column: 5, scope: !432)
!457 = !DILocation(line: 38, column: 1, scope: !432)
