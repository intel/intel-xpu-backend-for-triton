; ------------------------------------------------
; OCL_asm604c6b145d199062_optimized.ll
; ------------------------------------------------
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent nounwind
define spir_kernel void @matmul_kernel_with_tensor_descriptors(i8 addrspace(1)* align 1 %0, i8 addrspace(1)* align 1 %1, i8 addrspace(1)* align 1 %2, i8 addrspace(1)* nocapture readnone align 1 %3, i8 addrspace(1)* nocapture readnone align 1 %4, <8 x i32> %r0, <3 x i32> %globalOffset, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bindlessOffset, i32 %bindlessOffset5, i32 %bindlessOffset6, i32 %bindlessOffset7, i32 %bindlessOffset8) #0 !dbg !435 {
  %6 = extractelement <8 x i32> %r0, i64 1
  %q_appx = call i32 @llvm.genx.GenISA.umulH.i32(i32 %6, i32 -1431655765), !dbg !441
  %q_appx169 = lshr i32 %q_appx, 4, !dbg !441
  %7 = sub nsw i32 1, %q_appx169, !dbg !442, !spirv.Decorations !443
  %.neg = mul i32 %q_appx169, -24, !dbg !445
  %.decomposed = add i32 %.neg, %6, !dbg !445
  %8 = sdiv i32 %.decomposed, %7, !dbg !446
  %9 = mul i32 %8, %7, !dbg !447
  %.decomposed1 = sub i32 %.decomposed, %9, !dbg !447
  %10 = add nuw nsw i32 %q_appx169, %.decomposed1, !dbg !448, !spirv.Decorations !449
  %11 = shl nuw nsw i32 %10, 3, !dbg !451, !spirv.Decorations !449
  %12 = shl nsw i32 %8, 9, !dbg !452, !spirv.Decorations !443
  %13 = bitcast <8 x i32> %r0 to <32 x i8>, !dbg !453
  %14 = extractelement <32 x i8> %13, i64 8, !dbg !453
  %localThreadId17 = zext i8 %14 to i32, !dbg !453
  %15 = and i32 %localThreadId17, 48, !dbg !453
  %16 = shl nuw nsw i32 %localThreadId17, 5, !dbg !453
  %17 = and i32 %16, 480, !dbg !453
  %18 = ptrtoint i8 addrspace(1)* %1 to i64, !dbg !453
  %19 = and i64 %18, -64, !dbg !453
  %20 = trunc i64 %18 to i32, !dbg !453
  %21 = and i32 %20, 63, !dbg !453
  %22 = lshr i32 %21, 1, !dbg !453
  %23 = or i32 %22, %17, !dbg !453
  %24 = or i32 %23, %12, !dbg !453
  %25 = add nuw nsw i32 %21, 24575
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %15, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %26 = ptrtoint i8 addrspace(1)* %0 to i64
  %27 = and i64 %26, -64
  %28 = trunc i64 %26 to i32
  %29 = and i32 %28, 63
  %30 = lshr i32 %29, 1
  %31 = or i32 %12, %22
  %32 = shl nuw nsw i32 %localThreadId17, 4, !dbg !453
  %33 = and i32 %32, 496, !dbg !453
  %34 = add nuw nsw i32 %29, 8191
  %35 = add i32 %31, %33
  br label %._crit_edge, !dbg !454

._crit_edge:                                      ; preds = %._crit_edge, %5
  %36 = phi i32 [ 0, %5 ], [ %292, %._crit_edge ]
  %vectorized_phi = phi <8 x float> [ zeroinitializer, %5 ], [ %308, %._crit_edge ], !dbg !455
  %37 = or i32 %36, 64, !dbg !456
  %38 = or i32 %15, %37, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %38, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %39 = or i32 %36, %30, !dbg !457
  %Block2D_AddrPayload = call i32* @llvm.genx.GenISA.LSC2DBlockCreateAddrPayload.p0i32(i64 %27, i32 %34, i32 3, i32 8191, i32 0, i32 0, i32 16, i32 8, i32 2)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %39, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %40 = shufflevector <16 x i16> %Block2D_ReadAddrPayload, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %41 = shufflevector <16 x i16> %Block2D_ReadAddrPayload, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %42 = or i32 %36, 32, !dbg !457
  %43 = or i32 %42, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %43, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload44 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %44 = shufflevector <16 x i16> %Block2D_ReadAddrPayload44, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %45 = shufflevector <16 x i16> %Block2D_ReadAddrPayload44, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %Block2D_AddrPayload45 = call i32* @llvm.genx.GenISA.LSC2DBlockCreateAddrPayload.p0i32(i64 %19, i32 %25, i32 4095, i32 24575, i32 0, i32 0, i32 16, i32 32, i32 1)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %36, i1 false)
  %Block2D_ReadAddrPayload46 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %46 = shufflevector <16 x i32> %Block2D_ReadAddrPayload46, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %47 = shufflevector <16 x i32> %Block2D_ReadAddrPayload46, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %42, i1 false)
  %Block2D_ReadAddrPayload48 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %48 = shufflevector <16 x i32> %Block2D_ReadAddrPayload48, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %49 = shufflevector <16 x i32> %Block2D_ReadAddrPayload48, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %50 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %vectorized_phi, <8 x i16> %40, <8 x i32> %46, i32 11, i32 11, i32 8, i32 8, i1 false)
  %51 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %50, <8 x i16> %41, <8 x i32> %47, i32 11, i32 11, i32 8, i32 8, i1 false)
  %52 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %51, <8 x i16> %44, <8 x i32> %48, i32 11, i32 11, i32 8, i32 8, i1 false)
  %53 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %52, <8 x i16> %45, <8 x i32> %49, i32 11, i32 11, i32 8, i32 8, i1 false)
  %54 = or i32 %36, 128, !dbg !456
  %55 = or i32 %15, %54, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %55, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %56 = or i32 %37, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %56, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload50 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %57 = shufflevector <16 x i16> %Block2D_ReadAddrPayload50, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %58 = shufflevector <16 x i16> %Block2D_ReadAddrPayload50, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %59 = or i32 %36, 96, !dbg !457
  %60 = or i32 %59, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %60, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload52 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %61 = shufflevector <16 x i16> %Block2D_ReadAddrPayload52, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %62 = shufflevector <16 x i16> %Block2D_ReadAddrPayload52, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %37, i1 false)
  %Block2D_ReadAddrPayload54 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %63 = shufflevector <16 x i32> %Block2D_ReadAddrPayload54, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %64 = shufflevector <16 x i32> %Block2D_ReadAddrPayload54, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %59, i1 false)
  %Block2D_ReadAddrPayload56 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %65 = shufflevector <16 x i32> %Block2D_ReadAddrPayload56, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %66 = shufflevector <16 x i32> %Block2D_ReadAddrPayload56, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %67 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %53, <8 x i16> %57, <8 x i32> %63, i32 11, i32 11, i32 8, i32 8, i1 false)
  %68 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %67, <8 x i16> %58, <8 x i32> %64, i32 11, i32 11, i32 8, i32 8, i1 false)
  %69 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %68, <8 x i16> %61, <8 x i32> %65, i32 11, i32 11, i32 8, i32 8, i1 false)
  %70 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %69, <8 x i16> %62, <8 x i32> %66, i32 11, i32 11, i32 8, i32 8, i1 false)
  %71 = or i32 %36, 192, !dbg !456
  %72 = or i32 %15, %71, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %72, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %73 = or i32 %54, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %73, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload58 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %74 = shufflevector <16 x i16> %Block2D_ReadAddrPayload58, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %75 = shufflevector <16 x i16> %Block2D_ReadAddrPayload58, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %76 = or i32 %36, 160, !dbg !457
  %77 = or i32 %76, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %77, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload60 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %78 = shufflevector <16 x i16> %Block2D_ReadAddrPayload60, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %79 = shufflevector <16 x i16> %Block2D_ReadAddrPayload60, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %54, i1 false)
  %Block2D_ReadAddrPayload62 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %80 = shufflevector <16 x i32> %Block2D_ReadAddrPayload62, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %81 = shufflevector <16 x i32> %Block2D_ReadAddrPayload62, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %76, i1 false)
  %Block2D_ReadAddrPayload64 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %82 = shufflevector <16 x i32> %Block2D_ReadAddrPayload64, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %83 = shufflevector <16 x i32> %Block2D_ReadAddrPayload64, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %84 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %70, <8 x i16> %74, <8 x i32> %80, i32 11, i32 11, i32 8, i32 8, i1 false)
  %85 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %84, <8 x i16> %75, <8 x i32> %81, i32 11, i32 11, i32 8, i32 8, i1 false)
  %86 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %85, <8 x i16> %78, <8 x i32> %82, i32 11, i32 11, i32 8, i32 8, i1 false)
  %87 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %86, <8 x i16> %79, <8 x i32> %83, i32 11, i32 11, i32 8, i32 8, i1 false)
  %88 = or i32 %36, 256, !dbg !456
  %89 = or i32 %15, %88, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %89, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %90 = or i32 %71, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %90, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload66 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %91 = shufflevector <16 x i16> %Block2D_ReadAddrPayload66, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %92 = shufflevector <16 x i16> %Block2D_ReadAddrPayload66, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %93 = or i32 %36, 224, !dbg !457
  %94 = or i32 %93, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %94, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload68 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %95 = shufflevector <16 x i16> %Block2D_ReadAddrPayload68, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %96 = shufflevector <16 x i16> %Block2D_ReadAddrPayload68, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %71, i1 false)
  %Block2D_ReadAddrPayload70 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %97 = shufflevector <16 x i32> %Block2D_ReadAddrPayload70, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %98 = shufflevector <16 x i32> %Block2D_ReadAddrPayload70, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %93, i1 false)
  %Block2D_ReadAddrPayload72 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %99 = shufflevector <16 x i32> %Block2D_ReadAddrPayload72, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %100 = shufflevector <16 x i32> %Block2D_ReadAddrPayload72, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %101 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %87, <8 x i16> %91, <8 x i32> %97, i32 11, i32 11, i32 8, i32 8, i1 false)
  %102 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %101, <8 x i16> %92, <8 x i32> %98, i32 11, i32 11, i32 8, i32 8, i1 false)
  %103 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %102, <8 x i16> %95, <8 x i32> %99, i32 11, i32 11, i32 8, i32 8, i1 false)
  %104 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %103, <8 x i16> %96, <8 x i32> %100, i32 11, i32 11, i32 8, i32 8, i1 false)
  %105 = or i32 %36, 320, !dbg !456
  %106 = or i32 %15, %105, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %106, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %107 = or i32 %88, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %107, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload74 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %108 = shufflevector <16 x i16> %Block2D_ReadAddrPayload74, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %109 = shufflevector <16 x i16> %Block2D_ReadAddrPayload74, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %110 = or i32 %36, 288, !dbg !457
  %111 = or i32 %110, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %111, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload76 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %112 = shufflevector <16 x i16> %Block2D_ReadAddrPayload76, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %113 = shufflevector <16 x i16> %Block2D_ReadAddrPayload76, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %88, i1 false)
  %Block2D_ReadAddrPayload78 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %114 = shufflevector <16 x i32> %Block2D_ReadAddrPayload78, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %115 = shufflevector <16 x i32> %Block2D_ReadAddrPayload78, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %110, i1 false)
  %Block2D_ReadAddrPayload80 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %116 = shufflevector <16 x i32> %Block2D_ReadAddrPayload80, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %117 = shufflevector <16 x i32> %Block2D_ReadAddrPayload80, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %118 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %104, <8 x i16> %108, <8 x i32> %114, i32 11, i32 11, i32 8, i32 8, i1 false)
  %119 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %118, <8 x i16> %109, <8 x i32> %115, i32 11, i32 11, i32 8, i32 8, i1 false)
  %120 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %119, <8 x i16> %112, <8 x i32> %116, i32 11, i32 11, i32 8, i32 8, i1 false)
  %121 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %120, <8 x i16> %113, <8 x i32> %117, i32 11, i32 11, i32 8, i32 8, i1 false)
  %122 = or i32 %36, 384, !dbg !456
  %123 = or i32 %15, %122, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %123, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %124 = or i32 %105, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %124, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload82 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %125 = shufflevector <16 x i16> %Block2D_ReadAddrPayload82, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %126 = shufflevector <16 x i16> %Block2D_ReadAddrPayload82, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %127 = or i32 %36, 352, !dbg !457
  %128 = or i32 %127, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %128, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload84 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %129 = shufflevector <16 x i16> %Block2D_ReadAddrPayload84, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %130 = shufflevector <16 x i16> %Block2D_ReadAddrPayload84, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %105, i1 false)
  %Block2D_ReadAddrPayload86 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %131 = shufflevector <16 x i32> %Block2D_ReadAddrPayload86, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %132 = shufflevector <16 x i32> %Block2D_ReadAddrPayload86, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %127, i1 false)
  %Block2D_ReadAddrPayload88 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %133 = shufflevector <16 x i32> %Block2D_ReadAddrPayload88, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %134 = shufflevector <16 x i32> %Block2D_ReadAddrPayload88, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %135 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %121, <8 x i16> %125, <8 x i32> %131, i32 11, i32 11, i32 8, i32 8, i1 false)
  %136 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %135, <8 x i16> %126, <8 x i32> %132, i32 11, i32 11, i32 8, i32 8, i1 false)
  %137 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %136, <8 x i16> %129, <8 x i32> %133, i32 11, i32 11, i32 8, i32 8, i1 false)
  %138 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %137, <8 x i16> %130, <8 x i32> %134, i32 11, i32 11, i32 8, i32 8, i1 false)
  %139 = or i32 %36, 448, !dbg !456
  %140 = or i32 %15, %139, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %140, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %141 = or i32 %122, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %141, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload90 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %142 = shufflevector <16 x i16> %Block2D_ReadAddrPayload90, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %143 = shufflevector <16 x i16> %Block2D_ReadAddrPayload90, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %144 = or i32 %36, 416, !dbg !457
  %145 = or i32 %144, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %145, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload92 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %146 = shufflevector <16 x i16> %Block2D_ReadAddrPayload92, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %147 = shufflevector <16 x i16> %Block2D_ReadAddrPayload92, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %122, i1 false)
  %Block2D_ReadAddrPayload94 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %148 = shufflevector <16 x i32> %Block2D_ReadAddrPayload94, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %149 = shufflevector <16 x i32> %Block2D_ReadAddrPayload94, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %144, i1 false)
  %Block2D_ReadAddrPayload96 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %150 = shufflevector <16 x i32> %Block2D_ReadAddrPayload96, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %151 = shufflevector <16 x i32> %Block2D_ReadAddrPayload96, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %152 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %138, <8 x i16> %142, <8 x i32> %148, i32 11, i32 11, i32 8, i32 8, i1 false)
  %153 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %152, <8 x i16> %143, <8 x i32> %149, i32 11, i32 11, i32 8, i32 8, i1 false)
  %154 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %153, <8 x i16> %146, <8 x i32> %150, i32 11, i32 11, i32 8, i32 8, i1 false)
  %155 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %154, <8 x i16> %147, <8 x i32> %151, i32 11, i32 11, i32 8, i32 8, i1 false)
  %156 = or i32 %36, 512, !dbg !456
  %157 = or i32 %15, %156, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %157, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %158 = or i32 %139, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %158, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload98 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %159 = shufflevector <16 x i16> %Block2D_ReadAddrPayload98, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %160 = shufflevector <16 x i16> %Block2D_ReadAddrPayload98, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %161 = or i32 %36, 480, !dbg !457
  %162 = or i32 %161, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %162, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload100 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %163 = shufflevector <16 x i16> %Block2D_ReadAddrPayload100, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %164 = shufflevector <16 x i16> %Block2D_ReadAddrPayload100, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %139, i1 false)
  %Block2D_ReadAddrPayload102 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %165 = shufflevector <16 x i32> %Block2D_ReadAddrPayload102, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %166 = shufflevector <16 x i32> %Block2D_ReadAddrPayload102, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %161, i1 false)
  %Block2D_ReadAddrPayload104 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %167 = shufflevector <16 x i32> %Block2D_ReadAddrPayload104, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %168 = shufflevector <16 x i32> %Block2D_ReadAddrPayload104, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %169 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %155, <8 x i16> %159, <8 x i32> %165, i32 11, i32 11, i32 8, i32 8, i1 false)
  %170 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %169, <8 x i16> %160, <8 x i32> %166, i32 11, i32 11, i32 8, i32 8, i1 false)
  %171 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %170, <8 x i16> %163, <8 x i32> %167, i32 11, i32 11, i32 8, i32 8, i1 false)
  %172 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %171, <8 x i16> %164, <8 x i32> %168, i32 11, i32 11, i32 8, i32 8, i1 false)
  %173 = or i32 %36, 576, !dbg !456
  %174 = or i32 %15, %173, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %174, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %175 = or i32 %156, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %175, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload106 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %176 = shufflevector <16 x i16> %Block2D_ReadAddrPayload106, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %177 = shufflevector <16 x i16> %Block2D_ReadAddrPayload106, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %178 = or i32 %36, 544, !dbg !457
  %179 = or i32 %178, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %179, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload108 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %180 = shufflevector <16 x i16> %Block2D_ReadAddrPayload108, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %181 = shufflevector <16 x i16> %Block2D_ReadAddrPayload108, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %156, i1 false)
  %Block2D_ReadAddrPayload110 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %182 = shufflevector <16 x i32> %Block2D_ReadAddrPayload110, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %183 = shufflevector <16 x i32> %Block2D_ReadAddrPayload110, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %178, i1 false)
  %Block2D_ReadAddrPayload112 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %184 = shufflevector <16 x i32> %Block2D_ReadAddrPayload112, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %185 = shufflevector <16 x i32> %Block2D_ReadAddrPayload112, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %186 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %172, <8 x i16> %176, <8 x i32> %182, i32 11, i32 11, i32 8, i32 8, i1 false)
  %187 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %186, <8 x i16> %177, <8 x i32> %183, i32 11, i32 11, i32 8, i32 8, i1 false)
  %188 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %187, <8 x i16> %180, <8 x i32> %184, i32 11, i32 11, i32 8, i32 8, i1 false)
  %189 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %188, <8 x i16> %181, <8 x i32> %185, i32 11, i32 11, i32 8, i32 8, i1 false)
  %190 = or i32 %36, 640, !dbg !456
  %191 = or i32 %15, %190, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %191, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %192 = or i32 %173, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %192, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload114 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %193 = shufflevector <16 x i16> %Block2D_ReadAddrPayload114, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %194 = shufflevector <16 x i16> %Block2D_ReadAddrPayload114, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %195 = or i32 %36, 608, !dbg !457
  %196 = or i32 %195, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %196, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload116 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %197 = shufflevector <16 x i16> %Block2D_ReadAddrPayload116, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %198 = shufflevector <16 x i16> %Block2D_ReadAddrPayload116, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %173, i1 false)
  %Block2D_ReadAddrPayload118 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %199 = shufflevector <16 x i32> %Block2D_ReadAddrPayload118, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %200 = shufflevector <16 x i32> %Block2D_ReadAddrPayload118, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %195, i1 false)
  %Block2D_ReadAddrPayload120 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %201 = shufflevector <16 x i32> %Block2D_ReadAddrPayload120, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %202 = shufflevector <16 x i32> %Block2D_ReadAddrPayload120, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %203 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %189, <8 x i16> %193, <8 x i32> %199, i32 11, i32 11, i32 8, i32 8, i1 false)
  %204 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %203, <8 x i16> %194, <8 x i32> %200, i32 11, i32 11, i32 8, i32 8, i1 false)
  %205 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %204, <8 x i16> %197, <8 x i32> %201, i32 11, i32 11, i32 8, i32 8, i1 false)
  %206 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %205, <8 x i16> %198, <8 x i32> %202, i32 11, i32 11, i32 8, i32 8, i1 false)
  %207 = or i32 %36, 704, !dbg !456
  %208 = or i32 %15, %207, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %208, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %209 = or i32 %190, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %209, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload122 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %210 = shufflevector <16 x i16> %Block2D_ReadAddrPayload122, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %211 = shufflevector <16 x i16> %Block2D_ReadAddrPayload122, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %212 = or i32 %36, 672, !dbg !457
  %213 = or i32 %212, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %213, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload124 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %214 = shufflevector <16 x i16> %Block2D_ReadAddrPayload124, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %215 = shufflevector <16 x i16> %Block2D_ReadAddrPayload124, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %190, i1 false)
  %Block2D_ReadAddrPayload126 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %216 = shufflevector <16 x i32> %Block2D_ReadAddrPayload126, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %217 = shufflevector <16 x i32> %Block2D_ReadAddrPayload126, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %212, i1 false)
  %Block2D_ReadAddrPayload128 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %218 = shufflevector <16 x i32> %Block2D_ReadAddrPayload128, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %219 = shufflevector <16 x i32> %Block2D_ReadAddrPayload128, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %220 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %206, <8 x i16> %210, <8 x i32> %216, i32 11, i32 11, i32 8, i32 8, i1 false)
  %221 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %220, <8 x i16> %211, <8 x i32> %217, i32 11, i32 11, i32 8, i32 8, i1 false)
  %222 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %221, <8 x i16> %214, <8 x i32> %218, i32 11, i32 11, i32 8, i32 8, i1 false)
  %223 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %222, <8 x i16> %215, <8 x i32> %219, i32 11, i32 11, i32 8, i32 8, i1 false)
  %224 = or i32 %36, 768, !dbg !456
  %225 = or i32 %15, %224, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %225, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %226 = or i32 %207, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %226, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload130 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %227 = shufflevector <16 x i16> %Block2D_ReadAddrPayload130, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %228 = shufflevector <16 x i16> %Block2D_ReadAddrPayload130, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %229 = or i32 %36, 736, !dbg !457
  %230 = or i32 %229, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %230, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload132 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %231 = shufflevector <16 x i16> %Block2D_ReadAddrPayload132, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %232 = shufflevector <16 x i16> %Block2D_ReadAddrPayload132, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %207, i1 false)
  %Block2D_ReadAddrPayload134 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %233 = shufflevector <16 x i32> %Block2D_ReadAddrPayload134, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %234 = shufflevector <16 x i32> %Block2D_ReadAddrPayload134, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %229, i1 false)
  %Block2D_ReadAddrPayload136 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %235 = shufflevector <16 x i32> %Block2D_ReadAddrPayload136, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %236 = shufflevector <16 x i32> %Block2D_ReadAddrPayload136, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %237 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %223, <8 x i16> %227, <8 x i32> %233, i32 11, i32 11, i32 8, i32 8, i1 false)
  %238 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %237, <8 x i16> %228, <8 x i32> %234, i32 11, i32 11, i32 8, i32 8, i1 false)
  %239 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %238, <8 x i16> %231, <8 x i32> %235, i32 11, i32 11, i32 8, i32 8, i1 false)
  %240 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %239, <8 x i16> %232, <8 x i32> %236, i32 11, i32 11, i32 8, i32 8, i1 false)
  %241 = or i32 %36, 832, !dbg !456
  %242 = or i32 %15, %241, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %242, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %243 = or i32 %224, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %243, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload138 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %244 = shufflevector <16 x i16> %Block2D_ReadAddrPayload138, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %245 = shufflevector <16 x i16> %Block2D_ReadAddrPayload138, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %246 = or i32 %36, 800, !dbg !457
  %247 = or i32 %246, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %247, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload140 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %248 = shufflevector <16 x i16> %Block2D_ReadAddrPayload140, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %249 = shufflevector <16 x i16> %Block2D_ReadAddrPayload140, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %224, i1 false)
  %Block2D_ReadAddrPayload142 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %250 = shufflevector <16 x i32> %Block2D_ReadAddrPayload142, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %251 = shufflevector <16 x i32> %Block2D_ReadAddrPayload142, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %246, i1 false)
  %Block2D_ReadAddrPayload144 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %252 = shufflevector <16 x i32> %Block2D_ReadAddrPayload144, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %253 = shufflevector <16 x i32> %Block2D_ReadAddrPayload144, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %254 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %240, <8 x i16> %244, <8 x i32> %250, i32 11, i32 11, i32 8, i32 8, i1 false)
  %255 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %254, <8 x i16> %245, <8 x i32> %251, i32 11, i32 11, i32 8, i32 8, i1 false)
  %256 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %255, <8 x i16> %248, <8 x i32> %252, i32 11, i32 11, i32 8, i32 8, i1 false)
  %257 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %256, <8 x i16> %249, <8 x i32> %253, i32 11, i32 11, i32 8, i32 8, i1 false)
  %258 = or i32 %36, 896, !dbg !456
  %259 = or i32 %15, %258, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %259, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %260 = or i32 %241, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %260, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload146 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %261 = shufflevector <16 x i16> %Block2D_ReadAddrPayload146, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %262 = shufflevector <16 x i16> %Block2D_ReadAddrPayload146, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %263 = or i32 %36, 864, !dbg !457
  %264 = or i32 %263, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %264, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload148 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %265 = shufflevector <16 x i16> %Block2D_ReadAddrPayload148, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %266 = shufflevector <16 x i16> %Block2D_ReadAddrPayload148, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %241, i1 false)
  %Block2D_ReadAddrPayload150 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %267 = shufflevector <16 x i32> %Block2D_ReadAddrPayload150, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %268 = shufflevector <16 x i32> %Block2D_ReadAddrPayload150, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %263, i1 false)
  %Block2D_ReadAddrPayload152 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %269 = shufflevector <16 x i32> %Block2D_ReadAddrPayload152, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %270 = shufflevector <16 x i32> %Block2D_ReadAddrPayload152, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %271 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %257, <8 x i16> %261, <8 x i32> %267, i32 11, i32 11, i32 8, i32 8, i1 false)
  %272 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %271, <8 x i16> %262, <8 x i32> %268, i32 11, i32 11, i32 8, i32 8, i1 false)
  %273 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %272, <8 x i16> %265, <8 x i32> %269, i32 11, i32 11, i32 8, i32 8, i1 false)
  %274 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %273, <8 x i16> %266, <8 x i32> %270, i32 11, i32 11, i32 8, i32 8, i1 false)
  %275 = or i32 %36, 960, !dbg !456
  %276 = or i32 %15, %275, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %276, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %277 = or i32 %258, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %277, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload154 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %278 = shufflevector <16 x i16> %Block2D_ReadAddrPayload154, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %279 = shufflevector <16 x i16> %Block2D_ReadAddrPayload154, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %280 = or i32 %36, 928, !dbg !457
  %281 = or i32 %280, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %281, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload156 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %282 = shufflevector <16 x i16> %Block2D_ReadAddrPayload156, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %283 = shufflevector <16 x i16> %Block2D_ReadAddrPayload156, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %258, i1 false)
  %Block2D_ReadAddrPayload158 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %284 = shufflevector <16 x i32> %Block2D_ReadAddrPayload158, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %285 = shufflevector <16 x i32> %Block2D_ReadAddrPayload158, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %280, i1 false)
  %Block2D_ReadAddrPayload160 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %286 = shufflevector <16 x i32> %Block2D_ReadAddrPayload160, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %287 = shufflevector <16 x i32> %Block2D_ReadAddrPayload160, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %288 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %274, <8 x i16> %278, <8 x i32> %284, i32 11, i32 11, i32 8, i32 8, i1 false)
  %289 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %288, <8 x i16> %279, <8 x i32> %285, i32 11, i32 11, i32 8, i32 8, i1 false)
  %290 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %289, <8 x i16> %282, <8 x i32> %286, i32 11, i32 11, i32 8, i32 8, i1 false)
  %291 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %290, <8 x i16> %283, <8 x i32> %287, i32 11, i32 11, i32 8, i32 8, i1 false)
  %292 = add nuw nsw i32 %36, 1024, !dbg !456, !spirv.Decorations !449
  %293 = or i32 %15, %292, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %19, i32 %25, i32 4095, i32 24575, i32 %24, i32 %293, i32 16, i32 32, i32 16, i32 1, i1 false, i1 false, i32 4)
  %294 = or i32 %275, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %294, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload162 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %295 = shufflevector <16 x i16> %Block2D_ReadAddrPayload162, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %296 = shufflevector <16 x i16> %Block2D_ReadAddrPayload162, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  %297 = or i32 %36, 992, !dbg !457
  %298 = or i32 %297, %30, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 5, i32 %298, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload, i32 6, i32 %11, i1 false)
  %Block2D_ReadAddrPayload164 = call <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32* %Block2D_AddrPayload, i32 0, i32 0, i32 16, i32 16, i32 8, i32 2, i1 false, i1 false, i32 0)
  %299 = shufflevector <16 x i16> %Block2D_ReadAddrPayload164, <16 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !457
  %300 = shufflevector <16 x i16> %Block2D_ReadAddrPayload164, <16 x i16> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !457
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %275, i1 false)
  %Block2D_ReadAddrPayload166 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %301 = shufflevector <16 x i32> %Block2D_ReadAddrPayload166, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %302 = shufflevector <16 x i32> %Block2D_ReadAddrPayload166, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 5, i32 %35, i1 false)
  call void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32* %Block2D_AddrPayload45, i32 6, i32 %297, i1 false)
  %Block2D_ReadAddrPayload168 = call <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32* %Block2D_AddrPayload45, i32 0, i32 0, i32 16, i32 16, i32 32, i32 1, i1 false, i1 true, i32 0)
  %303 = shufflevector <16 x i32> %Block2D_ReadAddrPayload168, <16 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !453
  %304 = shufflevector <16 x i32> %Block2D_ReadAddrPayload168, <16 x i32> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, !dbg !453
  %305 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %291, <8 x i16> %295, <8 x i32> %301, i32 11, i32 11, i32 8, i32 8, i1 false)
  %306 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %305, <8 x i16> %296, <8 x i32> %302, i32 11, i32 11, i32 8, i32 8, i1 false)
  %307 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %306, <8 x i16> %299, <8 x i32> %303, i32 11, i32 11, i32 8, i32 8, i1 false)
  %308 = call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %307, <8 x i16> %300, <8 x i32> %304, i32 11, i32 11, i32 8, i32 8, i1 false)
  %309 = icmp ult i32 %275, 4032, !dbg !454
  br i1 %309, label %._crit_edge, label %310, !dbg !454

310:                                              ; preds = %._crit_edge
  %311 = and i16 %localIdX, 512, !dbg !458
  %312 = icmp eq i16 %311, 0, !dbg !458
  %313 = select i1 %312, i32 %11, i32 4, !dbg !458
  %314 = bitcast <8 x float> %308 to <8 x i32>, !dbg !458
  %315 = ptrtoint i8 addrspace(1)* %2 to i64, !dbg !458
  %316 = and i64 %315, -64, !dbg !458
  %317 = trunc i64 %315 to i32, !dbg !458
  %318 = and i32 %317, 63, !dbg !458
  %319 = lshr i32 %318, 2, !dbg !458
  %320 = or i32 %319, %33, !dbg !458
  %321 = or i32 %320, %12, !dbg !458
  %322 = add nuw nsw i32 %318, 49151
  call void @llvm.genx.GenISA.LSC2DBlockWrite.v8i32(i64 %316, i32 %322, i32 3, i32 49151, i32 %321, i32 %313, i32 32, i32 16, i32 8, i32 1, i1 false, i1 false, i32 0, <8 x i32> %314)
  ret void, !dbg !459
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
; Function Attrs: nounwind speculatable willreturn memory(none)
declare i32* @llvm.genx.GenISA.LSC2DBlockCreateAddrPayload.p0i32(i64, i32, i32, i32, i32, i32, i32, i32, i32) #7

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Arg 2: 
; Arg 3: 
; Function Attrs: nounwind speculatable willreturn memory(write)
declare void @llvm.genx.GenISA.LSC2DBlockSetAddrPayloadField.p0i32.i32(i32*, i32, i32, i1) #8

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
; Function Attrs: nounwind willreturn memory(readwrite)
declare <16 x i16> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i16.p0i32(i32*, i32, i32, i32, i32, i32, i32, i1, i1, i32) #9

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
; Function Attrs: nounwind willreturn memory(readwrite)
declare <16 x i32> @llvm.genx.GenISA.LSC2DBlockReadAddrPayload.v16i32.p0i32(i32*, i32, i32, i32, i32, i32, i32, i1, i1, i32) #9

; Function Desc: 
; Output: 
; Arg 0: 
; Arg 1: 
; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.genx.GenISA.umulH.i32(i32, i32) #6

attributes #0 = { convergent nounwind "less-precise-fpmad"="true" }
attributes #1 = { convergent nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress nofree nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #4 = { convergent nounwind willreturn memory(none) }
attributes #5 = { nounwind memory(readwrite) }
attributes #6 = { nounwind willreturn memory(none) }
attributes #7 = { nounwind speculatable willreturn memory(none) }
attributes #8 = { nounwind speculatable willreturn memory(write) }
attributes #9 = { nounwind willreturn memory(readwrite) }

!llvm.module.flags = !{!0, !1, !2}
!llvm.dbg.cu = !{!3}
!spirv.MemoryModel = !{!5}
!spirv.Source = !{!6}
!spirv.Generator = !{!7}
!igc.functions = !{!8}
!IGCMetadata = !{!35}
!opencl.ocl.version = !{!433, !433, !433, !433, !433, !433, !433, !433, !433, !433, !433, !433, !433}
!opencl.spir.version = !{!433, !433, !433, !433, !433, !433, !433, !433, !433, !433, !433, !433, !433}
!llvm.ident = !{!434, !434, !434, !434, !434, !434, !434, !434, !434, !434, !434, !434, !434}

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
!35 = !{!"ModuleMD", !36, !37, !143, !267, !298, !315, !336, !346, !348, !349, !364, !365, !366, !367, !371, !372, !379, !380, !381, !382, !383, !384, !385, !386, !387, !388, !389, !391, !395, !396, !397, !398, !399, !400, !401, !402, !403, !404, !405, !406, !407, !408, !214, !409, !410, !411, !413, !415, !418, !419, !420, !422, !423, !424, !429, !430, !431, !432}
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
!263 = !{!"m_OptsToDisablePerFunc", !264, !265, !266}
!264 = !{!"m_OptsToDisablePerFuncSet[0]", !"IGC-AddressArithmeticSinking"}
!265 = !{!"m_OptsToDisablePerFuncSet[1]", !"IGC-AllowSimd32Slicing"}
!266 = !{!"m_OptsToDisablePerFuncSet[2]", !"IGC-SinkLoadOpt"}
!267 = !{!"pushInfo", !268, !269, !270, !274, !275, !276, !277, !278, !279, !280, !281, !294, !295, !296, !297}
!268 = !{!"pushableAddresses"}
!269 = !{!"bindlessPushInfo"}
!270 = !{!"dynamicBufferInfo", !271, !272, !273}
!271 = !{!"firstIndex", i32 0}
!272 = !{!"numOffsets", i32 0}
!273 = !{!"forceDisabled", i1 false}
!274 = !{!"MaxNumberOfPushedBuffers", i32 0}
!275 = !{!"inlineConstantBufferSlot", i32 -1}
!276 = !{!"inlineConstantBufferOffset", i32 -1}
!277 = !{!"inlineConstantBufferGRFOffset", i32 -1}
!278 = !{!"constants"}
!279 = !{!"inputs"}
!280 = !{!"constantReg"}
!281 = !{!"simplePushInfoArr", !282, !291, !292, !293}
!282 = !{!"simplePushInfoArrVec[0]", !283, !284, !285, !286, !287, !288, !289, !290}
!283 = !{!"cbIdx", i32 0}
!284 = !{!"pushableAddressGrfOffset", i32 -1}
!285 = !{!"pushableOffsetGrfOffset", i32 -1}
!286 = !{!"offset", i32 0}
!287 = !{!"size", i32 0}
!288 = !{!"isStateless", i1 false}
!289 = !{!"isBindless", i1 false}
!290 = !{!"simplePushLoads"}
!291 = !{!"simplePushInfoArrVec[1]", !283, !284, !285, !286, !287, !288, !289, !290}
!292 = !{!"simplePushInfoArrVec[2]", !283, !284, !285, !286, !287, !288, !289, !290}
!293 = !{!"simplePushInfoArrVec[3]", !283, !284, !285, !286, !287, !288, !289, !290}
!294 = !{!"simplePushBufferUsed", i32 0}
!295 = !{!"pushAnalysisWIInfos"}
!296 = !{!"inlineRTGlobalPtrOffset", i32 0}
!297 = !{!"rtSyncSurfPtrOffset", i32 0}
!298 = !{!"psInfo", !299, !300, !301, !302, !303, !304, !305, !306, !307, !308, !309, !310, !311, !312, !313, !314}
!299 = !{!"BlendStateDisabledMask", i8 0}
!300 = !{!"SkipSrc0Alpha", i1 false}
!301 = !{!"DualSourceBlendingDisabled", i1 false}
!302 = !{!"ForceEnableSimd32", i1 false}
!303 = !{!"DisableSimd32WithDiscard", i1 false}
!304 = !{!"outputDepth", i1 false}
!305 = !{!"outputStencil", i1 false}
!306 = !{!"outputMask", i1 false}
!307 = !{!"blendToFillEnabled", i1 false}
!308 = !{!"forceEarlyZ", i1 false}
!309 = !{!"hasVersionedLoop", i1 false}
!310 = !{!"forceSingleSourceRTWAfterDualSourceRTW", i1 false}
!311 = !{!"NumSamples", i8 0}
!312 = !{!"blendOptimizationMode"}
!313 = !{!"colorOutputMask"}
!314 = !{!"WaDisableVRS", i1 false}
!315 = !{!"csInfo", !316, !317, !318, !319, !77, !53, !54, !320, !55, !321, !322, !323, !324, !325, !326, !327, !328, !329, !330, !331, !87, !332, !333, !334, !335}
!316 = !{!"maxWorkGroupSize", i32 0}
!317 = !{!"waveSize", i32 0}
!318 = !{!"ComputeShaderSecondCompile"}
!319 = !{!"forcedSIMDSize", i8 0}
!320 = !{!"VISAPreSchedScheduleExtraGRF", i32 0}
!321 = !{!"forceSpillCompression", i1 false}
!322 = !{!"allowLowerSimd", i1 false}
!323 = !{!"disableSimd32Slicing", i1 false}
!324 = !{!"disableSplitOnSpill", i1 false}
!325 = !{!"enableNewSpillCostFunction", i1 false}
!326 = !{!"forceVISAPreSched", i1 false}
!327 = !{!"disableLocalIdOrderOptimizations", i1 false}
!328 = !{!"disableDispatchAlongY", i1 false}
!329 = !{!"neededThreadIdLayout", i1* null}
!330 = !{!"forceTileYWalk", i1 false}
!331 = !{!"atomicBranch", i32 0}
!332 = !{!"disableEarlyOut", i1 false}
!333 = !{!"walkOrderEnabled", i1 false}
!334 = !{!"walkOrderOverride", i32 0}
!335 = !{!"ResForHfPacking"}
!336 = !{!"msInfo", !337, !338, !339, !340, !341, !342, !343, !344, !345}
!337 = !{!"PrimitiveTopology", i32 3}
!338 = !{!"MaxNumOfPrimitives", i32 0}
!339 = !{!"MaxNumOfVertices", i32 0}
!340 = !{!"MaxNumOfPerPrimitiveOutputs", i32 0}
!341 = !{!"MaxNumOfPerVertexOutputs", i32 0}
!342 = !{!"WorkGroupSize", i32 0}
!343 = !{!"WorkGroupMemorySizeInBytes", i32 0}
!344 = !{!"IndexFormat", i32 6}
!345 = !{!"SubgroupSize", i32 0}
!346 = !{!"taskInfo", !347, !342, !343, !345}
!347 = !{!"MaxNumOfOutputs", i32 0}
!348 = !{!"NBarrierCnt", i32 0}
!349 = !{!"rtInfo", !350, !351, !352, !353, !354, !355, !356, !357, !358, !359, !360, !361, !362, !363, !172}
!350 = !{!"RayQueryAllocSizeInBytes", i32 0}
!351 = !{!"NumContinuations", i32 0}
!352 = !{!"RTAsyncStackAddrspace", i32 -1}
!353 = !{!"RTAsyncStackSurfaceStateOffset", i1* null}
!354 = !{!"SWHotZoneAddrspace", i32 -1}
!355 = !{!"SWHotZoneSurfaceStateOffset", i1* null}
!356 = !{!"SWStackAddrspace", i32 -1}
!357 = !{!"SWStackSurfaceStateOffset", i1* null}
!358 = !{!"RTSyncStackAddrspace", i32 -1}
!359 = !{!"RTSyncStackSurfaceStateOffset", i1* null}
!360 = !{!"doSyncDispatchRays", i1 false}
!361 = !{!"MemStyle", !"Xe"}
!362 = !{!"GlobalDataStyle", !"Xe"}
!363 = !{!"uberTileDimensions", i1* null}
!364 = !{!"CurUniqueIndirectIdx", i32 0}
!365 = !{!"inlineDynTextures"}
!366 = !{!"inlineResInfoData"}
!367 = !{!"immConstant", !368, !369, !370}
!368 = !{!"data"}
!369 = !{!"sizes"}
!370 = !{!"zeroIdxs"}
!371 = !{!"stringConstants"}
!372 = !{!"inlineBuffers", !373, !377, !378}
!373 = !{!"inlineBuffersVec[0]", !374, !375, !376}
!374 = !{!"alignment", i32 0}
!375 = !{!"allocSize", i64 0}
!376 = !{!"Buffer"}
!377 = !{!"inlineBuffersVec[1]", !374, !375, !376}
!378 = !{!"inlineBuffersVec[2]", !374, !375, !376}
!379 = !{!"GlobalPointerProgramBinaryInfos"}
!380 = !{!"ConstantPointerProgramBinaryInfos"}
!381 = !{!"GlobalBufferAddressRelocInfo"}
!382 = !{!"ConstantBufferAddressRelocInfo"}
!383 = !{!"forceLscCacheList"}
!384 = !{!"SrvMap"}
!385 = !{!"RasterizerOrderedByteAddressBuffer"}
!386 = !{!"RasterizerOrderedViews"}
!387 = !{!"MinNOSPushConstantSize", i32 0}
!388 = !{!"inlineProgramScopeOffsets"}
!389 = !{!"shaderData", !390}
!390 = !{!"numReplicas", i32 0}
!391 = !{!"URBInfo", !392, !393, !394}
!392 = !{!"has64BVertexHeaderInput", i1 false}
!393 = !{!"has64BVertexHeaderOutput", i1 false}
!394 = !{!"hasVertexHeader", i1 true}
!395 = !{!"UseBindlessImage", i1 true}
!396 = !{!"UseBindlessImageWithSamplerTracking", i1 false}
!397 = !{!"enableRangeReduce", i1 false}
!398 = !{!"allowMatchMadOptimizationforVS", i1 false}
!399 = !{!"disableMatchMadOptimizationForCS", i1 false}
!400 = !{!"disableMemOptforNegativeOffsetLoads", i1 false}
!401 = !{!"enableThreeWayLoadSpiltOpt", i1 false}
!402 = !{!"statefulResourcesNotAliased", i1 false}
!403 = !{!"disableMixMode", i1 false}
!404 = !{!"genericAccessesResolved", i1 false}
!405 = !{!"disableSeparateSpillPvtScratchSpace", i1 false}
!406 = !{!"enableSeparateSpillPvtScratchSpace", i1 false}
!407 = !{!"disableSeparateScratchWA", i1 false}
!408 = !{!"enableRemoveUnusedTGMFence", i1 false}
!409 = !{!"PrivateMemoryPerFG"}
!410 = !{!"m_OptsToDisable"}
!411 = !{!"capabilities", !412}
!412 = !{!"globalVariableDecorationsINTEL", i1 false}
!413 = !{!"extensions", !414}
!414 = !{!"spvINTELBindlessImages", i1 false}
!415 = !{!"m_ShaderResourceViewMcsMask", !416, !417}
!416 = !{!"m_ShaderResourceViewMcsMaskVec[0]", i64 0}
!417 = !{!"m_ShaderResourceViewMcsMaskVec[1]", i64 0}
!418 = !{!"computedDepthMode", i32 0}
!419 = !{!"isHDCFastClearShader", i1 false}
!420 = !{!"argRegisterReservations", !421}
!421 = !{!"argRegisterReservationsVec[0]", i32 0}
!422 = !{!"SIMD16_SpillThreshold", i8 0}
!423 = !{!"SIMD32_SpillThreshold", i8 0}
!424 = !{!"m_CacheControlOption", !425, !426, !427, !428}
!425 = !{!"LscLoadCacheControlOverride", i8 0}
!426 = !{!"LscStoreCacheControlOverride", i8 0}
!427 = !{!"TgmLoadCacheControlOverride", i8 0}
!428 = !{!"TgmStoreCacheControlOverride", i8 0}
!429 = !{!"ModuleUsesBindless", i1 false}
!430 = !{!"predicationMap"}
!431 = !{!"lifeTimeStartMap"}
!432 = !{!"HitGroups"}
!433 = !{i32 2, i32 0}
!434 = !{!"clang version 16.0.6"}
!435 = distinct !DISubprogram(name: "matmul_kernel_with_tensor_descriptors", linkageName: "matmul_kernel_with_tensor_descriptors", scope: null, file: !4, line: 38, type: !436, scopeLine: 38, spFlags: DISPFlagDefinition | DISPFlagOptimized | DISPFlagMainSubprogram, unit: !3, templateParams: !440, retainedNodes: !440)
!436 = !DISubroutineType(types: !437)
!437 = !{null, !438, !438, !438, !438, !438}
!438 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !439, size: 64, dwarfAddressSpace: 1)
!439 = !DIBasicType(name: "unknown_type", encoding: DW_ATE_signed)
!440 = !{}
!441 = !DILocation(line: 54, column: 16, scope: !435)
!442 = !DILocation(line: 56, column: 24, scope: !435)
!443 = !{!444}
!444 = !{i32 4469}
!445 = !DILocation(line: 57, column: 29, scope: !435)
!446 = !DILocation(line: 58, column: 13, scope: !435)
!447 = !DILocation(line: 57, column: 28, scope: !435)
!448 = !DILocation(line: 57, column: 13, scope: !435)
!449 = !{!444, !450}
!450 = !{i32 4470}
!451 = !DILocation(line: 79, column: 30, scope: !435)
!452 = !DILocation(line: 83, column: 37, scope: !435)
!453 = !DILocation(line: 83, column: 17, scope: !435)
!454 = !DILocation(line: 75, column: 5, scope: !435)
!455 = !DILocation(line: 84, column: 24, scope: !435)
!456 = !DILocation(line: 85, column: 9, scope: !435)
!457 = !DILocation(line: 79, column: 17, scope: !435)
!458 = !DILocation(line: 90, column: 5, scope: !435)
!459 = !DILocation(line: 38, column: 1, scope: !435)
