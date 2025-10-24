; ------------------------------------------------
; OCL_asmb5106d4cf5402500_simd32_entry_0001.visa.ll
; ------------------------------------------------
; Function Attrs: convergent nounwind null_pointer_is_valid
define spir_kernel void @kernel(float addrspace(1)* align 4 %0, float addrspace(1)* align 4 %1, float addrspace(1)* align 4 %2, float addrspace(1)* align 4 %3, i8 addrspace(1)* align 1 %4, i8 addrspace(1)* align 1 %5, <8 x i32> %r0, <8 x i32> %payloadHeader, i16 %localIdX, i16 %localIdY, i16 %localIdZ, i8* %privateBase, i32 %bufferOffset, i32 %bufferOffset1, i32 %bufferOffset2, i32 %bufferOffset3, i32 %bufferOffset4, i32 %bufferOffset5) #0 {
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
  %18 = bitcast float %17 to i32
  %19 = and i32 %18, 2139095040		; visa id: 9
  %20 = icmp eq i32 %19, 0		; visa id: 10
  %21 = select i1 %20, float 0x41F0000000000000, float 1.000000e+00		; visa id: 11
  %22 = icmp uge i32 %19, 1677721600		; visa id: 12
  %23 = select i1 %22, float 0x3DF0000000000000, float %21		; visa id: 13
  %24 = fmul float %17, %23		; visa id: 14
  %25 = fdiv float 1.000000e+00, %24		; visa id: 15
  %26 = fmul float %25, %13		; visa id: 16
  %27 = fmul float %26, %23		; visa id: 17
  %28 = and i32 %18, 8388607
  %29 = icmp eq i32 %19, 0
  %30 = icmp eq i32 %28, 0		; visa id: 18
  %31 = or i1 %29, %30		; visa id: 20
  %32 = xor i1 %31, true		; visa id: 22
  %33 = fcmp oeq float %13, %17
  %34 = and i1 %33, %32		; visa id: 23
  %35 = select i1 %34, float 1.000000e+00, float %27		; visa id: 25
  %36 = ptrtoint float addrspace(1)* %2 to i64		; visa id: 26
  %37 = add i64 %10, %36		; visa id: 26
  %38 = inttoptr i64 %37 to float addrspace(1)*		; visa id: 27
  store float %35, float addrspace(1)* %38, align 4		; visa id: 27
  %39 = ptrtoint float addrspace(1)* %3 to i64		; visa id: 28
  %40 = add i64 %10, %39		; visa id: 28
  %41 = inttoptr i64 %40 to float addrspace(1)*		; visa id: 29
  store float %35, float addrspace(1)* %41, align 4		; visa id: 29
  ret void, !stats.blockFrequency.digits !410, !stats.blockFrequency.scale !411		; visa id: 30
}
