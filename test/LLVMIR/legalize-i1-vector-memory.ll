; RUN: triton-llvm-opt -legalize-i1-vector-memory %s | FileCheck %s
; RUN: triton-llvm-opt -legalize-i1-vector-memory %s | FileCheck %s --check-prefix=NOI1MEM

; `<N x i1>` loads and stores are rewritten as integer accesses of the same
; bytes, with the bit (un)packing done in registers. LLVM bit-packs `<N x i1>`
; in memory, IGC gives every i1 a byte, and SPIR-V defines no memory layout for
; Booleans, so no such access may reach the SPIR-V translator.

; NOI1MEM-NOT: load {{(volatile )?}}<{{[0-9]+}} x i1>
; NOI1MEM-NOT: store {{(volatile )?}}<{{[0-9]+}} x i1>

; The shape InstCombine forms from an i1 relayout through shared memory: one
; `<32 x i1>` load read back at the LSB of each byte. It becomes one i32 load.
; CHECK-LABEL: @load_extracts
; CHECK:       [[RAW:%.*]] = load i32, ptr addrspace(3) %p, align 4
; CHECK:       [[M0:%.*]] = and i32 [[RAW]], 1
; CHECK:       [[B0:%.*]] = icmp ne i32 [[M0]], 0
; CHECK:       [[M8:%.*]] = and i32 [[RAW]], 256
; CHECK:       [[B8:%.*]] = icmp ne i32 [[M8]], 0
; CHECK:       [[M16:%.*]] = and i32 [[RAW]], 65536
; CHECK:       [[B16:%.*]] = icmp ne i32 [[M16]], 0
; CHECK:       [[M24:%.*]] = and i32 [[RAW]], 16777216
; CHECK:       [[B24:%.*]] = icmp ne i32 [[M24]], 0
; CHECK:       zext i1 [[B0]] to i32
; CHECK:       zext i1 [[B8]] to i32
; CHECK:       zext i1 [[B16]] to i32
; CHECK:       zext i1 [[B24]] to i32
define <4 x i32> @load_extracts(ptr addrspace(3) %p) {
  %v = load <32 x i1>, ptr addrspace(3) %p, align 4
  %b0 = extractelement <32 x i1> %v, i64 0
  %b8 = extractelement <32 x i1> %v, i64 8
  %b16 = extractelement <32 x i1> %v, i64 16
  %b24 = extractelement <32 x i1> %v, i64 24
  %z0 = zext i1 %b0 to i32
  %z8 = zext i1 %b8 to i32
  %z16 = zext i1 %b16 to i32
  %z24 = zext i1 %b24 to i32
  %r0 = insertelement <4 x i32> poison, i32 %z0, i64 0
  %r1 = insertelement <4 x i32> %r0, i32 %z8, i64 1
  %r2 = insertelement <4 x i32> %r1, i32 %z16, i64 2
  %r3 = insertelement <4 x i32> %r2, i32 %z24, i64 3
  ret <4 x i32> %r3
}

; Any other user gets the whole vector rebuilt from the loaded bits.
; CHECK-LABEL: @load_whole_vector
; CHECK:       [[RAW:%.*]] = load i8, ptr addrspace(3) %p, align 1
; CHECK:       [[M0:%.*]] = and i8 [[RAW]], 1
; CHECK:       [[B0:%.*]] = icmp ne i8 [[M0]], 0
; CHECK:       [[V0:%.*]] = insertelement <8 x i1> poison, i1 [[B0]], i64 0
; CHECK:       [[M7:%.*]] = and i8 [[RAW]], -128
; CHECK:       [[B7:%.*]] = icmp ne i8 [[M7]], 0
; CHECK:       [[V7:%.*]] = insertelement <8 x i1> {{%.*}}, i1 [[B7]], i64 7
; CHECK:       select <8 x i1> [[V7]], <8 x i32> %a, <8 x i32> %b
define <8 x i32> @load_whole_vector(ptr addrspace(3) %p, <8 x i32> %a, <8 x i32> %b) {
  %v = load <8 x i1>, ptr addrspace(3) %p, align 1
  %r = select <8 x i1> %v, <8 x i32> %a, <8 x i32> %b
  ret <8 x i32> %r
}

; Wider vectors are accessed as a vector of i32: element 65 is bit 1 of word 2.
; Volatility is preserved.
; CHECK-LABEL: @load_wide
; CHECK:       [[RAW:%.*]] = load volatile <4 x i32>, ptr addrspace(3) %p, align 16
; CHECK:       [[W:%.*]] = extractelement <4 x i32> [[RAW]], i64 2
; CHECK:       [[M:%.*]] = and i32 [[W]], 2
; CHECK:       [[B:%.*]] = icmp ne i32 [[M]], 0
; CHECK:       ret i1 [[B]]
define i1 @load_wide(ptr addrspace(3) %p) {
  %v = load volatile <128 x i1>, ptr addrspace(3) %p, align 16
  %b = extractelement <128 x i1> %v, i64 65
  ret i1 %b
}

; A store packs the elements into the integer it writes.
; CHECK-LABEL: @store_bits
; CHECK:       [[E0:%.*]] = extractelement <4 x i1> %v, i64 0
; CHECK:       [[Z0:%.*]] = zext i1 [[E0]] to i8
; CHECK:       [[E1:%.*]] = extractelement <4 x i1> %v, i64 1
; CHECK:       [[Z1:%.*]] = zext i1 [[E1]] to i8
; CHECK:       [[S1:%.*]] = shl i8 [[Z1]], 1
; CHECK:       [[O1:%.*]] = or i8 [[Z0]], [[S1]]
; CHECK:       [[E2:%.*]] = extractelement <4 x i1> %v, i64 2
; CHECK:       [[Z2:%.*]] = zext i1 [[E2]] to i8
; CHECK:       [[S2:%.*]] = shl i8 [[Z2]], 2
; CHECK:       [[O2:%.*]] = or i8 [[O1]], [[S2]]
; CHECK:       [[E3:%.*]] = extractelement <4 x i1> %v, i64 3
; CHECK:       [[Z3:%.*]] = zext i1 [[E3]] to i8
; CHECK:       [[S3:%.*]] = shl i8 [[Z3]], 3
; CHECK:       [[O3:%.*]] = or i8 [[O2]], [[S3]]
; CHECK:       store i8 [[O3]], ptr addrspace(3) %p, align 1
define void @store_bits(<4 x i1> %v, ptr addrspace(3) %p) {
  store <4 x i1> %v, ptr addrspace(3) %p, align 1
  ret void
}

; Constant elements fold into the stored integer: bits 0, 2 and 7.
; CHECK-LABEL: @store_constant
; CHECK:       store i8 -123, ptr addrspace(3) %p, align 1
define void @store_constant(ptr addrspace(3) %p) {
  store <8 x i1> <i1 true, i1 false, i1 true, i1 false, i1 false, i1 false, i1 false, i1 true>, ptr addrspace(3) %p, align 1
  ret void
}

; Element 32 of a 64-bit store lands in bit 32 of one i64.
; CHECK-LABEL: @store_wide
; CHECK:       [[E32:%.*]] = extractelement <64 x i1> %v, i64 32
; CHECK:       [[Z32:%.*]] = zext i1 [[E32]] to i64
; CHECK:       shl i64 [[Z32]], 32
; CHECK:       store i64 {{%.*}}, ptr addrspace(3) %p, align 8
define void @store_wide(<64 x i1> %v, ptr addrspace(3) %p) {
  store <64 x i1> %v, ptr addrspace(3) %p, align 8
  ret void
}

; Byte vectors and scalar i1 are left untouched.
; CHECK-LABEL: @untouched
; CHECK:       load <4 x i8>, ptr addrspace(3) %p, align 4
; CHECK:       load i1, ptr addrspace(3) %q, align 1
define i1 @untouched(ptr addrspace(3) %p, ptr addrspace(3) %q) {
  %v = load <4 x i8>, ptr addrspace(3) %p, align 4
  %e = extractelement <4 x i8> %v, i64 0
  %t = trunc i8 %e to i1
  %s = load i1, ptr addrspace(3) %q, align 1
  %r = and i1 %t, %s
  ret i1 %r
}
