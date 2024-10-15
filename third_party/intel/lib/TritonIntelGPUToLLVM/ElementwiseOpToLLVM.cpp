#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

using mlir::triton::gpu::ElementwiseOpConversionBase;

namespace {

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format
static SmallVector<Value>
Fp16_to_Fp8E5M2_func(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));

  Value a0 = bitcast(fp16x2Vec0, i32_ty);
  Value a1 = bitcast(fp16x2Vec1, i32_ty);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = bitcast(a0, fp8x4VecTy);
  a1 = bitcast(a1, fp8x4VecTy);

  return {extract_element(i8_ty, a0, i32_val(1)),
          extract_element(i8_ty, a0, i32_val(3)),
          extract_element(i8_ty, a1, i32_val(1)),
          extract_element(i8_ty, a1, i32_val(3))};
}

static SmallVector<Value>
Fp16_to_Fp8E5M2_RTNE_func(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {

  Value val = zext(i32_ty, bitcast(v[0], i16_ty));
  Value sign = and_(i32_ty, val, i32_val(0x8000));
  Value nosign = and_(i32_ty, val, i32_val(0x7fff));

  Value truncated = and_(i32_ty, nosign, i32_val(0x7f00));
  Value tail = and_(i32_ty, nosign, i32_val(0xff));
  Value odd_trunc =
      icmp_ne(and_(i32_ty, truncated, i32_val(0x100)), i32_val(0));
  Value round_up = or_(icmp_ugt(tail, i32_val(0x80)),
                       and_(icmp_eq(tail, i32_val(0x80)), odd_trunc));
  // Skip round-up if it leads to inf/nan.
  round_up = and_(round_up, icmp_ult(truncated, i32_val(0x7b00)));
  truncated = select(round_up, add(truncated, i32_val(0x100)), truncated);

  Value res_val = or_(i32_ty, truncated, sign);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value res = bitcast(res_val, fp8x4VecTy);

  return {extract_element(i8_ty, res, i32_val(1))};
}

static SmallVector<Value>
Fp8E5M2_to_Fp16_func(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);
  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = bitcast(a1, fp16x2VecTy);

  return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
}

static SmallVector<Value>
Fp8E5M2_to_Bf16_func(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  // In i32 original fp8 exponent is b0 >> 26
  // bf16's 5-bit exponent in the top 2 bytes of i32 is at b0 >> 23
  // 2^5-1 << 23 = 0xf800000
  b0 = lshr(i32_ty, b0, i32_val(3));
  b1 = lshr(i32_ty, b1, i32_val(3));

  Value c0 = and_(i32_ty, b0, i32_val(0xffff0000));
  Value c1 = shl(i32_ty, b0, i32_val(16));
  Value c2 = and_(i32_ty, b1, i32_val(0xffff0000));
  Value c3 = shl(i32_ty, b1, i32_val(16));

  auto i32x4VecTy = vec_ty(i32_ty, 4);
  Value predefined = undef(i32x4VecTy);
  predefined = insert_element(i32x4VecTy, predefined, i32_val(0x0), i32_val(0));
  predefined =
      insert_element(i32x4VecTy, predefined, i32_val(0x37800000), i32_val(1));
  predefined =
      insert_element(i32x4VecTy, predefined, i32_val(0x38000000), i32_val(2));
  predefined =
      insert_element(i32x4VecTy, predefined, i32_val(0x38400000), i32_val(3));
  // Check if the exponent is zero, i.e. subnormal number.
  // depending on the significand value normalization goes like:
  //  [00] -> 0x0
  //  [01] -> exp=127-16, sig=0x0
  //  [10] -> exp=127-15, sig=0x0
  //  [11] -> exp=127-15, sig=b1000...
  Value cmp0 = icmp_eq(and_(c0, i32_val(0xf800000)), i32_val(0));
  Value cmp1 = icmp_eq(and_(c1, i32_val(0xf800000)), i32_val(0));
  Value cmp2 = icmp_eq(and_(c2, i32_val(0xf800000)), i32_val(0));
  Value cmp3 = icmp_eq(and_(c3, i32_val(0xf800000)), i32_val(0));

  Value predef_idx0 = lshr(and_(c0, i32_val(3 << 21)), i32_val(21));
  Value predef_idx1 = lshr(and_(c1, i32_val(3 << 21)), i32_val(21));
  Value predef_idx2 = lshr(and_(c2, i32_val(3 << 21)), i32_val(21));
  Value predef_idx3 = lshr(and_(c3, i32_val(3 << 21)), i32_val(21));

  Value normalized0 = extract_element(i32_ty, predefined, predef_idx0);
  Value normalized1 = extract_element(i32_ty, predefined, predef_idx1);
  Value normalized2 = extract_element(i32_ty, predefined, predef_idx2);
  Value normalized3 = extract_element(i32_ty, predefined, predef_idx3);

  Value d0 = add(i32_ty, c0, i32_val(0x38000000));
  Value d1 = add(i32_ty, c1, i32_val(0x38000000));
  Value d2 = add(i32_ty, c2, i32_val(0x38000000));
  Value d3 = add(i32_ty, c3, i32_val(0x38000000));

  Value res0 = select(cmp0, normalized0, d0);
  Value res1 = select(cmp1, normalized1, d1);
  Value res2 = select(cmp2, normalized2, d2);
  Value res3 = select(cmp3, normalized3, d3);

  Value f0 = or_(i32_ty, res0, lshr(i32_ty, res1, i32_val(16)));
  Value f1 = or_(i32_ty, res2, lshr(i32_ty, res3, i32_val(16)));

  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, a1, i32_val(0x80008000));

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, f0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, f1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return {extract_element(bf16_ty, bf16x2Vec0, i32_val(0)),
          extract_element(bf16_ty, bf16x2Vec0, i32_val(1)),
          extract_element(bf16_ty, bf16x2Vec1, i32_val(0)),
          extract_element(bf16_ty, bf16x2Vec1, i32_val(1))};
}

static SmallVector<Value>
Bf16_to_Fp8E5M2_func(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = undef(bf16x2VecTy);
  Value bf16x2Vec1 = undef(bf16x2VecTy);
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[0], i32_val(0));
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[1], i32_val(1));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[2], i32_val(0));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[3], i32_val(1));
  bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = undef(fp8x4VecTy);
  sign0 = bitcast(sign0, fp8x4VecTy);
  sign1 = bitcast(sign1, fp8x4VecTy);
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(1)), i32_val(0));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(3)), i32_val(1));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(1)), i32_val(2));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(3)), i32_val(3));
  sign = bitcast(sign, i32_ty);

  Value nosign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x7fff7fff));
  Value nosign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x7fff7fff));

  Value nosign_0_0 = and_(i32_ty, nosign0, i32_val(0xffff0000));
  nosign_0_0 = umax(i32_ty, nosign_0_0, i32_val(0x38000000));
  nosign_0_0 = umin(i32_ty, nosign_0_0, i32_val(0x57e00000));
  Value nosign_0_1 = and_(i32_ty, nosign0, i32_val(0x0000ffff));
  nosign_0_1 = umax(i32_ty, nosign_0_1, i32_val(0x3800));
  nosign_0_1 = umin(i32_ty, nosign_0_1, i32_val(0x57e0));
  nosign0 = or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = and_(i32_ty, nosign1, i32_val(0xffff0000));
  nosign_1_0 = umax(i32_ty, nosign_1_0, i32_val(0x38000000));
  nosign_1_0 = umin(i32_ty, nosign_1_0, i32_val(0x57e00000));
  Value nosign_1_1 = and_(i32_ty, nosign1, i32_val(0x0000ffff));
  nosign_1_1 = umax(i32_ty, nosign_1_1, i32_val(0x3800));
  nosign_1_1 = umin(i32_ty, nosign_1_1, i32_val(0x57e0));
  nosign1 = or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = add(i32_ty, nosign0, i32_val(0x00100010));
  nosign1 = add(i32_ty, nosign1, i32_val(0x00100010));
  nosign0 = sub(i32_ty, nosign0, i32_val(0x38003800));
  nosign1 = sub(i32_ty, nosign1, i32_val(0x38003800));
  nosign0 = shl(i32_ty, nosign0, i32_val(3));
  nosign1 = shl(i32_ty, nosign1, i32_val(3));

  nosign0 = bitcast(nosign0, fp8x4VecTy);
  nosign1 = bitcast(nosign1, fp8x4VecTy);
  Value nosign = undef(fp8x4VecTy);
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(1)), i32_val(0));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(3)), i32_val(1));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(1)), i32_val(2));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(3)), i32_val(3));
  nosign = bitcast(nosign, i32_ty);

  Value fp8x4Vec = or_(i32_ty, nosign, sign);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);
  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
          extract_element(i8_ty, fp8x4Vec, i32_val(1)),
          extract_element(i8_ty, fp8x4Vec, i32_val(2)),
          extract_element(i8_ty, fp8x4Vec, i32_val(3))};
}

static SmallVector<Value>
Bf16_to_Fp8E5M2_RTNE_func(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  Value val = zext(i32_ty, bitcast(v[0], i16_ty));
  Value sign = and_(i32_ty, val, i32_val(0x8000));
  Value nosign = and_(i32_ty, val, i32_val(0x7fff));

  Value exp = and_(i32_ty, lshr(nosign, i32_val(7)), i32_val(0xff));
  // Check if we need a translation to a subnormal value. This happens when
  // exp value is in range [110, 112].
  Value is_subnormal =
      and_(icmp_uge(exp, i32_val(110)), icmp_ule(exp, i32_val(112)));
  Value shift = sub(i32_ty, exp, i32_val(110));
  Value subnormal = and_(i32_ty, nosign, i32_val(0x7f));
  subnormal = or_(i32_ty, subnormal, i32_val(0x80));
  // Make rounding with respect to bits we are going to shift and cut off.
  Value round_step = lshr(i32_ty, i32_val(0x100), shift);
  Value tail_mask = sub(i32_ty, round_step, i32_val(1));
  Value tail = and_(i32_ty, subnormal, tail_mask);
  Value threshold = lshr(i32_ty, i32_val(0x80), shift);
  Value odd_truncated =
      icmp_ne(and_(i32_ty, subnormal, round_step), i32_val(0));
  Value round_up = or_(icmp_ugt(tail, threshold),
                       and_(icmp_eq(tail, threshold), odd_truncated));
  subnormal = select(round_up, add(i32_ty, subnormal, round_step), subnormal);
  // Now shift to get the final result.
  subnormal = shl(i32_ty, subnormal, shift);

  // Normalized case. Start with rounding, then apply exp range to fit 5 bits,
  // adjust bias and shift left.
  // TODO: NaN values might be mishandled.
  tail = and_(i32_ty, nosign, i32_val(0x1f));
  odd_truncated = icmp_ne(and_(i32_ty, nosign, i32_val(0x20)), i32_val(0));
  round_up = or_(icmp_ugt(tail, i32_val(0x10)),
                 and_(icmp_eq(tail, i32_val(0x10)), odd_truncated));
  Value rounded =
      and_(i32_ty, add(i32_ty, nosign, i32_val(0x20)), i32_val(0x7fe0));
  nosign = select(round_up, rounded, nosign);

  nosign = umax(i32_ty, nosign, i32_val(0x3800));
  nosign = umin(i32_ty, nosign, i32_val(0x57e0));
  nosign = sub(i32_ty, nosign, i32_val(0x3800));
  nosign = shl(i32_ty, nosign, i32_val(3));

  // Choose between subnormal and normal values.
  nosign = select(is_subnormal, subnormal, nosign);

  Value res_val = or_(i32_ty, nosign, sign);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value res = bitcast(res_val, fp8x4VecTy);

  return {extract_element(i8_ty, res, i32_val(1))};
}

/* ----- FP8E4M3B15 ------ */
// This data-type is a variant of the standard FP8E4M3 format.
// It was designed for fast software conversion to FP16 on GPUs that do not
// support it natively. Specifically, this data-type:
//    - has infinities
//    - has multiple nans (when all exponent bits are 1)
//    - has an exponent bias of 15 (vs. 7 for fp8e4m3)
static SmallVector<Value>
Fp8E4M3B15_to_Fp16_func(Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));

  b0 = lshr(i32_ty, b0, i32_val(1));
  b1 = lshr(i32_ty, b1, i32_val(1));

  b0 = or_(i32_ty, b0, and_(i32_ty, a0, i32_val(0x80008000)));
  b1 = or_(i32_ty, b1, and_(i32_ty, a1, i32_val(0x80008000)));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = bitcast(b0, fp16x2VecTy);
  auto fp16x2Vec1 = bitcast(b1, fp16x2VecTy);

  return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
}

static SmallVector<Value>
Fp16_to_Fp8E4M3B15_func(Location loc, ConversionPatternRewriter &rewriter,
                        const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);

  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));

  Value fp16x2VecMin = i32_val(0xBF80BF80);
  Value fp16x2VecMax = i32_val(0x3F803F80);
  fp16x2VecMin = bitcast(fp16x2VecMin, fp16x2VecTy);
  fp16x2VecMax = bitcast(fp16x2VecMax, fp16x2VecTy);
  fp16x2Vec0 = fmax(fp16x2VecTy, fp16x2Vec0, fp16x2VecMin);
  fp16x2Vec1 = fmax(fp16x2VecTy, fp16x2Vec1, fp16x2VecMin);
  fp16x2Vec0 = fmin(fp16x2VecTy, fp16x2Vec0, fp16x2VecMax);
  fp16x2Vec1 = fmin(fp16x2VecTy, fp16x2Vec1, fp16x2VecMax);

  fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

  Value a0 = shl(i32_ty, fp16x2Vec0, i32_val(1));
  Value a1 = shl(i32_ty, fp16x2Vec1, i32_val(1));
  a0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  a1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  a0 = add(i32_ty, a0, i32_val(0x00800080));
  a1 = add(i32_ty, a1, i32_val(0x00800080));
  Value b0 = or_(i32_ty, and_(i32_ty, fp16x2Vec0, i32_val(0x80008000)), a0);
  Value b1 = or_(i32_ty, and_(i32_ty, fp16x2Vec1, i32_val(0x80008000)), a1);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  b0 = bitcast(b0, fp8x4VecTy);
  b1 = bitcast(b1, fp8x4VecTy);

  return {extract_element(i8_ty, b0, i32_val(1)),
          extract_element(i8_ty, b0, i32_val(3)),
          extract_element(i8_ty, b1, i32_val(1)),
          extract_element(i8_ty, b1, i32_val(3))};
}

/* ----- FP8E4M3 ------ */
// Note: when handled by software, this format
// has more than a single NaN values.

// Fp8E4M3 -> Fp16 (packed)
static SmallVector<Value>
Fp8E4M3Nv_to_Fp16_func(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));

  b0 = lshr(i32_ty, b0, i32_val(1));

  Value c0 = and_(i32_ty, b0, i32_val(0xffff0000));
  Value c1 = shl(i32_ty, b0, i32_val(16));

  // Check if the exponent is zero, i.e. subnormal number.
  // fp8e4 has a bias of 7
  // depending on the significand value normalization goes like:
  //  [000] -> 0x0
  //  [001] -> exp=15-9, sig=0x0
  //  [010] -> exp=15-8, sig=0x0
  //  [011] -> exp=15-8, sig=b100...
  //  [100] -> exp=15-7, sig=0x0
  //  [101] -> exp=15-7, sig=b010...
  //  [110] -> exp=15-7, sig=b100...
  //  [111] -> exp=15-7, sig=b110...
  Value cmp0 = icmp_eq(and_(c0, i32_val(0x7c000000)), i32_val(0));
  Value cmp1 = icmp_eq(and_(c1, i32_val(0x7c000000)), i32_val(0));

  auto i32x8VecTy = vec_ty(i32_ty, 8);
  Value predefined = undef(i32x8VecTy);
  predefined = insert_element(i32x8VecTy, predefined, i32_val(0x0), i32_val(0));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x18000000), i32_val(1));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x1C000000), i32_val(2));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x1E000000), i32_val(3));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x20000000), i32_val(4));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x21000000), i32_val(5));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x22000000), i32_val(6));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x23000000), i32_val(7));

  Value predef_idx0 = lshr(and_(c0, i32_val(7 << 23)), i32_val(23));
  Value predef_idx1 = lshr(and_(c1, i32_val(7 << 23)), i32_val(23));

  Value normalized0 = extract_element(i32_ty, predefined, predef_idx0);
  Value normalized1 = extract_element(i32_ty, predefined, predef_idx1);

  Value d0 = add(i32_ty, c0, i32_val(0x20000000));
  Value d1 = add(i32_ty, c1, i32_val(0x20000000));

  Value res0 = select(cmp0, normalized0, d0);
  Value res1 = select(cmp1, normalized1, d1);

  Value f0 = or_(i32_ty, res0, lshr(i32_ty, res1, i32_val(16)));
  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = or_(i32_ty, sign0, f0);
  fp16x2Vec0 = bitcast(fp16x2Vec0, fp16x2VecTy);

  return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec0, i32_val(1))};
}

// Fp16 -> Fp8E4M3 (packed)
static SmallVector<Value>
Fp16_to_Fp8E4M3Nv_func(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);

  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));

  fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec0 = sub(i32_ty, fp16x2Vec0, i32_val(0x20002000));

  Value a0 = shl(i32_ty, fp16x2Vec0, i32_val(1));
  a0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  a0 = add(i32_ty, a0, i32_val(0x00800080));
  Value b0 = or_(i32_ty, and_(i32_ty, fp16x2Vec0, i32_val(0x80008000)), a0);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  b0 = bitcast(b0, fp8x4VecTy);

  return {extract_element(i8_ty, b0, i32_val(1)),
          extract_element(i8_ty, b0, i32_val(3))};
}

static SmallVector<Value>
Fp16_to_Fp8E4M3Nv_RTNE_func(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v) {
  Value val = zext(i32_ty, bitcast(v[0], i16_ty));
  Value sign = and_(i32_ty, val, i32_val(0x8000));
  Value nosign = and_(i32_ty, val, i32_val(0x7fff));

  Value exp = and_(i32_ty, lshr(nosign, i32_val(10)), i32_val(0x1f));
  // Check if we need a translation to a subnormal value. This happens when
  // exp value is in range [5, 8].
  Value is_subnormal =
      and_(icmp_uge(exp, i32_val(5)), icmp_ule(exp, i32_val(8)));
  Value shift = sub(i32_ty, i32_val(8), exp);
  Value subnormal = and_(i32_ty, nosign, i32_val(0x3ff));
  subnormal = or_(i32_ty, subnormal, i32_val(0x400));
  // Make rounding with respect to bits we are going to shift and cut off.
  Value round_step = shl(i32_ty, i32_val(0x100), shift);
  Value tail_mask = sub(i32_ty, round_step, i32_val(1));
  Value tail = and_(i32_ty, subnormal, tail_mask);
  Value threshold = shl(i32_ty, i32_val(0x80), shift);
  Value odd_truncated =
      icmp_ne(and_(i32_ty, subnormal, round_step), i32_val(0));
  Value round_up = or_(icmp_ugt(tail, threshold),
                       and_(icmp_eq(tail, threshold), odd_truncated));
  subnormal = select(round_up, add(i32_ty, subnormal, round_step), subnormal);
  // Now shift to get the final result.
  subnormal = lshr(i32_ty, subnormal, shift);

  // Normalized case. Start with rounding, then apply exp range to fit 4 bits,
  // adjust bias and shift left.
  // TODO: NaN values might be mishandled.
  tail = and_(i32_ty, nosign, i32_val(0x7f));
  odd_truncated = icmp_ne(and_(i32_ty, nosign, i32_val(0x80)), i32_val(0));
  round_up = or_(icmp_ugt(tail, i32_val(0x40)),
                 and_(icmp_eq(tail, i32_val(0x40)), odd_truncated));
  Value rounded =
      and_(i32_ty, add(i32_ty, nosign, i32_val(0x80)), i32_val(0x7f80));
  nosign = select(round_up, rounded, nosign);

  nosign = umax(i32_ty, nosign, i32_val(0x2000));
  nosign = umin(i32_ty, nosign, i32_val(0x5c00));
  nosign = sub(i32_ty, nosign, i32_val(0x2000));
  nosign = shl(i32_ty, nosign, i32_val(1));

  // Choose between subnormal and normal values.
  nosign = select(is_subnormal, subnormal, nosign);

  Value res_val = or_(i32_ty, nosign, sign);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value res = bitcast(res_val, fp8x4VecTy);

  return {extract_element(i8_ty, res, i32_val(1))};
}

static SmallVector<Value>
Fp8E4M3Nv_to_Bf16_func(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = undef(fp8x4VecTy);
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(0));
  a0 = insert_element(fp8x4VecTy, a0, v[0], i32_val(1));
  a0 = insert_element(fp8x4VecTy, a0, int_val(8, 0), i32_val(2));
  a0 = insert_element(fp8x4VecTy, a0, v[1], i32_val(3));
  a0 = bitcast(a0, i32_ty);

  Value a1 = undef(fp8x4VecTy);
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(0));
  a1 = insert_element(fp8x4VecTy, a1, v[2], i32_val(1));
  a1 = insert_element(fp8x4VecTy, a1, int_val(8, 0), i32_val(2));
  a1 = insert_element(fp8x4VecTy, a1, v[3], i32_val(3));
  a1 = bitcast(a1, i32_ty);

  Value b0 = and_(i32_ty, a0, i32_val(0x7fff7fff));
  Value b1 = and_(i32_ty, a1, i32_val(0x7fff7fff));
  b0 = lshr(i32_ty, b0, i32_val(4));
  b1 = lshr(i32_ty, b1, i32_val(4));

  Value c0 = and_(i32_ty, b0, i32_val(0xffff0000));
  Value c1 = shl(i32_ty, b0, i32_val(16));
  Value c2 = and_(i32_ty, b1, i32_val(0xffff0000));
  Value c3 = shl(i32_ty, b1, i32_val(16));

  // Check if the exponent is zero, i.e. subnormal number.
  // fp8e4 has a bias of 7
  // depending on the significand value normalization goes like:
  //  [000] -> 0x0
  //  [001] -> exp=127-9, sig=0x0
  //  [010] -> exp=127-8, sig=0x0
  //  [011] -> exp=127-8, sig=b1000...
  //  [100] -> exp=127-7, sig=0x0
  //  [101] -> exp=127-7, sig=b0100...
  //  [110] -> exp=127-7, sig=b1000...
  //  [111] -> exp=127-7, sig=b1100...
  Value cmp0 = icmp_eq(and_(c0, i32_val(0xf800000)), i32_val(0));
  Value cmp1 = icmp_eq(and_(c1, i32_val(0xf800000)), i32_val(0));
  Value cmp2 = icmp_eq(and_(c2, i32_val(0xf800000)), i32_val(0));
  Value cmp3 = icmp_eq(and_(c3, i32_val(0xf800000)), i32_val(0));

  auto i32x8VecTy = vec_ty(i32_ty, 8);
  Value predefined = undef(i32x8VecTy);
  predefined = insert_element(i32x8VecTy, predefined, i32_val(0x0), i32_val(0));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3B000000), i32_val(1));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3B800000), i32_val(2));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3BC00000), i32_val(3));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3C000000), i32_val(4));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3C200000), i32_val(5));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3C400000), i32_val(6));
  predefined =
      insert_element(i32x8VecTy, predefined, i32_val(0x3C600000), i32_val(7));

  Value predef_idx0 = lshr(and_(c0, i32_val(7 << 20)), i32_val(20));
  Value predef_idx1 = lshr(and_(c1, i32_val(7 << 20)), i32_val(20));
  Value predef_idx2 = lshr(and_(c2, i32_val(7 << 20)), i32_val(20));
  Value predef_idx3 = lshr(and_(c3, i32_val(7 << 20)), i32_val(20));

  Value normalized0 = extract_element(i32_ty, predefined, predef_idx0);
  Value normalized1 = extract_element(i32_ty, predefined, predef_idx1);
  Value normalized2 = extract_element(i32_ty, predefined, predef_idx2);
  Value normalized3 = extract_element(i32_ty, predefined, predef_idx3);

  Value d0 = add(i32_ty, c0, i32_val(0x3c000000));
  Value d1 = add(i32_ty, c1, i32_val(0x3c000000));
  Value d2 = add(i32_ty, c2, i32_val(0x3c000000));
  Value d3 = add(i32_ty, c3, i32_val(0x3c000000));

  Value res0 = select(cmp0, normalized0, d0);
  Value res1 = select(cmp1, normalized1, d1);
  Value res2 = select(cmp2, normalized2, d2);
  Value res3 = select(cmp3, normalized3, d3);

  Value f0 = or_(i32_ty, res0, lshr(i32_ty, res1, i32_val(16)));
  Value f1 = or_(i32_ty, res2, lshr(i32_ty, res3, i32_val(16)));

  Value sign0 = and_(i32_ty, a0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, a1, i32_val(0x80008000));

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, f0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, f1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return {extract_element(bf16_ty, bf16x2Vec0, i32_val(0)),
          extract_element(bf16_ty, bf16x2Vec0, i32_val(1)),
          extract_element(bf16_ty, bf16x2Vec1, i32_val(0)),
          extract_element(bf16_ty, bf16x2Vec1, i32_val(1))};
}

static SmallVector<Value>
Bf16_to_Fp8E4M3Nv_func(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = undef(bf16x2VecTy);
  Value bf16x2Vec1 = undef(bf16x2VecTy);
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[0], i32_val(0));
  bf16x2Vec0 = insert_element(bf16x2VecTy, bf16x2Vec0, v[1], i32_val(1));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[2], i32_val(0));
  bf16x2Vec1 = insert_element(bf16x2VecTy, bf16x2Vec1, v[3], i32_val(1));
  bf16x2Vec0 = bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x80008000));
  Value sign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = undef(fp8x4VecTy);
  sign0 = bitcast(sign0, fp8x4VecTy);
  sign1 = bitcast(sign1, fp8x4VecTy);
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(1)), i32_val(0));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign0, i32_val(3)), i32_val(1));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(1)), i32_val(2));
  sign = insert_element(fp8x4VecTy, sign,
                        extract_element(i8_ty, sign1, i32_val(3)), i32_val(3));
  sign = bitcast(sign, i32_ty);

  Value nosign0 = and_(i32_ty, bf16x2Vec0, i32_val(0x7fff7fff));
  Value nosign1 = and_(i32_ty, bf16x2Vec1, i32_val(0x7fff7fff));

  Value nosign_0_0 = and_(i32_ty, nosign0, i32_val(0xffff0000));
  nosign_0_0 = umax(i32_ty, nosign_0_0, i32_val(0x3c000000));
  nosign_0_0 = umin(i32_ty, nosign_0_0, i32_val(0x43f00000));
  Value nosign_0_1 = and_(i32_ty, nosign0, i32_val(0x0000ffff));
  nosign_0_1 = umax(i32_ty, nosign_0_1, i32_val(0x3c00));
  nosign_0_1 = umin(i32_ty, nosign_0_1, i32_val(0x43f0));
  nosign0 = or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = and_(i32_ty, nosign1, i32_val(0xffff0000));
  nosign_1_0 = umax(i32_ty, nosign_1_0, i32_val(0x3c000000));
  nosign_1_0 = umin(i32_ty, nosign_1_0, i32_val(0x43f00000));
  Value nosign_1_1 = and_(i32_ty, nosign1, i32_val(0x0000ffff));
  nosign_1_1 = umax(i32_ty, nosign_1_1, i32_val(0x3c00));
  nosign_1_1 = umin(i32_ty, nosign_1_1, i32_val(0x43f0));
  nosign1 = or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = add(i32_ty, nosign0, i32_val(0x80008));
  nosign1 = add(i32_ty, nosign1, i32_val(0x80008));
  nosign0 = sub(i32_ty, nosign0, i32_val(0x3c003c00));
  nosign1 = sub(i32_ty, nosign1, i32_val(0x3c003c00));
  nosign0 = lshr(i32_ty, nosign0, i32_val(4));
  nosign1 = lshr(i32_ty, nosign1, i32_val(4));

  nosign0 = bitcast(nosign0, fp8x4VecTy);
  nosign1 = bitcast(nosign1, fp8x4VecTy);
  Value nosign = undef(fp8x4VecTy);
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(0)), i32_val(0));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign0, i32_val(2)), i32_val(1));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(0)), i32_val(2));
  nosign =
      insert_element(fp8x4VecTy, nosign,
                     extract_element(i8_ty, nosign1, i32_val(2)), i32_val(3));
  nosign = bitcast(nosign, i32_ty);

  Value fp8x4Vec = or_(i32_ty, nosign, sign);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);
  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
          extract_element(i8_ty, fp8x4Vec, i32_val(1)),
          extract_element(i8_ty, fp8x4Vec, i32_val(2)),
          extract_element(i8_ty, fp8x4Vec, i32_val(3))};
}

static SmallVector<Value>
Bf16_to_Fp8E4M3Nv_RTNE_func(Location loc, ConversionPatternRewriter &rewriter,
                            const SmallVector<Value> &v) {
  Value val = zext(i32_ty, bitcast(v[0], i16_ty));
  Value sign = and_(i32_ty, val, i32_val(0x8000));
  Value nosign = and_(i32_ty, val, i32_val(0x7fff));

  Value exp = and_(i32_ty, lshr(nosign, i32_val(7)), i32_val(0xff));
  // Check if we need a translation to a subnormal value. This happens when
  // exp value is in range [117, 120].
  Value is_subnormal =
      and_(icmp_uge(exp, i32_val(117)), icmp_ule(exp, i32_val(120)));
  Value shift = sub(i32_ty, exp, i32_val(117));
  Value subnormal = and_(i32_ty, nosign, i32_val(0x7f));
  subnormal = or_(i32_ty, subnormal, i32_val(0x80));
  // Make rounding with respect to bits we are going to shift and cut off.
  Value round_step = lshr(i32_ty, i32_val(0x100), shift);
  Value tail_mask = sub(i32_ty, round_step, i32_val(1));
  Value tail = and_(i32_ty, subnormal, tail_mask);
  Value threshold = lshr(i32_ty, i32_val(0x80), shift);
  Value odd_truncated =
      icmp_ne(and_(i32_ty, subnormal, round_step), i32_val(0));
  Value round_up = or_(icmp_ugt(tail, threshold),
                       and_(icmp_eq(tail, threshold), odd_truncated));
  subnormal = select(round_up, add(i32_ty, subnormal, round_step), subnormal);
  // Now shift to get the final result.
  subnormal = shl(i32_ty, subnormal, shift);

  // Normalized case. Start with rounding, then apply exp range to fit 4 bits,
  // adjust bias and shift left.
  // TODO: NaN values might be mishandled.
  tail = and_(i32_ty, nosign, i32_val(0xf));
  odd_truncated = icmp_ne(and_(i32_ty, nosign, i32_val(0x10)), i32_val(0));
  round_up = or_(icmp_ugt(tail, i32_val(0x8)),
                 and_(icmp_eq(tail, i32_val(0x8)), odd_truncated));
  Value rounded =
      and_(i32_ty, add(i32_ty, nosign, i32_val(0x10)), i32_val(0x7ff0));
  nosign = select(round_up, rounded, nosign);

  nosign = umax(i32_ty, nosign, i32_val(0x3c00));
  nosign = umin(i32_ty, nosign, i32_val(0x4380));
  nosign = sub(i32_ty, nosign, i32_val(0x3c00));
  nosign = shl(i32_ty, nosign, i32_val(4));

  // Choose between subnormal and normal values.
  nosign = select(is_subnormal, subnormal, nosign);

  Value res_val = or_(i32_ty, nosign, sign);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value res = bitcast(res_val, fp8x4VecTy);

  return {extract_element(i8_ty, res, i32_val(1))};
}

static SmallVector<Value> Bf16_to_Fp16_func(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);

  Value bf16x2Vec = undef(bf16x2VecTy);
  bf16x2Vec = insert_element(bf16x2VecTy, bf16x2Vec, v[0], i32_val(0));
  bf16x2Vec = insert_element(bf16x2VecTy, bf16x2Vec, v[1], i32_val(1));
  bf16x2Vec = bitcast(bf16x2Vec, i32_ty);

  Value sign = and_(i32_ty, bf16x2Vec, i32_val(0x80008000));
  Value nosign = and_(i32_ty, bf16x2Vec, i32_val(0x7fff7fff));

  // BF16 exp range is 0..255 with bias 127
  // FP16 exp range is 0..31 with bias 15
  // So, BF16 exp values has to be adjusted by subtracting 112.
  // Min BF16 value we can convert is 112 << 7 = 0x3800
  // Max BF16 value we can convert is 143 << 7 + <max fraction> =
  // 0x4780 + 0x7F = 0x47FF
  Value nosign_0 = and_(i32_ty, nosign, i32_val(0xffff0000));
  nosign_0 = umax(i32_ty, nosign_0, i32_val(0x38000000));
  nosign_0 = umin(i32_ty, nosign_0, i32_val(0x47ff0000));
  Value nosign_1 = and_(i32_ty, nosign, i32_val(0xffff));
  nosign_1 = umax(i32_ty, nosign_1, i32_val(0x3800));
  nosign_1 = umin(i32_ty, nosign_1, i32_val(0x47ff));
  nosign = or_(i32_ty, nosign_0, nosign_1);

  nosign = sub(i32_ty, nosign, i32_val(0x38003800));
  nosign = shl(i32_ty, nosign, i32_val(3));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec = or_(i32_ty, nosign, sign);
  fp16x2Vec = bitcast(fp16x2Vec, fp16x2VecTy);
  return {extract_element(f16_ty, fp16x2Vec, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec, i32_val(1))};
}

inline Type getFunctionType(Type resultType, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

inline LLVM::LLVMFuncOp
appendOrGetExternFuncOp(ConversionPatternRewriter &rewriter, Operation *op,
                        StringRef funcName, Type funcType,
                        StringRef libname = "", StringRef libpath = "") {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  auto parent = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder b(parent);
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  ret.getOperation()->setAttr("libname",
                              StringAttr::get(op->getContext(), libname));
  ret.getOperation()->setAttr("libpath",
                              StringAttr::get(op->getContext(), libpath));
  return ret;
}

inline Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    return tensorType.getElementType();
  return type;
}

inline SmallVector<Value> unpackI32(const SmallVector<Value> &inValues,
                                    Type srcTy,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    TypeConverter *typeConverter) {
  auto tensorTy = dyn_cast<RankedTensorType>(srcTy);
  if (!tensorTy)
    return inValues;
  auto encoding = dyn_cast<DotOperandEncodingAttr>(tensorTy.getEncoding());
  if (!(encoding && isa<NvidiaMmaEncodingAttr>(encoding.getParent())))
    return inValues;
  SmallVector<Value> outValues;
  for (auto v : inValues) {
    // cast i32 to appropriate eltType vector and extract elements
    auto eltType = typeConverter->convertType(tensorTy.getElementType());
    auto vecType = vec_ty(eltType, 32 / eltType.getIntOrFloatBitWidth());
    auto vec = bitcast(v, vecType);
    for (int i = 0; i < 32 / eltType.getIntOrFloatBitWidth(); i++) {
      outValues.push_back(extract_element(vec, i32_val(i)));
    }
  }
  return outValues;
}

inline SmallVector<Value> packI32(const SmallVector<Value> &inValues,
                                  Type srcTy,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc, TypeConverter *typeConverter) {
  auto tensorTy = dyn_cast<RankedTensorType>(srcTy);
  if (!tensorTy)
    return inValues;
  auto encoding = dyn_cast<DotOperandEncodingAttr>(tensorTy.getEncoding());
  if (!(encoding && isa<NvidiaMmaEncodingAttr>(encoding.getParent())))
    return inValues;
  SmallVector<Value> outValues;
  auto eltType = typeConverter->convertType(tensorTy.getElementType());
  int vecWidth = 32 / eltType.getIntOrFloatBitWidth();
  auto vecType = vec_ty(eltType, vecWidth);
  for (int i = 0; i < inValues.size(); i += vecWidth) {
    Value vec = undef(vecType);
    for (int j = 0; j < vecWidth; j++) {
      vec = insert_element(vec, inValues[i + j], i32_val(j));
    }
    outValues.push_back(bitcast(vec, i32_ty));
  }
  return outValues;
}

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

class MultipleOperandsRange
    : public iterator_range<SmallVector<SmallVector<Value>>::iterator> {
  using ContainerT = SmallVector<SmallVector<Value>>;

public:
  using iterator_range<ContainerT::iterator>::iterator_range;
  ContainerT::reference operator[](ContainerT::size_type idx) {
    return begin()[idx];
  }
  ContainerT::const_reference operator[](ContainerT::size_type idx) const {
    return begin()[idx];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

// Attempts to use vectorized conversions via inline PTX when possible.
struct FpToFpOpConversion
    : public ElementwiseOpConversionBase<FpToFpOp, FpToFpOpConversion> {
  using ElementwiseOpConversionBase<
      FpToFpOp, FpToFpOpConversion>::ElementwiseOpConversionBase;

  explicit FpToFpOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit) {}

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    return rewriter.create<LLVM::FPExtOp>(loc, f32_ty, v);
  }

  static LLVM::RoundingMode
  convertTritonRoundingModeToLLVM(const RoundingMode rounding) {
    LLVM::RoundingMode roundingMode;
    switch (rounding) {
    case RoundingMode::RTNE:
      return LLVM::RoundingMode::NearestTiesToEven;
    case RoundingMode::RTZ:
      return LLVM::RoundingMode::TowardZero;
    default:
      llvm::errs() << "WARNING: unsupported rounding mode for f32->f16 "
                      "conversion: "
                   << stringifyRoundingMode(rounding) << "\n";
      llvm_unreachable("");
    }
  }

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    MLIRContext *ctx = rewriter.getContext();
    return rewriter.create<LLVM::ConstrainedFPTruncIntr>(
        loc, f16_ty, v,
        LLVM::RoundingModeAttr::get(ctx,
                                    convertTritonRoundingModeToLLVM(rounding)),
        arith::getLLVMDefaultFPExceptionBehavior(*ctx));
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3B15TyID = TypeID::get<Float8E4M3B11FNUZType>();
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    if (srcTy.getTypeID() == dstTy.getTypeID()) {
      constexpr auto identityFn = [](Location,
                                     ConversionPatternRewriter &rewriter,
                                     const SmallVector<Value> &v) { return v; };
      return {identityFn, (srcTy.getTypeID() == F8E4M3TyID ||
                           dstTy.getTypeID() == F8E4M3TyID)
                              ? 2
                              : 4};
    }

    auto undefRounding = static_cast<RoundingMode>(-1);
    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>,
                    std::pair<ConverterT, size_t>>
        srcMap = {
            // F8 -> F16
            {{F8E4M3B15TyID, F16TyID, undefRounding},
             {Fp8E4M3B15_to_Fp16_func, 4}},
            {{F8E4M3TyID, F16TyID, undefRounding}, {Fp8E4M3Nv_to_Fp16_func, 2}},
            {{F8E5M2TyID, F16TyID, undefRounding}, {Fp8E5M2_to_Fp16_func, 4}},
            // F16 -> F8
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E4M3B15_func, 4}},
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTNE},
             // TODO: provide proper implementation for RTNE rounding.
             {Fp16_to_Fp8E4M3B15_func, 4}},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E4M3Nv_func, 2}},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {Fp16_to_Fp8E4M3Nv_RTNE_func, 1}},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E5M2_func, 4}},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {Fp16_to_Fp8E5M2_RTNE_func, 1}},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding}, {Fp8E5M2_to_Bf16_func, 4}},
            {{F8E4M3TyID, BF16TyID, undefRounding},
             {Fp8E4M3Nv_to_Bf16_func, 4}},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTZ},
             {Bf16_to_Fp8E5M2_func, 4}},
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {Bf16_to_Fp8E5M2_RTNE_func, 1}},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTZ},
             {Bf16_to_Fp8E4M3Nv_func, 4}},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {Bf16_to_Fp8E4M3Nv_RTNE_func, 1}},
            // BF16 -> F16
            {{BF16TyID, F16TyID, undefRounding}, {Bf16_to_Fp16_func, 2}},
        };

    std::tuple<TypeID, TypeID, RoundingMode> key = {
        srcTy.getTypeID(), dstTy.getTypeID(),
        roundingMode.value_or(undefRounding)};
    if (srcMap.count(key) == 0) {
      llvm::errs() << "Unsupported conversion from " << srcTy << " to "
                   << dstTy;
      if (roundingMode.has_value())
        llvm::errs() << " with rounding mode "
                     << stringifyRoundingMode(roundingMode.value());
      llvm::errs() << "\n";
      llvm_unreachable("");
    }
    return srcMap.lookup(key);
  }

  SmallVector<Value> createDestOps(FpToFpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (dstElementType.isFloat8E5M2() || dstElementType.isFloat8E4M3FN()) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for convertsions to fp8");

      // For now only RTNE is supported for conversions from fp16 to fp8
      if (!srcElementType.isF32() &&
          roundingMode.value() != RoundingMode::RTNE) {
        llvm::errs() << "Unsupported rounding mode for conversion to fp8: "
                     << stringifyRoundingMode(roundingMode.value()) << "\n";
        llvm_unreachable("");
      }
    }

    if (srcElementType.isF32() && dstElementType.isF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->fp16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            convertFp32ToFp16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    if (srcElementType.isF32() && dstElementType.isBF16()) {
      assert(roundingMode.has_value() &&
             "rounding mode must be specified for fp32->bf16 conversion");
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(
            intel::convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
      }
      return outVals;
    }

    bool useFP16IntermediateSrc = srcElementType.isF32();
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, roundingMode.value());
    inVals.resize(numElements, undef(typeConverter->convertType(srcType)));
    SmallVector<Value> outVals = cvtFunc(loc, rewriter, inVals);
    assert(outVals.size() == inVals.size());
    outVals.resize(std::min(numElements, operands.size()));
    if (isDstFP32)
      for (Value &v : outVals)
        v = convertFp16ToFp32(loc, rewriter, v);
    // Pack values
    return outVals;
  }
};

template <typename OP>
Value EmitDualBF16ElementwiseOp(Location loc,
                                ConversionPatternRewriter &rewriter,
                                MultipleOperandsRange operands) {
  auto v0 = intel::convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 = intel::convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  auto undefRounding = static_cast<RoundingMode>(-1);
  return intel::convertFp32ToBf16(loc, rewriter, result, undefRounding);
}

struct ExternElementwiseOpConversion
    : public ElementwiseOpConversionBase<ExternElementwiseOp,
                                         ExternElementwiseOpConversion> {
  using Base = ElementwiseOpConversionBase<ExternElementwiseOp,
                                           ExternElementwiseOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  SmallVector<Value> createDestOps(ExternElementwiseOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    StringRef funcName = op.getSymbol();
    if (funcName.empty())
      llvm::errs() << "ExternElementwiseOpConversion";

    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp = appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType, op.getLibname(), op.getLibpath());

    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

    return {callOp.getResult()};
  }
};

struct FDivOpConversion
    : ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::DivFOp, FDivOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::DivFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {rewriter.create<LLVM::FDivOp>(loc, elemTy, operands[0][0],
                                          operands[0][1])};
  }
};

struct FMulOpConversion
    : ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::MulFOp, FMulOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::MulFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());

    bool lhsAndRhsAreBF16 = lhsElemTy.isBF16() && rhsElemTy.isBF16();

    if (lhsAndRhsAreBF16)
      return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};

    return {rewriter.create<LLVM::FMulOp>(loc, elemTy, operands[0][0],
                                          operands[0][1])};
  }
};

struct FAddOpConversion
    : ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::AddFOp, FAddOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::AddFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    bool lhsAndRhsAreBF16 = lhsElemTy.isBF16() && rhsElemTy.isBF16();

    if (lhsAndRhsAreBF16)
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};

    return {rewriter.create<LLVM::FAddOp>(loc, elemTy, operands[0][0],
                                          operands[0][1])};
  }
};

struct FSubOpConversion
    : ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SubFOp, FSubOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SubFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto lhsElemTy = getElementType(op.getLhs());
    auto rhsElemTy = getElementType(op.getRhs());
    bool lhsAndRhsAreBF16 = lhsElemTy.isBF16() && rhsElemTy.isBF16();

    if (lhsAndRhsAreBF16)
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};

    return {rewriter.create<LLVM::FSubOp>(loc, elemTy, operands[0][0],
                                          operands[0][1])};
  }
};

// Uses inline ptx to convert s8/u8 to bf16, since the
struct SIToFPOpConversion
    : ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SIToFPOp, SIToFPOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SIToFPOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type inElemTy = getElementType(op.getIn());
    Type outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16() && inElemTy.isInteger(8) && operands.size() >= 4) {
      SmallVector<Value> outVals;
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {
          intel::convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
      llvm_unreachable("");
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {
          intel::convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
    }

    return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
  }
};

struct FPToSIOpConversion
    : ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::FPToSIOp, FPToSIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::FPToSIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto value = intel::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value)};
    }

    return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
  }
};

struct ExtFOpConversion
    : ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::ExtFOp, ExtFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::ExtFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy = getElementType(op.getIn());
    if (inElemTy.isBF16()) {
      auto outElemTy = getElementType(op.getOut());
      assert(outElemTy.isF32() && "unsupported conversion");
      return {intel::convertBf16ToFp32(loc, rewriter, operands[0][0])};
    }

    return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
  }
};

struct TruncFOpConversion
    : ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::TruncFOp, TruncFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::TruncFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto outElemTy = getElementType(op.getOut());
    if (outElemTy.isBF16()) {
      auto inElemTy = getElementType(op.getIn());
      assert(inElemTy.isF32() && "unsupported conversion");
      return {// Trunc uses the default rounding mode: RTNE
              intel::convertFp32ToBf16(loc, rewriter, operands[0][0],
                                       RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
    }
  }
};

struct ExpOpConversionApprox
    : ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox> {
  using Base = ElementwiseOpConversionBase<math::ExpOp, ExpOpConversionApprox>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::ExpOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // For non-FP32 input, call math library expf for higher-precision
    // calculation
    if (elemTy.getIntOrFloatBitWidth() != 32)
      return {};

    const double log2e = 1.4426950408889634;
    Value prod = fmul(f32_ty, operands[0][0], f32_val(log2e));

    return {rewriter.create<math::Exp2Op>(loc, f32_ty, prod,
                                          adaptor.getAttributes().getValue())};
  }
};

struct AbsFOpConversion
    : ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsFOp, AbsFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // FIXME: Remove bitcast to and from i16 once SPIRV-LLVM-Translator supports
    // LLVM::FAbsOp with bf16.
    Value v = operands[0][0];
    Type orig_type = elemTy;
    if (llvm::isa<BFloat16Type>(orig_type)) {
      v = bitcast(v, i16_ty);
      elemTy = i16_ty;
    }
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      Value res = and_(v, maskConst);
      if (llvm::isa<BFloat16Type>(orig_type))
        res = bitcast(res, orig_type);
      return {res};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, v)};
  }
};

struct MulhiUIOpConversion
    : public ElementwiseOpConversionBase<MulhiUIOp, MulhiUIOpConversion> {
  using Base = ElementwiseOpConversionBase<MulhiUIOp, MulhiUIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  explicit MulhiUIOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit),
        targetInfo(targetInfo) {}
  SmallVector<Value> createDestOps(MulhiUIOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {

    Type resultElementTy = getElementTypeOrSelf(op.getResult().getType());
    assert(resultElementTy.isInteger(32) || resultElementTy.isInteger(64));

    std::string funcName = targetInfo.getMulhiFuncName(resultElementTy);
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    return {callOp.getResult()};
  }

protected:
  const TargetInfoBase &targetInfo;
};

template <typename TritonOp>
struct OpToExternCallConversion
    : public ElementwiseOpConversionBase<TritonOp,
                                         OpToExternCallConversion<TritonOp>> {
  using Base =
      ElementwiseOpConversionBase<TritonOp, OpToExternCallConversion<TritonOp>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit OpToExternCallConversion(LLVMTypeConverter &typeConverter,
                                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                                    StringRef externFuncName,
                                    PatternBenefit benefit)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit),
        funcName(externFuncName) {}

  SmallVector<Value> createDestOps(TritonOp op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    return {callOp.getResult()};
  }

private:
  StringRef funcName;
};

} // namespace

namespace mlir::triton::intel {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "__imf_sqrtf", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "__imf_fdiv_rn", benefit);

  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
  patterns.add<MulhiUIOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                    benefit);
  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);

  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // a vendor specific math library for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  // TODO(FIXME): spirv's OpenCL extension (fmin/fmax) does not support
  // nan propagation. Set these conversion benefit to the max benefit:
  // PatternBenefit::ImpossibleToMatchSentinel - 1 to make sure the
  // correctness
  PatternBenefit benefitForPropNan = 65534;
  mlir::triton::populateMinMaxFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis,
      /*hwNanPropagationSupported=*/false, benefitForPropNan);
  mlir::triton::populateClampFOpToLLVMPattern(
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
}

} // namespace mlir::triton::intel
