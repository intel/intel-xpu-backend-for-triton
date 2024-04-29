#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace {
static SmallVector<Value> identity_func(Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        const SmallVector<Value> &v) {
  return v;
}
} // namespace
namespace mlir::triton {

namespace gpu {
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

  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, f0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, f1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return {extract_element(i16_ty, bf16x2Vec0, i32_val(0)),
          extract_element(i16_ty, bf16x2Vec0, i32_val(1)),
          extract_element(i16_ty, bf16x2Vec1, i32_val(0)),
          extract_element(i16_ty, bf16x2Vec1, i32_val(1))};
}

static SmallVector<Value>
Bf16_to_Fp8E5M2_func(Location loc, ConversionPatternRewriter &rewriter,
                     const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(i16_ty, 2);
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

/* ----- FP8E4M3B15X4 ------ */
// NOTE: NOT USED RIGHT NOW
// Packed variant of FP8E4M3B15
// A little bit more efficient but elements need are not
// serialized as you expect when 4 are packed into int32.

// fast conversion code provided by Scott Gray @ OpenAI
// $0 = (($2 << 1) & 0x80008000u) | (($2 << 7) & 0x3f803f80u);
// $1 = (($2 << 0) & 0x80008000u) | (($2 << 0) & 0x3f803f80u);
// WARN: subnormal (0bs0000xxx) are not handled
static SmallVector<Value>
Fp8E4M3B15x4_to_Fp16_func(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value fp8x4Vec = undef(fp8x4VecTy);
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[0], i32_val(0));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[1], i32_val(1));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[2], i32_val(2));
  fp8x4Vec = insert_element(fp8x4VecTy, fp8x4Vec, v[3], i32_val(3));
  fp8x4Vec = bitcast(fp8x4Vec, i32_ty);

  Value a0 = add(i32_ty, fp8x4Vec, fp8x4Vec);
  Value a1 = shl(i32_ty, fp8x4Vec, i32_val(7));

  Value fp16x2Vec0 = and_(i32_ty, a0, i32_val(0x80008000));
  fp16x2Vec0 = or_(i32_ty, fp16x2Vec0, and_(i32_ty, a1, i32_val(0x3f803f80)));
  Value fp16x2Vec1 = and_(i32_ty, fp8x4Vec, i32_val(0xbf80bf80));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  fp16x2Vec0 = bitcast(fp16x2Vec0, fp16x2VecTy);
  fp16x2Vec1 = bitcast(fp16x2Vec1, fp16x2VecTy);

  return {extract_element(f16_ty, fp16x2Vec0, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec0, i32_val(1)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(0)),
          extract_element(f16_ty, fp16x2Vec1, i32_val(1))};
}

// Fp16 -> Fp8E4M3B15 (packed)
// fast conversion code provided by Scott Gray @ OpenAI
// ret = ((e4.x >> 1) & (0x80008000u >> 1)) |
//       ((e4.x >> 7) & (0x3f803f80u >> 7)) |
//       ((e4.y >> 0) & (0x80008000u >> 0)) |
//       ((e4.y >> 0) & (0x3f803f80u >> 0)) ;
// WARN: subnormal (0bs0000xxx) are not handled

static SmallVector<Value>
Fp16_to_Fp8E4M3B15x4_func(Location loc, ConversionPatternRewriter &rewriter,
                          const SmallVector<Value> &v) {
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = undef(fp16x2VecTy);
  Value fp16x2Vec1 = undef(fp16x2VecTy);

  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[0], i32_val(0));
  fp16x2Vec0 = insert_element(fp16x2VecTy, fp16x2Vec0, v[1], i32_val(1));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[2], i32_val(0));
  fp16x2Vec1 = insert_element(fp16x2VecTy, fp16x2Vec1, v[3], i32_val(1));

  fp16x2Vec0 = bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec1 = bitcast(fp16x2Vec1, i32_ty);

  Value a0 = lshr(i32_ty, fp16x2Vec0, i32_val(1));
  Value a1 = lshr(i32_ty, fp16x2Vec0, i32_val(7));

  Value fp8x4Vec = and_(i32_ty, a0, i32_val(0x40004000));
  fp8x4Vec = or_(i32_ty, fp8x4Vec, and_(i32_ty, a1, i32_val(0x007f007f)));
  fp8x4Vec =
      or_(i32_ty, fp8x4Vec, and_(i32_ty, fp16x2Vec1, i32_val(0xbf80bf80)));

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  fp8x4Vec = bitcast(fp8x4Vec, fp8x4VecTy);

  return {extract_element(i8_ty, fp8x4Vec, i32_val(0)),
          extract_element(i8_ty, fp8x4Vec, i32_val(1)),
          extract_element(i8_ty, fp8x4Vec, i32_val(2)),
          extract_element(i8_ty, fp8x4Vec, i32_val(3))};
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

  auto bf16x2VecTy = vec_ty(i16_ty, 2);
  Value bf16x2Vec0 = or_(i32_ty, sign0, f0);
  Value bf16x2Vec1 = or_(i32_ty, sign1, f1);
  bf16x2Vec0 = bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = bitcast(bf16x2Vec1, bf16x2VecTy);

  return {extract_element(i16_ty, bf16x2Vec0, i32_val(0)),
          extract_element(i16_ty, bf16x2Vec0, i32_val(1)),
          extract_element(i16_ty, bf16x2Vec1, i32_val(0)),
          extract_element(i16_ty, bf16x2Vec1, i32_val(1))};
}

static SmallVector<Value>
Bf16_to_Fp8E4M3Nv_func(Location loc, ConversionPatternRewriter &rewriter,
                       const SmallVector<Value> &v) {
  auto bf16x2VecTy = vec_ty(i16_ty, 2);
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
  auto bf16x2VecTy = vec_ty(i16_ty, 2);

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

// MMA encoding has a different order depending on the element's bit width;
// reorder if we're in this case.
static SmallVector<Value> reorderValues(const SmallVector<Value> &values,
                                        Type inType, Type ouType) {
  auto inTensorTy = inType.dyn_cast<RankedTensorType>();
  auto ouTensorTy = ouType.dyn_cast<RankedTensorType>();
  if (!inTensorTy || !ouTensorTy)
    return values;
  auto inEncoding = dyn_cast<DotOperandEncodingAttr>(inTensorTy.getEncoding());
  auto ouEncoding = dyn_cast<DotOperandEncodingAttr>(ouTensorTy.getEncoding());
  assert(inEncoding == ouEncoding);
  if (!inEncoding)
    return values;

  // If the parent of the dot operand is in block encoding, we don't need to
  // reorder elements
  auto parentEncoding = dyn_cast<NvidiaMmaEncodingAttr>(ouEncoding.getParent());
  if (!parentEncoding)
    return values;

  size_t inBitWidth = inTensorTy.getElementType().getIntOrFloatBitWidth();
  size_t ouBitWidth = ouTensorTy.getElementType().getIntOrFloatBitWidth();
  auto ouEltTy = ouTensorTy.getElementType();
  if (inBitWidth == ouBitWidth)
    return values;
  if (inBitWidth == 16 && ouBitWidth == 32) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 8) {
      ret.push_back(values[i]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
    }
    return ret;
  }
  if (inBitWidth == 8 && ouBitWidth == 16) {
    SmallVector<Value> ret;
    for (unsigned i = 0; i < values.size(); i += 16) {
      ret.push_back(values[i + 0]);
      ret.push_back(values[i + 1]);
      ret.push_back(values[i + 2]);
      ret.push_back(values[i + 3]);
      ret.push_back(values[i + 8]);
      ret.push_back(values[i + 9]);
      ret.push_back(values[i + 10]);
      ret.push_back(values[i + 11]);
      ret.push_back(values[i + 4]);
      ret.push_back(values[i + 5]);
      ret.push_back(values[i + 6]);
      ret.push_back(values[i + 7]);
      ret.push_back(values[i + 12]);
      ret.push_back(values[i + 13]);
      ret.push_back(values[i + 14]);
      ret.push_back(values[i + 15]);
    }
    return ret;
  }
  llvm_unreachable("unimplemented code path");
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
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType();
  return type;
}

inline SmallVector<Value> unpackI32(const SmallVector<Value> &inValues,
                                    Type srcTy,
                                    ConversionPatternRewriter &rewriter,
                                    Location loc,
                                    TypeConverter *typeConverter) {
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
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
  auto tensorTy = srcTy.dyn_cast<RankedTensorType>();
  if (!tensorTy)
    return inValues;
  auto encoding = tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>();
  if (!(encoding && encoding.getParent().isa<NvidiaMmaEncodingAttr>()))
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

// Base pattern for elementwise conversion using ConcreteT. Unpacks individual
// elements from a `!llvm.struct` via `llvm.extactvalue`, calls
// ConcreteT::createDestOps on each element, and packs them back into an
// `!llvm.struct` using `llvm.insertvalue`.
//
// Also supports processing the inputs in a vectorized form by consuming and
// producing multiple operand sets in ConcreteT::createDestOps.
template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase
    : public ConvertTritonGPUOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(LLVMTypeConverter &typeConverter,
                                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                                       PatternBenefit benefit = 1)
      : ConvertTritonGPUOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        axisAnalysisPass(axisAnalysisPass) {}

  // Try to deduplicate the resultVals based on the
  // constancy properties of the result discovered by
  // the axis analysis pass. If possible, redundant
  // computation is eliminated.
  SmallVector<Value> maybeDeduplicate(SourceOp op,
                                      SmallVector<Value> resultVals) const {
    if (!isMemoryEffectFree(op))
      // the op has side effects: can't dedup
      return resultVals;
    SmallVector<Value> results = op->getResults();
    if (results.size() == 0 || results.size() > 1)
      // there must be exactly 1 result
      return resultVals;
    Value result = results[0];
    Type type = result.getType();
    if (!type)
      return resultVals;
    RankedTensorType rtType = type.dyn_cast<RankedTensorType>();
    if (!rtType)
      // the result must be a tensor
      return resultVals;
    Attribute encoding = rtType.getEncoding();
    if (!encoding)
      // encoding not available
      return resultVals;
    if (!encoding.dyn_cast<BlockedEncodingAttr>() &&
        !encoding.dyn_cast<SliceEncodingAttr>()) {
      // TODO: constraining the ecndoing type here is necessary for avoiding
      // crashes in the getElemsPerThread call below happening in the
      // test_core::test_fp8_dot_acc
      return resultVals;
    }

    SmallVector<unsigned> elemsPerThread = getElemsPerThread(rtType);
    int rank = elemsPerThread.size();
    if (product<unsigned>(elemsPerThread) != resultVals.size())
      return resultVals;
    AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(result);
    if (!axisInfo)
      // axis info (e.g., constancy) not available
      return resultVals;
    SmallVector<unsigned> sizePerThread = getSizePerThread(encoding);
    if (rank != sizePerThread.size())
      return resultVals;

    SmallVector<int64_t> constancy = axisInfo->getConstancy();
    if (rank != constancy.size())
      return resultVals;
    bool hasConstancy = false;
    for (int i = 0; i < rank; ++i) {
      if (constancy[i] > sizePerThread[i]) {
        if (constancy[i] % sizePerThread[i] != 0)
          // constancy is not evenly covered by sizePerThread
          return resultVals;
        // can't move the values across different
        // "sizePerThread"-sized blocks
        constancy[i] = sizePerThread[i];
      }
      if (elemsPerThread[i] < 1 || constancy[i] < 1)
        return resultVals;
      if (!(elemsPerThread[i] % constancy[i] == 0 ||
            constancy[i] % elemsPerThread[i] == 0))
        // either the constancy along each dimension must fit
        // into the elemsPerThread or the other way around
        return resultVals;
      if (constancy[i] > 1)
        hasConstancy = true;
    }
    if (!hasConstancy)
      // nothing to deduplicate
      return resultVals;

    if (rank > 1) {
      // reorder the shape and constancy vectors by the axis order:
      // from the fastest-changing to the smallest-changing axis
      SmallVector<unsigned> order = getOrder(encoding);
      if (rank != order.size())
        return resultVals;
      elemsPerThread = applyPermutation(elemsPerThread, order);
      constancy = applyPermutation(constancy, order);
    }

    SmallVector<unsigned> strides(rank, 1);
    for (int i = 1; i < rank; ++i) {
      strides[i] = strides[i - 1] * elemsPerThread[i - 1];
    }
    SmallVector<Value> dedupResultVals;
    dedupResultVals.reserve(resultVals.size());
    for (int i = 0; i < resultVals.size(); ++i) {
      // each coordinate of the orig_idx is "coarsened" using the
      // constancy along this dimension: the resulting dedup_idx
      // points to the reused value in the original resultsVal
      int orig_idx = i;
      int dedup_idx = 0;
      for (int j = 0; j < rank; ++j) {
        int coord_j = orig_idx % elemsPerThread[j];
        dedup_idx += (coord_j / constancy[j] * constancy[j]) * strides[j];
        orig_idx /= elemsPerThread[j];
      }
      dedupResultVals.push_back(resultVals[dedup_idx]);
    }

    return dedupResultVals;
  }

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      subOperands = unpackI32(subOperands, argTy, rewriter, loc,
                              this->getTypeConverter());
      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands))
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});

    SmallVector<Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
      auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
          op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
      if (curr.size() == 0)
        return failure();
      for (auto v : curr) {
        if (!static_cast<bool>(v))
          return failure();
        resultVals.push_back(v);
      }
      it += curr.size();
    }
    if (op->getNumOperands() > 0) {
      auto argTy = op->getOperand(0).getType();
      resultVals = reorderValues(resultVals, argTy, resultTy);
    }
    resultVals = maybeDeduplicate(op, resultVals);
    resultVals =
        packI32(resultVals, resultTy, rewriter, loc, this->getTypeConverter());
    Value view = packLLElements(loc, this->getTypeConverter(), resultVals,
                                rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {rewriter.create<DestOp>(loc, elemTy, operands[0],
                                    adaptor.getAttributes().getValue())};
  }
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

  static Value convertBf16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    auto as_int16 = bitcast(v, i16_ty);
    auto as_int32 = zext(i32_ty, as_int16);
    auto shifted = shl(i32_ty, as_int32, i32_val(16));
    return (bitcast(shifted, f32_ty));
  }

  static Value convertFp16ToFp32(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v) {
    auto ctx = rewriter.getContext();
    return rewriter.create<LLVM::FPExtOp>(loc, f32_ty, v);
  }

  static Value convertFp32ToBf16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v, const RoundingMode rounding) {
    auto as_uint32 = bitcast(v, i32_ty);
    auto check_exponent =
        and_(i32_ty, xor_(i32_ty, as_uint32, i32_val(0xffffffff)),
             i32_val(0x7f800000));
    auto exponent_not_all1s = icmp_ne(check_exponent, i32_val(0));
    auto exponent_all1s = icmp_eq(check_exponent, i32_val(0));
    Value rounded = as_uint32;
    if (rounding == RoundingMode::RTNE) {
      rounded =
          add(i32_ty, i32_val(0x7fff),
              and_(i32_ty, lshr(i32_ty, as_uint32, i32_val(16)), i32_val(1)));
      rounded = add(i32_ty, rounded, as_uint32);
      rounded = select(exponent_not_all1s, rounded, as_uint32);
    }

    auto preserve_nan =
        and_(i1_ty, exponent_all1s,
             icmp_ne(and_(i32_ty, as_uint32, i32_val(0xffff)), i32_val(0)));
    auto nan = or_(i32_ty, as_uint32, i32_val(0x10000));
    Value res = select(preserve_nan, nan, rounded);

    auto shifted = lshr(i32_ty, res, i32_val(16));
    auto truncated = trunc(i16_ty, shifted);
    return truncated;
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
    auto F8E4M3TyID = TypeID::get<Float8E4M3FNUZType>();
    auto F8E5M2TyID = TypeID::get<Float8E5M2Type>();
    auto F8E4M3FNTyID = TypeID::get<Float8E4M3FNType>();
    auto F16TyID = TypeID::get<Float16Type>();
    auto BF16TyID = TypeID::get<BFloat16Type>();
    auto F32TyID = TypeID::get<Float32Type>();
    auto F64TyID = TypeID::get<Float64Type>();

    if (srcTy.getTypeID() == dstTy.getTypeID()) {
      if (srcTy.getTypeID() == F8E4M3TyID || dstTy.getTypeID() == F8E4M3TyID)
        return {identity_func, 2};
      else
        return {identity_func, 4};
    }

    auto undefRounding = static_cast<RoundingMode>(-1);
    static DenseMap<std::tuple<TypeID, TypeID, RoundingMode>,
                    std::pair<ConverterT, size_t>>
        srcMap = {
            // F8 -> F16
            {{F8E4M3B15TyID, F16TyID, undefRounding},
             {Fp8E4M3B15_to_Fp16_func, 4}},
            {{F8E4M3FNTyID, F16TyID, undefRounding},
             {Fp8E4M3B15x4_to_Fp16_func, 4}},
            {{F8E4M3TyID, F16TyID, undefRounding}, {Fp8E4M3Nv_to_Fp16_func, 2}},
            {{F8E5M2TyID, F16TyID, undefRounding}, {Fp8E5M2_to_Fp16_func, 4}},
            // F16 -> F8
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E4M3B15_func, 4}},
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTNE},
             // TODO: provide proper implementation for RTNE rounding.
             {Fp16_to_Fp8E4M3B15_func, 4}},
            {{F16TyID, F8E4M3FNTyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E4M3B15x4_func, 4}},
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

    if (dstElementType.isFloat8E5M2() || dstElementType.isFloat8E4M3FNUZ()) {
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
            convertFp32ToBf16(loc, rewriter, v, roundingMode.value()));
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
  auto v0 =
      FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
  auto v1 =
      FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][1]);
  auto result = rewriter.create<OP>(loc, f32_ty, v0, v1);
  auto undefRounding = static_cast<RoundingMode>(-1);
  return FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, result,
                                               undefRounding);
}

struct CmpIOpConversion
    : public ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<LLVM::ICmpOp> createDestOps(arith::CmpIOp op, OpAdaptor adaptor,
                                          ConversionPatternRewriter &rewriter,
                                          Type elemTy,
                                          MultipleOperandsRange operands,
                                          Location loc) const {
    return {rewriter.create<LLVM::ICmpOp>(
        loc, elemTy, ArithCmpIPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::ICmpPredicate
  ArithCmpIPredicateToLLVM(arith::CmpIPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__)                                                    \
  case arith::CmpIPredicate::item__:                                           \
    return LLVM::ICmpPredicate::item__

      __PRED_ENUM(eq);
      __PRED_ENUM(ne);
      __PRED_ENUM(sgt);
      __PRED_ENUM(sge);
      __PRED_ENUM(slt);
      __PRED_ENUM(sle);
      __PRED_ENUM(ugt);
      __PRED_ENUM(uge);
      __PRED_ENUM(ult);
      __PRED_ENUM(ule);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpIPredicate");
  }
};

struct CmpFOpConversion
    : public ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::CmpFOp, CmpFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  static SmallVector<LLVM::FCmpOp>
  createDestOps(arith::CmpFOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter, Type elemTy,
                MultipleOperandsRange operands, Location loc) {
    return {rewriter.create<LLVM::FCmpOp>(
        loc, elemTy, ArithCmpFPredicateToLLVM(op.getPredicate()),
        operands[0][0], operands[0][1])};
  }

  static LLVM::FCmpPredicate
  ArithCmpFPredicateToLLVM(arith::CmpFPredicate predicate) {
    switch (predicate) {
#define __PRED_ENUM(item__, item1__)                                           \
  case arith::CmpFPredicate::item__:                                           \
    return LLVM::FCmpPredicate::item1__

      __PRED_ENUM(OEQ, oeq);
      __PRED_ENUM(ONE, one);
      __PRED_ENUM(OGT, ogt);
      __PRED_ENUM(OGE, oge);
      __PRED_ENUM(OLT, olt);
      __PRED_ENUM(OLE, ole);
      __PRED_ENUM(ORD, ord);
      __PRED_ENUM(UEQ, ueq);
      __PRED_ENUM(UGT, ugt);
      __PRED_ENUM(UGE, uge);
      __PRED_ENUM(ULT, ult);
      __PRED_ENUM(ULE, ule);
      __PRED_ENUM(UNE, une);
      __PRED_ENUM(UNO, uno);
      __PRED_ENUM(AlwaysTrue, _true);
      __PRED_ENUM(AlwaysFalse, _false);

#undef __PRED_ENUM
    }
    llvm_unreachable("Unknown arith::CmpFPredicate");
  }
};

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

    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

    return {callOp.getResult()};
  }
};

struct ElementwiseInlineAsmOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<ElementwiseInlineAsmOp> {
  using Base = ConvertTritonGPUOpToLLVMPattern<ElementwiseInlineAsmOp>;

  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;
  typedef typename Base::OpAdaptor OpAdaptor;

  // If operand size is smaller than 32 bits, pack in groups of 32 bits.
  SmallVector<Value> packOperands(ElementwiseInlineAsmOp op,
                                  MultipleOperandsRange operands,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) const {
    SmallVector<Value> packedOperands;
    unsigned numPackedElements = op.getPackedElement();
    for (int i = 0, e = op.getNumOperands(); i < e; i++) {
      Type elemTy = getElementType(op.getOperand(i));
      unsigned bitWidth =
          elemTy.isIntOrFloat() ? elemTy.getIntOrFloatBitWidth() : 64;
      unsigned numElementPerReg = bitWidth < 32 ? 32 / bitWidth : 1;
      numElementPerReg = std::min(numElementPerReg, numPackedElements);
      for (int j = 0; j < numPackedElements; j += numElementPerReg) {
        if (numElementPerReg == 1) {
          packedOperands.push_back(operands[j][i]);
          continue;
        }
        Type t =
            vec_ty(getTypeConverter()->convertType(elemTy), numElementPerReg);
        Value packed = undef(t);
        for (int k = 0; k < numElementPerReg; k++) {
          packed = insert_element(packed, operands[j + k][i], i32_val(k));
        }
        packedOperands.push_back(packed);
      }
    }
    return packedOperands;
  }

  SmallVector<SmallVector<Value>>
  createDestOps(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter,
                MultipleOperandsRange operands, Location loc) const {
    auto ctx = op->getContext();

    if (operands.size() % op.getPackedElement() != 0)
      llvm::report_fatal_error("Inline asm op has more packed elements than "
                               "number of elements per thread.");

    // Pack elems smaller than 32 bits into 32-bit registers.
    SmallVector<Value> packedOperands =
        packOperands(op, operands, rewriter, loc);

    // Types returned by the LLVM asm op.  If there's more than one, they'll be
    // wrapped in a struct.
    SmallVector<Type> asmRetTypes;
    for (auto result : op.getResult()) {
      auto ty = getTypeConverter()->convertType(getElementType(result));

      // Pack return elements into 32-bits.
      unsigned bitWidth = ty.isIntOrFloat() ? ty.getIntOrFloatBitWidth() : 64;
      unsigned numElemsPerReg =
          std::min(bitWidth < 32 ? 32 / bitWidth : 1, op.getPackedElement());
      assert(op.getPackedElement() % numElemsPerReg == 0);
      if (numElemsPerReg > 1) {
        ty = vec_ty(ty, numElemsPerReg);
      }
      for (unsigned i = 0; i < op.getPackedElement() / numElemsPerReg; i++) {
        asmRetTypes.push_back(ty);
      }
    }
    Type asmRetType =
        asmRetTypes.size() > 1 ? struct_ty(asmRetTypes) : asmRetTypes[0];

    Value asmResults =
        rewriter
            .create<LLVM::InlineAsmOp>(
                loc, asmRetType,
                /*operands=*/packedOperands,
                /*asm_string=*/op.getAsmString(),
                /*constraints=*/op.getConstraints(),
                /*has_side_effects=*/!op.getPure(),
                /*is_align_stack=*/false,
                /*asm_dialect=*/
                LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                          LLVM::AsmDialect::AD_ATT),
                /*operand_attrs=*/ArrayAttr())
            ->getResult(0);

    // asmResults is a flat struct; pack its values into
    // [return_value][op.getPackedElement()].
    SmallVector<SmallVector<Value>> ret(op->getNumResults());
    for (int i = 0; i < op->getNumResults(); i++) {
      for (int j = 0; j < op.getPackedElement(); j++) {
        auto val = asmRetTypes.size() > 1
                       ? extract_val(asmResults, i * op.getPackedElement() + j)
                       : asmResults;
        if (auto vectorTy = val.getType().dyn_cast<VectorType>()) {
          for (int k = 0; k < vectorTy.getNumElements(); k++) {
            ret[i].push_back(extract_element(val, i32_val(k)));
          }
          j += vectorTy.getNumElements() - 1;
        } else {
          ret[i].push_back(val);
        }
      }
    }
    return ret;
  }

  LogicalResult
  matchAndRewrite(ElementwiseInlineAsmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Layout is unpackedOperands[operand][elem].
    SmallVector<SmallVector<Value>> unpackedOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      unpackedOperands.push_back(
          unpackI32(subOperands, argTy, rewriter, loc, getTypeConverter()));
    }
    if (unpackedOperands.empty())
      unpackedOperands.push_back({});

    // Although we ensure that all operands and results to this op have the same
    // encoding, MMA layouts have a different physical ordering depending on the
    // bit width of the underlying element.
    //
    // Thus if the inputs to the inline asm op are MMA with different widths, we
    // need to reorder them so we iterate over the operands' elements in the
    // same logical order.
    for (unsigned i = 1; i < unpackedOperands.size(); ++i) {
      unpackedOperands[i] = reorderValues(
          unpackedOperands[i], /*inType=*/op->getOperand(i).getType(),
          /*ouType=*/op->getResult(0).getType());
    }

    // Number of (unpacked) elements to process per operand.  Normally this
    // equals the number of output elements per return value, except when the
    // asm has no inputs, in which case there's 1 output element.
    size_t numInputElems = unpackedOperands[0].size();

    // These are checked by the verifier, so we don't need to raise a nice
    // error.
    assert(all_of(unpackedOperands, [&](auto &operands) {
      return operands.size() == numInputElems;
    }));
    assert(numInputElems % op.getPackedElement() == 0);

    // Run the inline asm op on each block of elements.
    //
    // Layout is unpackedResults[result_idx][elem].
    //
    // This loop always runs at least once, even when the asm has no input
    // elements.
    SmallVector<SmallVector<Value>> unpackedResults(op->getNumResults());
    for (unsigned i = 0; i < std::max(numInputElems, size_t{1});
         i += op.getPackedElement()) {
      // Block of elements to process with one call to the inline asm.  This is
      // ordered opposite `unpackedResults`: The outer dim is
      // op.getPackedElement(), and the inner dim is the operand.
      SmallVector<SmallVector<Value>> block(op.getPackedElement());
      if (numInputElems > 0) {
        for (auto &os : unpackedOperands) {
          for (int j = 0; j < op.getPackedElement(); j++) {
            block[j].push_back(os[i + j]);
          }
        }
      }
      auto cur = createDestOps(op, adaptor, rewriter, block, loc);
      assert(cur.size() == unpackedResults.size());
      for (unsigned j = 0; j < cur.size(); j++) {
        unpackedResults[j].insert(unpackedResults[j].end(), cur[j].begin(),
                                  cur[j].end());
      }
    }

    // Reorder and pack the results.
    SmallVector<Value> outs;
    for (int i = 0; i < unpackedResults.size(); i++) {
      // We reordered all the inputs so they match operand 0.  Reorder the
      // outputs accordingly.
      if (op->getNumOperands() > 0) {
        unpackedResults[i] = reorderValues(
            unpackedResults[i], /*inType=*/op->getOperand(0).getType(),
            /*ouType=*/op->getResult(i).getType());
      }
      auto packed = packI32(unpackedResults[i], op->getResult(i).getType(),
                            rewriter, loc, getTypeConverter());
      outs.push_back(packLLElements(loc, getTypeConverter(), unpackedResults[i],
                                    rewriter, op->getResult(i).getType()));
    }

    rewriter.replaceOp(op, outs);
    return success();
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

    if (lhsAndRhsAreBF16) {
      return {EmitDualBF16ElementwiseOp<LLVM::FMulOp>(loc, rewriter, operands)};
    }

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

    if (lhsAndRhsAreBF16) {
      return {EmitDualBF16ElementwiseOp<LLVM::FAddOp>(loc, rewriter, operands)};
    }

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

    if (lhsAndRhsAreBF16) {
      return {EmitDualBF16ElementwiseOp<LLVM::FSubOp>(loc, rewriter, operands)};
    }
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
      return {FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, value,
                                                    RoundingMode::RTNE)};
      llvm_unreachable("");
    } else if (outElemTy.isBF16()) {
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {FpToFpOpConversion::convertFp32ToBf16(loc, rewriter, value,
                                                    RoundingMode::RTNE)};
    } else {
      return {rewriter.create<LLVM::SIToFPOp>(loc, elemTy, operands[0][0])};
    }
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
      auto value =
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0]);
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, value)};
    } else {
      return {rewriter.create<LLVM::FPToSIOp>(loc, elemTy, operands[0][0])};
    }
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
      return {
          FpToFpOpConversion::convertBf16ToFp32(loc, rewriter, operands[0][0])};
    } else {
      return {rewriter.create<LLVM::FPExtOp>(loc, elemTy, operands[0][0])};
    }
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
              FpToFpOpConversion::convertFp32ToBf16(
                  loc, rewriter, operands[0][0], RoundingMode::RTNE)};
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

struct AbsIOpConversion
    : ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion> {
  using Base = ElementwiseOpConversionBase<math::AbsIOp, AbsIOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(math::AbsIOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto boolFalse = rewriter.getBoolAttr(false);
    auto constFalse = rewriter.create<LLVM::ConstantOp>(loc, boolFalse);
    return {rewriter.create<LLVM::AbsOp>(loc, elemTy, operands[0][0],
                                         /*is_int_min_poison=*/constFalse)};
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
    if (llvm::isa<IntegerType>(elemTy)) {
      // Mask out the sign bit
      auto num_bits =
          getElementTypeOrSelf(op.getType()).getIntOrFloatBitWidth();
      assert(num_bits <= 16);
      auto mask = (1u << (num_bits - 1u)) - 1u;
      auto maskAttr = rewriter.getIntegerAttr(elemTy, mask);
      auto maskConst = rewriter.create<LLVM::ConstantOp>(loc, maskAttr);
      return {and_(operands[0][0], maskConst)};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, operands[0][0])};
  }
};

template <typename OpTy>
struct MinMaxFOpConversion
    : ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>> {
  using Base = ElementwiseOpConversionBase<OpTy, MinMaxFOpConversion<OpTy>>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  static_assert(std::is_same<OpTy, arith::MinimumFOp>::value ||
                    std::is_same<OpTy, arith::MaximumFOp>::value,
                "OpTy must be arith::MinimumFOp or arith::MaximumFOp");

  // Choose the destination op based on the OpTy.
  using DestOpNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinimumOp, LLVM::MaximumOp>::type;
  using DestOpNoNanProp =
      typename std::conditional<std::is_same<OpTy, arith::MinimumFOp>::value,
                                LLVM::MinNumOp, LLVM::MaxNumOp>::type;

  explicit MinMaxFOpConversion(LLVMTypeConverter &typeConverter,
                               ModuleAxisInfoAnalysis &axisAnalysisPass,
                               PatternBenefit benefit = 1)
      : Base::ElementwiseOpConversionBase(typeConverter, axisAnalysisPass,
                                          benefit) {}

  SmallVector<Value> createDestOps(OpTy op, Adaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // If any of the operands is NaN, return NaN.
    auto lhs = operands[0][0];
    auto rhs = operands[0][1];
    auto lhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, lhs, lhs);
    auto rhsIsNan =
        rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une, rhs, rhs);
    auto isNan = rewriter.create<LLVM::OrOp>(loc, lhsIsNan, rhsIsNan);
    auto nonNanRes = rewriter.create<DestOpNoNanProp>(loc, elemTy, lhs, rhs);

    auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);

    // Select the result based on the isNan flag.
    return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
  }
};

struct ClampFOpConversion
    : ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion> {
  using Base = ElementwiseOpConversionBase<ClampFOp, ClampFOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  explicit ClampFOpConversion(LLVMTypeConverter &typeConverter,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit = 1)
      : ElementwiseOpConversionBase(typeConverter, axisAnalysisPass, benefit) {}

  SmallVector<Value> createDestOps(ClampFOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    // Pattern matching the sequence of clamp(x, -limit, limit) to generate more
    // efficient PTX code.
    // NOTE: This pattern matching is not general enough, but it is sufficient.
    // We detect only two cases here:
    // 1. where the "-limit" is computed as 0 - limit:
    //   %cst = arith.constant dense<0.000000e+00>
    //   %8 = tt.load %7, %2
    //   %11 = arith.subf %cst, %8
    //   %12 = tt.clamp %5, %11, %8
    // 2. where "-limit" and "limit" are constants.
    //   %cst_6 = arith.constant dense<-6.0000e+00>
    //   %cst_7 = arith.constant dense<6.0000e+00>
    //   %160 = tt.clamp %158, %cst_6, %cst_7

    auto getSplatInitializer = [](Value v) -> std::optional<double> {
      if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto attr =
                constOp.getValueAttr().dyn_cast<DenseIntOrFPElementsAttr>()) {
          if (attr.isSplat()) {
            return attr.getSplatValue<APFloat>().convertToDouble();
          }
        }
      }
      return std::nullopt;
    };

    assert(elemTy.isF32() || elemTy.isF16());

    if (op.getPropagateNan() == PropagateNan::ALL) {
      // handle NaN propagation manually. We need to check only the first
      // operand for clamp.
      auto lhs = operands[0][0];
      auto isNan = rewriter.create<LLVM::FCmpOp>(loc, LLVM::FCmpPredicate::une,
                                                 lhs, lhs);
      auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                               operands[0][1]);
      auto nonNanRes = rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2]);
      auto nan = LLVM::createNaNConstant(loc, rewriter, elemTy);
      // Select the result based on the isNan flag.
      return {rewriter.create<LLVM::SelectOp>(loc, isNan, nan, nonNanRes)};
    }

    // No NaN propagation.
    assert(op.getPropagateNan() == PropagateNan::NONE);
    auto v = rewriter.create<LLVM::MaxNumOp>(loc, elemTy, operands[0][0],
                                             operands[0][1]);
    return {rewriter.create<LLVM::MinNumOp>(loc, v, operands[0][2])};
  }
};

/// The lowering of index_cast becomes an integer conversion since index
/// becomes an integer.  If the bit width of the source and target integer
/// types is the same, just erase the cast.  If the target type is wider,
/// sign-extend the value, otherwise truncate it.
struct IndexCastOpLowering
    : public ElementwiseOpConversionBase<arith::IndexCastOp,
                                         IndexCastOpLowering> {
  using Base =
      ElementwiseOpConversionBase<arith::IndexCastOp, IndexCastOpLowering>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::IndexCastOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    auto inElemTy =
        this->getTypeConverter()->convertType(getElementType(op.getIn()));
    unsigned targetBits = elemTy.getIntOrFloatBitWidth();
    unsigned sourceBits = inElemTy.getIntOrFloatBitWidth();

    if (targetBits == sourceBits)
      return {operands[0][0]};
    if (targetBits < sourceBits)
      return {rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, elemTy,
                                                         operands[0][0])};
    return {
        rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, elemTy, operands[0][0])};
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

    StringRef funcName = targetInfo.getMulhiFuncName(resultElementTy);
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]);
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
    auto callOp = rewriter.create<LLVM::CallOp>(loc, funcOp, operands[0]);
    callOp.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    return {callOp.getResult()};
  }

private:
  StringRef funcName;
};

struct SelectOpConversion
    : ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion> {
  using Base = ElementwiseOpConversionBase<arith::SelectOp, SelectOpConversion>;
  using Base::Base;
  using Adaptor = typename Base::OpAdaptor;

  SmallVector<Value> createDestOps(arith::SelectOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    std::array<Value, 3> llvmOperands;
    if (operands[0].size() == 2) {
      // Case of scalar condition with tensor operands.
      assert(op.getCondition().getType().isInteger(1));
      llvmOperands = {adaptor.getCondition(), operands[0][0], operands[0][1]};
    } else {
      llvmOperands = {operands[0][0], operands[0][1], operands[0][2]};
    }
    return {rewriter.create<LLVM::SelectOp>(
        loc, llvmOperands[1].getType(), llvmOperands,
        adaptor.getAttributes().getValue())};
  }
};

struct AddPtrOpConversion : public ConvertTritonGPUOpToLLVMPattern<AddPtrOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      AddPtrOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto typeConverter = getTypeConverter();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy = typeConverter->convertType(
          resultTensorTy.getElementType().cast<PointerType>().getPointeeType());
      Type ptrTy = typeConverter->convertType(resultTensorTy.getElementType());
      auto ptrs = unpackLLElements(loc, adaptor.getPtr(), rewriter);
      auto offsets = unpackLLElements(loc, adaptor.getOffset(), rewriter);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(ptrTy, elemTy, ptrs[i], offsets[i]);
      }
      Value view =
          packLLElements(loc, typeConverter, resultVals, rewriter, resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<PointerType>());
      auto resultPtrTy = typeConverter->convertType(resultTy);
      auto resultElemTy = typeConverter->convertType(
          resultTy.cast<PointerType>().getPointeeType());
      Value result =
          gep(resultPtrTy, resultElemTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

} // namespace
} // namespace gpu

namespace intel {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {
  using namespace mlir::triton::gpu;

#define POPULATE_BINARY_OP(SRC_OP, DST_OP)                                     \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_BINARY_OP(arith::SubIOp, LLVM::SubOp) // -
  POPULATE_BINARY_OP(arith::AddIOp, LLVM::AddOp) // +
  POPULATE_BINARY_OP(arith::MulIOp, LLVM::MulOp) // *
  POPULATE_BINARY_OP(arith::DivSIOp, LLVM::SDivOp)
  POPULATE_BINARY_OP(arith::DivUIOp, LLVM::UDivOp)
  POPULATE_BINARY_OP(arith::RemFOp, LLVM::FRemOp) // %
  POPULATE_BINARY_OP(arith::RemSIOp, LLVM::SRemOp)
  POPULATE_BINARY_OP(arith::RemUIOp, LLVM::URemOp)
  POPULATE_BINARY_OP(arith::AndIOp, LLVM::AndOp)   // &
  POPULATE_BINARY_OP(arith::OrIOp, LLVM::OrOp)     // |
  POPULATE_BINARY_OP(arith::XOrIOp, LLVM::XOrOp)   // ^
  POPULATE_BINARY_OP(arith::ShLIOp, LLVM::ShlOp)   // <<
  POPULATE_BINARY_OP(arith::ShRSIOp, LLVM::AShrOp) // >>
  POPULATE_BINARY_OP(arith::ShRUIOp, LLVM::LShrOp) // >>
  POPULATE_BINARY_OP(
      arith::MinNumFOp,
      LLVM::MinNumOp) // fmin (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(
      arith::MaxNumFOp,
      LLVM::MaxNumOp) // fmax (return non-NaN if either op is non-NaN)
  POPULATE_BINARY_OP(arith::MinSIOp, LLVM::SMinOp) // smin
  POPULATE_BINARY_OP(arith::MaxSIOp, LLVM::SMaxOp) // smax
  POPULATE_BINARY_OP(arith::MinUIOp, LLVM::UMinOp) // umin
  POPULATE_BINARY_OP(arith::MaxUIOp, LLVM::UMaxOp) // umax
#undef POPULATE_BINARY_OP

#define POPULATE_UNARY_OP(SRC_OP, DST_OP)                                      \
  patterns.add<ElementwiseOpConversion<SRC_OP, DST_OP>>(                       \
      typeConverter, axisInfoAnalysis, benefit);
  POPULATE_UNARY_OP(arith::TruncIOp, LLVM::TruncOp)
  POPULATE_UNARY_OP(arith::ExtSIOp, LLVM::SExtOp)
  POPULATE_UNARY_OP(arith::ExtUIOp, LLVM::ZExtOp)
  POPULATE_UNARY_OP(arith::FPToUIOp, LLVM::FPToUIOp)
  POPULATE_UNARY_OP(arith::UIToFPOp, LLVM::UIToFPOp)
  POPULATE_UNARY_OP(math::FloorOp, math::FloorOp)
  POPULATE_UNARY_OP(math::LogOp, math::LogOp)
  POPULATE_UNARY_OP(math::Log2Op, math::Log2Op)
  POPULATE_UNARY_OP(math::CosOp, math::CosOp)
  POPULATE_UNARY_OP(math::SinOp, math::SinOp)
  POPULATE_UNARY_OP(math::SqrtOp, math::SqrtOp)
  POPULATE_UNARY_OP(math::RsqrtOp, math::RsqrtOp)
  POPULATE_UNARY_OP(math::ExpOp, math::ExpOp)
  POPULATE_UNARY_OP(math::Exp2Op, math::Exp2Op)
  POPULATE_UNARY_OP(math::ErfOp, math::ErfOp)
  POPULATE_UNARY_OP(triton::BitcastOp, LLVM::BitcastOp)
  POPULATE_UNARY_OP(triton::IntToPtrOp, LLVM::IntToPtrOp)
  POPULATE_UNARY_OP(triton::PtrToIntOp, LLVM::PtrToIntOp)
#undef POPULATE_UNARY_OP

  patterns.add<ElementwiseOpConversion<math::FmaOp, LLVM::FMAOp>>(
      typeConverter, axisInfoAnalysis, benefit);

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "__imf_sqrtf", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "__imf_fdiv_rn", benefit);

  patterns.add<AddPtrOpConversion>(typeConverter, benefit);
  patterns.add<AbsIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<CmpFOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FDivOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FSubOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FAddOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FMulOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<SelectOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ExtFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<TruncFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<FPToSIOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<SIToFPOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<IndexCastOpLowering>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<FpToFpOpConversion>(typeConverter, axisInfoAnalysis, benefit);

  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);
  patterns.add<ElementwiseInlineAsmOpConversion>(typeConverter, benefit);
  // ExpOpConversionApprox will try using ex2.approx if the input type is
  // FP32. For other input types, ExpOpConversionApprox will return failure and
  // ElementwiseOpConversion<math::ExpOp, math::ExpOp> defined below will call
  // a vendor specific math library for higher-precision calculation
  patterns.add<ExpOpConversionApprox>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<MulhiUIOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                    benefit);
  patterns.add<ClampFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  PatternBenefit benefitForPropNan = benefit;
  // TODO(FIXME): spirv's OpenCL extension (fmin/fmax) does not support
  // nan propagation. Set these conversion benefit to the max benefit:
  // PatternBenefit::ImpossibleToMatchSentinel - 1 to make sure the
  // correctness
  benefitForPropNan = 65534;
  patterns.add<MinMaxFOpConversion<arith::MinimumFOp>>(
      typeConverter, axisInfoAnalysis, benefitForPropNan);
  patterns.add<MinMaxFOpConversion<arith::MaximumFOp>>(
      typeConverter, axisInfoAnalysis, benefitForPropNan);
}
} // namespace intel
} // namespace mlir::triton
