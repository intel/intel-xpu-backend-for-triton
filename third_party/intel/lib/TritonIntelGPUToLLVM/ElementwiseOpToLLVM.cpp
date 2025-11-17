#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/ArithCommon/AttrToLLVMConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "third_party/intel/include/Dialect/TritonIntelGPU/IR/Utils.h"
#include "third_party/intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "third_party/intel/lib/Utils/Mangling.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

using mlir::triton::gpu::ElementwiseOpConversionBase;
using mlir::triton::gpu::MultipleOperandsRange;

namespace {

/* ----- FP8E5M2 ------ */
// This data-type is the standard FP8E5M2 format
static SmallVector<Value>
Fp16_to_Fp8E5M2_RTZ(Location loc, ConversionPatternRewriter &rewriter,
                    const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.undef(fp16x2VecTy);
  Value fp16x2Vec1 = b.undef(fp16x2VecTy);
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[0], b.i32_val(0));
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[1], b.i32_val(1));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[2], b.i32_val(0));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[3], b.i32_val(1));

  Value a0 = b.bitcast(fp16x2Vec0, i32_ty);
  Value a1 = b.bitcast(fp16x2Vec1, i32_ty);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  a0 = b.bitcast(a0, fp8x4VecTy);
  a1 = b.bitcast(a1, fp8x4VecTy);

  return {b.extract_element(i8_ty, a0, b.i32_val(1)),
          b.extract_element(i8_ty, a0, b.i32_val(3)),
          b.extract_element(i8_ty, a1, b.i32_val(1)),
          b.extract_element(i8_ty, a1, b.i32_val(3))};
}

static SmallVector<Value> Fp8E5M2_to_Fp16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);
  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = b.bitcast(a0, fp16x2VecTy);
  auto fp16x2Vec1 = b.bitcast(a1, fp16x2VecTy);

  return {b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(1)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(1))};
}

static SmallVector<Value> Fp8E5M2_to_Bf16(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  Value b1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));
  // In i32 original fp8 exponent is b0 >> 26
  // bf16's 5-bit exponent in the top 2 bytes of i32 is at b0 >> 23
  // 2^5-1 << 23 = 0xf800000
  b0 = b.lshr(i32_ty, b0, b.i32_val(3));
  b1 = b.lshr(i32_ty, b1, b.i32_val(3));

  Value c0 = b.and_(i32_ty, b0, b.i32_val(0xffff0000));
  Value c1 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c2 = b.and_(i32_ty, b1, b.i32_val(0xffff0000));
  Value c3 = b.shl(i32_ty, b1, b.i32_val(16));

  auto i32x4VecTy = vec_ty(i32_ty, 4);
  Value predefined = b.undef(i32x4VecTy);
  predefined =
      b.insert_element(i32x4VecTy, predefined, b.i32_val(0x0), b.i32_val(0));
  predefined = b.insert_element(i32x4VecTy, predefined, b.i32_val(0x37800000),
                                b.i32_val(1));
  predefined = b.insert_element(i32x4VecTy, predefined, b.i32_val(0x38000000),
                                b.i32_val(2));
  predefined = b.insert_element(i32x4VecTy, predefined, b.i32_val(0x38400000),
                                b.i32_val(3));
  // Check if the exponent is zero, i.e. subnormal number.
  // depending on the significand value normalization goes like:
  //  [00] -> 0x0
  //  [01] -> exp=127-16, sig=0x0
  //  [10] -> exp=127-15, sig=0x0
  //  [11] -> exp=127-15, sig=b1000...
  Value cmp0 = b.icmp_eq(b.and_(c0, b.i32_val(0xf800000)), b.i32_val(0));
  Value cmp1 = b.icmp_eq(b.and_(c1, b.i32_val(0xf800000)), b.i32_val(0));
  Value cmp2 = b.icmp_eq(b.and_(c2, b.i32_val(0xf800000)), b.i32_val(0));
  Value cmp3 = b.icmp_eq(b.and_(c3, b.i32_val(0xf800000)), b.i32_val(0));

  Value predef_idx0 = b.lshr(b.and_(c0, b.i32_val(3 << 21)), b.i32_val(21));
  Value predef_idx1 = b.lshr(b.and_(c1, b.i32_val(3 << 21)), b.i32_val(21));
  Value predef_idx2 = b.lshr(b.and_(c2, b.i32_val(3 << 21)), b.i32_val(21));
  Value predef_idx3 = b.lshr(b.and_(c3, b.i32_val(3 << 21)), b.i32_val(21));

  Value normalized0 = b.extract_element(i32_ty, predefined, predef_idx0);
  Value normalized1 = b.extract_element(i32_ty, predefined, predef_idx1);
  Value normalized2 = b.extract_element(i32_ty, predefined, predef_idx2);
  Value normalized3 = b.extract_element(i32_ty, predefined, predef_idx3);

  Value d0 = b.add(i32_ty, c0, b.i32_val(0x38000000));
  Value d1 = b.add(i32_ty, c1, b.i32_val(0x38000000));
  Value d2 = b.add(i32_ty, c2, b.i32_val(0x38000000));
  Value d3 = b.add(i32_ty, c3, b.i32_val(0x38000000));

  Value res0 = b.select(cmp0, normalized0, d0);
  Value res1 = b.select(cmp1, normalized1, d1);
  Value res2 = b.select(cmp2, normalized2, d2);
  Value res3 = b.select(cmp3, normalized3, d3);

  Value f0 = b.or_(i32_ty, res0, b.lshr(i32_ty, res1, b.i32_val(16)));
  Value f1 = b.or_(i32_ty, res2, b.lshr(i32_ty, res3, b.i32_val(16)));

  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, a1, b.i32_val(0x80008000));

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = b.or_(i32_ty, sign0, f0);
  Value bf16x2Vec1 = b.or_(i32_ty, sign1, f1);
  bf16x2Vec0 = b.bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = b.bitcast(bf16x2Vec1, bf16x2VecTy);

  return {b.extract_element(bf16_ty, bf16x2Vec0, b.i32_val(0)),
          b.extract_element(bf16_ty, bf16x2Vec0, b.i32_val(1)),
          b.extract_element(bf16_ty, bf16x2Vec1, b.i32_val(0)),
          b.extract_element(bf16_ty, bf16x2Vec1, b.i32_val(1))};
}

static SmallVector<Value> Bf16_to_Fp8E5M2(Location loc,
                                          ConversionPatternRewriter &rewriter,
                                          const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = b.undef(bf16x2VecTy);
  Value bf16x2Vec1 = b.undef(bf16x2VecTy);
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[0], b.i32_val(0));
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[1], b.i32_val(1));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[2], b.i32_val(0));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[3], b.i32_val(1));
  bf16x2Vec0 = b.bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = b.bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = b.undef(fp8x4VecTy);
  sign0 = b.bitcast(sign0, fp8x4VecTy);
  sign1 = b.bitcast(sign1, fp8x4VecTy);
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(1)),
                          b.i32_val(0));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(3)),
                          b.i32_val(1));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(1)),
                          b.i32_val(2));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(3)),
                          b.i32_val(3));
  sign = b.bitcast(sign, i32_ty);

  Value nosign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x7fff7fff));
  Value nosign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x7fff7fff));

  Value nosign_0_0 = b.and_(i32_ty, nosign0, b.i32_val(0xffff0000));
  nosign_0_0 = b.umax(i32_ty, nosign_0_0, b.i32_val(0x38000000));
  nosign_0_0 = b.umin(i32_ty, nosign_0_0, b.i32_val(0x57e00000));
  Value nosign_0_1 = b.and_(i32_ty, nosign0, b.i32_val(0x0000ffff));
  nosign_0_1 = b.umax(i32_ty, nosign_0_1, b.i32_val(0x3800));
  nosign_0_1 = b.umin(i32_ty, nosign_0_1, b.i32_val(0x57e0));
  nosign0 = b.or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = b.and_(i32_ty, nosign1, b.i32_val(0xffff0000));
  nosign_1_0 = b.umax(i32_ty, nosign_1_0, b.i32_val(0x38000000));
  nosign_1_0 = b.umin(i32_ty, nosign_1_0, b.i32_val(0x57e00000));
  Value nosign_1_1 = b.and_(i32_ty, nosign1, b.i32_val(0x0000ffff));
  nosign_1_1 = b.umax(i32_ty, nosign_1_1, b.i32_val(0x3800));
  nosign_1_1 = b.umin(i32_ty, nosign_1_1, b.i32_val(0x57e0));
  nosign1 = b.or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = b.add(i32_ty, nosign0, b.i32_val(0x00100010));
  nosign1 = b.add(i32_ty, nosign1, b.i32_val(0x00100010));
  nosign0 = b.sub(i32_ty, nosign0, b.i32_val(0x38003800));
  nosign1 = b.sub(i32_ty, nosign1, b.i32_val(0x38003800));
  nosign0 = b.shl(i32_ty, nosign0, b.i32_val(3));
  nosign1 = b.shl(i32_ty, nosign1, b.i32_val(3));

  nosign0 = b.bitcast(nosign0, fp8x4VecTy);
  nosign1 = b.bitcast(nosign1, fp8x4VecTy);
  Value nosign = b.undef(fp8x4VecTy);
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(1)),
                            b.i32_val(0));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(3)),
                            b.i32_val(1));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(1)),
                            b.i32_val(2));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(3)),
                            b.i32_val(3));
  nosign = b.bitcast(nosign, i32_ty);

  Value fp8x4Vec = b.or_(i32_ty, nosign, sign);
  fp8x4Vec = b.bitcast(fp8x4Vec, fp8x4VecTy);
  return {b.extract_element(i8_ty, fp8x4Vec, b.i32_val(0)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(1)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(2)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(3))};
}

/* ----- FP8E4M3B15 ------ */
// This data-type is a variant of the standard FP8E4M3 format.
// It was designed for fast software conversion to FP16 on GPUs that do not
// support it natively. Specifically, this data-type:
//    - has infinities
//    - has multiple nans (when all exponent bits are 1)
//    - has an exponent bias of 15 (vs. 7 for fp8e4m3)
static SmallVector<Value>
Fp8E4M3B15_to_Fp16(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  Value b1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));

  b0 = b.lshr(i32_ty, b0, b.i32_val(1));
  b1 = b.lshr(i32_ty, b1, b.i32_val(1));

  b0 = b.or_(i32_ty, b0, b.and_(i32_ty, a0, b.i32_val(0x80008000)));
  b1 = b.or_(i32_ty, b1, b.and_(i32_ty, a1, b.i32_val(0x80008000)));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  auto fp16x2Vec0 = b.bitcast(b0, fp16x2VecTy);
  auto fp16x2Vec1 = b.bitcast(b1, fp16x2VecTy);

  return {b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(1)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec1, b.i32_val(1))};
}

static SmallVector<Value>
Fp16_to_Fp8E4M3B15(Location loc, ConversionPatternRewriter &rewriter,
                   const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.undef(fp16x2VecTy);
  Value fp16x2Vec1 = b.undef(fp16x2VecTy);

  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[0], b.i32_val(0));
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[1], b.i32_val(1));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[2], b.i32_val(0));
  fp16x2Vec1 = b.insert_element(fp16x2VecTy, fp16x2Vec1, v[3], b.i32_val(1));

  Value fp16x2VecMin = b.i32_val(0xBF80BF80);
  Value fp16x2VecMax = b.i32_val(0x3F803F80);
  fp16x2VecMin = b.bitcast(fp16x2VecMin, fp16x2VecTy);
  fp16x2VecMax = b.bitcast(fp16x2VecMax, fp16x2VecTy);
  fp16x2Vec0 = b.fmax(fp16x2VecTy, fp16x2Vec0, fp16x2VecMin);
  fp16x2Vec1 = b.fmax(fp16x2VecTy, fp16x2Vec1, fp16x2VecMin);
  fp16x2Vec0 = b.fmin(fp16x2VecTy, fp16x2Vec0, fp16x2VecMax);
  fp16x2Vec1 = b.fmin(fp16x2VecTy, fp16x2Vec1, fp16x2VecMax);

  fp16x2Vec0 = b.bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec1 = b.bitcast(fp16x2Vec1, i32_ty);

  Value a0 = b.shl(i32_ty, fp16x2Vec0, b.i32_val(1));
  Value a1 = b.shl(i32_ty, fp16x2Vec1, b.i32_val(1));
  a0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  a1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));
  a0 = b.add(i32_ty, a0, b.i32_val(0x00800080));
  a1 = b.add(i32_ty, a1, b.i32_val(0x00800080));
  Value b0 =
      b.or_(i32_ty, b.and_(i32_ty, fp16x2Vec0, b.i32_val(0x80008000)), a0);
  Value b1 =
      b.or_(i32_ty, b.and_(i32_ty, fp16x2Vec1, b.i32_val(0x80008000)), a1);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  b0 = b.bitcast(b0, fp8x4VecTy);
  b1 = b.bitcast(b1, fp8x4VecTy);

  return {b.extract_element(i8_ty, b0, b.i32_val(1)),
          b.extract_element(i8_ty, b0, b.i32_val(3)),
          b.extract_element(i8_ty, b1, b.i32_val(1)),
          b.extract_element(i8_ty, b1, b.i32_val(3))};
}

/* ----- FP8E4M3 ------ */
// Note: when handled by software, this format
// has more than a single NaN values.

// Fp8E4M3 -> Fp16 (packed)
static SmallVector<Value> Fp8E4M3Nv_to_Fp16(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));

  b0 = b.lshr(i32_ty, b0, b.i32_val(1));

  Value c0 = b.and_(i32_ty, b0, b.i32_val(0xffff0000));
  Value c1 = b.shl(i32_ty, b0, b.i32_val(16));

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
  Value cmp0 = b.icmp_eq(b.and_(c0, b.i32_val(0x7c000000)), b.i32_val(0));
  Value cmp1 = b.icmp_eq(b.and_(c1, b.i32_val(0x7c000000)), b.i32_val(0));

  auto i32x8VecTy = vec_ty(i32_ty, 8);
  Value predefined = b.undef(i32x8VecTy);
  predefined =
      b.insert_element(i32x8VecTy, predefined, b.i32_val(0x0), b.i32_val(0));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x18000000),
                                b.i32_val(1));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x1C000000),
                                b.i32_val(2));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x1E000000),
                                b.i32_val(3));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x20000000),
                                b.i32_val(4));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x21000000),
                                b.i32_val(5));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x22000000),
                                b.i32_val(6));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x23000000),
                                b.i32_val(7));

  Value predef_idx0 = b.lshr(b.and_(c0, b.i32_val(7 << 23)), b.i32_val(23));
  Value predef_idx1 = b.lshr(b.and_(c1, b.i32_val(7 << 23)), b.i32_val(23));

  Value normalized0 = b.extract_element(i32_ty, predefined, predef_idx0);
  Value normalized1 = b.extract_element(i32_ty, predefined, predef_idx1);

  Value d0 = b.add(i32_ty, c0, b.i32_val(0x20000000));
  Value d1 = b.add(i32_ty, c1, b.i32_val(0x20000000));

  Value res0 = b.select(cmp0, normalized0, d0);
  Value res1 = b.select(cmp1, normalized1, d1);

  Value f0 = b.or_(i32_ty, res0, b.lshr(i32_ty, res1, b.i32_val(16)));
  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.or_(i32_ty, sign0, f0);
  fp16x2Vec0 = b.bitcast(fp16x2Vec0, fp16x2VecTy);

  return {b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec0, b.i32_val(1))};
}

// Fp16 -> Fp8E4M3 (packed)
static SmallVector<Value> Fp16_to_Fp8E4M3Nv(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec0 = b.undef(fp16x2VecTy);

  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[0], b.i32_val(0));
  fp16x2Vec0 = b.insert_element(fp16x2VecTy, fp16x2Vec0, v[1], b.i32_val(1));

  fp16x2Vec0 = b.bitcast(fp16x2Vec0, i32_ty);
  fp16x2Vec0 = b.sub(i32_ty, fp16x2Vec0, b.i32_val(0x20002000));

  Value a0 = b.shl(i32_ty, fp16x2Vec0, b.i32_val(1));
  a0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  a0 = b.add(i32_ty, a0, b.i32_val(0x00800080));
  Value b0 =
      b.or_(i32_ty, b.and_(i32_ty, fp16x2Vec0, b.i32_val(0x80008000)), a0);

  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  b0 = b.bitcast(b0, fp8x4VecTy);

  return {b.extract_element(i8_ty, b0, b.i32_val(1)),
          b.extract_element(i8_ty, b0, b.i32_val(3))};
}

template <typename SrcTy, typename DstTy,
          int64_t SrcBits = std::is_same_v<SrcTy, Float16Type> ? 16 : 32,
          int64_t SrcEBits = std::is_same_v<SrcTy, Float16Type> ? 5 : 8,
          int64_t DstEBits = std::is_same_v<DstTy, Float8E4M3Type> ? 4 : 5,
          int64_t SrcMBits = SrcBits - SrcEBits - 1,
          int64_t DstMBits = 8 - DstEBits - 1,
          int64_t SrcBias = (1L << (SrcEBits - 1)) - 1,
          int64_t DstBias = (1L << (DstEBits - 1)) - 1>
static SmallVector<Value> Fp_to_Fp8_RTNE(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         const SmallVector<Value> &v) {
  static_assert((std::is_same_v<SrcTy, Float16Type>) ||
                (std::is_same_v<SrcTy, Float32Type>) ||
                (std::is_same_v<SrcTy, BFloat16Type>));
  static_assert((std::is_same_v<DstTy, Float8E4M3Type>) ||
                (std::is_same_v<DstTy, Float8E5M2Type>));
  constexpr int64_t SRC_MASK = (1ULL << (SrcBits - 1)) - 1;
  constexpr int64_t SRC_MMASK = (1L << SrcMBits) - 1;
  constexpr int64_t DST_NAN = 0x7F;
  constexpr int64_t DST_MAX =
      std::is_same_v<DstTy, Float8E5M2Type> ? 0x7B : DST_NAN - 1;
  constexpr bool IS_FP16 = std::is_same_v<SrcTy, Float16Type>;
  TritonLLVMIRRewriter b(loc, rewriter);
  auto srcTy = IS_FP16 ? f16_ty : f32_ty; // BF16 is converted to FP32 below
  auto srcITy = rewriter.getIntegerType(SrcBits);
  auto fval = [&b](float v) { return IS_FP16 ? b.f16_val(v) : b.f32_val(v); };
  auto ival = [&](int64_t v) { return b.int_val(SrcBits, v); };
  auto zero = ival(0);

  Value val = v[0];
  if constexpr (std::is_same_v<SrcTy, BFloat16Type>) {
    // Convert to FP32 since the nearbyint intrinsic does not support BF16
    val = b.shl(b.zext(srcITy, b.bitcast(val, i16_ty)), ival(16));
  } else {
    val = b.bitcast(val, srcITy);
  }
  Value sign = b.and_(val, ival(1L << (SrcBits - 1)));
  Value nosign = b.and_(val, ival(SRC_MASK));
  Value exp = b.lshr(nosign, ival(SrcMBits));
  Value man = b.and_(nosign, ival(SRC_MMASK));
  // NaN if exp is all ones and man is all zeroes
  Value isNan = b.icmp_ugt(nosign, ival(SRC_MASK ^ SRC_MMASK));

  // dstExp = max(0, srcExp - SrcBias + DstBias)
  // if (dstExp) dstMan = srcMan * 2^(DstMBits - SrcMBits)
  // else dstMan = src * 2^(DstMBits + DstBias - 1)
  Value scale = fval(1.0 / static_cast<float>(1 << (SrcMBits - DstMBits)));
  man = b.create<LLVM::UIToFPOp>(srcTy, man);
  if constexpr (SrcBias != DstBias) {
    exp = b.smax(b.sub(exp, ival(SrcBias - DstBias)), zero);
    Value isSubnorm = b.icmp_eq(exp, zero);
    man = b.select(isSubnorm, b.bitcast(nosign, srcTy), man);
    scale = b.select(isSubnorm,
                     fval(static_cast<float>(1 << (DstMBits + DstBias - 1))),
                     scale);
  }
  man = b.fmul(man, scale, LLVM::FastmathFlags::fast);
  man = b.create<LLVM::FPToUIOp>(srcITy, b.create<LLVM::NearbyintOp>(man));

  val = b.add(b.shl(exp, ival(DstMBits)), man);
  val = b.umin(ival(DST_MAX), val);
  val = b.or_(b.lshr(sign, ival(SrcBits - 8)), val);
  return {b.select(isNan, b.i8_val(DST_NAN), b.trunc(i8_ty, val))};
}

static SmallVector<Value> Fp8E4M3Nv_to_Bf16(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value a0 = b.undef(fp8x4VecTy);
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(0));
  a0 = b.insert_element(fp8x4VecTy, a0, v[0], b.i32_val(1));
  a0 = b.insert_element(fp8x4VecTy, a0, b.int_val(8, 0), b.i32_val(2));
  a0 = b.insert_element(fp8x4VecTy, a0, v[1], b.i32_val(3));
  a0 = b.bitcast(a0, i32_ty);

  Value a1 = b.undef(fp8x4VecTy);
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(0));
  a1 = b.insert_element(fp8x4VecTy, a1, v[2], b.i32_val(1));
  a1 = b.insert_element(fp8x4VecTy, a1, b.int_val(8, 0), b.i32_val(2));
  a1 = b.insert_element(fp8x4VecTy, a1, v[3], b.i32_val(3));
  a1 = b.bitcast(a1, i32_ty);

  Value b0 = b.and_(i32_ty, a0, b.i32_val(0x7fff7fff));
  Value b1 = b.and_(i32_ty, a1, b.i32_val(0x7fff7fff));
  b0 = b.lshr(i32_ty, b0, b.i32_val(4));
  b1 = b.lshr(i32_ty, b1, b.i32_val(4));

  Value c0 = b.and_(i32_ty, b0, b.i32_val(0xffff0000));
  Value c1 = b.shl(i32_ty, b0, b.i32_val(16));
  Value c2 = b.and_(i32_ty, b1, b.i32_val(0xffff0000));
  Value c3 = b.shl(i32_ty, b1, b.i32_val(16));

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
  Value cmp0 = b.icmp_eq(b.and_(c0, b.i32_val(0xf800000)), b.i32_val(0));
  Value cmp1 = b.icmp_eq(b.and_(c1, b.i32_val(0xf800000)), b.i32_val(0));
  Value cmp2 = b.icmp_eq(b.and_(c2, b.i32_val(0xf800000)), b.i32_val(0));
  Value cmp3 = b.icmp_eq(b.and_(c3, b.i32_val(0xf800000)), b.i32_val(0));

  auto i32x8VecTy = vec_ty(i32_ty, 8);
  Value predefined = b.undef(i32x8VecTy);
  predefined =
      b.insert_element(i32x8VecTy, predefined, b.i32_val(0x0), b.i32_val(0));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3B000000),
                                b.i32_val(1));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3B800000),
                                b.i32_val(2));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3BC00000),
                                b.i32_val(3));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3C000000),
                                b.i32_val(4));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3C200000),
                                b.i32_val(5));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3C400000),
                                b.i32_val(6));
  predefined = b.insert_element(i32x8VecTy, predefined, b.i32_val(0x3C600000),
                                b.i32_val(7));

  Value predef_idx0 = b.lshr(b.and_(c0, b.i32_val(7 << 20)), b.i32_val(20));
  Value predef_idx1 = b.lshr(b.and_(c1, b.i32_val(7 << 20)), b.i32_val(20));
  Value predef_idx2 = b.lshr(b.and_(c2, b.i32_val(7 << 20)), b.i32_val(20));
  Value predef_idx3 = b.lshr(b.and_(c3, b.i32_val(7 << 20)), b.i32_val(20));

  Value normalized0 = b.extract_element(i32_ty, predefined, predef_idx0);
  Value normalized1 = b.extract_element(i32_ty, predefined, predef_idx1);
  Value normalized2 = b.extract_element(i32_ty, predefined, predef_idx2);
  Value normalized3 = b.extract_element(i32_ty, predefined, predef_idx3);

  Value d0 = b.add(i32_ty, c0, b.i32_val(0x3c000000));
  Value d1 = b.add(i32_ty, c1, b.i32_val(0x3c000000));
  Value d2 = b.add(i32_ty, c2, b.i32_val(0x3c000000));
  Value d3 = b.add(i32_ty, c3, b.i32_val(0x3c000000));

  Value res0 = b.select(cmp0, normalized0, d0);
  Value res1 = b.select(cmp1, normalized1, d1);
  Value res2 = b.select(cmp2, normalized2, d2);
  Value res3 = b.select(cmp3, normalized3, d3);

  Value f0 = b.or_(i32_ty, res0, b.lshr(i32_ty, res1, b.i32_val(16)));
  Value f1 = b.or_(i32_ty, res2, b.lshr(i32_ty, res3, b.i32_val(16)));

  Value sign0 = b.and_(i32_ty, a0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, a1, b.i32_val(0x80008000));

  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = b.or_(i32_ty, sign0, f0);
  Value bf16x2Vec1 = b.or_(i32_ty, sign1, f1);
  bf16x2Vec0 = b.bitcast(bf16x2Vec0, bf16x2VecTy);
  bf16x2Vec1 = b.bitcast(bf16x2Vec1, bf16x2VecTy);

  return {b.extract_element(bf16_ty, bf16x2Vec0, b.i32_val(0)),
          b.extract_element(bf16_ty, bf16x2Vec0, b.i32_val(1)),
          b.extract_element(bf16_ty, bf16x2Vec1, b.i32_val(0)),
          b.extract_element(bf16_ty, bf16x2Vec1, b.i32_val(1))};
}

static SmallVector<Value> Bf16_to_Fp8E4M3Nv(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);
  Value bf16x2Vec0 = b.undef(bf16x2VecTy);
  Value bf16x2Vec1 = b.undef(bf16x2VecTy);
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[0], b.i32_val(0));
  bf16x2Vec0 = b.insert_element(bf16x2VecTy, bf16x2Vec0, v[1], b.i32_val(1));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[2], b.i32_val(0));
  bf16x2Vec1 = b.insert_element(bf16x2VecTy, bf16x2Vec1, v[3], b.i32_val(1));
  bf16x2Vec0 = b.bitcast(bf16x2Vec0, i32_ty);
  bf16x2Vec1 = b.bitcast(bf16x2Vec1, i32_ty);

  Value sign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x80008000));
  Value sign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x80008000));
  auto fp8x4VecTy = vec_ty(i8_ty, 4);
  Value sign = b.undef(fp8x4VecTy);
  sign0 = b.bitcast(sign0, fp8x4VecTy);
  sign1 = b.bitcast(sign1, fp8x4VecTy);
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(1)),
                          b.i32_val(0));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign0, b.i32_val(3)),
                          b.i32_val(1));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(1)),
                          b.i32_val(2));
  sign = b.insert_element(fp8x4VecTy, sign,
                          b.extract_element(i8_ty, sign1, b.i32_val(3)),
                          b.i32_val(3));
  sign = b.bitcast(sign, i32_ty);

  Value nosign0 = b.and_(i32_ty, bf16x2Vec0, b.i32_val(0x7fff7fff));
  Value nosign1 = b.and_(i32_ty, bf16x2Vec1, b.i32_val(0x7fff7fff));

  Value nosign_0_0 = b.and_(i32_ty, nosign0, b.i32_val(0xffff0000));
  nosign_0_0 = b.umax(i32_ty, nosign_0_0, b.i32_val(0x3c000000));
  nosign_0_0 = b.umin(i32_ty, nosign_0_0, b.i32_val(0x43f00000));
  Value nosign_0_1 = b.and_(i32_ty, nosign0, b.i32_val(0x0000ffff));
  nosign_0_1 = b.umax(i32_ty, nosign_0_1, b.i32_val(0x3c00));
  nosign_0_1 = b.umin(i32_ty, nosign_0_1, b.i32_val(0x43f0));
  nosign0 = b.or_(i32_ty, nosign_0_0, nosign_0_1);

  Value nosign_1_0 = b.and_(i32_ty, nosign1, b.i32_val(0xffff0000));
  nosign_1_0 = b.umax(i32_ty, nosign_1_0, b.i32_val(0x3c000000));
  nosign_1_0 = b.umin(i32_ty, nosign_1_0, b.i32_val(0x43f00000));
  Value nosign_1_1 = b.and_(i32_ty, nosign1, b.i32_val(0x0000ffff));
  nosign_1_1 = b.umax(i32_ty, nosign_1_1, b.i32_val(0x3c00));
  nosign_1_1 = b.umin(i32_ty, nosign_1_1, b.i32_val(0x43f0));
  nosign1 = b.or_(i32_ty, nosign_1_0, nosign_1_1);

  nosign0 = b.add(i32_ty, nosign0, b.i32_val(0x80008));
  nosign1 = b.add(i32_ty, nosign1, b.i32_val(0x80008));
  nosign0 = b.sub(i32_ty, nosign0, b.i32_val(0x3c003c00));
  nosign1 = b.sub(i32_ty, nosign1, b.i32_val(0x3c003c00));
  nosign0 = b.lshr(i32_ty, nosign0, b.i32_val(4));
  nosign1 = b.lshr(i32_ty, nosign1, b.i32_val(4));

  nosign0 = b.bitcast(nosign0, fp8x4VecTy);
  nosign1 = b.bitcast(nosign1, fp8x4VecTy);
  Value nosign = b.undef(fp8x4VecTy);
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(0)),
                            b.i32_val(0));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign0, b.i32_val(2)),
                            b.i32_val(1));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(0)),
                            b.i32_val(2));
  nosign = b.insert_element(fp8x4VecTy, nosign,
                            b.extract_element(i8_ty, nosign1, b.i32_val(2)),
                            b.i32_val(3));
  nosign = b.bitcast(nosign, i32_ty);

  Value fp8x4Vec = b.or_(i32_ty, nosign, sign);
  fp8x4Vec = b.bitcast(fp8x4Vec, fp8x4VecTy);
  return {b.extract_element(i8_ty, fp8x4Vec, b.i32_val(0)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(1)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(2)),
          b.extract_element(i8_ty, fp8x4Vec, b.i32_val(3))};
}

static SmallVector<Value> Bf16_to_Fp16(Location loc,
                                       ConversionPatternRewriter &rewriter,
                                       const SmallVector<Value> &v) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto bf16x2VecTy = vec_ty(bf16_ty, 2);

  Value bf16x2Vec = b.undef(bf16x2VecTy);
  bf16x2Vec = b.insert_element(bf16x2VecTy, bf16x2Vec, v[0], b.i32_val(0));
  bf16x2Vec = b.insert_element(bf16x2VecTy, bf16x2Vec, v[1], b.i32_val(1));
  bf16x2Vec = b.bitcast(bf16x2Vec, i32_ty);

  Value sign = b.and_(i32_ty, bf16x2Vec, b.i32_val(0x80008000));
  Value nosign = b.and_(i32_ty, bf16x2Vec, b.i32_val(0x7fff7fff));

  // BF16 exp range is 0..255 with bias 127
  // FP16 exp range is 0..31 with bias 15
  // So, BF16 exp values has to be adjusted by subtracting 112.
  // Min BF16 value we can convert is 112 << 7 = 0x3800
  // Max BF16 value we can convert is 143 << 7 + <max fraction> =
  // 0x4780 + 0x7F = 0x47FF
  Value nosign_0 = b.and_(i32_ty, nosign, b.i32_val(0xffff0000));
  nosign_0 = b.umax(i32_ty, nosign_0, b.i32_val(0x38000000));
  nosign_0 = b.umin(i32_ty, nosign_0, b.i32_val(0x47ff0000));
  Value nosign_1 = b.and_(i32_ty, nosign, b.i32_val(0xffff));
  nosign_1 = b.umax(i32_ty, nosign_1, b.i32_val(0x3800));
  nosign_1 = b.umin(i32_ty, nosign_1, b.i32_val(0x47ff));
  nosign = b.or_(i32_ty, nosign_0, nosign_1);

  nosign = b.sub(i32_ty, nosign, b.i32_val(0x38003800));
  nosign = b.shl(i32_ty, nosign, b.i32_val(3));

  auto fp16x2VecTy = vec_ty(f16_ty, 2);
  Value fp16x2Vec = b.or_(i32_ty, nosign, sign);
  fp16x2Vec = b.bitcast(fp16x2Vec, fp16x2VecTy);
  return {b.extract_element(f16_ty, fp16x2Vec, b.i32_val(0)),
          b.extract_element(f16_ty, fp16x2Vec, b.i32_val(1))};
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

typedef std::function<SmallVector<Value>(Location, ConversionPatternRewriter &,
                                         const SmallVector<Value> &)>
    ConverterT;

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

  static Value convertFp32ToFp16(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 const Value &v,
                                 const triton::RoundingMode rounding) {
    MLIRContext *ctx = rewriter.getContext();
    return rewriter.create<LLVM::ConstrainedFPTruncIntr>(
        loc, f16_ty, v,
        LLVM::RoundingModeAttr::get(
            ctx, LLVM::intel::convertTritonRoundingModeToLLVM(rounding)),
        arith::getLLVMDefaultFPExceptionBehavior(*ctx));
  }

  std::pair<ConverterT, size_t>
  getConversionFunc(Type srcTy, Type dstTy,
                    std::optional<RoundingMode> roundingMode) const {
    auto F8E4M3B15TyID = TypeID::get<Float8E4M3FNUZType>();
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
            {{F8E4M3B15TyID, F16TyID, undefRounding}, {Fp8E4M3B15_to_Fp16, 4}},
            {{F8E4M3TyID, F16TyID, undefRounding}, {Fp8E4M3Nv_to_Fp16, 2}},
            {{F8E5M2TyID, F16TyID, undefRounding}, {Fp8E5M2_to_Fp16, 4}},
            // F16 -> F8
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E4M3B15, 4}},
            {{F16TyID, F8E4M3B15TyID, RoundingMode::RTNE},
             // TODO: provide proper implementation for RTNE rounding.
             {Fp16_to_Fp8E4M3B15, 4}},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTZ}, {Fp16_to_Fp8E4M3Nv, 2}},
            {{F16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {Fp_to_Fp8_RTNE<Float16Type, Float8E4M3Type>, 1}},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTZ},
             {Fp16_to_Fp8E5M2_RTZ, 4}},
            {{F16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {Fp_to_Fp8_RTNE<Float16Type, Float8E5M2Type>, 1}},
            // F8 -> BF16
            {{F8E5M2TyID, BF16TyID, undefRounding}, {Fp8E5M2_to_Bf16, 4}},
            {{F8E4M3TyID, BF16TyID, undefRounding}, {Fp8E4M3Nv_to_Bf16, 4}},
            // BF16 -> F8
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTZ}, {Bf16_to_Fp8E5M2, 4}},
            {{BF16TyID, F8E5M2TyID, RoundingMode::RTNE},
             {Fp_to_Fp8_RTNE<BFloat16Type, Float8E5M2Type>, 1}},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTZ}, {Bf16_to_Fp8E4M3Nv, 4}},
            {{BF16TyID, F8E4M3TyID, RoundingMode::RTNE},
             {Fp_to_Fp8_RTNE<BFloat16Type, Float8E4M3Type>, 1}},
            // BF16 -> F16
            {{BF16TyID, F16TyID, undefRounding}, {Bf16_to_Fp16, 2}},
            // F32 -> F8
            {{F32TyID, F8E4M3TyID, RoundingMode::RTNE},
             {Fp_to_Fp8_RTNE<Float32Type, Float8E4M3Type>, 1}},
            {{F32TyID, F8E5M2TyID, RoundingMode::RTNE},
             {Fp_to_Fp8_RTNE<Float32Type, Float8E5M2Type>, 1}},
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcElementType = getElementType(op.getSrc());
    auto dstElementType = getElementType(op.getResult());
    auto roundingMode = op.getRounding();

    if (isa<Float8E5M2Type, Float8E4M3FNType>(dstElementType)) {
      assert(roundingMode.has_value() &&
             "Rounding mode must be specified for conversions to fp8");

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

    if (srcElementType.isBF16() && dstElementType.isF32()) {
      SmallVector<Value> outVals;
      for (Value v : operands[0]) {
        outVals.push_back(intel::convertBf16ToFp32(loc, rewriter, v));
      }
      return outVals;
    }

    bool useFP16IntermediateSrc =
        srcElementType.isF32() &&
        !((roundingMode == RoundingMode::RTNE) &&
          (dstElementType.getTypeID() == TypeID::get<Float8E4M3FNType>() ||
           dstElementType.getTypeID() == TypeID::get<Float8E5M2Type>()));
    bool isDstFP32 = dstElementType.isF32();
    Type srcType = useFP16IntermediateSrc ? f16_ty : srcElementType;
    Type dstType = isDstFP32 ? f16_ty : dstElementType;
    auto [cvtFunc, numElements] =
        getConversionFunc(srcType, dstType, roundingMode);
    SmallVector<Value> inVals;
    inVals.reserve(std::min(numElements, operands.size()));
    for (unsigned i = 0; i < std::min(numElements, operands.size()); i++) {
      inVals.push_back(operands[i][0]);
    }
    if (useFP16IntermediateSrc)
      for (Value &v : inVals)
        v = convertFp32ToFp16(loc, rewriter, v, roundingMode.value());
    inVals.resize(numElements, b.undef(typeConverter->convertType(srcType)));
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

template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    assert((!getElementType(operands[0][0]).isBF16() &&
            !getElementType(operands[0][1]).isBF16()) &&
           "unsupported conversion");
    return {
        rewriter.create<DestOp>(loc, elemTy, operands[0][0], operands[0][1])};
  }
};

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
      auto value = rewriter.create<LLVM::SIToFPOp>(loc, f32_ty, operands[0][0]);
      return {
          intel::convertFp32ToBf16(loc, rewriter, value, RoundingMode::RTNE)};
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
    }
    return {rewriter.create<LLVM::FPTruncOp>(loc, elemTy, operands[0][0])};
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

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    const double log2e = 1.4426950408889634;
    Value prod = b.fmul(f32_ty, operands[0][0], b.f32_val(log2e));

    // Here we use llvm.exp2.f32 instead of math::Exp2Op. The latter
    // flushes denorms by default, but we want to preserve denorms by default
    // for expOp.
    StringRef funcName = "llvm.exp2.f32";
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);

    return {LLVM::createLLVMCallOp(rewriter, loc, funcOp, prod).getResult()};
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
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value v = operands[0][0];
    Type origTy = elemTy;
    if (llvm::isa<BFloat16Type>(origTy)) {
      v = b.bitcast(v, i16_ty);
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
      Value res = b.and_(v, maskConst);
      if (llvm::isa<BFloat16Type>(origTy))
        res = b.bitcast(res, origTy);
      return {res};
    }

    return {rewriter.create<LLVM::FAbsOp>(loc, elemTy, v)};
  }
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
    SmallVector<Type> operandTypes(ValueRange(operands[0]).getTypes());
    std::string fnName =
        mlir::triton::gpu::intel::mangle(funcName, operandTypes);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, fnName, funcType);
    funcOp.setCConv(triton::gpu::intel::getDefaultCConv(op));
    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]);
    callOp.setCConv(funcOp.getCConv());
    return {callOp.getResult()};
  }

private:
  StringRef funcName;
};

// Following two patterns are copied from the common part to fix-up calling
// convention for created function declaration.
// TODO: propose changes in the common part to use CC provided by target.
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

    auto funcName = targetInfo.getMulhiFuncName(resultElementTy);
    Type funcType = getFunctionType(elemTy, operands[0]);
    LLVM::LLVMFuncOp funcOp =
        appendOrGetExternFuncOp(rewriter, op, funcName, funcType);
    funcOp.setCConv(triton::gpu::intel::getDefaultCConv(op));
    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]);
    callOp.setCConv(funcOp.getCConv());
    return {callOp.getResult()};
  }

protected:
  const TargetInfoBase &targetInfo;
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
    funcOp.setCConv(triton::gpu::intel::getDefaultCConv(op));
    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, operands[0]);
    callOp.setCConv(funcOp.getCConv());
    return {callOp.getResult()};
  }
};

} // namespace

namespace mlir::triton::intel {
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit) {

  patterns.add<OpToExternCallConversion<triton::PreciseSqrtOp>>(
      typeConverter, axisInfoAnalysis, "sqrt_cr", benefit);
  patterns.add<OpToExternCallConversion<triton::PreciseDivFOp>>(
      typeConverter, axisInfoAnalysis, "divide_cr", benefit);
  patterns.add<MulhiUIOpConversion>(typeConverter, axisInfoAnalysis, targetInfo,
                                    benefit);
  patterns.add<ExternElementwiseOpConversion>(typeConverter, axisInfoAnalysis,
                                              benefit);

  // Use lower benefit for common patterns to prioritize our versions.
  assert(benefit > 0);
  mlir::triton::populateElementwiseOpToLLVMPatterns(
      typeConverter, patterns, axisInfoAnalysis, targetInfo,
      benefit.getBenefit() - 1);

  patterns.add<AbsFOpConversion>(typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<arith::DivFOp, LLVM::FDivOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<arith::MulFOp, LLVM::FMulOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<arith::AddFOp, LLVM::FAddOp>>(
      typeConverter, axisInfoAnalysis, benefit);
  patterns.add<ElementwiseOpConversion<arith::SubFOp, LLVM::FSubOp>>(
      typeConverter, axisInfoAnalysis, benefit);

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
