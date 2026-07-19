#include "Dialect/TritonIntelGPU/Transforms/DecomposeScaledBlocked.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

SmallVector<int, 2> getTransposeOrder(int rank) {
  assert(rank >= 2);
  auto transOrder = llvm::to_vector<2>(llvm::seq<int>(rank - 2));
  transOrder.push_back(rank - 1);
  transOrder.push_back(rank - 2);
  return transOrder;
}

// Build a BlockedEncodingAttr where sizePerThread[kDim] is doubled by pulling
// a factor of 2 out of threadsPerWarp[kDim] and giving it to threadsPerWarp on
// a donor axis (some non-K axis with slack).  Returns std::nullopt if no valid
// donor axis exists — caller should fall back to non-tiled path.
std::optional<BlockedEncodingAttr>
makeSplitFriendlyBlockedEnc(BlockedEncodingAttr srcEnc, ArrayRef<int64_t> shape,
                            int kDim) {
  auto spt = srcEnc.getSizePerThread();
  auto tpw = srcEnc.getThreadsPerWarp();
  auto wpc = srcEnc.getWarpsPerCTA();
  int rank = spt.size();

  SmallVector<unsigned> newSpt(spt.begin(), spt.end());
  SmallVector<unsigned> newTpw(tpw.begin(), tpw.end());
  SmallVector<unsigned> newWpc(wpc.begin(), wpc.end());

  // If sizePerThread[kDim] is already >= 2, the source is already split-
  // friendly — no changes needed.
  if (spt[kDim] >= 2)
    return srcEnc;

  // Try to pull a factor of 2 from threadsPerWarp[kDim] first, then
  // warpsPerCTA[kDim].
  if (tpw[kDim] % 2 == 0) {
    newSpt[kDim] = spt[kDim] * 2;
    newTpw[kDim] = tpw[kDim] / 2;
  } else if (wpc[kDim] % 2 == 0) {
    newSpt[kDim] = spt[kDim] * 2;
    newWpc[kDim] = wpc[kDim] / 2;
  } else {
    return std::nullopt;
  }

  // We halved some factor on kDim.  Preserve total thread/warp count by
  // giving that factor to a non-K "donor" axis (in the same tpw or wpc bucket
  // we halved from).  Check whether the donor can accept it (i.e. its new
  // used product doesn't exceed the shape on that axis).
  bool halvedTpw = (newTpw[kDim] != tpw[kDim]);
  bool halvedWpc = (newWpc[kDim] != wpc[kDim]);

  auto findDonor = [&](bool doubleTpw) -> int {
    int best = -1;
    int64_t bestSlack = 0;
    for (int d = 0; d < rank; ++d) {
      if (d == kDim)
        continue;
      int64_t used = int64_t(newTpw[d]) * newSpt[d] * newWpc[d];
      int64_t newUsed = used * 2;
      if (newUsed > shape[d])
        continue;
      int64_t slack = shape[d] / std::max<int64_t>(newUsed, 1);
      if (slack > bestSlack) {
        bestSlack = slack;
        best = d;
      }
    }
    if (best != -1)
      return best;
    // Second pass: allow tight fit (slack==0 but newUsed==shape[d]).
    for (int d = 0; d < rank; ++d) {
      if (d == kDim)
        continue;
      int64_t used = int64_t(newTpw[d]) * newSpt[d] * newWpc[d];
      if (used * 2 <= shape[d])
        return d;
    }
    return -1;
  };

  int donor = findDonor(halvedTpw);
  if (donor == -1)
    return std::nullopt;

  if (halvedTpw)
    newTpw[donor] = tpw[donor] * 2;
  else if (halvedWpc)
    newWpc[donor] = wpc[donor] * 2;

  return BlockedEncodingAttr::get(srcEnc.getContext(), newSpt, newTpw, newWpc,
                                  srcEnc.getOrder(), srcEnc.getCGALayout());
}

class DecomposeScaledBlocked : public OpRewritePattern<DotScaledOp> {

public:
  DecomposeScaledBlocked(MLIRContext *context, int benefit)
      : OpRewritePattern<DotScaledOp>(context, benefit) {}

  LogicalResult matchAndRewrite(DotScaledOp scaledDotOp,
                                PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = scaledDotOp.getType();
    // Skip if already converted to DPAS encoding
    if (oldRetType.getEncoding() &&
        isa<intel::DpasEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // Types
    auto computeType = getComputeType(scaledDotOp.getAElemType(),
                                      scaledDotOp.getBElemType(), rewriter);
    auto loc = scaledDotOp.getLoc();

    // Decide if K-tiling is needed
    int numTiles = computeNumKTiles(scaledDotOp, computeType);

    auto cvtDotOperand = [&](TypedValue<RankedTensorType> v,
                             int opIdx) -> TypedValue<RankedTensorType> {
      auto *ctx = rewriter.getContext();
      auto retEnc = scaledDotOp.getType().getEncoding();
      auto vType = v.getType();
      auto encoding = DotOperandEncodingAttr::get(ctx, opIdx, retEnc,
                                                  vType.getElementType());
      RankedTensorType retTy = vType.cloneWithEncoding(encoding);
      return ConvertLayoutOp::create(rewriter, loc, retTy, v);
    };

    if (numTiles == 1) {
      // No tiling - original path
      auto scaledA = scaleArg(rewriter, scaledDotOp, 0, computeType);
      scaledA = cvtDotOperand(scaledA, 0);
      auto scaledB = scaleArg(rewriter, scaledDotOp, 1, computeType);
      scaledB = cvtDotOperand(scaledB, 1);
      auto newDot =
          DotOp::create(rewriter, scaledDotOp.getLoc(), scaledA, scaledB,
                        scaledDotOp.getC(), InputPrecision::TF32, 0);

      rewriter.replaceOpWithNewOp<ConvertLayoutOp>(
          scaledDotOp, scaledDotOp.getType(), newDot);
      return success();
    }

    // K-tiling path: split operands, process each tile sequentially.  If the
    // split cannot be made layout-friendly, fall back to the non-tiled path.
    auto aUpcast = upcastOperand(rewriter, scaledDotOp, 0, computeType);
    auto bUpcast = upcastOperand(rewriter, scaledDotOp, 1, computeType);

    auto aTiles = splitAlongK(rewriter, scaledDotOp, aUpcast, 0, numTiles);
    auto bTiles = splitAlongK(rewriter, scaledDotOp, bUpcast, 1, numTiles);

    // Scale split (skipped if absent).  If scales are present but their split
    // fails, we must also bail out.
    SmallVector<Value> aScaleTiles, bScaleTiles;
    bool aScaleOk = true, bScaleOk = true;
    if (scaledDotOp.getAScale()) {
      aScaleTiles = splitScaleAlongK(rewriter, scaledDotOp, 0, numTiles);
      aScaleOk = (int)aScaleTiles.size() == numTiles;
    }
    if (scaledDotOp.getBScale()) {
      bScaleTiles = splitScaleAlongK(rewriter, scaledDotOp, 1, numTiles);
      bScaleOk = (int)bScaleTiles.size() == numTiles;
    }

    bool tilesOk = (int)aTiles.size() == numTiles &&
                   (int)bTiles.size() == numTiles && aScaleOk && bScaleOk;

    if (!tilesOk) {
      auto scaledA = scaleArg(rewriter, scaledDotOp, 0, computeType);
      scaledA = cvtDotOperand(scaledA, 0);
      auto scaledB = scaleArg(rewriter, scaledDotOp, 1, computeType);
      scaledB = cvtDotOperand(scaledB, 1);
      auto newDot =
          DotOp::create(rewriter, scaledDotOp.getLoc(), scaledA, scaledB,
                        scaledDotOp.getC(), InputPrecision::TF32, 0);
      rewriter.replaceOpWithNewOp<ConvertLayoutOp>(
          scaledDotOp, scaledDotOp.getType(), newDot);
      return success();
    }

    // Process each tile: scale + dot
    Value acc = scaledDotOp.getC();
    for (int i = 0; i < numTiles; i++) {
      auto scaledATile = applyScaleToTile(
          rewriter, scaledDotOp, aTiles[i],
          aScaleTiles.empty() ? Value() : aScaleTiles[i], 0, computeType);
      scaledATile = cvtDotOperand(scaledATile, 0);

      auto scaledBTile = applyScaleToTile(
          rewriter, scaledDotOp, bTiles[i],
          bScaleTiles.empty() ? Value() : bScaleTiles[i], 1, computeType);
      scaledBTile = cvtDotOperand(scaledBTile, 1);

      acc = DotOp::create(rewriter, loc, scaledATile, scaledBTile, acc,
                          InputPrecision::TF32, 0);
    }

    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(scaledDotOp,
                                                 scaledDotOp.getType(), acc);
    return success();
  }

private:
  // Compute how many K-tiles are needed to reduce register pressure
  int computeNumKTiles(DotScaledOp scaledDotOp, FloatType computeType) const {
    auto aType = scaledDotOp.getA().getType();
    auto rank = aType.getRank();
    int64_t M = aType.getShape()[rank - 2];
    int64_t K = aType.getShape()[rank - 1];

    // Adjust K for FP4 packing
    if (scaledDotOp.getAElemType() == ScaleDotElemType::E2M1 &&
        scaledDotOp.getLhsKPack())
      K *= 2;

    // Estimate intermediate size per operand:
    // upcast (K elems) + scale_broadcast (K elems) + mulf (K elems) + nan_mask
    // (K elems) = 4 * M * K * sizeof(bf16) per operand
    int64_t bytesPerElem = computeType.getIntOrFloatBitWidth() / 8;
    int64_t perOperandBytes = 4 * M * K * bytesPerElem;
    int64_t totalBytes = perOperandBytes * 2; // both A and B

    // Target: fit in 64KB (half of 128KB register file, leave room for
    // accumulator)
    constexpr int64_t targetMaxBytes = 65536;

    if (totalBytes <= targetMaxBytes)
      return 1; // No tiling needed

    // Calculate number of tiles needed
    int numTiles = 2;
    int32_t scaleFactor = scaledDotOp.deduceScaleFactor();

    // Check if tiling is beneficial
    if (totalBytes / numTiles > targetMaxBytes && numTiles * scaleFactor <= K) {
      return 2;
    }

    return 1; // No tiling
  }

  // Upcast operand to compute type without scaling
  TypedValue<RankedTensorType> upcastOperand(PatternRewriter &rewriter,
                                             DotScaledOp scaledDotOp, int opIdx,
                                             FloatType computeType) const {
    auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    auto res = scaledDotOp.getD();
    auto isFp4 =
        ScaleDotElemType::E2M1 ==
        (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

    auto loc = v.getLoc();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

    if (isFp4) {
      auto resShape = res.getType().getShape();
      auto vShape = v.getType().getShape();
      auto packDim = kDim;
      if ((opIdx == 0 && resShape[rank - 2] != vShape[rank - 2]) ||
          (opIdx == 1 && resShape[rank - 1] != vShape[rank - 1])) {
        packDim = (packDim + 1) % 2;
      }
      return Fp4ToFpOp::create(rewriter, loc, v, computeType, packDim);
    } else {
      auto vType16 = v.getType().clone(computeType);
      return cast<TypedValue<RankedTensorType>>(
          FpToFpOp::create(rewriter, loc, vType16, v).getResult());
    }
  }

  // Split tensor along K dimension.  Returns empty vector to signal that the
  // source layout cannot be made split-friendly and the caller should fall
  // back to the non-tiled path.
  //
  // Approach: ConvertLayoutOp to a split-friendly BlockedEncodingAttr (where
  // sizePerThread[kDim] >= 2) BEFORE reshape+split, then ConvertLayoutOp each
  // split half back to the original source encoding.  Mirrors upstream
  // WGMMAPipeline::splitLhs.
  SmallVector<Value> splitAlongK(PatternRewriter &rewriter,
                                 DotScaledOp scaledDotOp,
                                 TypedValue<RankedTensorType> tensor, int opIdx,
                                 int numTiles) const {
    auto tensorType = tensor.getType();
    auto shape = tensorType.getShape();
    int rank = shape.size();
    auto loc = tensor.getLoc();

    // K dimension is last for opIdx=0, second-to-last for opIdx=1
    int kDim = (opIdx == 0) ? rank - 1 : rank - 2;
    int64_t K = shape[kDim];
    assert(K % numTiles == 0 && "K must be divisible by numTiles");
    int64_t tileK = K / numTiles;

    // The tt.split fast path requires sizePerThread.back() == 2 on the
    // pre-split tensor.  Rewrite the source encoding so that kDim has enough
    // registers, via a ConvertLayoutOp.
    auto srcEnc =
        dyn_cast_or_null<BlockedEncodingAttr>(tensorType.getEncoding());
    if (!srcEnc)
      return {};

    auto maybeFriendly = makeSplitFriendlyBlockedEnc(srcEnc, shape, kDim);
    if (!maybeFriendly.has_value())
      return {};
    auto friendlyEnc = *maybeFriendly;

    // Step 0: Convert to split-friendly layout
    auto friendlyType =
        RankedTensorType::get(shape, tensorType.getElementType(), friendlyEnc);
    Value src = ConvertLayoutOp::create(rewriter, loc, friendlyType, tensor);

    // Step 1: Reshape [..., K, ...] -> [..., K/2, 2, ...]
    // MLIR will infer the reshaped encoding via inferReshapeOpEncoding.  With
    // sizePerThread[kDim] = 2*spt[kDim] on the friendly encoding, the inserted
    // size-2 axis will receive sizePerThread=2.
    SmallVector<int64_t> viewShape(shape.begin(), shape.end());
    viewShape[kDim] = tileK;
    viewShape.insert(viewShape.begin() + kDim + 1, 2);
    auto viewed = ReshapeOp::create(rewriter, loc, viewShape,
                                    cast<TypedValue<RankedTensorType>>(src));

    // Result type of the tiles (source encoding, reduced K).
    SmallVector<int64_t> tileShape(shape.begin(), shape.end());
    tileShape[kDim] = tileK;
    auto tileType =
        RankedTensorType::get(tileShape, tensorType.getElementType(), srcEnc);

    // Step 2: If the size-2 axis is NOT last, transpose to move it there.
    // Then split.  ConvertLayoutOp on each half handles any permutation
    // needed to get back to the source encoding on tileShape.
    Value lhs, rhs;
    if (kDim + 1 < rank) {
      SmallVector<int32_t> order;
      for (int i = 0; i <= rank; ++i) {
        if (i == kDim + 1)
          continue;
        order.push_back(i);
      }
      order.push_back(kDim + 1);

      auto transposed = TransOp::create(rewriter, loc, viewed, order);
      auto split = SplitOp::create(rewriter, loc, transposed);

      // Split results have rank == rank (original), but their axes are in
      // the permuted order given by `order[0..rank-1]`.  ConvertLayoutOp
      // cannot re-order axes — we need a TransOp to permute back first.
      // Build the inverse of the prefix permutation.
      // If original axes are [0..rank-1] and permuted axes are
      // order[0..rank-1], then invPerm[p] = i where order[i] == p, for p in
      // [0..rank-1].
      SmallVector<int32_t> invPerm(rank);
      for (int i = 0; i < rank; ++i) {
        int32_t p = order[i]; // p is in [0..rank] but not equal to kDim+1
        // Map p to its position in the "kept" axes (order[0..rank-1]).
        // p < kDim+1 → position p; p > kDim+1 → position p-1.
        int32_t origAxis = (p < kDim + 1) ? p : (p - 1);
        invPerm[origAxis] = i;
      }

      lhs = TransOp::create(rewriter, loc, split.getResult(0), invPerm);
      rhs = TransOp::create(rewriter, loc, split.getResult(1), invPerm);
    } else {
      auto split = SplitOp::create(rewriter, loc, viewed);
      lhs = split.getResult(0);
      rhs = split.getResult(1);
    }
    // Convert back to source encoding.
    return {ConvertLayoutOp::create(rewriter, loc, tileType, lhs),
            ConvertLayoutOp::create(rewriter, loc, tileType, rhs)};
  }

  // Split scale tensor along K.  Same layout-friendly approach as
  // splitAlongK: ConvertLayoutOp to a friendly encoding before reshape+split,
  // then ConvertLayoutOp back.  Scale K is always the trailing dimension, so
  // no transpose is needed after reshape.  Returns empty vector on fallback.
  SmallVector<Value> splitScaleAlongK(PatternRewriter &rewriter,
                                      DotScaledOp scaledDotOp, int opIdx,
                                      int numTiles) const {
    auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
    if (!scale)
      return {};

    auto scaleType = cast<RankedTensorType>(scale.getType());
    auto shape = scaleType.getShape();
    int rank = shape.size();
    int kDim = rank - 1; // Scale K is always trailing
    int64_t scaleK = shape[kDim];
    int64_t tileK = scaleK / numTiles;
    auto loc = scale.getLoc();

    auto srcEnc =
        dyn_cast_or_null<BlockedEncodingAttr>(scaleType.getEncoding());
    if (!srcEnc)
      return {};

    auto maybeFriendly = makeSplitFriendlyBlockedEnc(srcEnc, shape, kDim);
    if (!maybeFriendly.has_value())
      return {};
    auto friendlyEnc = *maybeFriendly;

    // Step 0: Convert to split-friendly layout.
    auto friendlyType =
        RankedTensorType::get(shape, scaleType.getElementType(), friendlyEnc);
    Value src = ConvertLayoutOp::create(
        rewriter, loc, friendlyType, cast<TypedValue<RankedTensorType>>(scale));

    // Step 1: Reshape [..., K] -> [..., K/2, 2].  Size-2 axis is trailing.
    SmallVector<int64_t> viewShape(shape.begin(), shape.end());
    viewShape[kDim] = tileK;
    viewShape.push_back(2);
    auto viewed = ReshapeOp::create(rewriter, loc, viewShape,
                                    cast<TypedValue<RankedTensorType>>(src));

    // Step 2: Split on trailing size-2 dim.
    auto split = SplitOp::create(rewriter, loc, viewed);

    // Step 3: Convert each half back to source encoding on reduced-K shape.
    SmallVector<int64_t> tileShape(shape.begin(), shape.end());
    tileShape[kDim] = tileK;
    auto tileType =
        RankedTensorType::get(tileShape, scaleType.getElementType(), srcEnc);
    return {
        ConvertLayoutOp::create(rewriter, loc, tileType, split.getResult(0)),
        ConvertLayoutOp::create(rewriter, loc, tileType, split.getResult(1))};
  }

  // Apply scale to a single tile
  TypedValue<RankedTensorType> applyScaleToTile(PatternRewriter &rewriter,
                                                DotScaledOp scaledDotOp,
                                                Value upcastTile,
                                                Value scaleTile, int opIdx,
                                                FloatType computeType) const {
    auto v = cast<TypedValue<RankedTensorType>>(upcastTile);
    if (!scaleTile)
      return v;

    auto loc = v.getLoc();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;
    auto scale = cast<TypedValue<RankedTensorType>>(scaleTile);

    // Transpose scale for RHS operand
    if (opIdx == 1) {
      auto order = getTransposeOrder(rank);
      scale = TransOp::create(rewriter, loc, scale, order);
    }

    // Cast scale to compute type
    auto scale16 = scaleTo16(rewriter, scale, computeType);

    // Broadcast scale to match tile shape
    auto mod = scaledDotOp->getParentOfType<ModuleOp>();
    auto reshapeScale =
        broadcastScale(rewriter, scaledDotOp, mod, scale16, kDim);
    reshapeScale =
        ConvertLayoutOp::create(rewriter, loc, v.getType(), reshapeScale);

    // Multiply
    auto mxfp = cast<TypedValue<RankedTensorType>>(
        arith::MulFOp::create(rewriter, loc, v, reshapeScale).getResult());

    // Apply NaN mask
    return maskNan(rewriter, scaledDotOp, mxfp, scale, kDim);
  }

  FloatType getComputeType(ScaleDotElemType aType, ScaleDotElemType bType,
                           PatternRewriter &rewriter) const {
    if (aType == ScaleDotElemType::FP16 || bType == ScaleDotElemType::FP16)
      return rewriter.getF16Type();
    return rewriter.getBF16Type();
  }

  TypedValue<RankedTensorType> scaleTo16(PatternRewriter &rewriter,
                                         TypedValue<RankedTensorType> scale,
                                         FloatType computeType) const {
    auto loc = scale.getLoc();
    auto scaleTy = scale.getType();
    assert(computeType == rewriter.getBF16Type() ||
           computeType == rewriter.getF16Type());

    if (isa<FloatType>(scaleTy.getElementType())) {
      auto scaleType = scaleTy.clone(computeType);
      return cast<TypedValue<RankedTensorType>>(
          FpToFpOp::create(rewriter, loc, scaleType, scale).getResult());
    }

    // Choose an fp type that can fit the scale value.
    FloatType largeFpType = computeType == rewriter.getF16Type()
                                ? rewriter.getF32Type()
                                : computeType;
    int intWidth = largeFpType.getIntOrFloatBitWidth();
    auto intType = rewriter.getIntegerType(intWidth);

    auto zexted =
        arith::ExtUIOp::create(rewriter, loc, scaleTy.clone(intType), scale);
    // getFpMantissaWidth() returns the number of bits in the mantissa plus the
    // sign bit!
    int shiftValue = largeFpType.getFPMantissaWidth() - 1;
    auto shiftConst =
        arith::ConstantIntOp::create(rewriter, loc, shiftValue, intWidth);
    auto shift =
        SplatOp::create(rewriter, loc, scaleTy.clone(intType), shiftConst);
    auto shlRes = arith::ShLIOp::create(rewriter, loc, zexted, shift);
    Value scaleFP =
        BitcastOp::create(rewriter, loc, scaleTy.clone(largeFpType), shlRes);
    if (largeFpType != computeType) {
      scaleFP = arith::TruncFOp::create(rewriter, loc,
                                        scaleTy.clone(computeType), scaleFP);
    }
    return cast<TypedValue<RankedTensorType>>(scaleFP);
  }

  TypedValue<RankedTensorType>
  broadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                 ModuleOp mod, TypedValue<RankedTensorType> scale,
                 int dim) const {
    auto *ctx = rewriter.getContext();
    auto loc = scale.getLoc();
    auto scaleTy = scale.getType();
    auto rank = scaleTy.getRank();
    // 2.1) Expand dims along the last dimension
    {
      // 2.1.1) Find default encoding for ExpandDims
      auto shape = to_vector(scaleTy.getShape());
      shape.insert(shape.end(), 1);
      auto nWarps = lookupNumWarps(scaledDotOp);
      auto threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
      auto numCTAs = TritonGPUDialect::getNumCTAs(mod);
      auto blockedEnc = getDefaultBlockedEncoding(ctx, shape, nWarps,
                                                  threadsPerWarp, numCTAs);
      // 2.1.2) Cast scale16 to SliceEncoding
      auto sliceEnc = SliceEncodingAttr::get(ctx, rank, blockedEnc);
      RankedTensorType sliceType = scaleTy.cloneWithEncoding(sliceEnc);
      scale = ConvertLayoutOp::create(rewriter, loc, sliceType, scale);
    }
    auto expandScale = ExpandDimsOp::create(rewriter, loc, scale, rank);
    int32_t scaleFactor = scaledDotOp.deduceScaleFactor();
    // 2.2) Broadcast the dimension to the microscaling factor.
    auto scaleShape = to_vector(scaleTy.getShape());
    scaleShape.push_back(scaleFactor);
    auto broadcastScale = BroadcastOp::create(
        rewriter, loc, expandScale.getType().clone(scaleShape), expandScale);
    // 2.3) Transpose the dimension to the scaled dimension
    auto transposeOrder = llvm::to_vector(llvm::seq<int32_t>(rank));
    transposeOrder.insert(transposeOrder.begin() + dim + 1, rank);
    auto transposedScale =
        TransOp::create(rewriter, loc, broadcastScale, transposeOrder);
    // 2.4) Reshape to the shape of v
    scaleShape.pop_back();
    scaleShape[dim] *= scaleFactor;
    auto reshapeScale =
        ReshapeOp::create(rewriter, loc, scaleShape, transposedScale);
    return reshapeScale;
  }

  TypedValue<RankedTensorType>
  extendAndBroadcastScale(PatternRewriter &rewriter, DotScaledOp scaledDotOp,
                          TypedValue<RankedTensorType> &scale,
                          FloatType computeType, RankedTensorType dstType,
                          int opIdx) const {
    auto loc = scale.getLoc();
    auto mod = scaledDotOp->getParentOfType<ModuleOp>();
    auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // Transpose scale for RHS operand (inplace — caller sees the change).
    if (opIdx == 1) {
      auto order = getTransposeOrder(rank);
      scale = TransOp::create(rewriter, loc, scale, order);
    }

    // 1) Cast scale to compute type (fp16/bf16)
    auto scale16 = scaleTo16(rewriter, scale, computeType);

    // 2) Broadcast scale to the same shape as v and convert the layout
    auto reshapeScale =
        broadcastScale(rewriter, scaledDotOp, mod, scale16, kDim);
    return ConvertLayoutOp::create(rewriter, loc, dstType, reshapeScale);
  }

  TypedValue<RankedTensorType> maskNan(PatternRewriter &rewriter,
                                       DotScaledOp scaledDotOp,
                                       TypedValue<RankedTensorType> mxfp,
                                       TypedValue<RankedTensorType> scale,
                                       int dim) const {
    // Skip NaN checks if fastMath
    if (scaledDotOp.getFastMath())
      return mxfp;

    // Implement tl.where(scale == 0xFF, float("nan"), mxfp)
    auto loc = scale.getLoc();
    auto mod = scaledDotOp->getParentOfType<ModuleOp>();

    // Scale is NaN
    auto scaleTy = scale.getType();
    TypedValue<RankedTensorType> scaleIsNan;
    if (isa<FloatType>(scaleTy.getElementType())) {
      auto computeType = cast<FloatType>(mxfp.getType().getElementType());
      auto scaleFp = scaleTo16(rewriter, scale, computeType);
      scaleIsNan = cast<TypedValue<RankedTensorType>>(
          arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::UNO,
                                scaleFp, scaleFp)
              .getResult());
    } else {
      auto constFF = arith::ConstantOp::create(
          rewriter, loc, scaleTy,
          DenseElementsAttr::get(
              scaleTy, APInt(scaleTy.getElementTypeBitWidth(), 0xff)));
      scaleIsNan = cast<TypedValue<RankedTensorType>>(
          arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, scale,
                                constFF)
              .getResult());
    }
    auto cond = broadcastScale(rewriter, scaledDotOp, mod, scaleIsNan, dim);
    // Make scale is NaN compatible with mxfp
    auto condTy = cond.getType();
    condTy = condTy.cloneWithEncoding(mxfp.getType().getEncoding());
    cond = ConvertLayoutOp::create(rewriter, loc, condTy, cond);

    // Create NaN
    auto mxfpTy = mxfp.getType();
    auto nan = APFloat::getNaN(
        cast<FloatType>(mxfpTy.getElementType()).getFloatSemantics());
    auto constNan = arith::ConstantOp::create(
        rewriter, loc, mxfpTy, DenseElementsAttr::get(mxfpTy, nan));

    auto result = arith::SelectOp::create(rewriter, loc, cond, constNan, mxfp);
    return cast<TypedValue<RankedTensorType>>(result.getResult());
  }

  TypedValue<RankedTensorType> scaleArg(PatternRewriter &rewriter,
                                        DotScaledOp scaledDotOp, int opIdx,
                                        FloatType computeType) const {
    auto v = opIdx == 0 ? scaledDotOp.getA() : scaledDotOp.getB();
    auto res = scaledDotOp.getD();
    auto scale = opIdx == 0 ? scaledDotOp.getAScale() : scaledDotOp.getBScale();
    auto isFp4 =
        ScaleDotElemType::E2M1 ==
        (opIdx == 0 ? scaledDotOp.getAElemType() : scaledDotOp.getBElemType());

    auto loc = v.getLoc();
    auto rank = v.getType().getRank();
    auto kDim = opIdx == 0 ? rank - 1 : rank - 2;

    // 0) Upcast value to computeType (fp16/bf16)
    if (isFp4) {
      auto resShape = res.getType().getShape();
      auto vShape = v.getType().getShape();
      auto packDim = kDim;
      if ((opIdx == 0 && resShape[rank - 2] != vShape[rank - 2]) ||
          (opIdx == 1 && resShape[rank - 1] != vShape[rank - 1])) {
        packDim = (packDim + 1) % 2;
      }
      v = Fp4ToFpOp::create(rewriter, loc, v, computeType, packDim);
    } else {
      auto vType16 = v.getType().clone(computeType);
      v = cast<TypedValue<RankedTensorType>>(
          FpToFpOp::create(rewriter, loc, vType16, v).getResult());
    }
    if (!scale)
      return v;

    // 1) Cast scale to fp16/bf16, broadcast it and convert its layout
    auto reshapeScale = extendAndBroadcastScale(
        rewriter, scaledDotOp, scale, computeType, v.getType(), opIdx);

    // 2) Multiply
    auto mxfp = cast<TypedValue<RankedTensorType>>(
        arith::MulFOp::create(rewriter, loc, v, reshapeScale).getResult());

    // 3) If the scale is NaN, return NaN, else return the scaled value.
    return maskNan(rewriter, scaledDotOp, mxfp, scale, kDim);
  }
};

} // namespace

namespace mlir::triton::gpu::intel {

void populateDecomposeScaledBlockedPatterns(RewritePatternSet &patterns,
                                            int benefit) {
  patterns.add<DecomposeScaledBlocked>(patterns.getContext(), benefit);
}

} // namespace mlir::triton::gpu::intel
