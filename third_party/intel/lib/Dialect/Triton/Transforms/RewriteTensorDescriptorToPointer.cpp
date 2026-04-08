#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>

namespace mlir::triton::intel {

#define GEN_PASS_DEF_TRITONREWRITETENSORDESCRIPTORTOPOINTER
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

using namespace mlir;

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute>
filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs) {
  mlir::SmallVector<NamedAttribute> ret;
  llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
    auto attrName = attr.getName().getValue();
    return attrName != "operandSegmentSizes";
  });
  return ret;
}

struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
  Value roundF32ToTF32;
};

Descriptor unpackDescriptor(TensorDescType type, ValueRange pack) {
  int rank = type.getBlockType().getRank();
  assert(pack.size() == 1 + 2 * static_cast<size_t>(rank) + 2 &&
         "Expected tensor descriptors to consist of a pointer, "
         "followed by 'rank' shape values and 'rank' stride values, "
         "followed by padding and TF32 rounding option values.");

  Descriptor res;
  res.base = pack[0];
  res.shape = pack.slice(1, rank);
  res.strides = pack.slice(1 + rank, rank);
  res.paddingOption = pack[1 + 2 * rank];
  res.roundF32ToTF32 = pack[2 + 2 * rank];
  return res;
}

Value expandOffsets(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockShape, Value offsets, unsigned dim) {
  Value expandedResult = offsets;
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == dim) {
      continue;
    }
    expandedResult =
        triton::ExpandDimsOp::create(builder, loc, expandedResult, j);
  }

  return expandedResult;
}

Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 Value offset, unsigned dim) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI64Type());
  Value splatOffset =
      triton::SplatOp::create(builder, loc, indexRowType, offset);
  Value range = triton::MakeRangeOp::create(builder, loc, indexI32RowType, 0,
                                            blockShape[dim]);
  Value i64Range = arith::ExtSIOp::create(builder, loc, indexRowType, range);

  Value offsets = arith::AddIOp::create(builder, loc, splatOffset, i64Range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = triton::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = triton::SplatOp::create(
        builder, loc, offsets[i].getType(), desc.strides[i]);
    Value offsetWithStride =
        arith::MulIOp::create(builder, loc, offsets[i], splatStride);
    Value broadcasted = triton::BroadcastOp::create(
        builder, loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        triton::AddPtrOp::create(builder, loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
}

Value generateMaskFromOffsetRanges(OpBuilder &builder, const Location &loc,
                                   ArrayRef<std::int64_t> blockShape,
                                   Descriptor &desc, ValueRange offsetRanges) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsetRanges.size());

  // Generate mask per dimension
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  Value mask;
  for (std::size_t i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange = offsetRanges[i];

    // Compare with lower bound
    Value lowerBound = mlir::arith::ConstantIntOp::create(
        builder, loc, builder.getI64Type(), 0);
    Value splatLowerBound = triton::SplatOp::create(
        builder, loc, offsetWithRange.getType(), lowerBound);
    Value cmpLower =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                              offsetWithRange, splatLowerBound);

    // Compare with upper bound
    Value splatUpperBound = triton::SplatOp::create(
        builder, loc, offsetWithRange.getType(), desc.shape[i]);
    Value cmpUpper =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                              offsetWithRange, splatUpperBound);

    // And and broadcast
    Value andResult = arith::AndIOp::create(builder, loc, cmpLower, cmpUpper);
    Value broadcasted =
        triton::BroadcastOp::create(builder, loc, maskTensorType, andResult);

    // And up all results
    if (!mask) {
      mask = broadcasted;
    } else {
      mask = arith::AndIOp::create(builder, loc, mask, broadcasted);
    }
  }

  return mask;
}

Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                   ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                      offsetRanges);
}

Value generateOther(OpBuilder &builder, Location loc, Type scalarTy,
                    ArrayRef<int64_t> blockShape,
                    Value paddingOption = nullptr) {
  auto blockTy = RankedTensorType::get(blockShape, scalarTy);
  if (paddingOption && mlir::isa<FloatType>(scalarTy)) {
    auto floatTy = mlir::cast<FloatType>(scalarTy);
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto nanValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getFloatAttr(floatTy, nan)));
    auto zeroValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getZeroAttr(floatTy)));
    return mlir::arith::SelectOp::create(builder, loc, paddingOption, nanValue,
                                         zeroValue);
  } else {
    auto attr = builder.getZeroAttr(blockTy);
    return arith::ConstantOp::create(builder, loc, attr);
  }
}

Value generateOther(OpBuilder &builder, Location loc, TensorDescType descTy,
                    Value paddingOption = nullptr) {
  auto blockTy = descTy.getSignlessBlockType();
  return generateOther(builder, loc, blockTy.getElementType(),
                       blockTy.getShape(), paddingOption);
}

Type getI32TypeLike(OpBuilder &builder, Type ty) {
  if (auto shapedTy = dyn_cast<ShapedType>(ty))
    return shapedTy.clone(builder.getI32Type());
  return builder.getI32Type();
}

Value getI32ConstLike(OpBuilder &builder, Location loc, Type likeType,
                      int32_t value) {
  auto i32Ty = getI32TypeLike(builder, likeType);
  if (auto shapedTy = dyn_cast<ShapedType>(i32Ty)) {
    auto attr =
        DenseElementsAttr::get(shapedTy, builder.getI32IntegerAttr(value));
    return arith::ConstantOp::create(builder, loc, shapedTy, attr);
  }
  return arith::ConstantOp::create(builder, loc, i32Ty,
                                   builder.getI32IntegerAttr(value));
}

Value roundF32ToTF32(OpBuilder &builder, Location loc, Value value) {
  auto valueTy = value.getType();
  auto i32Ty = getI32TypeLike(builder, valueTy);
  auto bits = triton::BitcastOp::create(builder, loc, i32Ty, value);

  auto expMask = getI32ConstLike(builder, loc, i32Ty, 0x7F800000);
  auto exp = arith::AndIOp::create(builder, loc, bits, expMask);
  auto isSpecial = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         exp, expMask);

  auto shift = getI32ConstLike(builder, loc, i32Ty, 13);
  auto lsb = arith::AndIOp::create(
      builder, loc, arith::ShRUIOp::create(builder, loc, bits, shift),
      getI32ConstLike(builder, loc, i32Ty, 1));
  auto roundBias = arith::AddIOp::create(
      builder, loc, lsb, getI32ConstLike(builder, loc, i32Ty, 0x00000FFF));
  auto rounded = arith::AndIOp::create(
      builder, loc, arith::AddIOp::create(builder, loc, bits, roundBias),
      getI32ConstLike(builder, loc, i32Ty, 0xFFFFE000));
  auto outBits =
      arith::SelectOp::create(builder, loc, isSpecial, bits, rounded);
  return triton::BitcastOp::create(builder, loc, valueTy, outBits);
}

SmallVector<mlir::Value> castToI64(OpBuilder &builder,
                                   mlir::ValueRange values) {
  auto i64Type = builder.getI64Type();
  return llvm::map_to_vector(values, [&](mlir::Value v) {
    return builder.createOrFold<arith::ExtSIOp>(v.getLoc(), i64Type, v);
  });
}

/// Result of analyzing x_offsets for gather/scatter descriptor ops.
struct ContiguousOffsetInfo {
  Value baseOffset; // Start offset — constant or dynamic Value (i32)
  int64_t count;    // Number of consecutive offsets (N)
};

/// Analyze whether x_offsets form a contiguous range
/// [base, base+1, ..., base+N-1].
/// Returns nullopt if offsets are not provably contiguous.
///
/// Walks the SSA def chain of xOffsets looking for the pattern:
///   arith.addi(tt.splat(scalar), arith.extsi(tt.make_range(start, end)))
/// or subsets thereof (e.g. just tt.make_range, or with nested extsi/addi).
static std::optional<ContiguousOffsetInfo>
analyzeContiguousOffsets(Value xOffsets) {
  auto tensorTy = dyn_cast<RankedTensorType>(xOffsets.getType());
  if (!tensorTy || tensorTy.getRank() != 1)
    return std::nullopt;

  int64_t tensorSize = tensorTy.getDimSize(0);

  // Track accumulated dynamic base offset from splat-adds.
  Value dynamicBase = nullptr;

  // Walk through the def chain.
  Value current = xOffsets;
  while (true) {
    Operation *defOp = current.getDefiningOp();
    if (!defOp)
      return std::nullopt;

    // Strip arith.extsi wrappers.
    if (auto extsi = dyn_cast<arith::ExtSIOp>(defOp)) {
      current = extsi.getIn();
      continue;
    }

    // Strip arith.addi — if one side is tt.splat(scalar), accumulate it.
    if (auto addi = dyn_cast<arith::AddIOp>(defOp)) {
      Value lhs = addi.getLhs();
      Value rhs = addi.getRhs();

      // Check if rhs is a splat of a scalar.
      auto rhsSplat = rhs.getDefiningOp<triton::SplatOp>();
      auto lhsSplat = lhs.getDefiningOp<triton::SplatOp>();

      auto splatOp = rhsSplat ? rhsSplat : lhsSplat;
      Value splatSide = splatOp ? splatOp.getSrc() : nullptr;
      Value otherSide = rhsSplat ? lhs : (lhsSplat ? rhs : nullptr);

      if (splatSide) {
        // Accumulate scalar from splat and recurse on the other side.
        if (dynamicBase) {
          if (dynamicBase.getType() != splatSide.getType())
            return std::nullopt;
          // Both dynamicBase and splatSide are operands of defOp, so
          // inserting before defOp guarantees both are defined.
          OpBuilder builder(defOp);
          dynamicBase = arith::AddIOp::create(builder, defOp->getLoc(),
                                              dynamicBase, splatSide);
        } else {
          dynamicBase = splatSide;
        }
        current = otherSide;
        continue;
      }

      // Neither side is a splat — cannot analyze further.
      return std::nullopt;
    }

    // Look for tt.make_range at the root.
    if (auto makeRange = dyn_cast<triton::MakeRangeOp>(defOp)) {
      int64_t start = makeRange.getStartAttr().getInt();
      int64_t end = makeRange.getEndAttr().getInt();
      int64_t count = end - start;

      // Verify the range covers the full tensor dimension with stride 1.
      if (count != tensorSize)
        return std::nullopt;

      // Build the final baseOffset combining constant start and dynamic base.
      // The result must be an i32 scalar (DescriptorLoadOp index type).
      OpBuilder builder(makeRange);
      Location loc = makeRange.getLoc();

      // Cast an integer Value to i32 for use as a DescriptorLoadOp index.
      // Inputs are typically i32 (from make_range/splat) or i64 (after extsi).
      // Uses ExtSI for narrow types, TruncI for wider — truncation is safe
      // here since row indices fit in i32.
      auto castToI32 = [&](Value val) -> Value {
        unsigned srcWidth = cast<IntegerType>(val.getType()).getWidth();
        IntegerType i32Ty = builder.getI32Type();
        if (srcWidth == 32)
          return val;
        if (srcWidth < 32)
          return arith::ExtSIOp::create(builder, loc, i32Ty, val);
        return arith::TruncIOp::create(builder, loc, i32Ty, val);
      };

      Value baseOffset;
      if (dynamicBase) {
        builder.setInsertionPointAfterValue(dynamicBase);
        baseOffset = castToI32(dynamicBase);
        if (start != 0) {
          Value startVal = arith::ConstantOp::create(
              builder, loc, builder.getI32IntegerAttr(start));
          baseOffset =
              arith::AddIOp::create(builder, loc, baseOffset, startVal);
        }
      } else {
        baseOffset = arith::ConstantOp::create(
            builder, loc, builder.getI32IntegerAttr(start));
      }

      return ContiguousOffsetInfo{baseOffset, count};
    }

    // Unknown op — cannot analyze.
    return std::nullopt;
  }
}

/// A contiguous sub-range within compile-time-constant x_offsets.
struct ContiguousRange {
  int64_t start;        // First row index in the surface
  int64_t count;        // Number of consecutive rows
  int64_t resultOffset; // Starting row index in the gather result
};

/// Analyze compile-time-constant x_offsets for contiguous sub-ranges.
/// Returns nullopt if offsets are not all constants.
///
/// Example: [0,1,2,3, 8,9,10,11] →
///   [{start=0, count=4, resultOffset=0}, {start=8, count=4, resultOffset=4}]
static std::optional<SmallVector<ContiguousRange>>
analyzeConstantOffsetRanges(Value xOffsets) {
  auto defOp = xOffsets.getDefiningOp<arith::ConstantOp>();
  if (!defOp)
    return std::nullopt;

  auto denseAttr = dyn_cast<DenseIntElementsAttr>(defOp.getValue());
  if (!denseAttr)
    return std::nullopt;

  SmallVector<int64_t> values;
  for (const APInt &val : denseAttr.getValues<APInt>())
    values.push_back(val.getSExtValue());

  if (values.empty())
    return std::nullopt;

  // Group consecutive values into sub-ranges.
  SmallVector<ContiguousRange> ranges;
  int64_t rangeStart = values[0];
  int64_t resultOffset = 0;
  int64_t count = 1;

  for (size_t i = 1; i < values.size(); ++i) {
    if (values[i] == values[i - 1] + 1) {
      ++count;
    } else {
      ranges.push_back({rangeStart, count, resultOffset});
      resultOffset += count;
      rangeStart = values[i];
      count = 1;
    }
  }
  ranges.push_back({rangeStart, count, resultOffset});

  return ranges;
}

/// Rewrite contiguous DescriptorGatherOps to DescriptorLoadOps.
/// When x_offsets form a contiguous range [base, base+N-1], the gather can
/// be replaced with a single 2D block load which is significantly more
/// efficient than scattered scalar loads.
///
/// Example — single contiguous range:
///   %desc = tt.make_tensor_desc %ptr, [%H, %W], [%s0, %s1]
///              : !tt.tensordesc<tensor<1x64xbf16>>
///   %range = tt.make_range {start = 0, end = 16} : tensor<16xi32>
///   %off = tt.splat %base : (i32) -> tensor<16xi32>
///   %x = arith.addi %range, %off : tensor<16xi32>
///   %v = tt.descriptor_gather %desc[%x, %y]
///          : (!tt.tensordesc<tensor<1x64xbf16>>, tensor<16xi32>, i32)
///            -> tensor<16x64xbf16>
/// Becomes:
///   %new_desc = tt.make_tensor_desc %ptr, [%H, %W], [%s0, %s1]
///                  : !tt.tensordesc<tensor<16x64xbf16>>
///   %v = tt.descriptor_load %new_desc[%base, %y]
///          : !tt.tensordesc<tensor<16x64xbf16>> -> tensor<16x64xbf16>
struct RewriteContiguousGather
    : public OpRewritePattern<triton::DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorGatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // 1. Check if x_offsets are contiguous.
    std::optional<ContiguousOffsetInfo> contiguousInfo =
        analyzeContiguousOffsets(gatherOp.getXOffsets());
    if (!contiguousInfo)
      return rewriter.notifyMatchFailure(gatherOp,
                                         "x_offsets are not contiguous");

    // 2. Get result type shape: N rows, W columns.
    auto resultTy = cast<RankedTensorType>(gatherOp.getResult().getType());
    if (resultTy.getRank() != 2)
      return rewriter.notifyMatchFailure(gatherOp, "result is not rank 2");

    int64_t numRows = resultTy.getShape()[0];
    int64_t rowWidth = resultTy.getShape()[1];
    Type elemTy = resultTy.getElementType();

    // 3. Find the MakeTensorDescOp that defines the descriptor.
    triton::MakeTensorDescOp makeTensorDescOp =
        gatherOp.getDesc().getDefiningOp<triton::MakeTensorDescOp>();
    if (!makeTensorDescOp)
      return rewriter.notifyMatchFailure(
          gatherOp, "descriptor not from MakeTensorDescOp");

    // 4. Create a new MakeTensorDescOp with block shape [N, W].
    Location loc = gatherOp.getLoc();
    auto newBlockTy = RankedTensorType::get({numRows, rowWidth}, elemTy);
    auto newDescTy =
        triton::TensorDescType::get(rewriter.getContext(), newBlockTy);
    auto newMakeDesc = triton::MakeTensorDescOp::create(
        rewriter, loc, newDescTy, makeTensorDescOp.getBase(),
        makeTensorDescOp.getShape(), makeTensorDescOp.getStrides(),
        makeTensorDescOp.getPadding());

    // 5. Create tt.descriptor_load with offsets [baseOffset, yOffset].
    SmallVector<Value> indices = {contiguousInfo->baseOffset,
                                  gatherOp.getYOffset()};
    auto descLoadOp = triton::DescriptorLoadOp::create(rewriter, loc, resultTy,
                                                       newMakeDesc, indices);

    // TF32 rounding for f32 types is handled by RewriteLoadPattern
    // when it converts the DescriptorLoadOp in the subsequent phase.
    rewriter.replaceOp(gatherOp, descLoadOp.getResult());
    return success();
  }
};

/// Rewrite DescriptorGatherOps with compile-time-constant x_offsets that form
/// multiple contiguous sub-ranges into one descriptor_load per sub-range,
/// concatenated with tt.cat.
///
/// Example — constant offsets [0,1,2,3, 8,9,10,11] on tensor<1x64xbf16>:
///   %x = arith.constant dense<[0, 1, 2, 3, 8, 9, 10, 11]> : tensor<8xi32>
///   %v = tt.descriptor_gather %desc[%x, %y]
///          : (!tt.tensordesc<tensor<1x64xbf16>>, tensor<8xi32>, i32)
///            -> tensor<8x64xbf16>
/// Becomes (2 block loads + concatenation):
///   %d0 = tt.make_tensor_desc ... : !tt.tensordesc<tensor<4x64xbf16>>
///   %v0 = tt.descriptor_load %d0[%c0, %y] : ... -> tensor<4x64xbf16>
///   %d1 = tt.make_tensor_desc ... : !tt.tensordesc<tensor<4x64xbf16>>
///   %v1 = tt.descriptor_load %d1[%c8, %y] : ... -> tensor<4x64xbf16>
///   %v  = tt.cat %v0, %v1 : tensor<4x64xbf16> -> tensor<8x64xbf16>
///
/// Constraint: all sub-ranges must have equal size (tt.cat requires
/// SameTypeOperands). Falls back if sub-ranges differ in count.
struct RewriteMultiRangeGather
    : public OpRewritePattern<triton::DescriptorGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DescriptorGatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    // 1. Try to extract constant offset sub-ranges.
    std::optional<SmallVector<ContiguousRange>> rangesOpt =
        analyzeConstantOffsetRanges(gatherOp.getXOffsets());
    if (!rangesOpt)
      return rewriter.notifyMatchFailure(gatherOp,
                                         "x_offsets are not constant");

    SmallVector<ContiguousRange> &ranges = *rangesOpt;

    // Single range is handled by RewriteContiguousGather (higher benefit).
    if (ranges.size() <= 1)
      return rewriter.notifyMatchFailure(
          gatherOp, "single range, defer to contiguous pattern");

    // Cap sub-range count to avoid excessive code generation.
    constexpr size_t kMaxSubRanges = 4;
    if (ranges.size() > kMaxSubRanges)
      return rewriter.notifyMatchFailure(
          gatherOp, "too many sub-ranges, would generate excessive loads");

    // 2. All sub-ranges must have equal count (tt.cat requires
    // SameTypeOperands).
    int64_t rangeCount = ranges[0].count;
    for (const auto &range : ranges) {
      if (range.count != rangeCount)
        return rewriter.notifyMatchFailure(
            gatherOp,
            "sub-ranges have unequal sizes, tt.cat requires same type");
    }

    // 3. Get result shape and element type.
    auto resultTy = cast<RankedTensorType>(gatherOp.getResult().getType());
    if (resultTy.getRank() != 2)
      return rewriter.notifyMatchFailure(gatherOp, "result is not rank 2");
    int64_t rowWidth = resultTy.getShape()[1];
    Type elemTy = resultTy.getElementType();

    // 4. Find the MakeTensorDescOp.
    auto makeTensorDescOp =
        gatherOp.getDesc().getDefiningOp<triton::MakeTensorDescOp>();
    if (!makeTensorDescOp)
      return rewriter.notifyMatchFailure(
          gatherOp, "descriptor not from MakeTensorDescOp");

    // 5. Emit one descriptor_load per sub-range.
    Location loc = gatherOp.getLoc();
    RankedTensorType sliceTy =
        RankedTensorType::get({rangeCount, rowWidth}, elemTy);
    triton::TensorDescType sliceDescTy =
        triton::TensorDescType::get(rewriter.getContext(), sliceTy);

    SmallVector<Value> loads;
    for (const auto &range : ranges) {
      auto desc = triton::MakeTensorDescOp::create(
          rewriter, loc, sliceDescTy, makeTensorDescOp.getBase(),
          makeTensorDescOp.getShape(), makeTensorDescOp.getStrides(),
          makeTensorDescOp.getPadding());

      Value startOffset = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(range.start));
      SmallVector<Value> indices = {startOffset, gatherOp.getYOffset()};
      auto load = triton::DescriptorLoadOp::create(rewriter, loc, sliceTy, desc,
                                                   indices);
      loads.push_back(load.getResult());
    }

    // 6. Concatenate with tt.cat using a balanced reduction tree.
    //    tt.cat requires SameTypeOperands, so we can only cat tensors of
    //    equal shape. A balanced tree ensures each pair has the same type.
    //    Requires the number of ranges to be a power of 2.
    if (loads.size() & (loads.size() - 1))
      return rewriter.notifyMatchFailure(
          gatherOp, "number of sub-ranges is not a power of 2");

    SmallVector<Value> current = std::move(loads);
    int64_t currentRows = rangeCount;
    while (current.size() > 1) {
      SmallVector<Value> next;
      int64_t nextRows = currentRows * 2;
      auto catTy = RankedTensorType::get({nextRows, rowWidth}, elemTy);
      for (size_t i = 0; i < current.size(); i += 2)
        next.push_back(triton::CatOp::create(rewriter, loc, catTy, current[i],
                                             current[i + 1]));
      current = std::move(next);
      currentRows = nextRows;
    }

    rewriter.replaceOp(gatherOp, current[0]);
    return success();
  }
};

struct RewriteMakeTensorDesc : OpConversionPattern<triton::MakeTensorDescOp> {
  using OpConversionPattern<triton::MakeTensorDescOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Value> ptrShapeStridesPaddingOption;
    llvm::append_values(ptrShapeStridesPaddingOption, adaptor.getBase());
    llvm::append_range(ptrShapeStridesPaddingOption,
                       castToI64(rewriter, adaptor.getShape()));
    llvm::append_range(ptrShapeStridesPaddingOption, adaptor.getStrides());
    auto paddingOption = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(adaptor.getPadding() ==
                             triton::PaddingOption::PAD_NAN));
    llvm::append_values(ptrShapeStridesPaddingOption, paddingOption);
    auto roundF32ToTF32 = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(false));
    llvm::append_values(ptrShapeStridesPaddingOption, roundF32ToTF32);
    rewriter.replaceOpWithMultiple(op, {ptrShapeStridesPaddingOption});
    return mlir::success();
  }
};

struct RewriteLoadPattern : OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto other = generateOther(rewriter, loc, descTy, desc.paddingOption);
    auto newLoad = triton::LoadOp::create(
        rewriter, loc, generatePtr(rewriter, loc, blockShape, desc, offsets),
        generateMask(rewriter, loc, blockShape, desc, offsets), other,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    Value result = newLoad.getResult();
    if (descTy.getBlockType().getElementType().isF32()) {

      auto ifOp = scf::IfOp::create(rewriter, loc, result.getType(),
                                    desc.roundF32ToTF32, /*withElse=*/true);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      auto rounded = roundF32ToTF32(rewriter, loc, result);
      scf::YieldOp::create(rewriter, loc, rounded);

      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      scf::YieldOp::create(rewriter, loc, result);
      result = ifOp.getResult(0);
    }

    rewriter.replaceOp(op, result);
    return llvm::success();
  }
};

struct RewriteStorePattern : OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::pair<Value, Value>
generateGatherScatterPtrMask(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> blockShape, Descriptor &desc,
                             Value xOffsets, Value yOffset) {
  Value xOffsetRange =
      expandOffsets(builder, loc, blockShape, xOffsets, /*dim=*/0);
  yOffset = castToI64(builder, {yOffset})[0];
  auto xOffsetI64Ty = RankedTensorType::get(
      cast<RankedTensorType>(xOffsetRange.getType()).getShape(),
      yOffset.getType());
  xOffsetRange =
      arith::ExtSIOp::create(builder, loc, xOffsetI64Ty, xOffsetRange);
  auto yOffsetRange =
      getExpandedOffsetWithRange(builder, loc, blockShape, yOffset, /*dim=*/1);
  auto ptr = generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                         {xOffsetRange, yOffsetRange});
  auto mask = generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                           {xOffsetRange, yOffsetRange});
  return {ptr, mask};
}

struct RewriteGatherPattern : OpConversionPattern<triton::DescriptorGatherOp> {
  using OpConversionPattern<triton::DescriptorGatherOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getResult().getType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto other = generateOther(rewriter, loc,
                               descTy.getSignlessBlockType().getElementType(),
                               blockShape, desc.paddingOption);
    auto newLoad = triton::LoadOp::create(
        rewriter, loc, ptr, mask, other, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    Value result = newLoad.getResult();
    if (descTy.getSignlessBlockType().getElementType().isF32()) {
      auto rounded = roundF32ToTF32(rewriter, loc, result);
      result = arith::SelectOp::create(rewriter, loc, desc.roundF32ToTF32,
                                       rounded, result);
    }

    rewriter.replaceOp(op, result);
    return llvm::success();
  }
};

struct RewriteScatterPattern
    : OpConversionPattern<triton::DescriptorScatterOp> {
  using OpConversionPattern<triton::DescriptorScatterOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getSrc().getType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, ptr, op.getSrc(), mask, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::optional<RMWOp> translateReduceKind(DescriptorReduceKind kind,
                                         TensorDescType ty) {
  auto scalarTy = ty.getBlockType().getElementType();
  switch (kind) {
  case DescriptorReduceKind::ADD:
    return scalarTy.isInteger() ? RMWOp::ADD : RMWOp::FADD;
  case DescriptorReduceKind::MIN:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMIN;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MIN;
    }
    return {};
  case DescriptorReduceKind::MAX:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMAX;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MAX;
    }
    return {};
  case DescriptorReduceKind::AND:
    return RMWOp::AND;
  case DescriptorReduceKind::OR:
    return RMWOp::OR;
  case DescriptorReduceKind::XOR:
    return RMWOp::XOR;
  default:
    break;
  }
  return {};
}

struct RewriteReducePattern : OpConversionPattern<triton::DescriptorReduceOp> {
  using OpConversionPattern<triton::DescriptorReduceOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorReduceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto rmwOp = translateReduceKind(op.getKind(), descTy);
    if (!rmwOp) {
      std::string msgstring;
      llvm::raw_string_ostream msg(msgstring);
      msg << "Cannot fallback on descriptor atomic op, unsupported for type "
          << descTy.getBlockType().getElementType();
      return op->emitError(msgstring);
    }

    triton::AtomicRMWOp::create(
        rewriter, loc, descTy.getSignlessBlockType(), *rmwOp,
        generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        MemSemantic::RELEASE, MemSyncScope::GPU);
    op.erase();
    return success();
  }
};

/**
 * @brief This implements the pass for converting triton tensor descriptor
 * loads/stores into indexed loads/stores.
 *
 * The key idea is that each tensor descriptor can be broken down into multiple
 * values. Suppose we have a tensor pointer with rank r, we can cast that tensor
 * descriptor value to and from 1+2r values: a tensor pointer value and two i32
 * value for each dimension representing the dynamic shape and strides.
 *
 * As in normal conversion patterns, individual operations can be converted
 * using casted tensor descriptors and offsets and casting the results back to
 * tensor pointers.
 *
 * We have special handling for TMA loads/stores and the make tensor descriptor
 * op.
 *
 * @note Why use the conversion pattern rewriter? In most cases the defining
 * operation of a tensor descriptor will be a make tensor descriptor op.
 * However, this isn't always true - for example, if the tensor descriptor is a
 * function argument or is in a conditional statement, we need better tracking
 * of the pointer, shape, and strides.
 */
class TritonRewriteTensorDescriptorToPointerPass
    : public triton::intel::impl::TritonRewriteTensorDescriptorToPointerBase<
          TritonRewriteTensorDescriptorToPointerPass> {
  void runOnOperation() override {
    Operation *op = getOperation();

    // Pre-pass: Rewrite contiguous DescriptorGatherOps to DescriptorLoadOps.
    // When x_offsets are provably contiguous, replace the gather with a single
    // 2D block load. For constant offsets with multiple contiguous sub-ranges,
    // emit one load per sub-range and concatenate with tt.cat.
    // Enabled by default. Set TRITON_INTEL_DISABLE_REWRITE_CONTIGUOUS_GATHER=1
    // to disable.
    if (!tools::getBoolEnv("TRITON_INTEL_DISABLE_REWRITE_CONTIGUOUS_GATHER")) {
      MLIRContext *ctx = op->getContext();
      RewritePatternSet patterns(ctx);
      patterns.add<RewriteContiguousGather>(ctx, /*benefit=*/2);
      patterns.add<RewriteMultiRangeGather>(ctx, /*benefit=*/1);
      // Failure here means no patterns matched (e.g. no gathers, or only
      // non-contiguous offsets). This is expected — unmatched gathers fall
      // through to the pointer-based conversion below.
      (void)applyPatternsGreedily(op, std::move(patterns));
    }

    llvm::SmallSetVector<triton::MakeTensorDescOp, 4>
        candidateMakeTensorDescOps;
    llvm::SmallSetVector<triton::MakeTensorDescOp, 4>
        unhandledMakeTensorDescOps;
    op->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<triton::DescriptorLoadOp,
                triton::DescriptorStoreOp>([&](auto op) {
            auto makeTensorDescOp =
                triton::intel::findDefiningOpOfType<triton::MakeTensorDescOp>(
                    op.getDesc());
            if (makeTensorDescOp.has_value())
              candidateMakeTensorDescOps.insert(*makeTensorDescOp);
          })
          .Case<triton::DescriptorGatherOp, triton::DescriptorScatterOp,
                triton::DescriptorReduceOp>([&](auto op) {
            auto makeTensorDescOp =
                triton::intel::findDefiningOpOfType<triton::MakeTensorDescOp>(
                    op.getDesc());
            if (makeTensorDescOp.has_value())
              unhandledMakeTensorDescOps.insert(*makeTensorDescOp);
          })
          .Default([](auto) {});
      return WalkResult::advance();
    });
    for (auto op : unhandledMakeTensorDescOps)
      candidateMakeTensorDescOps.remove(op);

    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<
        mlir::arith::ArithDialect, mlir::scf::SCFDialect,
        mlir::triton::TritonDialect>([&](mlir::Operation *op) {
      if (!hasATensorDescriptorType(op->getOperandTypes()) &&
          !hasATensorDescriptorType(op->getResultTypes()))
        return true;

      return TypeSwitch<Operation *, bool>(op)
          .Case<triton::MakeTensorDescOp>(
              [&](auto op) { return candidateMakeTensorDescOps.contains(op); })
          .Case<triton::DescriptorLoadOp,
                triton::DescriptorStoreOp>([&](auto op) {
            auto makeTensorDescOp =
                triton::intel::findDefiningOpOfType<triton::MakeTensorDescOp>(
                    op.getDesc());
            return makeTensorDescOp.has_value() &&
                   candidateMakeTensorDescOps.contains(*makeTensorDescOp);
          })
          .Default([](auto) { return false; });
    });
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    mlir::TypeConverter converter;

    converter.addConversion([](mlir::Type t) {
      // Most types don't require any conversion
      return t;
    });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into an pointer, and a shape and stride
      // for each dimension, and padding option. i.e., we create 1+2*rank+1
      // values. Note that tensor descriptors may be signed/unsigned integers
      // whereas pointers should always be signless.
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType.getElementType()));
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 64));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });

    FuncArgRenamer renamer(".");
    renamer.addRenamer([](mlir::triton::TensorDescType type,
                          llvm::SmallVectorImpl<std::string> &out_suffix) {
      auto tensorType = type.getSignlessBlockType();
      int dims = tensorType.getRank();
      out_suffix.push_back("");
      for (int i = 0; i < dims; i++) {
        out_suffix.push_back("shape." + std::to_string(i));
      }
      for (int i = 0; i < dims; i++) {
        out_suffix.push_back("stride." + std::to_string(i));
      }
      out_suffix.push_back("padding");
      out_suffix.push_back("roundF32ToTF32");
      return success();
    });

    mlir::RewritePatternSet patterns(op->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, renamer, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    patterns
        .add<RewriteMakeTensorDesc, RewriteLoadPattern, RewriteStorePattern,
             RewriteGatherPattern, RewriteScatterPattern, RewriteReducePattern>(
            converter, &getContext());

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            op, target, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::triton::intel
