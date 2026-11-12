#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-lower-to-2d-block-load"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
using LinearLayout = mlir::triton::LinearLayout;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPULOWERTO2DBLOCKLOAD
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

/// Check whether a load is eligible for 2D block IO lowering.
static bool isBlockIOEligible(tt::LoadOp op) {
  auto tensorTy =
      dyn_cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
  if (!tensorTy)
    return false;

  return ttgi::isBlockIOEligible(op, tensorTy);
}

/// Get the stride (in elements) for a given dimension from stride analysis.
/// Returns -1 if unknown, 0 if zero stride.
static int64_t getStride(tt::intel::ModuleStrideAnalysis &strideAnalysis,
                         Value ptr, unsigned dim) {
  tt::intel::StrideInfo *info = strideAnalysis.getStrideInfo(ptr);
  if (info) {
    const auto &stride = info->getStride();
    if (dim < stride.size())
      return stride[dim];
  }
  return -1;
}

/// Zero splat for tensorTy, used as the default `other` for a masked load
/// when the user did not pass one.
static Value createZeroSplat(OpBuilder &builder, Location loc,
                             RankedTensorType tensorTy) {
  Type elemType = tensorTy.getElementType();
  // Packed fp4 has no scalar constants and e8m0 cannot represent 0: splat
  // zero bits as i8 and bitcast, like upstream's descriptor zero padding.
  if (isa<tt::PackedFloat4E2M1Type, Float8E8M0FNUType>(elemType)) {
    auto bitsTy = tensorTy.clone(builder.getI8Type());
    Value zeroBits = createZeroSplat(builder, loc, bitsTy);
    return tt::BitcastOp::create(builder, loc, tensorTy, zeroBits);
  }
  Attribute zeroAttr;
  if (isa<FloatType>(elemType))
    zeroAttr = builder.getFloatAttr(elemType, 0.0);
  else if (isa<IntegerType>(elemType))
    zeroAttr = builder.getIntegerAttr(elemType, 0);
  else
    llvm_unreachable("unsupported element type for zero splat");
  Value zeroVal =
      arith::ConstantOp::create(builder, loc, cast<TypedAttr>(zeroAttr));
  return tt::SplatOp::create(builder, loc, tensorTy, zeroVal);
}

struct TritonIntelGPULowerTo2DBlockLoadPass
    : public ttgi::impl::TritonIntelGPULowerTo2DBlockLoadBase<
          TritonIntelGPULowerTo2DBlockLoadPass> {
public:
  using ttgi::impl::TritonIntelGPULowerTo2DBlockLoadBase<
      TritonIntelGPULowerTo2DBlockLoadPass>::
      TritonIntelGPULowerTo2DBlockLoadBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Without 2D block IO every tt.descriptor_load still becomes a tt.load.
    bool has2DBlockIO = mod->hasAttr(
        ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName());

    // FIXME: Remove once IGC can split large 2D block loads.
    // Read the env var once and materialize it as an attribute on the ops
    // so downstream passes only need to check the attribute.
    std::optional<bool> envOneMatrixPerLoad = tt::tools::isEnvValueBool(
        tt::tools::getStrEnv("TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT"));
    if (has2DBlockIO && envOneMatrixPerLoad.has_value()) {
      StringRef attrName =
          ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName();
      mod.walk([&](Operation *op) {
        if (!isa<tt::LoadOp, tt::DescriptorLoadOp>(op))
          return;
        if (*envOneMatrixPerLoad)
          op->setAttr(attrName, UnitAttr::get(mod.getContext()));
        else
          op->removeAttr(attrName);
      });
    }

    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    tt::intel::ModuleStrideAnalysis strideAnalysis(mod, axisInfoAnalysis);

    DominanceInfo domInfo(mod);
    SmallVector<tt::DescriptorLoadOp> descLoadOps;
    mod.walk([&](tt::DescriptorLoadOp op) { descLoadOps.push_back(op); });
    for (auto op : descLoadOps) {
      if (has2DBlockIO && convertDescriptorLoadOp(op, axisInfoAnalysis))
        continue;
      if (failed(lowerDescriptorLoadToLoad(op, domInfo)))
        signalPassFailure();
    }
    if (!has2DBlockIO)
      return;

    SmallVector<tt::LoadOp> loadOps;
    mod.walk([&](tt::LoadOp op) { loadOps.push_back(op); });
    for (auto op : loadOps)
      convertLoadOp(op, strideAnalysis, axisInfoAnalysis);
  }

private:
  /// Convert a tt.descriptor_load to ttig.2d_block_load. Returns false if the
  /// load needs another lowering.
  bool
  convertDescriptorLoadOp(tt::DescriptorLoadOp op,
                          tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    auto tensorTy = cast<RankedTensorType>(op.getType());
    if (!ttgi::isDescriptorLoadLowerableTo2DBlock(op, tensorTy,
                                                  axisInfoAnalysis))
      return false;

    unsigned rank = tensorTy.getRank();
    unsigned elemSizeInBits =
        getIntOrFloatOrPtrBitWidth(tensorTy.getElementType());

    // Find all MakeTensorDescOps that could define this descriptor.
    Value desc = op.getDesc();
    tt::intel::DescriptorDefinitions defs =
        tt::intel::findDescriptorDefinitions(desc);
    std::optional<tt::PaddingOption> padding = defs.consistentPadding();
    assert(padding && "isDescriptorLoadLowerableTo2DBlock checks padding");

    auto descType = cast<tt::TensorDescType>(desc.getType());
    unsigned descRank = descType.getBlockType().getRank();
    assert(descRank >= rank && "descriptor rank must be >= result tensor rank");

    bool memoryRowMajor = ttgi::isMemoryRowMajor(op);
    bool oneMatrixPerLoadForBT =
        op->hasAttr(ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName());

    // A misaligned pitch is an error whatever the layout, so this check is not
    // part of isDescriptorLoadLowerableTo2DBlock.
    int64_t elemBytesConst = elemSizeInBits / 8;
    for (tt::MakeTensorDescOp d : defs) {
      std::optional<int64_t> stride = tt::intel::getFoldedConstantValue(
          d->getOperand(/*pitch stride*/ 1 + descRank + (descRank - 2)));
      if (stride && (*stride * elemBytesConst) % 16 != 0) {
        op->emitOpError("descriptor pitch of ")
            << *stride * elemBytesConst
            << " bytes is not a multiple of 16 bytes";
        signalPassFailure();
        return true;
      }
    }

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Type i32Ty = builder.getI32Type();

    auto memLayout = memoryRowMajor ? ttgi::BlockIOMode::RowMajor
                                    : ttgi::BlockIOMode::ColumnMajor;

    // Extract all surface parameters from the runtime descriptor value.
    // This correctly handles loop-carried descriptors where fields change
    // per iteration. Struct layout: { shapes[rank], strides[rank], base_ptr }.
    Type i64Ty = builder.getI64Type();
    Type ptrType =
        tt::PointerType::get(descType.getBlockType().getElementType());
    SmallVector<Value> shapes(descRank);
    SmallVector<Value> strides(descRank);
    for (unsigned d = 0; d < descRank; ++d) {
      shapes[d] = ttgi::ExtractDescOp::create(builder, loc, i64Ty, desc,
                                              builder.getI32IntegerAttr(d));
      strides[d] = ttgi::ExtractDescOp::create(
          builder, loc, i64Ty, desc, builder.getI32IntegerAttr(descRank + d));
    }
    Value basePtr = ttgi::ExtractDescOp::create(
        builder, loc, ptrType, desc, builder.getI32IntegerAttr(2 * descRank));
    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    assert(indices.size() == descRank &&
           "descriptor index count must match descriptor rank");

    // Fold batch indices into base_ptr. This handles both rank-reducing loads
    // (descRank > rank: leading descriptor dims are dropped) and same-rank
    // loads with rank > 2 (batch dims before the inner-2 dims).
    unsigned numBatchDims = descRank - 2;
    for (unsigned d = 0; d < numBatchDims; ++d) {
      Value batchOffset = arith::MulIOp::create(
          builder, loc,
          arith::ExtSIOp::create(builder, loc, builder.getI64Type(),
                                 indices[d]),
          strides[d]);
      basePtr = tt::AddPtrOp::create(builder, loc, basePtr.getType(), basePtr,
                                     batchOffset);
    }

    // Helper to truncate to i32 only if needed.
    auto toI32 = [&](Value v) -> Value {
      if (v.getType().getIntOrFloatBitWidth() > 32)
        return arith::TruncIOp::create(builder, loc, i32Ty, v);
      return v;
    };

    // Surface width = inner dimension size * element bytes.
    // Surface height = second-to-last dimension size.
    // Pitch = stride of the second-to-last dimension * element bytes.
    // (In a tensor descriptor, the last stride is always 1.)
    Value elemBytes =
        arith::ConstantIntOp::create(builder, loc, elemSizeInBits / 8, 32);
    Value baseWidth = arith::MulIOp::create(
        builder, loc, toI32(shapes[descRank - 1]), elemBytes);
    Value baseHeight = toI32(shapes[descRank - 2]);
    Value basePitch = arith::MulIOp::create(
        builder, loc, toI32(strides[descRank - 2]), elemBytes);

    // Inner-2 indices: X = inner dim (descRank-1), Y = second-to-last.
    Value offsetX = indices[descRank - 1];
    Value offsetY = indices[descRank - 2];

    // The batch *indices* were folded into base_ptr above, but the result
    // layout also walks the batch dimensions during LLVM lowering and needs
    // their real strides, which are not derivable from the 2D surface params.
    // Result dim d is descriptor dim d + (descRank - rank).
    SmallVector<Value> batchStrides;
    for (unsigned d = 0; d + 2 < rank; ++d)
      batchStrides.push_back(strides[d + (descRank - rank)]);

    // Batch indices folded into base_ptr re-base the 2D surface, so they
    // escape the hardware's clamp; the lowering needs each index and its
    // declared extent to predicate the load. For that check only -- never for
    // addressing, since base_ptr already carries the offsets.
    SmallVector<Value> batchOffsets, batchShapes;
    for (unsigned d = 0; d < numBatchDims; ++d) {
      batchOffsets.push_back(indices[d]);
      batchShapes.push_back(toI32(shapes[d]));
    }

    // Determine padding mode from the descriptor.
    bool padNan = *padding == tt::PaddingOption::PAD_NAN;
    UnitAttr padNanAttr = padNan ? builder.getUnitAttr() : UnitAttr();

    auto blockLoadOp = ttgi::Subgroup2DBlockLoadOp::create(
        builder, loc, op.getType(), basePtr, baseWidth, baseHeight, basePitch,
        offsetX, offsetY, batchStrides, batchOffsets, batchShapes, padNanAttr,
        ttgi::BlockIOModeAttr::get(builder.getContext(), memLayout));

    // Propagate one_matrix_per_load attribute if present.
    if (oneMatrixPerLoadForBT)
      blockLoadOp->setAttr(
          ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName(),
          builder.getUnitAttr());

    op.replaceAllUsesWith(blockLoadOp.getResult());
    op.erase();
    LDBG("Converted descriptor load to ttig.2d_block_load: " << *blockLoadOp);
    return true;
  }

  /// Lower a tt.descriptor_load that cannot use a 2D block load to a masked
  /// tt.load of a tensor of pointers, so it takes the regular LoadOp lowering.
  LogicalResult lowerDescriptorLoadToLoad(tt::DescriptorLoadOp op,
                                          DominanceInfo &domInfo) {
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto encoding = dyn_cast_if_present<ttg::DistributedEncodingTrait>(
        resultTy.getEncoding());
    if (!encoding)
      return op.emitOpError("requires a distributed result encoding");

    Value desc = op.getDesc();
    auto descType = cast<tt::TensorDescType>(desc.getType());
    unsigned descRank = descType.getBlockType().getRank();
    unsigned rank = resultTy.getRank();
    // Column-major loads produce the inner two descriptor dims transposed.
    auto blockIOAttr = op->getAttrOfType<StringAttr>(
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
    bool columnMajor =
        blockIOAttr && ttgi::symbolizeBlockIOMode(blockIOAttr.getValue()) ==
                           ttgi::BlockIOMode::ColumnMajor;
    // Malformed (hand-written) IR; left for the LLVM conversion to reject.
    if (op.getIndices().size() != descRank) {
      LDBG("Index count does not match descriptor rank: " << *op);
      return success();
    }
    if (columnMajor && rank < 2)
      return op.emitOpError("column_major load requires a result of rank >= 2");
    if (failed(verifyInnerShape(op, descType.getBlockType().getShape(),
                                resultTy.getShape(), columnMajor)))
      return failure();
    unsigned numDroppedDims = descRank - rank;

    tt::intel::DescriptorDefinitions defs =
        tt::intel::findDescriptorDefinitions(desc);
    FailureOr<tt::PaddingOption> padding = resolvePadding(op, defs);
    if (failed(padding))
      return failure();

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Type i64Ty = builder.getI64Type();
    Type i1Ty = builder.getI1Type();

    // Use an operand shared by every MakeTensorDescOp, so AxisInfo sees it, if
    // it dominates where `desc` is defined (not a later loop iteration's
    // value); otherwise read the field from the descriptor.
    Operation *descDefPoint = desc.getDefiningOp()
                                  ? desc.getDefiningOp()
                                  : desc.getParentBlock()->getParentOp();
    // `fieldIdx` indexes the descriptor struct {shapes, strides, base}.
    auto getField = [&](unsigned operandIdx, unsigned fieldIdx,
                        Type extractTy) -> Value {
      if (!defs.empty()) {
        Value common;
        for (tt::MakeTensorDescOp d : defs) {
          Value v = d->getOperand(operandIdx);
          common = (!common || common == v) ? v : Value();
          if (!common)
            break;
        }
        if (common && domInfo.properlyDominates(common, descDefPoint))
          return common;
      }
      return ttgi::ExtractDescOp::create(builder, loc, extractTy, desc,
                                         builder.getI32IntegerAttr(fieldIdx));
    };
    auto toI64 = [&](Value v) -> Value {
      return v.getType() == i64Ty
                 ? v
                 : arith::ExtSIOp::create(builder, loc, i64Ty, v);
    };

    Type elemTy = resultTy.getElementType();
    Value base = getField(0, 2 * descRank, tt::PointerType::get(elemTy));
    SmallVector<Value> shapes, strides, indices;
    for (unsigned d = 0; d < descRank; ++d) {
      shapes.push_back(toI64(getField(1 + d, d, i64Ty)));
      strides.push_back(getField(1 + descRank + d, descRank + d, i64Ty));
      indices.push_back(toI64(op.getIndices()[d]));
    }

    SmallVector<unsigned> descDims =
        llvm::to_vector(llvm::seq<unsigned>(0, descRank));
    if (columnMajor)
      std::swap(descDims[descRank - 2], descDims[descRank - 1]);

    auto inBounds = [&](Value index, Value shape) -> Value {
      Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
      Value lower = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sge, index, zero);
      Value upper = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::slt, index, shape);
      return arith::AndIOp::create(builder, loc, lower, upper);
    };
    auto andMask = [&](Value lhs, Value rhs) -> Value {
      return lhs ? arith::AndIOp::create(builder, loc, lhs, rhs) : rhs;
    };

    // Fold every index into the scalar base so per-element offsets are only the
    // in-tile ranges, whose constant parts LLVM can then split off. Dropped
    // (unit) dims of a rank-reducing load predicate the whole tile.
    Value scalarPred;
    for (unsigned d = 0; d < descRank; ++d) {
      Value offset =
          arith::MulIOp::create(builder, loc, indices[d], strides[d]);
      base = tt::AddPtrOp::create(builder, loc, base.getType(), base, offset);
      if (d < numDroppedDims)
        scalarPred = andMask(scalarPred, inBounds(indices[d], shapes[d]));
    }

    // Bools are bytes in memory, as for tl.load.
    Type memElemTy = elemTy.isInteger(1) ? builder.getI8Type() : elemTy;
    Type ptrTy = tt::PointerType::get(memElemTy);
    if (memElemTy != elemTy)
      base = tt::BitcastOp::create(builder, loc, ptrTy, base);

    ArrayRef<int64_t> shape = resultTy.getShape();
    auto ptrTensorTy = RankedTensorType::get(shape, ptrTy, encoding);
    auto maskTy = RankedTensorType::get(shape, i1Ty, encoding);
    Value ptr = tt::SplatOp::create(builder, loc, ptrTensorTy, base);
    Value mask;
    for (unsigned dim = 0; dim < rank; ++dim) {
      unsigned descDim = descDims[dim + numDroppedDims];
      Value index = indices[descDim];
      Value dimShape = shapes[descDim];

      // 1D range `[0, shape[dim])`, sliced so expanding it back to full rank
      // yields `encoding`.
      Attribute enc1D = encoding;
      for (int j = rank - 1; j >= 0; --j)
        if (static_cast<unsigned>(j) != dim)
          enc1D = ttg::SliceEncodingAttr::get(
              builder.getContext(), j,
              cast<ttg::DistributedEncodingTrait>(enc1D));
      auto rangeTy =
          RankedTensorType::get({shape[dim]}, builder.getI32Type(), enc1D);
      auto offsetTy = RankedTensorType::get({shape[dim]}, i64Ty, enc1D);
      Value range = arith::ExtSIOp::create(
          builder, loc, offsetTy,
          tt::MakeRangeOp::create(builder, loc, rangeTy, 0, shape[dim]));
      for (unsigned j = 0; j < rank; ++j)
        if (j != dim)
          range = tt::ExpandDimsOp::create(builder, loc, range, j);
      auto expandedTy = cast<RankedTensorType>(range.getType());

      // Splat the stride into the expanded shape, not the 1D row, to keep its
      // divisibility.
      Value scaled = arith::MulIOp::create(
          builder, loc, range,
          tt::SplatOp::create(builder, loc, expandedTy, strides[descDim]));
      ptr = tt::AddPtrOp::create(
          builder, loc, ptrTensorTy, ptr,
          tt::BroadcastOp::create(builder, loc, ptrTensorTy.clone(i64Ty),
                                  scaled));

      // A tile whose shape and index are block multiples is entirely in or out
      // of bounds, so one check is exact and keeps the mask uniform.
      int64_t blockSize = shape[dim];
      if (defs.allSatisfy([&](tt::MakeTensorDescOp d) {
            return ttgi::isDivisible(d.getShape()[descDim], blockSize);
          }) &&
          ttgi::isDivisible(op.getIndices()[descDim], blockSize)) {
        scalarPred = andMask(scalarPred, inBounds(index, dimShape));
        continue;
      }

      Value offsets = arith::AddIOp::create(
          builder, loc, tt::SplatOp::create(builder, loc, expandedTy, index),
          range);
      Value zero = tt::SplatOp::create(
          builder, loc, expandedTy,
          arith::ConstantIntOp::create(builder, loc, 0, 64));
      Value lower = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::sge, offsets, zero);
      Value upper = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::slt, offsets,
          tt::SplatOp::create(builder, loc, expandedTy, dimShape));
      Value dimMask = arith::AndIOp::create(builder, loc, lower, upper);
      mask =
          andMask(mask, tt::BroadcastOp::create(builder, loc, maskTy, dimMask));
    }
    if (scalarPred)
      mask =
          andMask(mask, tt::SplatOp::create(builder, loc, maskTy, scalarPred));

    // Like the other descriptor paths, types without a NaN (e2m1) and e8m0 pad
    // with zero bits.
    auto memTy = resultTy.clone(memElemTy);
    auto floatTy = dyn_cast<FloatType>(memElemTy);
    Value other;
    if (*padding == tt::PaddingOption::PAD_NAN && floatTy &&
        !isa<Float8E8M0FNUType>(floatTy) &&
        APFloat::semanticsHasNaN(floatTy.getFloatSemantics())) {
      other = arith::ConstantOp::create(
          builder, loc,
          SplatElementsAttr::get(
              memTy,
              builder.getFloatAttr(
                  floatTy, APFloat::getNaN(floatTy.getFloatSemantics()))));
    } else {
      other = createZeroSplat(builder, loc, memTy);
    }

    auto load = tt::LoadOp::create(builder, loc, ptr, mask, other,
                                   op.getCachePolicyAttr());
    Value result = load.getResult();
    if (memElemTy != elemTy)
      result =
          arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, result,
                                createZeroSplat(builder, loc, memTy));

    op.replaceAllUsesWith(result);
    op.erase();
    LDBG("Lowered descriptor load to tt.load: " << *load);
    return success();
  }

  /// A rank-reducing load may only drop leading unit dims, and the remaining
  /// dims (inner two swapped for column_major) must match the result shape.
  static LogicalResult verifyInnerShape(Operation *op,
                                        ArrayRef<int64_t> descShape,
                                        ArrayRef<int64_t> shape,
                                        bool columnMajor) {
    size_t descRank = descShape.size(), rank = shape.size();
    if (descRank < rank)
      return op->emitOpError("descriptor rank must be >= result rank");
    size_t rankDelta = descRank - rank;
    for (size_t i = 0; i < rankDelta; ++i)
      if (descShape[i] != 1)
        return op->emitError()
               << "rank-reducing descriptor op only supports dropping leading "
                  "dimensions of size 1, but descriptor dim["
               << i << "] = " << descShape[i];
    for (size_t i = 0; i < rank; ++i) {
      size_t descDim = rankDelta + i;
      if (columnMajor && i >= rank - 2)
        descDim = (i == rank - 2) ? descRank - 1 : descRank - 2;
      if (descShape[descDim] != shape[i])
        return op->emitError()
               << "rank-reducing descriptor op requires that the result/source "
                  "tensor shape matches the inner dimensions of the descriptor "
                  "block type, but descriptor inner dim["
               << i << "] = " << descShape[descDim] << " != tensor dim[" << i
               << "] = " << shape[i];
    }
    return success();
  }

  /// The out-of-bounds fill: the defining ops must agree, and so must a stamped
  /// `ttig.desc_padding`. With no traceable definition, the attribute or
  /// PAD_ZERO, the default of `tt.make_tensor_descriptor`.
  static FailureOr<tt::PaddingOption>
  resolvePadding(tt::DescriptorLoadOp op,
                 const tt::intel::DescriptorDefinitions &defs) {
    std::optional<tt::PaddingOption> traced = defs.consistentPadding();
    auto attr = op->getAttrOfType<tt::PaddingOptionAttr>(
        ttgi::TritonIntelGPUDialect::getDescPaddingAttrName());
    if (!defs.empty() && !traced)
      return op.emitError(
          "descriptor padding is divergent: the operations defining this "
          "descriptor disagree, so there is no single out-of-bounds fill "
          "value; such a descriptor must be expanded to pointers by the "
          "'triton-intel-rewrite-tensor-descriptor-to-pointer' pass, which "
          "encodes the padding mode as a runtime value");
    if (attr && traced && attr.getValue() != *traced)
      return op.emitError("'ttig.desc_padding' disagrees with the padding of "
                          "the operations defining this descriptor");
    if (!attr && traced)
      return op.emitError(
          "'ttig.desc_padding' is missing but the operations defining this "
          "descriptor agree on a padding mode: the attribute is stamped by "
          "the 'tritonintelgpu-materialize-block-pointer' pass whenever they "
          "do, so add it to this operation (or run that pass) instead of "
          "relying on the lowering to re-derive it");
    return attr ? attr.getValue() : tt::PaddingOption::PAD_ZERO;
  }

  /// Convert a tt.load to ttig.2d_block_load_from_ptr.
  void convertLoadOp(tt::LoadOp op,
                     tt::intel::ModuleStrideAnalysis &strideAnalysis,
                     tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    if (!isBlockIOEligible(op))
      return;

    auto tensorTy =
        cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
    unsigned rank = tensorTy.getRank();
    unsigned elemSizeInBits =
        getIntOrFloatOrPtrBitWidth(tensorTy.getElementType());

    bool memoryRowMajor = ttgi::isMemoryRowMajor(op);
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;

    bool oneMatrixPerLoadForBT =
        op->hasAttr(ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName());

    // A row-limit mask can become `base_height` instead of per-element
    // masking (hardware zero-fill). Decided up front so tile-size
    // computation below can treat this load as unmasked.
    std::optional<ttgi::BoundaryMaskInfo> rowLimitMask;
    if (op.getMask() && rank >= 2) {
      unsigned pitchTensorDim =
          (contiguousDim == rank - 1) ? rank - 2 : rank - 1;
      bool pitchDimIsBroadcast =
          getStride(strideAnalysis, op.getPtr(), pitchTensorDim) == 0;
      if (auto m = ttgi::matchBoundaryCheckMask(
              op.getMask(), op.getOther(), tensorTy.getShape(), contiguousDim,
              pitchDimIsBroadcast))
        rowLimitMask = m;
    }

    // Retrieve mask axis info to validate tile constraints consistently
    // with the downstream LLVM lowering. Skipped for a row-limit mask: its
    // bound is handled via base_height below, not by clamping the tile.
    tt::AxisInfo *maskAxisInfo = nullptr;
    if (op.getMask() && !rowLimitMask)
      maskAxisInfo = axisInfoAnalysis.getAxisInfo(op.getMask());

    // Whether this load was annotated by the 1D→2D reshape in
    // MaterializeBlockPointer. Used for pitch and base-height computation
    // below.
    bool has1DReshapeStride =
        op->hasAttr(ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName());

    // Validate tile computation and get the actual row/col dims from the
    // encoding. These may differ from the conventional rank-2/rank-1 for
    // rank > 2 tensors.
    unsigned rowDim, colDim;
    int tileWidth = -1;
    int tileHeight = -1;
    int numPackedVals = -1;
    bool isTranspose = false;

    Attribute encoding = tensorTy.getEncoding();
    LinearLayout llEncoding =
        cast<ttg::DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorTy.getShape());
    if (!ttgi::validate2DBlockLoadTile(llEncoding, contiguousDim,
                                       elemSizeInBits, tensorTy,
                                       oneMatrixPerLoadForBT, maskAxisInfo)) {
      LDBG("Tile validation failed for load: " << *op);
      return;
    }
    auto sizeInfo =
        ttgi::getBlockIOLoadTileSize(llEncoding, contiguousDim, elemSizeInBits,
                                     maskAxisInfo, oneMatrixPerLoadForBT);
    rowDim = sizeInfo.rowDim;
    colDim = sizeInfo.colDim;
    tileWidth = sizeInfo.tileWidth;
    tileHeight = sizeInfo.tileHeight;
    isTranspose = sizeInfo.transpose;
    numPackedVals = sizeInfo.numElemPerPackedVal;

    // For the 2D block load surface, the pitch dimension is always the
    // non-contiguous memory direction. For transposed loads, rowDim is the
    // memory-contiguous dim and colDim is non-contiguous, so pitch uses colDim.
    unsigned pitchDim = isTranspose ? colDim : rowDim;

    // matchBoundaryCheckMask assumed the pitch dim is the conventional rank-2/
    // rank-1 dim, but for rank > 2 the tile's actual pitch dim (above) can
    // land on a batch dim instead (e.g. a layout with no register packing).
    // base_height only predicates `pitchDim`, so a mismatch here would
    // silently apply the bound to the wrong dimension; fall back to the
    // per-element masked path, redoing tile selection with the mask's real
    // axis info now that it is no longer absorbed into base_height.
    if (rowLimitMask && rowLimitMask->boundedDim != pitchDim) {
      LDBG("Row-limit mask bounds dim " << rowLimitMask->boundedDim
                                        << " but tile pitch dim is " << pitchDim
                                        << "; falling back to "
                                           "masked load: "
                                        << *op);
      rowLimitMask = std::nullopt;
      maskAxisInfo = axisInfoAnalysis.getAxisInfo(op.getMask());
      if (!ttgi::validate2DBlockLoadTile(llEncoding, contiguousDim,
                                         elemSizeInBits, tensorTy,
                                         oneMatrixPerLoadForBT, maskAxisInfo)) {
        LDBG("Tile validation failed for load (row-limit fallback): " << *op);
        return;
      }
      sizeInfo = ttgi::getBlockIOLoadTileSize(llEncoding, contiguousDim,
                                              elemSizeInBits, maskAxisInfo,
                                              oneMatrixPerLoadForBT);
      rowDim = sizeInfo.rowDim;
      colDim = sizeInfo.colDim;
      tileWidth = sizeInfo.tileWidth;
      tileHeight = sizeInfo.tileHeight;
      isTranspose = sizeInfo.transpose;
      numPackedVals = sizeInfo.numElemPerPackedVal;
      pitchDim = isTranspose ? colDim : rowDim;
    }

    // Check the final tile, after any row-limit fallback.
    tt::AxisInfo *ptrInfo = axisInfoAnalysis.getAxisInfo(op.getPtr());
    if (elemSizeInBits * numPackedVals > 32 &&
        (!ptrInfo || ptrInfo->getDivisibility(contiguousDim) % 8))
      return;

    // Compute pitch from stride analysis or the 1D->2D reshape attribute.
    // For the HW surface: width is along memory-contiguous direction, height
    // is along non-contiguous. For transposed loads, rowDim is the contiguous
    // dim and colDim is non-contiguous, so we swap which dim provides
    // width/height.
    unsigned surfaceWidthDim = isTranspose ? rowDim : colDim;
    unsigned surfaceHeightDim = isTranspose ? colDim : rowDim;
    // Dummy pitch for broadcast loads below; see isBlockPitchSupported for
    // the validity check.
    constexpr int64_t MIN_PITCH = 64;

    // Pitch is either a compile-time constant or a runtime SSA value,
    // never both.
    int64_t pitch = -1;
    Value pitchValue;
    bool isBroadcast = false;
    if (has1DReshapeStride) {
      auto strideAttr = op->getAttrOfType<IntegerAttr>(
          ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName());
      int64_t stride = strideAttr.getInt();
      isBroadcast = (stride == 0);
      pitch = stride * elemSizeInBits / 8;
    } else {
      // stride=0 along rowDim means every row is the same: treat it as a
      // broadcast and load height=1 with a dummy pitch.
      int64_t rowStride = getStride(strideAnalysis, op.getPtr(), rowDim);
      isBroadcast = (rowStride == 0);
    }

    int64_t perWarpWidth;
    if (has1DReshapeStride)
      perWarpWidth = tensorTy.getDimSize(surfaceWidthDim);
    else
      perWarpWidth = tileWidth * numPackedVals;
    int64_t baseWidthBytes = perWarpWidth * elemSizeInBits / 8;

    if (!has1DReshapeStride) {
      if (isBroadcast) {
        // For broadcast (height=1) loads, pitch is a dummy value (no row
        // advancement), but the HW still enforces pitch >= base_width.
        // Downstream lowering widens base_width by up to 63 bytes for 64-byte
        // pointer alignment compensation. Account for that here so the
        // constraint is never violated at runtime.
        int64_t fullRowBytes =
            tensorTy.getDimSize(surfaceWidthDim) * elemSizeInBits / 8;
        constexpr int64_t MAX_ALIGN_OVERHEAD = 63;
        int64_t maxAdjustedWidth = fullRowBytes + MAX_ALIGN_OVERHEAD;
        // Pitch must be a multiple of 16 bytes.
        pitch =
            llvm::alignTo(std::max(MIN_PITCH, maxAdjustedWidth), int64_t(16));
      } else {
        int64_t pitchStride = getStride(strideAnalysis, op.getPtr(), pitchDim);
        if (pitchStride < 0) {
          // No constant stride: use the runtime stride that StrideAnalysis
          // recovered. MaterializeBlockPointer already checked its 16-byte
          // alignment.
          Value lda = ttgi::getRuntimeStrideValue(strideAnalysis, op.getPtr(),
                                                  pitchDim);
          if (!lda) {
            LDBG("No runtime stride source for load: " << *op);
            return;
          }
          OpBuilder bld(op);
          pitchValue = ttgi::materializePitchBytes(bld, op.getLoc(), lda,
                                                   elemSizeInBits / 8);
          if (!pitchValue) {
            LDBG("Unsupported lda type: " << lda.getType());
            return;
          }
        } else {
          pitch = pitchStride * elemSizeInBits / 8;
        }
      }
    }

    // A runtime pitch can't be range-checked here; MaterializeBlockPointer
    // proved it 16-byte aligned and the row-width check covers the minimum.
    if (!ttgi::isBlockPitchSupported(
            pitchValue ? std::nullopt : std::optional<int64_t>(pitch),
            /*hasRuntimeStride=*/pitchValue != nullptr,
            tensorTy.getDimSize(surfaceWidthDim) * elemSizeInBits / 8)) {
      LDBG("Unsupported pitch for load: " << *op);
      return;
    }

    // For broadcast loads, the LLVM lowering's row replication
    // requires tileWidth >= threadsPerWarp or tileWidth * 2 == threadsPerWarp.
    // Reject unsupported configurations.
    if (isBroadcast && tileHeight > 1 && tileWidth > 0) {
      unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
          op->getParentOfType<ModuleOp>());
      if (tileWidth < (int)threadsPerWarp &&
          (unsigned)tileWidth * 2 != threadsPerWarp) {
        LDBG("Broadcast load tile width " << tileWidth
                                          << " incompatible with "
                                             "threadsPerWarp "
                                          << threadsPerWarp << " for: " << *op);
        return;
      }
    }

    OpBuilder builder(op);
    Location loc = op.getLoc();

    // Compute constant surface parameters.
    // For broadcast loads, height is always 1.
    // For non-broadcast, use the per-warp height from getBlockIOTileSize.
    int64_t baseHeightRows =
        isBroadcast
            ? 1
            : (has1DReshapeStride ? tensorTy.getDimSize(surfaceHeightDim)
                                  : tileHeight);

    auto memLayout = memoryRowMajor ? ttgi::BlockIOMode::RowMajor
                                    : ttgi::BlockIOMode::ColumnMajor;

    // Pitch is a single i32 operand: the runtime value when we have one, an
    // arith.constant otherwise (the lowering and asserts treat both the
    // same).
    Value pitchOperand =
        pitchValue
            ? pitchValue
            : arith::ConstantOp::create(
                  builder, loc,
                  builder.getI32IntegerAttr(static_cast<int32_t>(pitch)));
    Value mask = op.getMask();
    Value other = op.getOther();
    Value heightOperand;
    if (rowLimitMask) {
      // Drop the row-limit component: base_height replaces it. Keep the
      // uniform conjuncts, ANDed, as the mask operand for the LLVM lowering to
      // AND into its per-message select.
      auto maskTy = cast<RankedTensorType>(op.getMask().getType());
      mask = Value();
      for (Value pred : rowLimitMask->extraPreds) {
        pred = ttgi::materializeExtraPred(builder, loc, pred, maskTy);
        mask = mask ? arith::AndIOp::create(builder, loc, mask, pred) : pred;
      }
      other = mask ? createZeroSplat(builder, loc, tensorTy) : Value();
      // Relative to the tile's first row and may be <= 0; the LLVM lowering
      // adjusts per message and never forwards a non-positive base_height.
      heightOperand = ttgi::computeRowLimitHeight(
          builder, loc, rowLimitMask->rowBound, rowLimitMask->rowBase,
          tensorTy.getShape()[rowLimitMask->boundedDim]);
    } else {
      // If mask is present but other is absent, create a zero splat as the
      // default padding value (the verifier requires 'other' when 'mask' is
      // set).
      if (mask && !other)
        other = createZeroSplat(builder, loc, tensorTy);
      heightOperand = arith::ConstantOp::create(
          builder, loc,
          builder.getI32IntegerAttr(static_cast<int32_t>(baseHeightRows)));
    }

    Value widthOperand = arith::ConstantOp::create(
        builder, loc,
        builder.getI32IntegerAttr(static_cast<int32_t>(baseWidthBytes)));

    auto blockPtrLoadOp = ttgi::Subgroup2DBlockLoadFromPtrOp::create(
        builder, loc, op.getType(), op.getPtr(), pitchOperand, heightOperand,
        widthOperand, mask, other,
        ttgi::BlockIOModeAttr::get(builder.getContext(), memLayout));

    // Propagate attributes if present.
    if (oneMatrixPerLoadForBT)
      blockPtrLoadOp->setAttr(
          ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName(),
          builder.getUnitAttr());
    if (auto attr = op->getAttr(
            ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName()))
      blockPtrLoadOp->setAttr(
          ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName(), attr);
    if (rowLimitMask)
      blockPtrLoadOp->setAttr(
          ttgi::TritonIntelGPUDialect::getBlockIORuntimeRowLimitAttrName(),
          builder.getUnitAttr());

    op.replaceAllUsesWith(blockPtrLoadOp.getResult());
    op.erase();
    LDBG("Converted load to ttig.2d_block_load_from_ptr: " << *blockPtrLoadOp);
  }
};

} // namespace
