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
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"
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
template <typename OpTy,
          std::enable_if_t<
              llvm::is_one_of<OpTy, tt::LoadOp, tt::DescriptorLoadOp>::value,
              bool> = true>
static bool isBlockIOEligible(OpTy op) {
  RankedTensorType tensorTy;
  if constexpr (std::is_same_v<OpTy, tt::LoadOp>) {
    tensorTy =
        dyn_cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
    if (!tensorTy)
      return false;
  } else {
    tensorTy = cast<RankedTensorType>(op.getType());
  }

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

/// Compute the 1D SliceEncoding for a given dimension of a higher-rank
/// encoding.  This encoding is suitable for tt.make_range results that will
/// be expanded back to full rank via tt.expand_dims.
static Attribute get1DSliceEncoding(MLIRContext *ctx, unsigned dim,
                                    unsigned rank, Attribute encoding) {
  // Slice away all dimensions except `dim`, in reverse expansion order.
  // Expansion order is: j = 0, 1, ..., rank-1 (skipping dim).
  // So slice order is: j = rank-1, ..., 0 (skipping dim).
  Attribute enc = encoding;
  for (int j = rank - 1; j >= 0; --j) {
    if (static_cast<unsigned>(j) == dim)
      continue;
    enc = ttg::SliceEncodingAttr::get(ctx, j,
                                      cast<ttg::DistributedEncodingTrait>(enc));
  }
  return enc;
}

/// Create a padding value (zero or NaN) for the given tensor type.
static Value createPaddingValue(OpBuilder &builder, Location loc,
                                RankedTensorType tensorTy,
                                tt::PaddingOption padding) {
  Type elemType = tensorTy.getElementType();
  if (padding == tt::PaddingOption::PAD_NAN && isa<FloatType>(elemType)) {
    auto floatTy = cast<FloatType>(elemType);
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto attr =
        SplatElementsAttr::get(tensorTy, builder.getFloatAttr(floatTy, nan));
    return arith::ConstantOp::create(builder, loc, attr);
  }
  // Default: zero padding.
  Attribute zeroAttr;
  if (isa<FloatType>(elemType))
    zeroAttr = builder.getFloatAttr(elemType, 0.0);
  else if (isa<IntegerType>(elemType))
    zeroAttr = builder.getIntegerAttr(elemType, 0);
  else
    llvm_unreachable("unsupported element type for padding value");
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
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return;

    // FIXME: Remove once IGC can split large 2D block loads.
    // Read the env var once and materialize it as an attribute on the ops
    // so downstream passes only need to check the attribute.
    std::optional<bool> envOneMatrixPerLoad = tt::tools::isEnvValueBool(
        tt::tools::getStrEnv("TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT"));
    if (envOneMatrixPerLoad.has_value()) {
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

    SmallVector<tt::DescriptorLoadOp> descLoadOps;
    mod.walk([&](tt::DescriptorLoadOp op) { descLoadOps.push_back(op); });
    for (auto op : descLoadOps) {
      if (!convertDescriptorLoadTo2DBlockLoad(op))
        lowerDescriptorLoadToLoad(op);
    }

    SmallVector<tt::LoadOp> loadOps;
    mod.walk([&](tt::LoadOp op) { loadOps.push_back(op); });
    for (auto op : loadOps)
      convertLoadOp(op, strideAnalysis, axisInfoAnalysis);
  }

private:
  /// Try to convert a tt.descriptor_load to ttig.2d_block_load.
  /// Returns true if the conversion succeeded, false otherwise.
  bool convertDescriptorLoadTo2DBlockLoad(tt::DescriptorLoadOp op) {
    if (!isBlockIOEligible(op))
      return false;

    auto tensorTy = cast<RankedTensorType>(op.getType());
    unsigned rank = tensorTy.getRank();
    unsigned elemSizeInBits = tensorTy.getElementTypeBitWidth();

    // Find all MakeTensorDescOps that could define this descriptor.
    Value desc = op.getDesc();
    SmallVector<tt::MakeTensorDescOp> allDescs =
        tt::intel::findAllMakeTensorDescOps(desc);
    if (allDescs.empty()) {
      LDBG("Could not find MakeTensorDescOp for: " << *op);
      return false;
    }

    // All candidates must have the same padding.
    tt::PaddingOption padding = allDescs[0].getPadding();
    if (!llvm::all_of(allDescs, [&](tt::MakeTensorDescOp d) {
          return d.getPadding() == padding;
        })) {
      LDBG("Inconsistent padding across descriptor candidates for: " << *op);
      return false;
    }

    auto descType = cast<tt::TensorDescType>(desc.getType());
    unsigned descRank = descType.getBlockType().getRank();
    assert(descRank >= rank && "descriptor rank must be >= result tensor rank");

    bool memoryRowMajor = ttgi::isMemoryRowMajor(op);

    // Validate that tile computation will succeed during LLVM lowering.
    bool oneMatrixPerLoadForBT =
        op->hasAttr(ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName());

    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;
    Attribute encoding = tensorTy.getEncoding();
    LinearLayout llEncoding =
        cast<ttg::DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorTy.getShape());
    if (!ttgi::validate2DBlockLoadTile(llEncoding, contiguousDim,
                                       elemSizeInBits, tensorTy,
                                       oneMatrixPerLoadForBT)) {
      LDBG("Tile validation failed for descriptor load: " << *op);
      return false;
    }

    // For descriptor loads, the 2D block I/O tile must use only the inner 2
    // dims. Reject if rowDim or colDim falls in a batch dimension.
    if (rank > 2) {
      auto sizeInfo = ttgi::getBlockIOTileSize<true>(
          llEncoding, contiguousDim, elemSizeInBits,
          /*maskAxisInfo=*/nullptr, oneMatrixPerLoadForBT);
      int innerDimStart = static_cast<int>(rank - 2);
      if (sizeInfo.rowDim < innerDimStart || sizeInfo.colDim < innerDimStart) {
        LDBG("Batch dim in tile for descriptor load: " << *op);
        return false;
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
        tt::PointerType::get(descType.getBlockType().getElementType(), 1);
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

    // If the pitch stride is a known constant AND the descriptor/result ranks
    // match, validate HW constraints (>= 64 bytes, 16-byte aligned, encoded
    // in 24 bits per the `triton_gen.2Dblockload` verifier).
    // For rank-reducing loads, the stride interpretation may differ from the
    // 2D surface pitch, so skip static validation (runtime will handle it).
    if (rank == descRank) {
      std::optional<int64_t> pitchStride =
          tt::intel::getFoldedConstantValue(strides[descRank - 2]);
      if (pitchStride) {
        int64_t pitchBytes = *pitchStride * elemSizeInBits / 8;
        if (pitchBytes < 64 || (pitchBytes % 16) != 0 ||
            pitchBytes > (int64_t(1) << 24)) {
          LDBG("Invalid pitch " << pitchBytes
                                << " for descriptor load: " << *op);
          return false;
        }
      }
    }

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

    // Determine padding mode from the descriptor.
    bool padNan = padding == tt::PaddingOption::PAD_NAN;
    UnitAttr padNanAttr = padNan ? builder.getUnitAttr() : UnitAttr();

    auto blockLoadOp = ttgi::Subgroup2DBlockLoadOp::create(
        builder, loc, op.getType(), basePtr, baseWidth, baseHeight, basePitch,
        offsetX, offsetY, padNanAttr,
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

  /// Lower a tt.descriptor_load to tt.load when 2D block load is not possible.
  /// Constructs a pointer tensor and boundary-checking mask from the descriptor
  /// fields.
  void lowerDescriptorLoadToLoad(tt::DescriptorLoadOp op) {
    auto resultTy = cast<RankedTensorType>(op.getType());
    Attribute encoding = resultTy.getEncoding();
    if (!encoding)
      return;

    Type elemTy = resultTy.getElementType();
    unsigned rank = resultTy.getRank();
    ArrayRef<int64_t> shape = resultTy.getShape();

    auto descType = cast<tt::TensorDescType>(op.getDesc().getType());
    unsigned descRank = descType.getBlockType().getRank();

    OpBuilder builder(op);
    Location loc = op.getLoc();
    MLIRContext *ctx = builder.getContext();
    Type i64Ty = builder.getI64Type();
    Type ptrElemType = tt::PointerType::get(elemTy, 1);

    // Extract descriptor fields. Prefer using values from MakeTensorDescOp
    // directly when available — this preserves AxisInfo provenance so the
    // downstream LLVM lowering can infer contiguity and vectorize properly.
    // ExtractDescOp produces opaque runtime values that AxisInfo cannot
    // analyze, leading to scalar (vec=1) loads.
    Value desc = op.getDesc();
    SmallVector<Value> descShapes(descRank);
    SmallVector<Value> descStrides(descRank);
    Value basePtr;
    SmallVector<tt::MakeTensorDescOp> allDescs =
        tt::intel::findAllMakeTensorDescOps(desc);
    if (!allDescs.empty()) {
      tt::MakeTensorDescOp makeDesc = allDescs[0];
      basePtr = makeDesc.getBase();
      for (unsigned d = 0; d < descRank; ++d) {
        descShapes[d] =
            arith::ExtSIOp::create(builder, loc, i64Ty, makeDesc.getShape()[d]);
        descStrides[d] = makeDesc.getStrides()[d];
      }
    } else {
      // Opaque descriptor (function argument, untraceable control flow) —
      // fall back to ExtractDescOp.
      for (unsigned d = 0; d < descRank; ++d) {
        descShapes[d] = ttgi::ExtractDescOp::create(
            builder, loc, i64Ty, desc, builder.getI32IntegerAttr(d));
        descStrides[d] = ttgi::ExtractDescOp::create(
            builder, loc, i64Ty, desc, builder.getI32IntegerAttr(descRank + d));
      }
      basePtr =
          ttgi::ExtractDescOp::create(builder, loc, ptrElemType, desc,
                                      builder.getI32IntegerAttr(2 * descRank));
    }

    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    assert(indices.size() == descRank &&
           "descriptor index count must match descriptor rank");

    // Fold batch dimensions into base pointer.
    unsigned numBatchDims = descRank - rank;
    for (unsigned d = 0; d < numBatchDims; ++d) {
      Value batchOffset = arith::MulIOp::create(
          builder, loc, arith::ExtSIOp::create(builder, loc, i64Ty, indices[d]),
          descStrides[d]);
      basePtr = tt::AddPtrOp::create(builder, loc, basePtr.getType(), basePtr,
                                     batchOffset);
    }

    // Work with inner dimensions only.
    SmallVector<Value> innerShapes(descShapes.begin() + numBatchDims,
                                   descShapes.end());
    SmallVector<Value> innerStrides(descStrides.begin() + numBatchDims,
                                    descStrides.end());
    SmallVector<Value> innerIndices(indices.begin() + numBatchDims,
                                    indices.end());

    // For column-major descriptor loads, the result type has its inner-2
    // dimensions transposed relative to the descriptor's natural order (e.g.,
    // descriptor [N, K] produces result [K, N]). Swap the inner-2 dimensions
    // of shapes, strides, and indices so they align with the result type.
    auto blockIOAttr = op->getAttrOfType<StringAttr>(
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
    bool permuteDescDim =
        blockIOAttr && ttgi::symbolizeBlockIOMode(blockIOAttr.getValue()) ==
                           ttgi::BlockIOMode::ColumnMajor;
    if (permuteDescDim && rank >= 2) {
      std::swap(innerShapes[rank - 2], innerShapes[rank - 1]);
      std::swap(innerStrides[rank - 2], innerStrides[rank - 1]);
      std::swap(innerIndices[rank - 2], innerIndices[rank - 1]);
    }

    // Build pointer tensor: for each element (i0, i1, ..., iR-1),
    //   ptr[i0][i1]...[iR-1] = base + sum_d((indices[d] + id) * stride[d])
    auto ptrTensorTy = RankedTensorType::get(shape, ptrElemType, encoding);
    auto i64TensorTy = RankedTensorType::get(shape, i64Ty, encoding);

    Value ptrTensor = tt::SplatOp::create(builder, loc, ptrTensorTy, basePtr);
    Value mask;

    for (unsigned d = 0; d < rank; ++d) {
      // Compute 1D slice encoding for this dimension.
      Attribute enc1D = get1DSliceEncoding(ctx, d, rank, encoding);

      // Create range [0, 1, ..., shape[d]-1].
      auto range1DTy =
          RankedTensorType::get({shape[d]}, builder.getI32Type(), enc1D);
      Value range =
          tt::MakeRangeOp::create(builder, loc, range1DTy, 0, shape[d]);

      // Extend to i64.
      auto range1DI64Ty = RankedTensorType::get({shape[d]}, i64Ty, enc1D);
      Value rangeI64 =
          arith::ExtSIOp::create(builder, loc, range1DI64Ty, range);

      // Add index offset: indices[d] + range.
      Value indexI64 =
          arith::ExtSIOp::create(builder, loc, i64Ty, innerIndices[d]);
      Value splatIndex =
          tt::SplatOp::create(builder, loc, range1DI64Ty, indexI64);
      Value offsetRange =
          arith::AddIOp::create(builder, loc, splatIndex, rangeI64);

      // Expand to full rank via expand_dims (encoding inferred via
      // SliceEncoding -> parent).
      Value expanded = offsetRange;
      for (unsigned j = 0; j < rank; ++j) {
        if (j == d)
          continue;
        expanded = tt::ExpandDimsOp::create(builder, loc, expanded, j);
      }

      // Broadcast to full shape.
      Value broadcasted =
          tt::BroadcastOp::create(builder, loc, i64TensorTy, expanded);

      // Compute pointer contribution: broadcasted * stride[d].
      Value splatStride =
          tt::SplatOp::create(builder, loc, i64TensorTy, innerStrides[d]);
      Value offset =
          arith::MulIOp::create(builder, loc, broadcasted, splatStride);

      // Add to pointer tensor.
      ptrTensor =
          tt::AddPtrOp::create(builder, loc, ptrTensorTy, ptrTensor, offset);

      // Build mask for this dimension: 0 <= indices[d] + range < shape[d].
      // For the stride-1 (contiguous) dimension, use a scalar comparison to
      // preserve mask constancy for vectorization. Per-element comparison
      // along this dimension gives constancy=1, which forces scalar loads in
      // the LLVM lowering (getMaskAlignment reduces vec to 1).
      // A scalar check matches DescriptorLoadOpConversion's behavior where
      // the first element's predicate gates the entire vector chunk.
      std::optional<int64_t> strideVal =
          tt::intel::getFoldedConstantValue(innerStrides[d]);
      bool isContiguousDim = strideVal && *strideVal == 1;

      Value dimMask;
      if (isContiguousDim) {
        // Scalar: (offset + blockSize) <= shape → all elements in bounds.
        Value blockSize =
            arith::ConstantIntOp::create(builder, loc, shape[d], 64);
        Value end = arith::AddIOp::create(builder, loc, indexI64, blockSize);
        Value scalarCmp = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sle, end, innerShapes[d]);
        auto i1TensorTy =
            RankedTensorType::get(shape, builder.getI1Type(), encoding);
        dimMask = tt::SplatOp::create(builder, loc, i1TensorTy, scalarCmp);
      } else {
        Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
        Value splatZero = tt::SplatOp::create(builder, loc, i64TensorTy, zero);
        Value cmpLower = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::sge, broadcasted, splatZero);

        Value splatShape =
            tt::SplatOp::create(builder, loc, i64TensorTy, innerShapes[d]);
        Value cmpUpper = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::slt, broadcasted, splatShape);

        dimMask = arith::AndIOp::create(builder, loc, cmpLower, cmpUpper);
      }

      if (!mask)
        mask = dimMask;
      else
        mask = arith::AndIOp::create(builder, loc, mask, dimMask);
    }

    // Determine padding value from the descriptor.
    tt::PaddingOption padding = tt::PaddingOption::PAD_ZERO;
    if (!allDescs.empty())
      padding = allDescs[0].getPadding();

    Value other = createPaddingValue(builder, loc, resultTy, padding);

    // Create tt.load with pointer tensor, mask, and padding value.
    auto loadOp = tt::LoadOp::create(builder, loc, ptrTensor, mask, other,
                                     tt::CacheModifier::NONE,
                                     tt::EvictionPolicy::NORMAL, false);

    op.replaceAllUsesWith(loadOp.getResult());
    op.erase();
    LDBG("Lowered descriptor load to tt.load: " << *loadOp);
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
    unsigned elemSizeInBits = tensorTy.getElementTypeBitWidth();

    bool memoryRowMajor = ttgi::isMemoryRowMajor(op);
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;

    bool oneMatrixPerLoadForBT =
        op->hasAttr(ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName());

    // Retrieve mask axis info to validate tile constraints consistently
    // with the downstream LLVM lowering.
    tt::AxisInfo *maskAxisInfo = nullptr;
    if (op.getMask())
      maskAxisInfo = axisInfoAnalysis.getAxisInfo(op.getMask());

    // For 1D->2D reshape loads, skip tile validation and use the stride
    // attribute directly for pitch.
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
    if (has1DReshapeStride) {
      // 1D reshape: conventional dims, no tile validation needed.
      rowDim = memoryRowMajor ? rank - 2 : rank - 1;
      colDim = memoryRowMajor ? rank - 1 : rank - 2;
    } else {
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
      auto sizeInfo = ttgi::getBlockIOTileSize<true>(
          llEncoding, contiguousDim, elemSizeInBits, maskAxisInfo,
          oneMatrixPerLoadForBT);
      rowDim = sizeInfo.rowDim;
      colDim = sizeInfo.colDim;
      tileWidth = sizeInfo.tileWidth;
      tileHeight = sizeInfo.tileHeight;
      isTranspose = sizeInfo.transpose;
      numPackedVals = sizeInfo.numElemPerPackedVal;
    }

    // For the 2D block load surface, the pitch dimension is always the
    // non-contiguous memory direction. For transposed loads, rowDim is the
    // memory-contiguous dim and colDim is non-contiguous, so pitch uses colDim.
    unsigned pitchDim = isTranspose ? colDim : rowDim;

    // Compute pitch from stride analysis or the 1D->2D reshape attribute.
    // For the HW surface: width is along memory-contiguous direction, height
    // is along non-contiguous. For transposed loads, rowDim is the contiguous
    // dim and colDim is non-contiguous, so we swap which dim provides
    // width/height.
    unsigned surfaceWidthDim = isTranspose ? rowDim : colDim;
    unsigned surfaceHeightDim = isTranspose ? colDim : rowDim;
    constexpr int64_t MIN_PITCH = 64;
    // Surface pitch is encoded in 24 bits in the 2D block IO message
    // descriptor (see `triton_gen.2Dblockload` verifier in TritonGENOps.cpp).
    constexpr int64_t MAX_PITCH = int64_t(1) << 24;

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
        // Use the full surface row width (in bytes) as the baseline pitch.
        // Lowering may widen base_width (e.g. due to alignment), so ensure the
        // dummy pitch doesn't end up smaller than base_width.
        int64_t fullRowBytes =
            tensorTy.getDimSize(surfaceWidthDim) * elemSizeInBits / 8;
        pitch = std::max(MIN_PITCH, fullRowBytes);
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

    // The HW needs pitch >= 64 bytes, a multiple of 16, and within 24 bits.
    // For a constant pitch we just check it. For a runtime pitch we can't,
    // so we rely on the checks already done: MaterializeBlockPointer proved
    // the stride is 16-byte aligned, `lda >= K` keeps pitch >= the row width,
    // and the row-width check below covers the 64-byte minimum.
    if (!pitchValue) {
      if (pitch < MIN_PITCH || (pitch % 16) != 0 || pitch > MAX_PITCH) {
        LDBG("Invalid pitch " << pitch << " for load: " << *op);
        return;
      }
    } else {
      int64_t fullRowBytes =
          tensorTy.getDimSize(surfaceWidthDim) * elemSizeInBits / 8;
      if (fullRowBytes < MIN_PITCH) {
        LDBG("Runtime pitch: full row width " << fullRowBytes << " < "
                                              << MIN_PITCH << " bytes");
        return;
      }
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

    // If mask is present but other is absent, create a zero splat as the
    // default padding value (the verifier requires 'other' when 'mask' is set).
    Value mask = op.getMask();
    Value other = op.getOther();
    if (mask && !other)
      other = createZeroSplat(builder, loc, tensorTy);

    // Pitch is a single i32 operand: the runtime value when we have one, an
    // arith.constant otherwise (the lowering and asserts treat both the same).
    Value pitchOperand =
        pitchValue
            ? pitchValue
            : arith::ConstantOp::create(
                  builder, loc,
                  builder.getI32IntegerAttr(static_cast<int32_t>(pitch)));
    auto blockPtrLoadOp = ttgi::Subgroup2DBlockLoadFromPtrOp::create(
        builder, loc, op.getType(), op.getPtr(), pitchOperand, mask, other,
        builder.getI32IntegerAttr(baseWidthBytes),
        builder.getI32IntegerAttr(baseHeightRows),
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

    op.replaceAllUsesWith(blockPtrLoadOp.getResult());
    op.erase();
    LDBG("Converted load to ttig.2d_block_load_from_ptr: " << *blockPtrLoadOp);
  }
};

} // namespace
