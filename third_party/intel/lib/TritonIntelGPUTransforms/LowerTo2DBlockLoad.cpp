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
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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

/// True when `other` is absent or a splat of zero -- the only fill value the
/// 2D block message's hardware boundary check can produce, so the only case
/// where a row-limit mask (see ttgi::matchRowLimitMask) can be dropped in
/// favor of a dynamic `base_height` operand.
static bool isZeroFillOther(Value other) {
  if (!other)
    return true;
  DenseElementsAttr constAttr;
  if (!matchPattern(other, m_Constant(&constAttr)) || !constAttr.isSplat())
    return false;
  Type elemTy = constAttr.getElementType();
  if (isa<FloatType>(elemTy))
    return constAttr.getSplatValue<APFloat>().isZero();
  if (isa<IntegerType>(elemTy))
    return constAttr.getSplatValue<APInt>().isZero();
  return false;
}

/// True when `v` is a type `toI32` below can convert (or `v` is null, the
/// zero-base case). arith.cmpi operands are always integer or index, so this
/// is expected to always hold for a matched row-limit mask; kept as an
/// explicit, cheap guard rather than relying on that invariant silently.
static bool isI32Convertible(Value v) {
  return !v || v.getType().isIndex() || isa<IntegerType>(v.getType());
}

/// Normalize an integer or index scalar to i32 via index_cast/ext/trunc, so
/// it can feed an i32 block-load operand (base_pitch, base_height). `v` must
/// satisfy isI32Convertible.
static Value toI32(OpBuilder &builder, Location loc, Value v) {
  Type i32Ty = builder.getI32Type();
  if (v.getType() == i32Ty)
    return v;
  if (v.getType().isIndex())
    return arith::IndexCastOp::create(builder, loc, i32Ty, v);
  auto intTy = cast<IntegerType>(v.getType());
  if (intTy.getWidth() < 32)
    return arith::ExtSIOp::create(builder, loc, i32Ty, v);
  return arith::TruncIOp::create(builder, loc, i32Ty, v);
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
    for (auto op : descLoadOps)
      convertDescriptorLoadOp(op);

    SmallVector<tt::LoadOp> loadOps;
    mod.walk([&](tt::LoadOp op) { loadOps.push_back(op); });
    for (auto op : loadOps)
      convertLoadOp(op, strideAnalysis, axisInfoAnalysis);
  }

private:
  /// Convert a tt.descriptor_load to ttig.2d_block_load.
  void convertDescriptorLoadOp(tt::DescriptorLoadOp op) {
    if (!isBlockIOEligible(op))
      return;

    auto tensorTy = cast<RankedTensorType>(op.getType());
    unsigned rank = tensorTy.getRank();
    unsigned elemSizeInBits = tensorTy.getElementTypeBitWidth();

    // Find all MakeTensorDescOps that could define this descriptor.
    Value desc = op.getDesc();
    tt::intel::DescriptorDefinitions defs =
        tt::intel::findDescriptorDefinitions(desc);
    if (defs.empty()) {
      LDBG("Could not find MakeTensorDescOp for: " << *op);
      return;
    }

    std::optional<tt::PaddingOption> padding = defs.consistentPadding();
    if (!padding) {
      LDBG("Inconsistent padding across descriptor candidates for: " << *op);
      return;
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
      return;
    }

    // For descriptor loads, the 2D block I/O tile must use only the inner 2
    // dims. Reject if rowDim or colDim falls in a batch dimension.
    if (rank > 2) {
      auto sizeInfo = ttgi::getBlockIOLoadTileSize(
          llEncoding, contiguousDim, elemSizeInBits,
          /*maskAxisInfo=*/nullptr, oneMatrixPerLoadForBT);
      int innerDimStart = static_cast<int>(rank - 2);
      if (sizeInfo.rowDim < innerDimStart || sizeInfo.colDim < innerDimStart) {
        LDBG("Batch dim in tile for descriptor load: " << *op);
        return;
      }
    }

    // The 2D block message encodes base_width (bytes), base_height (rows) and
    // base_pitch (bytes) as value-1 in 24-bit fields, so each must be in
    // [1, 2^24]. Where the target needs 64-byte base alignment, the lowering
    // adds the base's misalignment (base & 63) to base_width, so base_width
    // must leave 63 bytes of room. This is the last point a descriptor load
    // can fall back: Subgroup2DBlockLoadOpConversion does not re-validate.
    // Bail if ANY candidate MakeTensorDescOp has a compile-time-foldable field
    // outside that range. Check before emitting any IR: the ttig.extract_desc
    // values built below do not fold, so read the defining MakeTensorDescOp(s).
    //
    // Not checked here:
    //  - Non-foldable (runtime) values are trusted to be in range.
    //  - Alignment (base 4B, width multiple of 4B):
    //    in the normal pipeline MaterializeBlockPointer checks it before it
    //    tags the load (runtime values rely on the tt.make_tensor_descriptor
    //    16-byte contract); hand-tagged IR is not re-checked here.
    //  - The HW 64-byte minimum for width/pitch is deliberately not enforced;
    //    it would move narrow descriptors off 2D block I/O.
    //  - pitch >= width (a TritonGEN verifier rule) is not a range check and
    //    is not enforced here either.
    constexpr int64_t kMax2DBlockField = int64_t(1) << 24;
    int64_t maxBaseWidth = ttgi::needs2DBlockIOAlignmentCompensation(op)
                               ? kMax2DBlockField - 63
                               : kMax2DBlockField;
    int64_t elemBytesConst = elemSizeInBits / 8;
    auto isOutOfRange = [&](unsigned operandIdx, int64_t scale,
                            int64_t maxValue, StringRef field) {
      return llvm::any_of(defs, [&](tt::MakeTensorDescOp d) {
        std::optional<int64_t> folded =
            tt::intel::getFoldedConstantValue(d->getOperand(operandIdx));
        if (!folded)
          return false;
        int64_t value;
        if (!llvm::MulOverflow(*folded, scale, value) && value >= 1 &&
            value <= maxValue)
          return false;
        LDBG("Invalid " << field << " (" << *folded << " x " << scale
                        << ") for descriptor load: " << *op);
        return true;
      });
    };
    // MakeTensorDescOp operands: base, shape[descRank], strides[descRank].
    if (isOutOfRange(/*inner shape*/ 1 + (descRank - 1), elemBytesConst,
                     maxBaseWidth, "base_width") ||
        isOutOfRange(/*outer shape*/ 1 + (descRank - 2), /*scale=*/1,
                     kMax2DBlockField, "base_height") ||
        isOutOfRange(/*pitch stride*/ 1 + descRank + (descRank - 2),
                     elemBytesConst, kMax2DBlockField, "base_pitch"))
      return;

    for (tt::MakeTensorDescOp d : defs) {
      std::optional<int64_t> stride = tt::intel::getFoldedConstantValue(
          d->getOperand(/*pitch stride*/ 1 + descRank + (descRank - 2)));
      if (stride && (*stride * elemBytesConst) % 16 != 0) {
        op->emitOpError("descriptor pitch of ")
            << *stride * elemBytesConst
            << " bytes is not a multiple of 16 bytes";
        signalPassFailure();
        return;
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

    // A mask of the form `(rowBase + arange) < bound`, constant along the
    // memory-contiguous dim, doesn't need per-element masking: the bound can
    // instead become the 2D block message's `base_height` operand, and the
    // hardware's own out-of-bounds zero-fill does what the mask would have
    // done. Gate on `other` being zero (the only fill value the hardware can
    // produce) and on the mask's bounded dim actually being the surface's
    // pitch dim, both decided up front so the tile-size computation below
    // can treat this load as unmasked consistently with the operand
    // construction further down.
    std::optional<ttgi::RowLimitMaskInfo> rowLimitMask;
    if (op.getMask() && rank == 2 && isZeroFillOther(op.getOther())) {
      if (auto m = ttgi::matchRowLimitMask(op.getMask(), tensorTy.getShape())) {
        unsigned pitchTensorDim =
            (contiguousDim == rank - 1) ? rank - 2 : rank - 1;
        if (m->boundedDim == pitchTensorDim && isI32Convertible(m->rowBase) &&
            isI32Convertible(m->bound))
          rowLimitMask = m;
        else
          LDBG("Row-limit mask does not match the pitch dim for: " << *op);
      }
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
      // Drop the mask/other: the hardware's own zero-fill at base_height
      // replaces the per-element predication.
      mask = Value();
      other = Value();
      Value boundI32 = toI32(builder, loc, rowLimitMask->bound);
      Value diff = boundI32;
      if (rowLimitMask->rowBase) {
        Value baseI32 = toI32(builder, loc, rowLimitMask->rowBase);
        diff = arith::SubIOp::create(builder, loc, boundI32, baseI32);
      }
      Value zero =
          arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
      heightOperand = arith::MaxSIOp::create(builder, loc, diff, zero);
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

    auto blockPtrLoadOp = ttgi::Subgroup2DBlockLoadFromPtrOp::create(
        builder, loc, op.getType(), op.getPtr(), pitchOperand, heightOperand,
        mask, other, builder.getI32IntegerAttr(baseWidthBytes),
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
