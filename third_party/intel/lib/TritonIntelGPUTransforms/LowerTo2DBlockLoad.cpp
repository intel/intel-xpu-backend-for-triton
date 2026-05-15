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
  if (!op->getAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName()))
    return false;

  RankedTensorType tensorTy;
  if constexpr (std::is_same_v<OpTy, tt::LoadOp>) {
    tensorTy =
        dyn_cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
    if (!tensorTy)
      return false;
  } else {
    tensorTy = cast<RankedTensorType>(op.getType());
  }

  if (tensorTy.getRank() < 2)
    return false;

  bool hasDpas =
      ttgi::hasDpasEncoding(tensorTy) || ttgi::hasDotDpasEncoding(tensorTy);

  std::optional<bool> enableBlockIOForAllLayout = tt::tools::isEnvValueBool(
      tt::tools::getStrEnv("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"));
  if (enableBlockIOForAllLayout.has_value() &&
      !enableBlockIOForAllLayout.value() && !hasDpas)
    return false;

  return true;
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

/// Create a zero-valued splat for the given tensor type. Used when a mask is
/// present but no explicit 'other' value was provided by the user.
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

/// Determine whether memory layout is row-major from the block_io attribute.
static bool isMemoryRowMajor(Operation *op) {
  auto blockIOAttr = op->getAttrOfType<StringAttr>(
      ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
  assert(blockIOAttr && "expected block_io attribute");
  auto mode = ttgi::symbolizeBlockIOMode(blockIOAttr.getValue());
  return !mode || *mode == ttgi::BlockIOMode::RowMajor;
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
    SmallVector<tt::MakeTensorDescOp> allDescs =
        tt::intel::findAllMakeTensorDescOps(desc);
    if (allDescs.empty()) {
      LDBG("Could not find MakeTensorDescOp for: " << *op);
      return;
    }

    // All candidates must have the same padding.
    tt::PaddingOption padding = allDescs[0].getPadding();
    if (!llvm::all_of(allDescs, [&](tt::MakeTensorDescOp d) {
          return d.getPadding() == padding;
        })) {
      LDBG("Inconsistent padding across descriptor candidates for: " << *op);
      return;
    }

    auto descType = cast<tt::TensorDescType>(desc.getType());
    unsigned descRank = descType.getBlockType().getRank();
    assert(descRank >= rank && "descriptor rank must be >= result tensor rank");

    bool memoryRowMajor = isMemoryRowMajor(op);

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
      auto sizeInfo = ttgi::getBlockIOTileSize<true>(
          llEncoding, contiguousDim, elemSizeInBits,
          /*maskAxisInfo=*/nullptr, oneMatrixPerLoadForBT);
      int innerDimStart = static_cast<int>(rank - 2);
      if (sizeInfo.rowDim < innerDimStart || sizeInfo.colDim < innerDimStart) {
        LDBG("Batch dim in tile for descriptor load: " << *op);
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
    // match, validate HW constraints (>= 64 bytes, 16-byte aligned).
    // For rank-reducing loads, the stride interpretation may differ from the
    // 2D surface pitch, so skip static validation (runtime will handle it).
    if (rank == descRank) {
      std::optional<int64_t> pitchStride =
          tt::intel::getFoldedConstantValue(strides[descRank - 2]);
      if (pitchStride) {
        int64_t pitchBytes = *pitchStride * elemSizeInBits / 8;
        if (pitchBytes < 64 || (pitchBytes % 16) != 0) {
          LDBG("Invalid pitch " << pitchBytes
                                << " for descriptor load: " << *op);
          return;
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

    bool memoryRowMajor = isMemoryRowMajor(op);
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
    int64_t baseWidthBytes =
        tensorTy.getDimSize(surfaceWidthDim) * elemSizeInBits / 8;
    constexpr int64_t MIN_PITCH = 64;

    int64_t pitch;
    bool isBroadcast = false;
    if (has1DReshapeStride) {
      auto strideAttr = op->getAttrOfType<IntegerAttr>(
          ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName());
      int64_t stride = strideAttr.getInt();
      isBroadcast = (stride == 0);
      pitch = stride * elemSizeInBits / 8;
    } else {
      // Check if the load is a broadcast: stride=0 along rowDim means all
      // rows are identical, so we only need height=1 with a dummy pitch.
      int64_t rowStride = getStride(strideAnalysis, op.getPtr(), rowDim);
      isBroadcast = (rowStride == 0);
      if (isBroadcast) {
        pitch = std::max((int64_t)MIN_PITCH, baseWidthBytes);
      } else {
        int64_t pitchStride = getStride(strideAnalysis, op.getPtr(), pitchDim);
        if (pitchStride < 0) {
          LDBG("Cannot compute constant stride for load: " << *op);
          return;
        }
        pitch = pitchStride * elemSizeInBits / 8;
      }
    }

    // HW requires pitch >= 64 bytes and pitch aligned to 16 bytes.
    if (pitch < MIN_PITCH || (pitch % 16) != 0) {
      LDBG("Invalid pitch " << pitch << " for load: " << *op);
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
    int64_t baseHeightRows =
        isBroadcast ? 1 : tensorTy.getDimSize(surfaceHeightDim);

    auto memLayout = memoryRowMajor ? ttgi::BlockIOMode::RowMajor
                                    : ttgi::BlockIOMode::ColumnMajor;

    // If mask is present but other is absent, create a zero splat as the
    // default padding value (the verifier requires 'other' when 'mask' is set).
    Value mask = op.getMask();
    Value other = op.getOther();
    if (mask && !other)
      other = createZeroSplat(builder, loc, tensorTy);

    auto blockPtrLoadOp = ttgi::Subgroup2DBlockLoadFromPtrOp::create(
        builder, loc, op.getType(), op.getPtr(), mask, other,
        builder.getI32IntegerAttr(baseWidthBytes),
        builder.getI32IntegerAttr(baseHeightRows),
        builder.getI32IntegerAttr(pitch),
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
