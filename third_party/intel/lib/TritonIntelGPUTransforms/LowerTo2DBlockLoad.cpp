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

/// Check whether a descriptor load is eligible for 2D block IO lowering.
static bool isBlockIOEligible(tt::DescriptorLoadOp op) {
  if (!op->getAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName()))
    return false;

  auto tensorTy = cast<RankedTensorType>(op.getType());
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

    SmallVector<tt::DescriptorLoadOp> descLoadOps;
    mod.walk([&](tt::DescriptorLoadOp op) { descLoadOps.push_back(op); });

    for (auto op : descLoadOps)
      convertDescriptorLoadOp(op);
  }

private:
  /// Convert a tt.descriptor_load to ttig.2d_block_load.
  void convertDescriptorLoadOp(tt::DescriptorLoadOp op) {
    if (!isBlockIOEligible(op))
      return;

    auto tensorTy = cast<RankedTensorType>(op.getType());
    unsigned rank = tensorTy.getRank();
    unsigned elemSizeInBits = tensorTy.getElementTypeBitWidth();

    // Find the MakeTensorDescOp that created the descriptor.
    Value desc = op.getDesc();
    std::optional<tt::MakeTensorDescOp> makeTensorDescOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(desc);
    if (!makeTensorDescOp) {
      LDBG("Could not find MakeTensorDescOp for: " << *op);
      return;
    }

    auto descType = cast<tt::TensorDescType>(desc.getType());
    unsigned descRank = descType.getBlockType().getRank();
    assert(descRank == rank &&
           "descriptor and result tensor must have same rank");

    bool memoryRowMajor = isMemoryRowMajor(op);

    // Validate that tile computation will succeed during LLVM lowering.
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;
    Attribute encoding = tensorTy.getEncoding();
    LinearLayout llEncoding =
        cast<ttg::DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorTy.getShape());
    if (!ttgi::validate2DBlockLoadTile(llEncoding, contiguousDim,
                                       elemSizeInBits, tensorTy)) {
      LDBG("Tile validation failed for descriptor load: " << *op);
      return;
    }

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Type i32Ty = builder.getI32Type();

    auto memLayout = memoryRowMajor ? ttgi::BlockIOMode::RowMajor
                                    : ttgi::BlockIOMode::ColumnMajor;

    // Extract surface parameters from MakeTensorDescOp.
    // The surface parameters describe the physical memory layout and are the
    // SAME regardless of row_major/column_major. The memory_layout attribute
    // tells the LLVM lowering to set contiguousDim, which triggers the
    // transpose flag in the HW instruction via getBlockIOTileSize.
    Value basePtr = makeTensorDescOp->getBase();
    Operation::operand_range shapes = makeTensorDescOp->getShape();
    Operation::operand_range strides = makeTensorDescOp->getStrides();
    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    assert(indices.size() == descRank &&
           "descriptor index count must match descriptor rank");

    // Fold batch indices into base_ptr for rank > 2 loads. The leading
    // dimensions (before the inner-2 dims) are batch dims whose offsets
    // are multiplied by the corresponding strides and added to the pointer.
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

    // If the pitch stride is a known constant, validate HW constraints
    // (>= 64 bytes, 16-byte aligned).
    std::optional<int64_t> pitchStride =
        tt::intel::getFoldedConstantValue(strides[descRank - 2]);
    if (pitchStride) {
      int64_t pitchBytes = *pitchStride * elemSizeInBits / 8;
      if (pitchBytes < 64 || (pitchBytes % 16) != 0) {
        LDBG("Invalid pitch " << pitchBytes << " for descriptor load: " << *op);
        return;
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
    bool padNan = makeTensorDescOp->getPadding() == tt::PaddingOption::PAD_NAN;
    UnitAttr padNanAttr = padNan ? builder.getUnitAttr() : UnitAttr();

    auto blockLoadOp = ttgi::Subgroup2DBlockLoadOp::create(
        builder, loc, op.getType(), basePtr, baseWidth, baseHeight, basePitch,
        offsetX, offsetY, padNanAttr,
        ttgi::BlockIOModeAttr::get(builder.getContext(), memLayout));

    // Propagate one_matrix_per_load attribute if present.
    if (op->hasAttr(ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName()))
      blockLoadOp->setAttr(
          ttgi::TritonIntelGPUDialect::getOneMatrixPerLoadAttrName(),
          builder.getUnitAttr());

    op.replaceAllUsesWith(blockLoadOp.getResult());
    op.erase();
    LDBG("Converted descriptor load to ttig.2d_block_load: " << *blockLoadOp);
  }
};

} // namespace
