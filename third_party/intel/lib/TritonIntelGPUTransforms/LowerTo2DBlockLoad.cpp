//===- LowerTo2DBlockLoad.cpp - Lower loads to ttig.2d_block_load ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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
#include "triton/Tools/Sys/GetEnv.hpp"
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
  if constexpr (std::is_same_v<OpTy, tt::LoadOp>)
    tensorTy =
        cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
  else
    tensorTy = cast<RankedTensorType>(op.getType());

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
static int getStride(tt::intel::ModuleStrideAnalysis &strideAnalysis, Value ptr,
                     unsigned dim) {
  tt::intel::StrideInfo *info = strideAnalysis.getStrideInfo(ptr);
  if (info) {
    const auto &stride = info->getStride();
    if (dim < stride.size())
      return stride[dim];
  }
  return -1;
}

/// Trace a pointer tensor back through addptr/broadcast/splat to find the
/// original scalar base pointer. Returns nullptr if the pattern is not
/// recognized.
static Value traceBasePtr(Value ptrTensor) {
  Value current = ptrTensor;
  for (;;) {
    if (auto addptr = current.getDefiningOp<tt::AddPtrOp>()) {
      current = addptr.getPtr();
      continue;
    }
    if (auto broadcast = current.getDefiningOp<tt::BroadcastOp>()) {
      current = broadcast.getSrc();
      continue;
    }
    if (auto splat = current.getDefiningOp<tt::SplatOp>()) {
      return splat.getSrc(); // Found the scalar pointer.
    }
    return nullptr; // Unrecognized pattern.
  }
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
    return nullptr;
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
  return *mode == ttgi::BlockIOMode::RowMajor;
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

    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    tt::intel::ModuleStrideAnalysis strideAnalysis(mod, axisInfoAnalysis);

    SmallVector<tt::LoadOp> loadOps;
    SmallVector<tt::DescriptorLoadOp> descLoadOps;
    mod.walk([&](tt::LoadOp op) { loadOps.push_back(op); });
    mod.walk([&](tt::DescriptorLoadOp op) { descLoadOps.push_back(op); });

    for (auto op : loadOps)
      convertLoadOp(op, strideAnalysis);
    for (auto op : descLoadOps)
      convertDescriptorLoadOp(op);
  }

private:
  /// Convert a tt.load with tensor-of-pointers to ttig.2d_block_load.
  void convertLoadOp(tt::LoadOp op,
                     tt::intel::ModuleStrideAnalysis &strideAnalysis) {
    if (!isBlockIOEligible(op))
      return;

    auto tensorTy =
        cast<RankedTensorType>(tt::getPointeeType(op.getPtr().getType()));
    unsigned rank = tensorTy.getRank();
    bool memoryRowMajor = isMemoryRowMajor(op);

    unsigned elemSizeInBits = tensorTy.getElementTypeBitWidth();

    // Compute pitch from stride analysis. For pointer-based 2D block IO, the
    // stride must be a compile-time constant.
    unsigned rowDim = memoryRowMajor ? rank - 2 : rank - 1;
    int stride = getStride(strideAnalysis, op.getPtr(), rowDim);
    if (stride < 0) {
      LDBG("Cannot compute constant stride for load: " << *op);
      return;
    }

    constexpr int MIN_PITCH = 64;
    unsigned colDim = memoryRowMajor ? rank - 1 : rank - 2;
    int64_t baseWidthBytes = tensorTy.getDimSize(colDim) * elemSizeInBits / 8;

    int pitch;
    if (stride == 0)
      pitch = std::max((int64_t)MIN_PITCH, baseWidthBytes);
    else
      pitch = stride * elemSizeInBits / 8;

    // HW requires pitch >= 64 bytes and pitch aligned to 16 bytes.
    if (pitch < MIN_PITCH || (pitch % 16) != 0) {
      LDBG("Invalid pitch " << pitch << " for load: " << *op);
      return;
    }

    // Validate that tile computation will succeed during LLVM lowering.
    Attribute encoding = tensorTy.getEncoding();
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;
    LinearLayout llEncoding =
        cast<ttg::DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorTy.getShape());
    if (!ttgi::validate2DBlockLoadTile(llEncoding, contiguousDim,
                                       elemSizeInBits, tensorTy)) {
      LDBG("Tile validation failed for load: " << *op);
      return;
    }

    // Trace back through addptr/broadcast/splat to find the scalar base
    // pointer. This gives a uniform value (no per-lane shuffle needed).
    Value basePtr = traceBasePtr(op.getPtr());
    if (!basePtr) {
      LDBG("Cannot trace base pointer for load: " << *op);
      return;
    }

    OpBuilder builder(op);
    Location loc = op.getLoc();

    // Compute constant surface parameters.
    int64_t baseHeightRows = stride == 0 ? 1 : tensorTy.getDimSize(rowDim);

    Value baseWidth =
        arith::ConstantIntOp::create(builder, loc, baseWidthBytes, 32);
    Value baseHeight =
        arith::ConstantIntOp::create(builder, loc, baseHeightRows, 32);
    Value basePitch = arith::ConstantIntOp::create(builder, loc, pitch, 32);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);

    auto memLayout = memoryRowMajor ? ttgi::BlockIOMode::RowMajor
                                    : ttgi::BlockIOMode::ColumnMajor;

    // If mask is present but other is absent, create a zero splat as the
    // default padding value (the verifier requires 'other' when 'mask' is set).
    Value mask = op.getMask();
    Value other = op.getOther();
    if (mask && !other) {
      other = createZeroSplat(builder, loc, tensorTy);
      if (!other) {
        LDBG("Cannot create default 'other' for masked load: " << *op);
        return;
      }
    }

    auto blockLoadOp = ttgi::Subgroup2DBlockLoadOp::create(
        builder, loc, op.getType(), basePtr, baseWidth, baseHeight, basePitch,
        /*offset_x=*/zero, /*offset_y=*/zero, mask, other,
        ttgi::BlockIOModeAttr::get(builder.getContext(), memLayout));

    op.replaceAllUsesWith(blockLoadOp.getResult());
    op.erase();
    LDBG("Converted load to ttig.2d_block_load: " << *blockLoadOp);
  }

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
    assert(descRank >= rank && "descriptor rank must be >= result tensor rank");

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

    // For descriptor loads, the 2D block I/O tile must use only the inner 2
    // dims. Reject if rowDim or colDim falls in a batch dimension.
    auto sizeInfo = ttgi::getBlockIOTileSize<true>(llEncoding, contiguousDim,
                                                   elemSizeInBits);
    int innerDimStart = static_cast<int>(rank - 2);
    if (sizeInfo.rowDim < innerDimStart || sizeInfo.colDim < innerDimStart) {
      LDBG("Tile dims outside inner-2 for descriptor load: " << *op);
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

    // Fold rank-reducing batch indices into base_ptr.
    unsigned rankDelta = descRank - rank;
    for (unsigned d = 0; d < rankDelta; ++d) {
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

    // Handle padding: for PAD_NAN, create a NaN splat as 'other'.
    Value other;
    tt::PaddingOption padding = makeTensorDescOp->getPadding();
    if (padding == tt::PaddingOption::PAD_NAN) {
      Type elemType = tensorTy.getElementType();
      Attribute nanAttr;
      if (isa<FloatType>(elemType)) {
        auto floatTy = cast<FloatType>(elemType);
        nanAttr = builder.getFloatAttr(
            floatTy, APFloat::getNaN(floatTy.getFloatSemantics()));
      }
      if (nanAttr) {
        Value nanVal =
            arith::ConstantOp::create(builder, loc, cast<TypedAttr>(nanAttr));
        other = tt::SplatOp::create(builder, loc, tensorTy, nanVal);
      }
    }

    auto blockLoadOp = ttgi::Subgroup2DBlockLoadOp::create(
        builder, loc, op.getType(), basePtr, baseWidth, baseHeight, basePitch,
        offsetX, offsetY,
        /*mask=*/Value(), /*other=*/other,
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
