#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-materialize-block-pointer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUMATERIALIZEBLOCKPOINTER
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

/// True if `TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS` is explicitly set to a
/// falsy value. When that env var is "0" (e.g. on the CRI simulator to avoid
/// a hang), `LoadStoreOpToLLVM.cpp::isBlockIOCandidate()` rejects non-DPAS
/// 2D block loads/stores and they fall through to the regular gather, which
/// cannot correctly handle the [H,W] register-strides-rows encoding produced
/// by the 1D->2D reshape rewrites.
static bool isBlockIOForAllLayoutsExplicitlyDisabled() {
  auto enableBlockIOForAllLayout = tt::tools::isEnvValueBool(
      tt::tools::getStrEnv("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"));
  return enableBlockIOForAllLayout.has_value() &&
         !enableBlockIOForAllLayout.value();
}

// Check a descriptor base pointer or pitch against an alignment requirement.
// A constant must be provably divisible. For a runtime value, divisibility 1
// means the alignment is unknown; we trust the 16-byte make_tensor_descriptor
// contract rather than reject the 2D block IO path. This silently miscompiles
// if a caller breaks that contract, so the LDBG traces when we rely on it.
static bool isDescriptorAligned(tt::intel::ModuleAxisInfoAnalysis &axisInfo,
                                Value v, unsigned divisor) {
  if (matchPattern(v, m_Constant()))
    return ttgi::isDivisible(v, divisor);
  const tt::AxisInfo *info = axisInfo.getAxisInfo(v);
  int64_t div = info ? info->getDivisibility(0) : 1;
  if (div == 1) {
    LDBG("Divisibility unknown; trusting the 16-byte make_tensor_descriptor "
         "alignment contract for runtime value (divisor="
         << divisor << "): " << v);
    return true;
  }
  return div % divisor == 0;
}

struct TritonIntelGPUMaterializeBlockPointerPass
    : public triton::gpu::intel::impl::
          TritonIntelGPUMaterializeBlockPointerBase<
              TritonIntelGPUMaterializeBlockPointerPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUMaterializeBlockPointerBase<
      TritonIntelGPUMaterializeBlockPointerPass>::
      TritonIntelGPUMaterializeBlockPointerBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return;

    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    tt::intel::ModuleStrideAnalysis strideAnalysis(mod, axisInfoAnalysis);
    MLIRContext *context = &getContext();
    mod.walk([&](tt::LoadOp op) {
      visit(op, axisInfoAnalysis, strideAnalysis, context);
    });
    SmallVector<tt::StoreOp> storeOps;
    mod.walk([&](tt::StoreOp op) { storeOps.push_back(op); });
    for (auto op : storeOps)
      visit(op, axisInfoAnalysis, strideAnalysis, context);
    mod.walk(
        [&](tt::DescriptorLoadOp op) { visit(op, axisInfoAnalysis, context); });
    mod.walk([&](tt::DescriptorStoreOp op) {
      visit(op, axisInfoAnalysis, context);
    });
  }

private:
  // Visit method for descriptor operations
  void visit(tt::DescriptorLoadOp op,
             tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
             MLIRContext *context) const {
    visitDescriptor(op, op.getResult().getType(), axisInfoAnalysis, context);
  }

  void visit(tt::DescriptorStoreOp op,
             tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
             MLIRContext *context) const {
    visitDescriptor(op, op.getSrc().getType(), axisInfoAnalysis, context);
  }

  // Implementation for descriptor operations
  template <typename OpType>
  void visitDescriptor(OpType op, RankedTensorType tensorType,
                       tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                       MLIRContext *context) const {
    LDBG("Considering descriptor op: " << *op);

    Value desc = op.getDesc();
    // Find all MakeTensorDescOps that could define this descriptor.
    SmallVector<tt::MakeTensorDescOp> allDescs =
        tt::intel::findAllMakeTensorDescOps(desc);
    if (allDescs.empty()) {
      LDBG("Could not find any make tensor desc op for: " << *op);
      return;
    }

    tt::MakeTensorDescOp makeTensorDescOp = allDescs[0];
    LDBG("Make tensor desc op: " << makeTensorDescOp);

    // All candidates must have the same padding.
    tt::PaddingOption padding = makeTensorDescOp.getPadding();
    if (!llvm::all_of(allDescs, [&](tt::MakeTensorDescOp d) {
          return d.getPadding() == padding;
        })) {
      LDBG("Inconsistent padding across candidates");
      return;
    }

    // Propagate padding from MakeTensorDescOp unconditionally so the LLVM
    // lowering can read it even after MakeTensorDescOp has been converted
    // in the same applyPartialConversion phase.
    op->setAttr(ttgi::TritonIntelGPUDialect::getDescPaddingAttrName(),
                tt::PaddingOptionAttr::get(context, padding));

    Operation::operand_range shape = makeTensorDescOp.getShape();
    unsigned rank = shape.size();
    LDBG("Rank: " << rank);
    if (rank == 1)
      return;

    if (!satisfies2DBlockReadAlignment(op, axisInfoAnalysis)) {
      LDBG("Alignment checks failed for: " << *op);
      return;
    }

    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    LDBG("elementWidth: " << elementWidth);

    Operation::operand_range strides = makeTensorDescOp.getStrides();
    // For tensor descriptors, the last stride is always one (row major).
    unsigned strideOneDimVal = rank - 1;

    // Verify that tensor descriptor has stride=1 in last dimension.
    Value fastChangeStride = strides[strideOneDimVal];
    assert(tt::intel::isConstant(fastChangeStride, 1) &&
           "Tensor descriptor must have stride=1 in last dimension");

    // Across Intel platforms, the strictest pitch restriction is to be a
    // multiple of OWord(128 bits). All candidates must satisfy this.
    unsigned pitchDivisor = llvm::divideCeil(128, elementWidth);
    if (!llvm::all_of(allDescs, [&](tt::MakeTensorDescOp d) {
          return isDescriptorAligned(axisInfoAnalysis, d.getStrides()[rank - 2],
                                     pitchDivisor);
        }))
      return;

    // Tensor descriptors are always row major.
    op->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                StringAttr::get(context, "row_major"));
  }

  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, tt::LoadOp, tt::StoreOp>::value>>
  void visit(OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
             tt::intel::ModuleStrideAnalysis &strideAnalysis,
             MLIRContext *context) const {
    LDBG("Considering op: " << *op);

    Value ptr = op.getPtr();
    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return;

    if constexpr (std::is_same_v<OpType, tt::LoadOp>) {
      if (Value mask = op.getMask()) {
        if (!matchPattern(mask, m_One())) {
          LDBG("Load op has non-trivial mask, skip block IO attribute");
          return;
        }
      }
    }

    unsigned rank = tensorTy.getRank();

    // For 1D ops, try to detect strided access patterns and reshape
    // to 2D for block IO lowering.
    if constexpr (std::is_same_v<OpType, tt::LoadOp>) {
      if (rank == 1) {
        reshape1DStridedLoad(op, tensorTy, context);
        return;
      }
    }
    if constexpr (std::is_same_v<OpType, tt::StoreOp>) {
      if (rank == 1) {
        reshape1DStridedStore(op, tensorTy, context);
        return;
      }
    }

    LDBG("Considering tensor of pointer of memory accessing op: " << op);

    // Axis info describes the value layout of the indices tensor.
    //
    // For example, consider an indices tensor of type tensor<8x16xi32> with
    // values:
    //   [[  0,  1,  2, ...,  15],
    //    [ 16, 17, 18, ...,  31],
    //    ...
    //    [112,113,114, ...,127]]
    //
    // In this case, the global memory referenced by the tensor pointer is
    // row-major contiguous.
    //
    // Axis info:
    //   stride:      [16, 1]
    //   contiguity:  [1, 16]
    //
    // The code inspects the last two dimensions to determine which dimension
    // changes the fastest in memory. The remaining outer dimensions are treated
    // as irrelevant batch dimensions.
    //
    // Case 1: The innermost dimension is the fast-changing one.
    //   This corresponds to a row-major contiguous access pattern per 2d slice.
    //   The axis info reflects this with stride [..., 1].
    //
    // Case 2: The second innermost dimension is the fast-changing one.
    //   This corresponds to a column-major contiguous access pattern per 2d
    //   slice. The axis info reflects this with stride [..., 1, X].
    const tt::AxisInfo *axisInfo = axisInfoAnalysis.getAxisInfo(ptr);

    if (rank < 2) {
      LDBG("Rank is < 2, skip block IO attribute");
      return;
    }

    // Determine if LoadOp is row-major or column-major.
    tt::intel::StrideInfo *strideInfo = strideAnalysis.getStrideInfo(ptr);
    auto isMajor = [rank, &strideInfo](RankedTensorType tensorTy,
                                       unsigned fastChangeDim,
                                       const tt::AxisInfo &axisInfo) {
      assert((fastChangeDim == rank - 1 || fastChangeDim == rank - 2) &&
             "fastChangeDim is expected to be rank - 1 or rank - 2");
      const unsigned otherDim =
          (fastChangeDim == rank - 1) ? rank - 2 : rank - 1;
      // Limit to full row being contiguous.
      if (axisInfo.getContiguity(fastChangeDim) !=
          tensorTy.getDimSize(fastChangeDim)) {
        LDBG("Found non-contiguous row: "
             << axisInfo.getContiguity(fastChangeDim));
        return false;
      }

      // Runtime stride (-1) is OK if AxisInfo proves 16-byte alignment.
      int64_t otherDimStride =
          strideInfo ? strideInfo->getStride(otherDim) : -1;
      Type elemTy =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned elemSizeInBytes = elemTy.getIntOrFloatBitWidth() / 8;
      if (otherDimStride >= 0) {
        if ((otherDimStride * elemSizeInBytes) % 16 != 0) {
          LDBG("Non 16-byte aligned stride: " << otherDimStride);
          return false;
        }
      } else if (axisInfo.getDivisibility(otherDim) % 16 != 0) {
        LDBG("Runtime stride: divisibility "
             << axisInfo.getDivisibility(otherDim) << " < 16 bytes");
        return false;
      }

      // Base pointer can be compensate by the offset and base width, where they
      // each has restriction that it has to be 4 bytes aligned.
      if (axisInfo.getDivisibility(fastChangeDim) % 4 != 0) {
        LDBG("Found Non 4 bytes aligned base: " << axisInfo.getDivisibility(1));
        return false;
      }

      return true;
    };

    const StringRef blockIOAttrName =
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName();
    const bool isRowMajor =
        isMajor(tensorTy, rank - 1 /*fastChangeDim*/, *axisInfo);
    if (isRowMajor)
      op->setAttr(blockIOAttrName,
                  StringAttr::get(
                      op.getContext(),
                      ttgi::stringifyBlockIOMode(ttgi::BlockIOMode::RowMajor)));

    const bool isColMajor =
        isMajor(tensorTy, rank - 2 /*fastChangeDim*/, *axisInfo);
    if (isColMajor)
      op->setAttr(blockIOAttrName,
                  StringAttr::get(op.getContext(),
                                  ttgi::stringifyBlockIOMode(
                                      ttgi::BlockIOMode::ColumnMajor)));
  }

  /// Look through cast wrappers (index_cast, extui, extsi, trunci, etc.)
  /// to find the underlying arithmetic operation.
  static Value lookThroughCasts(Value val) {
    while (Operation *def = val.getDefiningOp()) {
      if (!isa<CastOpInterface>(def) || def->getNumOperands() != 1)
        break;
      val = def->getOperand(0);
    }
    return val;
  }

  /// Check whether val is a unit-stride linear index of length expectedEnd.
  /// Accepts two forms:
  ///   1. tt.make_range(0, N)
  ///   2. arith.addi(tt.splat(scalar_offset), tt.make_range(0, N))
  ///      where scalar_offset is a multiple of W (so rem/div semantics are
  ///      preserved: (offset + i) % W == i % W when offset % W == 0).
  static bool isCanonicalLinearIndex(Value val, int64_t expectedEnd,
                                     int64_t W) {
    val = lookThroughCasts(val);
    // Form 1: bare make_range.
    if (auto makeRange = val.getDefiningOp<tt::MakeRangeOp>())
      return makeRange.getStart() == 0 && makeRange.getEnd() == expectedEnd;
    // Form 2: addi(splat(offset), make_range) or addi(make_range, splat(...)).
    auto addI = val.getDefiningOp<arith::AddIOp>();
    if (!addI)
      return false;
    Value lhs = lookThroughCasts(addI.getLhs());
    Value rhs = lookThroughCasts(addI.getRhs());
    // Try both orderings.
    for (int i = 0; i < 2; ++i) {
      auto makeRange = lhs.getDefiningOp<tt::MakeRangeOp>();
      auto splatOp = rhs.getDefiningOp<tt::SplatOp>();
      if (makeRange && splatOp && makeRange.getStart() == 0 &&
          makeRange.getEnd() == expectedEnd) {
        // The splat offset must be a compile-time multiple of W for
        // rem/div correctness: (offset + i) % W == i % W when offset % W == 0.
        // We check structurally that offset = muli(_, C) where C % W == 0.
        Value scalar = splatOp.getSrc();
        if (auto mulI = scalar.getDefiningOp<arith::MulIOp>()) {
          // Check if either operand is a constant divisible by W.
          for (Value operand : {mulI.getLhs(), mulI.getRhs()}) {
            if (auto cst = getScalarConstantValue(operand))
              if (*cst % W == 0)
                return true;
          }
        }
        return false;
      }
      std::swap(lhs, rhs);
    }
    return false;
  }

  /// Extract a scalar integer constant (not tensor), looking through casts.
  static std::optional<int64_t> getScalarConstantValue(Value val) {
    val = lookThroughCasts(val);
    if (!val)
      return std::nullopt;
    auto constOp = val.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return std::nullopt;
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().getSExtValue();
    return std::nullopt;
  }

  /// Try to extract a constant integer value from a Value, looking through
  /// casts. Handles both scalar IntegerAttr and splat DenseIntElementsAttr
  /// (tensor constants like `arith.constant dense<32> : tensor<Nxi32>`).
  static std::optional<int64_t> getConstantValue(Value val) {
    // Try scalar constant first.
    if (auto scalar = getScalarConstantValue(val))
      return scalar;
    // Try splat tensor constant (e.g., arith.constant dense<32> :
    // tensor<Nxi32>).
    val = lookThroughCasts(val);
    if (!val)
      return std::nullopt;
    auto constOp = val.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return std::nullopt;
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat())
        return denseAttr.getSplatValue<APInt>().getSExtValue();
    }
    return std::nullopt;
  }

  /// Information extracted from a 1D strided access pattern:
  ///   offset = (idx % W) + (idx / W) * S
  struct StridedPatternInfo {
    int64_t W;          // Tile width (contiguous elements per row)
    int64_t H;          // Tile height (number of rows)
    int64_t S;          // Row stride in elements (pitch = S * elemBytes)
    unsigned numWarps;  // Number of warps in the module
    unsigned elemBits;  // Element size in bits
    unsigned elemBytes; // Element size in bytes
    Value ptr;          // The AddPtrOp result (original 1D pointer tensor)
  };

  /// Try to match a 1D strided access pattern from a pointer tensor.
  /// The pattern is: offset = (idx % W) + (idx / W) * S
  /// where W is the tile width, H = numElements / W is the tile height,
  /// and S is the row stride.
  ///
  /// \p maxPerWarpHeight is the maximum per-warp tile height allowed
  /// by the HW (8 for stores, 32 for loads).
  std::optional<StridedPatternInfo>
  matchStridedPattern(Operation *op, RankedTensorType ptrTensorTy,
                      unsigned maxPerWarpHeight) const {
    // Trace pointer through tt.addptr to find the offset tensor.
    Value ptr = op->getOperand(0); // ptr is always operand 0 for Load/Store
    auto addPtrOp = ptr.getDefiningOp<tt::AddPtrOp>();
    if (!addPtrOp) {
      LDBG("Pointer not defined by tt.addptr, skip 1D reshape");
      return std::nullopt;
    }
    Value offset = addPtrOp.getOffset();

    // Pattern-match the offset for:
    //   arith.addi(arith.remui(idx, W), arith.muli(arith.divui(idx, W), S))
    Value offsetUnwrapped = lookThroughCasts(offset);
    auto addIOp = offsetUnwrapped.getDefiningOp<arith::AddIOp>();
    if (!addIOp) {
      LDBG("Offset not defined by arith.addi, skip 1D reshape");
      return std::nullopt;
    }

    // Match (remui, muli) in either order of the addi operands.
    Value lhs = lookThroughCasts(addIOp.getLhs());
    Value rhs = lookThroughCasts(addIOp.getRhs());
    auto remOp = lhs.getDefiningOp<arith::RemUIOp>();
    auto mulOp = rhs.getDefiningOp<arith::MulIOp>();
    if (!remOp || !mulOp) {
      remOp = rhs.getDefiningOp<arith::RemUIOp>();
      mulOp = lhs.getDefiningOp<arith::MulIOp>();
    }
    if (!remOp || !mulOp) {
      LDBG("Could not match remui/muli pattern, skip 1D reshape");
      return std::nullopt;
    }

    // Extract W from remui(idx, W).
    Value remIdx = lookThroughCasts(remOp.getLhs());
    std::optional<int64_t> wVal = getConstantValue(remOp.getRhs());
    if (!wVal || *wVal <= 0) {
      LDBG("Could not extract constant W from remui, skip 1D reshape");
      return std::nullopt;
    }
    int64_t W = *wVal;

    // Match muli(divui(idx, W'), S) in either operand order.
    Value mulLhs = lookThroughCasts(mulOp.getLhs());
    Value mulRhs = lookThroughCasts(mulOp.getRhs());
    auto divOp = mulLhs.getDefiningOp<arith::DivUIOp>();
    std::optional<int64_t> sVal =
        divOp ? getConstantValue(mulRhs) : std::nullopt;
    if (!divOp || !sVal) {
      divOp = mulRhs.getDefiningOp<arith::DivUIOp>();
      sVal = divOp ? getConstantValue(mulLhs) : std::nullopt;
    }
    if (!divOp || !sVal || *sVal <= 0) {
      LDBG("Could not match divui/constant in muli, skip 1D reshape");
      return std::nullopt;
    }
    int64_t S = *sVal;

    // Verify divui uses the same index and same W constant as remui.
    Value divIdx = lookThroughCasts(divOp.getLhs());
    std::optional<int64_t> divWVal = getConstantValue(divOp.getRhs());
    if (!divWVal || *divWVal != W) {
      LDBG("divui W constant (" << (divWVal ? std::to_string(*divWVal) : "?")
                                << ") does not match remui W (" << W
                                << "), skip 1D reshape");
      return std::nullopt;
    }

    // Verify both remui and divui use the same index.
    if (remIdx != divIdx) {
      LDBG("remui and divui use different index values, skip 1D reshape");
      return std::nullopt;
    }

    // Compute H = numElements / W and verify evenly divides.
    // Bail out on dynamic shapes — getDimSize returns ShapedType::kDynamic
    // (-1), which would make subsequent divisibility and H computation
    // invalid.
    if (ShapedType::isDynamic(ptrTensorTy.getDimSize(0))) {
      LDBG("Pointer tensor has dynamic shape, skip 1D reshape");
      return std::nullopt;
    }
    int64_t numElements = ptrTensorTy.getDimSize(0);

    // Verify idx is a canonical unit-stride linear index.
    if (!isCanonicalLinearIndex(remIdx, numElements, W)) {
      LDBG("Index is not a canonical linear index of length "
           << numElements << " with W=" << W);
      return std::nullopt;
    }
    if (numElements % W != 0) {
      LDBG("numElements (" << numElements << ") not divisible by W (" << W
                           << "), skip 1D reshape");
      return std::nullopt;
    }
    int64_t H = numElements / W;

    LDBG("Detected strided pattern: W=" << W << ", H=" << H << ", S=" << S);

    // Validate HW constraints common to both loads and stores.
    Type pointeeTy =
        cast<tt::PointerType>(ptrTensorTy.getElementType()).getPointeeType();
    unsigned elemBits = pointeeTy.getIntOrFloatBitWidth();
    unsigned elemBytes = elemBits / 8;

    // Minimum pitch is 64 bytes.
    if (S * elemBytes < 64) {
      LDBG("Pitch " << S * elemBytes << " bytes < 64, skip 1D reshape");
      return std::nullopt;
    }

    // Pitch must be 16-byte aligned (surface pitch HW requirement).
    if ((S * elemBytes) % 16 != 0) {
      LDBG("Pitch " << S * elemBytes
                    << " bytes not 16-byte aligned, skip 1D reshape");
      return std::nullopt;
    }

    // Surface pitch is encoded in 24 bits in the 2D block IO message
    // descriptor.
    constexpr int64_t maxPitchBytes = int64_t(1) << 24;
    if (S * elemBytes > maxPitchBytes) {
      LDBG("Pitch " << S * elemBytes
                    << " bytes exceeds 24-bit HW limit, skip 1D reshape");
      return std::nullopt;
    }

    // Tile width must satisfy HW limits per element size.
    // The reshape always produces numElemPerPackedVal == 1, so
    // packedElemSizeInBits == elemBits.
    if (!ttgi::check2DBlockAddressPayloadRestriction(elemBits, W)) {
      LDBG("Tile width " << W << " invalid for " << elemBits
                         << "-bit elements, skip 1D reshape");
      return std::nullopt;
    }

    // Per-warp tile height must be in [1, maxPerWarpHeight].
    unsigned numWarps = ttg::lookupNumWarps(op);
    if (H % numWarps != 0 ||
        static_cast<unsigned>(H / numWarps) > maxPerWarpHeight) {
      LDBG("Per-warp height " << H / numWarps << " exceeds max "
                              << maxPerWarpHeight << ", skip 1D reshape");
      return std::nullopt;
    }

    return StridedPatternInfo{W, H, S, numWarps, elemBits, elemBytes, ptr};
  }

  /// Copy discardable attributes from \p src to \p dst, skipping block IO
  /// attributes that are set explicitly by the caller.
  static void copyNonBlockIOAttrs(Operation *src, Operation *dst) {
    for (NamedAttribute attr : src->getDiscardableAttrs()) {
      StringRef name = attr.getName().getValue();
      if (name == ttgi::TritonIntelGPUDialect::getBlockIOAttrName() ||
          name == ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName())
        continue;
      dst->setDiscardableAttr(attr.getName(), attr.getValue());
    }
  }

  /// Set the standard block IO attributes (row_major + stride) on an op.
  static void setBlockIOAttrs(Operation *op, MLIRContext *ctx, int64_t stride) {
    assert(stride > 0 && "stride must be positive");
    op->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                StringAttr::get(ctx, "row_major"));
    op->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName(),
                IntegerAttr::get(IntegerType::get(ctx, 64), stride));
  }

  /// Detect 1D tensor-of-pointers StoreOp with strided access pattern
  /// and reshape to 2D store with block IO attributes.
  ///
  /// Inductor often flattens a 2D row-major tile into a 1D index:
  ///
  ///   offset = (idx % W) + (idx / W) * S
  ///
  /// where W is the tile width, H = numElements / W is the tile height,
  /// and S is the row stride in elements (S >= W when the tile is a slice
  /// of a wider allocation).
  ///
  /// Example — 3x4 tile (H=3, W=4) inside a buffer with stride S=6:
  ///
  ///   Memory layout (. = padding):
  ///   addr:  0  1  2  3  .  .  6  7  8  9  .  . 12 13 14 15  .  .
  ///          [── row 0 ──]     [── row 1 ──]    [── row 2 ──]
  ///
  ///   idx | col = idx%4 | row = idx/4 | offset = col + row*6
  ///   ----+-------------+-------------+----------------------
  ///    0  |      0      |      0      |   0
  ///    1  |      1      |      0      |   1
  ///    2  |      2      |      0      |   2
  ///    3  |      3      |      0      |   3
  ///    4  |      0      |      1      |   6
  ///    ...
  ///   11  |      3      |      2      |  15
  ///
  /// Without this pass the compiler sees 12 independent pointer
  /// computations and emits 12 scalar scatter writes.  This method
  /// recovers W, H, and S from the arithmetic, reshapes the store from
  /// [12] to [3, 4], and annotates it so that StoreOpToBlockIOConversion
  /// emits a single LSC2DBlockWrite(base, width=4, height=3, pitch=6).
  void reshape1DStridedStore(tt::StoreOp op, RankedTensorType ptrTensorTy,
                             MLIRContext *ctx) const {
    LDBG("Attempting 1D strided store reshape for: " << *op);

    if (isBlockIOForAllLayoutsExplicitlyDisabled()) {
      LDBG("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 disables non-DPAS 2D "
           "block stores; skip 1D->2D store reshape");
      return;
    }

    // Reject masked stores — we only handle unmasked or provably all-true.
    if (Value mask = op.getMask()) {
      if (!matchPattern(mask, m_One())) {
        LDBG("Store has non-trivial mask, skip 1D reshape");
        return;
      }
    }

    // MAX_TILE_HEIGHT_STORE = 8
    std::optional<StridedPatternInfo> info =
        matchStridedPattern(op, ptrTensorTy, /*maxPerWarpHeight=*/8);
    if (!info)
      return;

    // TODO: 2D block store does not support hardware transpose. With H > 1,
    // the encoding inference puts registers in columns while the store
    // hardware expects registers in rows. Fixing this requires inserting a
    // ConvertLayoutOp before the store.
    if (info->H != 1) {
      LDBG("H=" << info->H
                << " > 1 not yet supported for store, skip 1D reshape");
      return;
    }

    // Create reshaped tensors: [N] -> [H, W].
    Location loc = op.getLoc();
    OpBuilder builder(op);
    SmallVector<int64_t> newShape = {info->H, info->W};

    // Use the ReshapeOp builder that infers the 2D encoding automatically.
    auto ptrReshape = tt::ReshapeOp::create(builder, loc, newShape, info->ptr,
                                            /*allowReorder=*/false);
    Value val = op.getValue();
    auto valReshape = tt::ReshapeOp::create(builder, loc, newShape, val,
                                            /*allowReorder=*/false);

    // Create the new 2D store.
    auto newStore = tt::StoreOp::create(builder, loc, ptrReshape, valReshape,
                                        op.getCache(), op.getEvict());

    setBlockIOAttrs(newStore, ctx, info->S);
    copyNonBlockIOAttrs(op, newStore);

    LDBG("Created 2D block store: " << *newStore);

    op.erase();
  }

  /// Detect 1D tensor-of-pointers LoadOp with strided access pattern
  /// and reshape to 2D load with block IO attributes.
  ///
  /// Similar to reshape1DStridedStore, but for loads. Since 2D block loads
  /// deliver data in a specific layout (lane k = column k, registers stack
  /// rows), we construct an explicit "load encoding" matching this HW
  /// delivery, perform the load, then insert a ConvertLayoutOp to the
  /// natural "consumer encoding" that the rest of the pipeline expects.
  void reshape1DStridedLoad(tt::LoadOp op, RankedTensorType ptrTensorTy,
                            MLIRContext *ctx) const {
    LDBG("Attempting 1D strided load reshape for: " << *op);

    if (isBlockIOForAllLayoutsExplicitlyDisabled()) {
      LDBG("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS=0 disables non-DPAS 2D "
           "block loads; skip 1D->2D load reshape");
      return;
    }

    // For loads, we allow masks (the load path handles them).
    // However, the strided pattern matcher needs the ptr, not mask.

    // MAX_TILE_HEIGHT_LOAD = 32
    std::optional<StridedPatternInfo> info =
        matchStridedPattern(op, ptrTensorTy, /*maxPerWarpHeight=*/32);
    if (!info)
      return;

    Location loc = op.getLoc();
    OpBuilder builder(op);
    SmallVector<int64_t> newShape = {info->H, info->W};

    unsigned threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    unsigned numWarps = info->numWarps;
    unsigned perWarpH = static_cast<unsigned>(info->H / numWarps);
    unsigned W = static_cast<unsigned>(info->W);

    // The 2D block load HW delivers tile_width contiguous columns per row
    // into the first tile_width lanes of the subgroup.  When W < tpw, the
    // remaining lanes' delivery pattern does not match a plain row/col
    // layout, and constructing a BlockedEncoding with threadsPerWarp=[1,tpw]
    // would create a replicated layout that the reshape lowering rejects
    // as an "expensive view" (make_llir failure).  Bail out for now; this
    // case can be re-enabled once the lowering handles sub-subgroup tiles.
    if (W < threadsPerWarp) {
      LDBG("W=" << W << " < threadsPerWarp=" << threadsPerWarp
                << " not supported for 1D load reshape");
      return;
    }

    // Construct "load encoding" matching HW delivery:
    // lane k = column k, registers stack rows.
    // When W > tpw, each thread owns W/tpw consecutive columns.
    unsigned loadSpt1 = W / threadsPerWarp;
    auto loadEnc = ttg::BlockedEncodingAttr::get(
        ctx,
        /*sizePerThread=*/{perWarpH, loadSpt1},
        /*threadsPerWarp=*/{1, threadsPerWarp},
        /*warpsPerCTA=*/{numWarps, 1},
        /*order=*/{1, 0},
        ttg::CGAEncodingAttr::fromSplitParams(
            ctx, /*CTAsPerCGA=*/SmallVector<unsigned>(2, 1),
            /*CTASplitNum=*/SmallVector<unsigned>(2, 1),
            /*CTAOrder=*/{0, 1}));

    // Construct "consumer encoding" — the natural 2D reshape of the original
    // 1D encoding. For a 1D blocked encoding with sizePerThread=[spt],
    // threadsPerWarp=[tpw], warpsPerCTA=[wpc], order=[0], the natural reshape
    // to [H, W] with order=[1,0] distributes elements column-first.
    auto origEnc =
        dyn_cast<ttg::BlockedEncodingAttr>(ptrTensorTy.getEncoding());
    if (!origEnc || origEnc.getSizePerThread().size() != 1 ||
        origEnc.getOrder().size() != 1 || origEnc.getOrder()[0] != 0) {
      LDBG("Expected 1D blocked encoding with order=[0], skip 1D reshape");
      return;
    }
    unsigned spt = origEnc.getSizePerThread()[0];
    unsigned tpw = origEnc.getThreadsPerWarp()[0];
    unsigned wpc = origEnc.getWarpsPerCTA()[0];
    unsigned spt1 = std::min(spt, static_cast<unsigned>(info->W));
    if (spt1 == 0 || spt % spt1 != 0) {
      LDBG("Original sizePerThread ("
           << spt << ") not divisible by spt1=" << spt1 << ", skip 1D reshape");
      return;
    }
    unsigned spt0 = spt / spt1;
    unsigned tpw1 = std::min(tpw, static_cast<unsigned>(info->W) / spt1);
    if (tpw1 == 0 || tpw % tpw1 != 0) {
      LDBG("Original threadsPerWarp ("
           << tpw << ") not divisible by tpw1=" << tpw1 << ", skip 1D reshape");
      return;
    }
    unsigned tpw0 = tpw / tpw1;
    auto consumerEnc = ttg::BlockedEncodingAttr::get(
        ctx, {spt0, spt1}, {tpw0, tpw1}, {wpc, 1}, {1, 0},
        ttg::CGAEncodingAttr::fromSplitParams(
            ctx, /*CTAsPerCGA=*/SmallVector<unsigned>(2, 1),
            /*CTASplitNum=*/SmallVector<unsigned>(2, 1),
            /*CTAOrder=*/{0, 1}));

    // Reshape pointer to [H, W] with load encoding.
    // Mark efficient_layout so RemoveLayoutConversions does not
    // rematerialize these reshapes with a different encoding.
    auto loadPtrTy =
        RankedTensorType::get(newShape, ptrTensorTy.getElementType(), loadEnc);
    auto ptrReshape = tt::ReshapeOp::create(builder, loc, loadPtrTy, info->ptr,
                                            /*allowReorder=*/true,
                                            /*efficientLayout=*/true);

    // Reshape mask if present.
    Value mask2d;
    if (Value mask = op.getMask()) {
      auto maskTy = cast<RankedTensorType>(mask.getType());
      auto loadMaskTy =
          RankedTensorType::get(newShape, maskTy.getElementType(), loadEnc);
      mask2d = tt::ReshapeOp::create(builder, loc, loadMaskTy, mask,
                                     /*allowReorder=*/true,
                                     /*efficientLayout=*/true);
    }

    // Create 2D load with load encoding.
    Type pointeeTy =
        cast<tt::PointerType>(ptrTensorTy.getElementType()).getPointeeType();
    auto loadResultTy = RankedTensorType::get(newShape, pointeeTy, loadEnc);
    auto newLoad = tt::LoadOp::create(builder, loc, loadResultTy, ptrReshape,
                                      mask2d, op.getOther(), op.getCache(),
                                      op.getEvict(), op.getIsVolatile());

    // Set block IO attributes.
    setBlockIOAttrs(newLoad, ctx, info->S);
    copyNonBlockIOAttrs(op, newLoad);

    // ConvertLayoutOp: load encoding -> consumer encoding.
    auto consumerResultTy =
        RankedTensorType::get(newShape, pointeeTy, consumerEnc);
    auto converted =
        ttg::ConvertLayoutOp::create(builder, loc, consumerResultTy, newLoad);

    // Reshape back to 1D with original result type.
    auto origResultTy = cast<RankedTensorType>(op.getType());
    auto reshapeBack = tt::ReshapeOp::create(builder, loc, origResultTy,
                                             converted, /*allowReorder=*/false,
                                             /*efficientLayout=*/true);

    LDBG("Created 2D block load with layout conversion: " << *newLoad);

    // Replace and erase.
    op.replaceAllUsesWith(reshapeBack.getResult());
    op.erase();
  }

  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, tt::DescriptorLoadOp, tt::DescriptorStoreOp>::value>>
  bool satisfies2DBlockReadAlignment(
      OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    Value desc = op.getDesc();

    // Find all MakeTensorDescOps that could define this descriptor.
    SmallVector<tt::MakeTensorDescOp> allDescs =
        tt::intel::findAllMakeTensorDescOps(desc);
    if (allDescs.empty())
      return false;

    tt::MakeTensorDescOp makeTensorDescOp = allDescs[0];
    Operation::operand_range shape = makeTensorDescOp.getShape();
    // All candidates must have the same shape operands.
    if (!llvm::all_of(allDescs, [&](tt::MakeTensorDescOp d) {
          return d.getShape() == shape;
        })) {
      LDBG("Inconsistent shape across descriptor candidates");
      return false;
    }

    unsigned rank = shape.size();
    if (rank == 1)
      return false;

    // For tensor descriptors, the last stride is always one (row major).
    unsigned strideOneDimVal = rank - 1;

    // Get the tensor type from the descriptor
    tt::TensorDescType descType =
        cast<tt::TensorDescType>(makeTensorDescOp.getType());
    RankedTensorType tensorType = descType.getBlockType();
    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    LDBG("strideOneDim: " << strideOneDimVal);

    // Ensure the base ptr is 4-byte aligned.
    // Note: the HW requires the address to be 64-byte aligned, however we will
    // compensate by imposing restrictions on the offsetX and baseWidth.
    if (!llvm::all_of(allDescs, [&](tt::MakeTensorDescOp d) {
          return isDescriptorAligned(axisInfoAnalysis, d.getBase(), 4);
        })) {
      LDBG("Found non 4 bytes aligned base");
      return false;
    }

    // Analyze the shape of the stride one dimension to ensure it satisfies HW
    // constraints.
    Value baseWidth = tt::intel::getFinalValue(shape[strideOneDimVal]);
    unsigned divisor = llvm::divideCeil(32u, elementWidth);
    if (!ttgi::isDivisible(baseWidth, divisor)) {
      LLVM_DEBUG({
        llvm::dbgs() << "baseWidth does not satisfies HW constraint: ";
        baseWidth.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\ndivisor: " << divisor << "\n";
      });
      return false;
    }
    LDBG("baseWidth: " << baseWidth);

    // Analyze the load/store-time index in the stride-one dimension to ensure
    // it satisfies HW constraints.
    Value offset = tt::intel::getFinalValue(op.getIndices()[strideOneDimVal]);
    if (!ttgi::isDivisible(offset, divisor)) {
      LLVM_DEBUG({
        llvm::dbgs() << "descriptor index does not satisfy HW constraints: ";
        offset.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\ndivisor: " << divisor << "\n";
      });
      return false;
    }
    LDBG("offset: " << offset);

    return true;
  }
};

} // anonymous namespace
