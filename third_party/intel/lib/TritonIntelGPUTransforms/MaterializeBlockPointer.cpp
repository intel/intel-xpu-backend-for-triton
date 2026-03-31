#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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
    mod.walk([&](tt::StoreOp op) {
      visit(op, axisInfoAnalysis, strideAnalysis, context);
    });
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
    // Find the make tensor desc operation that created the descriptor.
    std::optional<tt::MakeTensorDescOp> defOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(desc);
    if (!defOp) {
      LDBG("Could not find make tensor desc op for: " << *op);
      return;
    }

    tt::MakeTensorDescOp makeTensorDescOp = *defOp;
    LDBG("Make tensor desc op: " << makeTensorDescOp);

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
    // multiple of OWord(128 bits).
    Value pitch = strides[rank - 2];
    LDBG("Pitch: " << pitch);
    if (!ttgi::isDivisible(pitch, llvm::divideCeil(128, elementWidth)))
      return;

    std::optional<ttg::DotOperandEncodingAttr> dotLayout = getDotLayout(op);
    if (dotLayout) {
      // Check if the load is being used by a tt.dot operation, and if so is
      // this the first operand and is it a transposed row major matrix. If
      // so, skip the block descriptor attribute as performance is worse than
      // if we remove the tensor descriptor.
      LDBG("dotLayout: " << *dotLayout);
      auto opIdx =
          static_cast<ttgi::DpasEncodingAttr::OpIdx>(dotLayout->getOpIdx());
      auto dotOrder = tt::gpu::getThreadOrder(tensorType);
      // Row-major means the last dim (rank-1) is the fastest-varying, i.e.,
      // it appears first in the thread order vector.
      const bool valueRowMajor =
          (dotOrder[0] == rank - 1 && dotOrder[1] == rank - 2);
      if (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA && !valueRowMajor) {
        LDBG("Skipping block descriptor attribute for transposed A matrix in "
             "dot operation");
        return;
      }
    }

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

    if constexpr (std::is_same_v<OpType, tt::LoadOp>) {
      if (op.getMask()) {
        LDBG("Load op has mask, skip block IO attribute");
        return;
      }
    }

    Value ptr = op.getPtr();
    assert(!tt::isTensorPointerType(ptr.getType()) &&
           "Expected pointer refer to a tensor.");

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return;

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
    unsigned rank = axisInfo->getRank();

    // For 1D StoreOps, try to detect strided access patterns and reshape
    // to 2D for block IO lowering.
    if constexpr (std::is_same_v<OpType, tt::StoreOp>) {
      if (rank == 1) {
        reshape1DStridedStore(op, tensorTy, context);
        return;
      }
    }

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

      // Value -1 is used to represent the unknown stride.
      int64_t otherDimStride =
          strideInfo ? strideInfo->getStride(otherDim) : -1;
      if (otherDimStride < 0) {
        LDBG("Found unknown stride: " << otherDimStride);
        return false;
      }

      // Surface pitch is required to be 16 bytes aligned.
      Type elemTy =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned elemSizeInBytes = elemTy.getIntOrFloatBitWidth() / 8;
      if ((otherDimStride * elemSizeInBytes) % 16 != 0) {
        LDBG("Found Non 16 bytes aligned stride: " << otherDimStride);
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

  /// Look through index_cast, extui, extsi, trunci wrappers to find the
  /// underlying arithmetic operation.
  static Value lookThroughCasts(Value val) {
    while (val) {
      if (auto castOp = val.getDefiningOp<arith::IndexCastOp>()) {
        val = castOp.getIn();
      } else if (auto extOp = val.getDefiningOp<arith::ExtUIOp>()) {
        val = extOp.getIn();
      } else if (auto extOp = val.getDefiningOp<arith::ExtSIOp>()) {
        val = extOp.getIn();
      } else if (auto truncOp = val.getDefiningOp<arith::TruncIOp>()) {
        val = truncOp.getIn();
      } else {
        break;
      }
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
      return makeRange.getStart() == 0 &&
             static_cast<int64_t>(makeRange.getEnd()) == expectedEnd;
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
          static_cast<int64_t>(makeRange.getEnd()) == expectedEnd) {
        // The splat offset must be a multiple of W for rem/div correctness.
        // We don't need to know the exact value — the Triton frontend always
        // generates program_id * XBLOCK where XBLOCK % W == 0 for these
        // patterns. We verify this structurally: the offset must be
        // muli(program_id, XBLOCK) where XBLOCK % W == 0.
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
    val = lookThroughCasts(val);
    if (!val)
      return std::nullopt;
    auto constOp = val.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return std::nullopt;
    // Scalar constant.
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().getSExtValue();
    // Splat tensor constant (e.g., arith.constant dense<32> : tensor<Nxi32>).
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat())
        return denseAttr.getSplatValue<APInt>().getSExtValue();
    }
    return std::nullopt;
  }

  /// Detect 1D tensor-of-pointers StoreOp with strided access pattern
  /// (address = base + remui(idx, W) + muli(divui(idx, W), S)) and reshape
  /// to 2D store with block IO attributes.
  ///
  /// This enables the existing StoreOpToBlockIOConversion lowering to use
  /// hardware 2D block store instructions for what would otherwise be
  /// scalar scatter writes.
  void reshape1DStridedStore(tt::StoreOp op, RankedTensorType ptrTensorTy,
                             MLIRContext *ctx) const {
    LDBG("Attempting 1D strided store reshape for: " << *op);

    // 1. Reject masked stores — we only handle unmasked or splat(true).
    if (Value mask = op.getMask()) {
      // Allow splat(true) masks — they are equivalent to no mask.
      bool isTrivialMask = false;
      if (auto splatOp = mask.getDefiningOp<tt::SplatOp>()) {
        if (auto constOp =
                splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
          if (auto boolAttr = dyn_cast<BoolAttr>(constOp.getValue()))
            isTrivialMask = boolAttr.getValue();
        }
      }
      if (!isTrivialMask) {
        LDBG("Store has non-trivial mask, skip 1D reshape");
        return;
      }
    }

    // 2. Trace pointer through tt.addptr to find the offset tensor.
    Value ptr = op.getPtr();
    auto addPtrOp = ptr.getDefiningOp<tt::AddPtrOp>();
    if (!addPtrOp) {
      LDBG("Pointer not defined by tt.addptr, skip 1D reshape");
      return;
    }
    Value offset = addPtrOp.getOffset();

    // 3. Pattern-match the offset for:
    //    arith.addi(arith.remui(idx, W), arith.muli(arith.divui(idx, W), S))
    //    Look through cast wrappers at each step.
    Value offsetUnwrapped = lookThroughCasts(offset);
    auto addIOp = offsetUnwrapped.getDefiningOp<arith::AddIOp>();
    if (!addIOp) {
      LDBG("Offset not defined by arith.addi, skip 1D reshape");
      return;
    }

    // Try both orderings of the addi operands: (remui, muli) or (muli, remui).
    arith::RemUIOp remOp = nullptr;
    arith::MulIOp mulOp = nullptr;
    for (int order = 0; order < 2; ++order) {
      Value lhs = lookThroughCasts(addIOp.getLhs());
      Value rhs = lookThroughCasts(addIOp.getRhs());
      if (order == 1)
        std::swap(lhs, rhs);
      remOp = lhs.getDefiningOp<arith::RemUIOp>();
      mulOp = rhs.getDefiningOp<arith::MulIOp>();
      if (remOp && mulOp)
        break;
    }
    if (!remOp || !mulOp) {
      LDBG("Could not match remui/muli pattern, skip 1D reshape");
      return;
    }

    // Extract W from remui(idx, W).
    Value remIdx = lookThroughCasts(remOp.getLhs());
    std::optional<int64_t> wVal = getConstantValue(remOp.getRhs());
    if (!wVal || *wVal <= 0) {
      LDBG("Could not extract constant W from remui, skip 1D reshape");
      return;
    }
    int64_t W = *wVal;

    // Match muli(divui(idx, W'), S) — extract divui and verify W' == W.
    Value mulLhs = lookThroughCasts(mulOp.getLhs());
    Value mulRhs = lookThroughCasts(mulOp.getRhs());

    // Try both orderings of muli operands: (divui, S) or (S, divui).
    arith::DivUIOp divOp = nullptr;
    std::optional<int64_t> sVal;
    divOp = mulLhs.getDefiningOp<arith::DivUIOp>();
    if (divOp) {
      sVal = getConstantValue(mulRhs);
    } else {
      divOp = mulRhs.getDefiningOp<arith::DivUIOp>();
      sVal = getConstantValue(mulLhs);
    }
    if (!divOp || !sVal || *sVal <= 0) {
      LDBG("Could not match divui/constant in muli, skip 1D reshape");
      return;
    }
    int64_t S = *sVal;

    // Verify divui uses the same index and same W constant as remui.
    Value divIdx = lookThroughCasts(divOp.getLhs());
    std::optional<int64_t> divWVal = getConstantValue(divOp.getRhs());
    if (!divWVal || *divWVal != W) {
      LDBG("divui W constant (" << (divWVal ? std::to_string(*divWVal) : "?")
                                << ") does not match remui W (" << W
                                << "), skip 1D reshape");
      return;
    }

    // Verify both remui and divui use the same index.
    if (remIdx != divIdx) {
      LDBG("remui and divui use different index values, skip 1D reshape");
      return;
    }

    // 4. Compute H = numElements / W and verify evenly divides.
    int64_t numElements = ptrTensorTy.getDimSize(0);

    // Verify idx is a canonical unit-stride linear index: tt.make_range(0, N).
    // Without this, a scaled or permuted index that happens to match the
    // algebraic shape would produce an incorrect reshape.
    if (!isCanonicalLinearIndex(remIdx, numElements, W)) {
      LDBG("Index is not a canonical linear index of length "
           << numElements << " with W=" << W);
      return;
    }
    if (numElements % W != 0) {
      LDBG("numElements (" << numElements << ") not divisible by W (" << W
                           << "), skip 1D reshape");
      return;
    }
    int64_t H = numElements / W;

    LDBG("Detected strided pattern: W=" << W << ", H=" << H << ", S=" << S);

    // 5. Create reshaped tensors: [N] -> [H, W].
    Location loc = op.getLoc();
    OpBuilder builder(op);
    SmallVector<int64_t> newShape = {H, W};

    // Use the ReshapeOp builder that infers the 2D encoding automatically.
    auto ptrReshape = tt::ReshapeOp::create(builder, loc, newShape, ptr,
                                            /*allowReorder=*/false);
    Value val = op.getValue();
    auto valReshape = tt::ReshapeOp::create(builder, loc, newShape, val,
                                            /*allowReorder=*/false);

    // 6. Create the new 2D store.
    auto newStore = tt::StoreOp::create(builder, loc, ptrReshape, valReshape,
                                        op.getCache(), op.getEvict());

    // Copy over boundaryCheck if non-empty.
    if (!op.getBoundaryCheck().empty())
      newStore->setAttr("boundaryCheck", op.getBoundaryCheckAttr());

    // Copy ignore_cta if present.
    if (op.getIgnoreCta())
      newStore->setAttr("ignore_cta", UnitAttr::get(ctx));

    // 7. Set block IO attributes on the new store.
    // Stride is validated positive at extraction; assert here so lowering
    // can trust the attribute without re-checking.
    assert(S > 0 && "stride must be positive");
    newStore->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                      StringAttr::get(ctx, "row_major"));
    newStore->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOStrideAttrName(),
                      IntegerAttr::get(IntegerType::get(ctx, 64), S));

    LDBG("Created 2D block store: " << *newStore);

    // 8. Erase the original 1D store.
    op.erase();
  }

  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, tt::DescriptorLoadOp, tt::DescriptorStoreOp>::value>>
  std::optional<ttg::DotOperandEncodingAttr> getDotLayout(OpType op) const {
    // Get the tensor type from the operation's result (load) or value (store)
    Type resultType;
    if constexpr (std::is_same_v<OpType, tt::DescriptorLoadOp>) {
      resultType = op.getResult().getType();
    } else {
      resultType = op.getSrc().getType();
    }
    RankedTensorType tensorType = ttgi::getRankedTensorType(resultType);
    if (!tensorType)
      return std::nullopt;

    auto dotLayout = ttgi::getDotEncoding(tensorType);
    if (dotLayout)
      return dotLayout;

    auto allUsersAreConvertOps = [](Operation::user_range users) {
      return llvm::all_of(users, [](Operation *user) {
        return isa<ttg::ConvertLayoutOp>(user);
      });
    };

    auto allUserHaveIdenticalLayout = [](Operation::user_range users) {
      Attribute firstUserLayout =
          cast<ttg::ConvertLayoutOp>(*users.begin()).getType().getEncoding();
      return llvm::all_of(users, [&firstUserLayout](Operation *user) {
        return firstUserLayout ==
               cast<ttg::ConvertLayoutOp>(user).getType().getEncoding();
      });
    };

    Operation::user_range users = op->getUsers();
    if (!users.empty() && allUsersAreConvertOps(users) &&
        allUserHaveIdenticalLayout(users)) {
      Attribute firstUserLayout =
          cast<ttg::ConvertLayoutOp>(*users.begin()).getType().getEncoding();
      if (isa<ttg::DotOperandEncodingAttr>(firstUserLayout))
        return dyn_cast<ttg::DotOperandEncodingAttr>(firstUserLayout);
      return std::nullopt;
    }

    return std::nullopt;
  }

  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, tt::DescriptorLoadOp, tt::DescriptorStoreOp>::value>>
  bool satisfies2DBlockReadAlignment(
      OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    Value desc = op.getDesc();

    // Find the make tensor desc operation that created the descriptor for the
    // load/store operation.
    std::optional<tt::MakeTensorDescOp> defOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(desc);
    assert(defOp && "Expected a make tensor desc op.");
    tt::MakeTensorDescOp makeTensorDescOp = *defOp;
    Operation::operand_range shape = makeTensorDescOp.getShape();
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
    const tt::AxisInfo *axisInfo = axisInfoAnalysis.getAxisInfo(desc);
    if (axisInfo->getDivisibility(strideOneDimVal) % 4 != 0) {
      LDBG("Found non 4 bytes aligned base: "
           << axisInfo->getDivisibility(strideOneDimVal));
      return false;
    }

    // Analyze the shape of the stride one dimension to ensure it satisfies HW
    // constraints.
    Value baseWidth = tt::intel::getFinalValue(shape[strideOneDimVal]);
    unsigned divisor = llvm::divideCeil(32, elementWidth);
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
