#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
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

    if (!satisfies2DBlockReadAlignmentForDesc(op, axisInfoAnalysis)) {
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

    std::optional<ttg::DotOperandEncodingAttr> dotLayout =
        getDotLayoutForDesc(op);
    if (dotLayout) {
      // Check if the load is being used by a tt.dot operation, and if so is
      // this the first operand and is it a transposed row major matrix. If
      // so, skip the block descriptor attribute as performance is worse than
      // if we remove the tensor descriptor.
      LDBG("dotLayout: " << *dotLayout);
      auto opIdx =
          static_cast<ttgi::DpasEncodingAttr::OpIdx>(dotLayout->getOpIdx());
      auto dotOrder = tt::gpu::getThreadOrder(tensorType);
      const bool valueRowMajor =
          (dotOrder[rank - 2] == 1 && dotOrder[rank - 1] == 0);
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

  // Return the load layout if it is a dot layout. If it is not, check if the
  // load result is converted to a dot layout. If so, return the dot layout,
  // otherwise return nullopt.
  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, tt::LoadOp, tt::StoreOp>::value>>
  std::optional<ttg::DotOperandEncodingAttr> getDotLayout(OpType op) const {
    Value ptr = op.getPtr();
    if (!tt::isTensorPointerType(ptr.getType()))
      return std::nullopt;

    RankedTensorType tensorType = ttgi::getRankedTensorType(ptr.getType());
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
  std::optional<ttg::DotOperandEncodingAttr>
  getDotLayoutForDesc(OpType op) const {
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
  bool satisfies2DBlockReadAlignmentForDesc(
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
