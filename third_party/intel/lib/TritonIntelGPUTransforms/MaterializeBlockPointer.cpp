#include "intel/include/Analysis/AxisInfo.h"
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
    mod.walk([&](tt::DescriptorLoadOp op) {
      visitDescriptor(op, axisInfoAnalysis, context);
    });
    mod.walk([&](tt::DescriptorStoreOp op) {
      visitDescriptor(op, axisInfoAnalysis, context);
    });
  }

private:
  // Visit method for descriptor operations
  void visitDescriptor(tt::DescriptorLoadOp op,
                       tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                       MLIRContext *context) const {
    visitDescriptorImpl(op, op.getResult().getType(), axisInfoAnalysis,
                        context);
  }

  void visitDescriptor(tt::DescriptorStoreOp op,
                       tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                       MLIRContext *context) const {
    visitDescriptorImpl(op, op.getSrc().getType(), axisInfoAnalysis, context);
  }

  // Implementation for descriptor operations
  template <typename OpType>
  void visitDescriptorImpl(OpType op, Type resultType,
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

    RankedTensorType tensorType = dyn_cast<RankedTensorType>(resultType);
    if (!tensorType) {
      LDBG("Result type is not a RankedTensorType");
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
      const bool valueRowMajor = (dotOrder[0] == 1 && dotOrder[1] == 0);
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

    Value ptr = op.getPtr();
    if (!tt::isTensorPointerType(ptr.getType()))
      return MaterializeTensorOfPointers(op, axisInfoAnalysis, strideAnalysis);

    // Find the make tensor ptr operation that created the base ptr.
    std::optional<tt::MakeTensorPtrOp> defOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(ptr);
    if (!defOp) {
      LDBG("Could not find make tensor ptr op for: " << *op);
      return;
    }

    tt::MakeTensorPtrOp makeTensorPtrOp = *defOp;
    LDBG("Make tensor ptr op: " << makeTensorPtrOp);

    Operation::operand_range shape = makeTensorPtrOp.getShape();
    unsigned rank = shape.size();
    LDBG("Rank: " << rank);
    if (rank == 1)
      return;

    if (!satisfies2DBlockReadAlignment(op, axisInfoAnalysis)) {
      LDBG("Alignment checks failed for: " << *op);
      return;
    }

    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    LDBG("elementWidth: " << elementWidth);

    Operation::operand_range strides = makeTensorPtrOp.getStrides();
    std::optional<unsigned> strideOneDim = getStrideOneDim(makeTensorPtrOp);
    assert((strideOneDim && strideOneDim.value() < strides.size()) &&
           "Expected strideOneDim to be set and less than strides.size()");
    unsigned strideOneDimVal = strideOneDim.value();

    if (strideOneDimVal == rank - 2 && elementWidth == 8) {
      // TODO: column major layout w/ fp8 has performance regression
      return;
    }

    if (strideOneDimVal >= (rank - 2)) {
      // HW 2D block read instruction only supports contiguous access.
      Value fastChangeStride = strides[strideOneDimVal];
      if (!tt::intel::isConstant(fastChangeStride, 1))
        return;

      // Across Intel platforms, the strictest pitch restriction is to be a
      // multiple of OWord(128 bits).
      Value pitch =
          strides[(strideOneDimVal == rank - 1) ? rank - 2 : rank - 1];
      LDBG("Pitch: " << pitch);
      if (!ttgi::isDivisible(pitch, llvm::divideCeil(128, elementWidth)))
        return;

      const bool isRowMajor = (strideOneDimVal == rank - 1);
      std::optional<ttg::DotOperandEncodingAttr> dotLayout = getDotLayout(op);
      if (dotLayout) {
        // Check if the load is being used by a tt.dot operation, and if so is
        // this the first operand and is it a transposed row major matrix. If
        // so, skip the block ptr attribute as performance is worse than if we
        // remove the tensor pointer.
        LDBG("dotLayout: " << *dotLayout);
        auto opIdx =
            static_cast<ttgi::DpasEncodingAttr::OpIdx>(dotLayout->getOpIdx());
        auto dotOrder = tt::gpu::getThreadOrder(tensorType);
        const bool valueRowMajor = (dotOrder[0] == 1 && dotOrder[1] == 0);
        if (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA &&
            valueRowMajor ^ isRowMajor) {
          LDBG("Skipping block pointer attribute for transposed A matrix in "
               "dot operation");
          return;
        }
      }

      op->setAttr(
          ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
          StringAttr::get(context, isRowMajor ? "row_major" : "column_major"));
    }
  }

  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, tt::LoadOp, tt::StoreOp>::value>>
  void MaterializeTensorOfPointers(
      OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
      tt::intel::ModuleStrideAnalysis &strideAnalysis) const {
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

    // The axis info gives the information about the value of the indices
    // tensor. For example, if the indices tensor is tensor<8x16xi32> and
    // its value is:
    //   [[ 0,  1,  2,  3, ..., 12, 13, 14, 15],
    //    [16, 17, 18, 19, ..., 28, 29, 30, 31],
    //    ...
    //    [ 96,  97,  98,  99, ..., 108, 109, 110, 111],
    //    [112, 113, 114, 115, ..., 124, 125, 126, 127]]
    // Then the global memory refer by the tensor pointer is row-major
    // contiguous. And the axis info will be: stride: [16, 1],
    // contiguity: [1, 16], divisibility: [1, 16], constancy: [1, 1].
    const tt::AxisInfo *axisInfo = axisInfoAnalysis.getAxisInfo(ptr);
    unsigned rank = axisInfo->getRank();
    if (rank != 2) {
      LDBG("Rank is not 2, skip block IO attribute");
      return;
    }

    // Determine if LoadOp is row-major or column-major.
    tt::intel::StrideInfo *strideInfo = strideAnalysis.getStrideInfo(ptr);
    auto isMajor = [&strideInfo](RankedTensorType tensorTy,
                                 unsigned fastChangeDim,
                                 const tt::AxisInfo &axisInfo) {
      assert((fastChangeDim == 0 || fastChangeDim == 1) &&
             "fastChangeDim is expected to be 0 or 1");
      const unsigned otherDim = !fastChangeDim;
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
    const bool isRowMajor = isMajor(tensorTy, 1 /*fastChangeDim*/, *axisInfo);
    if (isRowMajor)
      op->setAttr(blockIOAttrName,
                  StringAttr::get(op.getContext(), "row_major"));

    const bool isColMajor = isMajor(tensorTy, 0 /*fastChangeDim*/, *axisInfo);
    if (isColMajor)
      op->setAttr(blockIOAttrName,
                  StringAttr::get(op.getContext(), "column_major"));
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

  std::optional<unsigned>
  getStrideOneDim(tt::MakeTensorPtrOp makeTensorPtrOp) const {
    assert(makeTensorPtrOp && "Expected a make tensor ptr op.");
    Operation::operand_range strides = makeTensorPtrOp.getStrides();
    std::optional<unsigned> strideOneDim{std::nullopt};
    for (auto [idx, stride] : llvm::enumerate(strides)) {
      if (!tt::intel::isConstant(stride, 1))
        continue;
      strideOneDim = idx;
      break;
    }
    return strideOneDim;
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

    // TODO: Support higher rank tensors in AxisInfo.
    if (rank > 2)
      return false;

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

  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, tt::LoadOp, tt::StoreOp>::value>>
  bool satisfies2DBlockReadAlignment(
      OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    Value ptr = op.getPtr();
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expected a ptr to a tensor of ptrs.");

    // Find the make tensor ptr operation that created the base ptr for the load
    // operation.
    std::optional<tt::MakeTensorPtrOp> defOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(ptr);
    assert(defOp && "Expected a make tensor ptr op.");
    tt::MakeTensorPtrOp makeTensorPtrOp = *defOp;
    Operation::operand_range shape = makeTensorPtrOp.getShape();
    if (shape.size() == 1)
      return false;

    std::optional<unsigned> strideOneDim = getStrideOneDim(makeTensorPtrOp);
    if (!strideOneDim) {
      LDBG("Could not find stride one dimension in: " << makeTensorPtrOp);
      return false;
    }

    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    unsigned strideOneDimVal = strideOneDim.value();
    LDBG("strideOneDim: " << strideOneDimVal);

    // Ensure the base ptr is 4-byte aligned.
    // Note: the HW requires the address to be 64-byte aligned, however we will
    // compensate by imposing restrictions on the offsetX and baseWidth.
    const tt::AxisInfo *axisInfo = axisInfoAnalysis.getAxisInfo(ptr);
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

    // Analyze the initial offset corresponding to the stride one dimension to
    // ensure it satisfies HW constraints.
    Value offset =
        tt::intel::getFinalValue(makeTensorPtrOp.getOffsets()[strideOneDimVal]);
    if (!ttgi::isDivisible(offset, divisor)) {
      LLVM_DEBUG({
        llvm::dbgs() << "offset does not satisfies HW constraints: ";
        offset.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\ndivisor: " << divisor << "\n";
      });
      return false;
    }
    LDBG("offset: " << offset);

    Region *loadRgn = op->getParentRegion();
    Region *makeTensorPtrRgn = makeTensorPtrOp->getParentRegion();
    bool inSameRegion = (loadRgn == makeTensorPtrRgn);
    if (inSameRegion)
      return satisfies2DBlockReadAlignment(offset, divisor);

    // TODO: analyze tt.advance (issue #3762).

    return true;
  }

  bool satisfies2DBlockReadAlignment(Value offset, unsigned divisor) const {
    assert(divisor != 0 && "Expected divisor to be non-zero");

    auto checkUsers = [&](Value::user_range users) {
      return llvm::all_of(users, [&](Operation *user) {
        if (isa<tt::MakeTensorPtrOp>(user))
          return true;
        if (Operation *addOp = dyn_cast<arith::AddIOp>(user)) {
          auto other = llvm::find_if(addOp->getOperands(),
                                     [&](Value op) { return op != offset; });
          if (!ttgi::isDivisible(*other, divisor)) {
            LDBG("Found a non-divisible increment: " << *addOp);
            return false;
          }
          return true;
        }
        LDBG("Unhandled user kind: " << user);
        return false;
      });
    };

    // Ensure that the offset is incremented by a multiple of the divisor.
    if (auto blockArg = dyn_cast<BlockArgument>(offset))
      return checkUsers(blockArg.getUsers());

    return true;
  }
};

} // anonymous namespace
