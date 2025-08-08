#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
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

  static Value getPointerFromOp(Operation *op) {
    return TypeSwitch<Operation *, Value>(op)
        .Case<tt::LoadOp, tt::StoreOp>([](auto op) { return op.getPtr(); })
        .Default([&](auto) {
          llvm_unreachable(
              +("Invalid operation: " + op->getName().getStringRef())
                   .str()
                   .c_str());
          return Value{};
        });
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName()))
      return;

    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    MLIRContext *context = &getContext();
    mod.walk([&](Operation *op) {
      if (!isa<tt::LoadOp, tt::StoreOp>(op)) {
        return;
      }
      LDBG("Considering op: " << *op);

      Value ptr = getPointerFromOp(op);
      if (!tt::isTensorPointerType(ptr.getType()))
        return MaterializeTensorOfPointers(op, axisInfoAnalysis);

      // Find the make tensor ptr operation that created the base ptr.
      std::optional<tt::MakeTensorPtrOp> defOp =
          tt::intel::findDefiningMakeTensorPtrOp(ptr);
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

        op->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                    StringAttr::get(context,
                                    isRowMajor ? "row_major" : "column_major"));
      }
    });
  }

private:
  void MaterializeTensorOfPointers(
      Operation *op,
      tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    MLIRContext *context = op->getContext();
    Value ptr = getPointerFromOp(op);
    assert(!tt::isTensorPointerType(ptr.getType()) &&
           "Expected pointer refer to a tensor.");

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return;

    LDBG("Considering tensor of pointer of memory accessing op: " << *op);

    if (auto loadOp = dyn_cast<tt::LoadOp>(*op)) {
      if (loadOp.getMask()) {
        LDBG("Load op has mask, skip block IO attribute");
        return;
      }
    }

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
    auto isMajor = [&](unsigned fastChangeDim) {
      assert((fastChangeDim == 0 || fastChangeDim == 1) &&
             "fastChangeDim is expected to be 0 or 1");
      const unsigned otherDim = !fastChangeDim;
      // Limit to full row being contiguous.
      if (axisInfo->getContiguity(fastChangeDim) !=
          tensorTy.getDimSize(fastChangeDim)) {
        LDBG("Found non-contiguous row: "
             << axisInfo->getContiguity(fastChangeDim));
        return false;
      }

      // Value -1 is used to represent the unknown stride.
      if (axisInfo->getStride(otherDim) < 0) {
        LDBG("Found unknown stride: " << axisInfo->getStride(otherDim));
        return false;
      }

      // Surface pitch is required to be 16 bytes aligned.
      Type elemTy =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned elemSizeInBytes = elemTy.getIntOrFloatBitWidth() / 8;
      if ((axisInfo->getStride(otherDim) * elemSizeInBytes) % 16 != 0) {
        LDBG("Found Non 16 bytes aligned stride: "
             << axisInfo->getStride(otherDim));
        return false;
      }

      // Base pointer can be compensate by the offset and base width, where they
      // each has restriction that it has to be 4 bytes aligned.
      if (axisInfo->getDivisibility(fastChangeDim) % 4 != 0) {
        LDBG(
            "Found Non 4 bytes aligned base: " << axisInfo->getDivisibility(1));
        return false;
      }

      return true;
    };

    // Check if loadOp is row major, i.e., fast changing dimension is one.
    if (isMajor(1 /*fastChangeDim*/)) {
      LDBG("Setting row_major attribute\n");
      op->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                  StringAttr::get(context, "row_major"));
    }

    // TODO: set column_major attribute
  }

  // Return the load layout if it is a dot layout. If it is not, check if the
  // load result is converted to a dot layout. If so, return the dot layout,
  // otherwise return nullopt.
  std::optional<ttg::DotOperandEncodingAttr> getDotLayout(Operation *op) const {
    Value ptr = getPointerFromOp(op);
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

  bool satisfies2DBlockReadAlignment(
      Operation *op,
      tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    Value ptr = getPointerFromOp(op);
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expected a ptr to a tensor of ptrs.");

    // Find the make tensor ptr operation that created the base ptr for the load
    // operation.
    std::optional<tt::MakeTensorPtrOp> defOp =
        tt::intel::findDefiningMakeTensorPtrOp(ptr);
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
