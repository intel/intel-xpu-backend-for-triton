#include "intel/include/Analysis/AxisInfo.h"
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
    MLIRContext *context = &getContext();
    mod.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(op))
        return visit(loadOp, axisInfoAnalysis, context);
      if (auto storeOp = dyn_cast<tt::StoreOp>(op))
        return visit(storeOp, axisInfoAnalysis, context);
      if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op))
        return visit(loadOp, axisInfoAnalysis, context);
      if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op))
        return visit(storeOp, axisInfoAnalysis, context);
    });
  }

private:
  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, tt::MakeTensorPtrOp, tt::MakeTensorDescOp>::value>>
  static RankedTensorType getRankedTensorType(OpType makePointerOp) {
    if constexpr (std::is_same_v<OpType, tt::MakeTensorPtrOp>) {
      auto ptrType = cast<tt::PointerType>(makePointerOp.getType());
      return cast<RankedTensorType>(ptrType.getPointeeType());
    }

    if constexpr (std::is_same_v<OpType, tt::MakeTensorDescOp>) {
      return makePointerOp.getType().getBlockType();
    }
  }

  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, tt::LoadOp, tt::StoreOp, tt::DescriptorLoadOp,
                tt::DescriptorStoreOp>::value>>
  void visit(OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
             MLIRContext *context) const {
    LDBG("Considering op: " << *op);

    if constexpr (llvm::is_one_of<OpType, tt::LoadOp, tt::StoreOp>::value) {
      Value ptr = op.getPtr();
      if (!tt::isTensorPointerType(ptr.getType()))
        return MaterializeTensorOfPointers(op, axisInfoAnalysis);
      else {
        return MaterializeStructedPointer(
            op, tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(ptr),
            axisInfoAnalysis);
      }
    }

    if constexpr (llvm::is_one_of<OpType, tt::DescriptorLoadOp,
                                  tt::DescriptorStoreOp>::value)
      return MaterializeStructedPointer(
          op,
          tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(op.getDesc()),
          axisInfoAnalysis);
  }

  template <typename MakePtrOpType>
  void MaterializeStructedPointer(
      Operation *memoryAccessOp, std::optional<MakePtrOpType> defOp,
      tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    // Find the make tensor ptr operation that created the base ptr.
    if (!defOp) {
      LDBG("Could not find make tensor ptr op for: " << *memoryAccessOp);
      return;
    }

    MakePtrOpType makePointerOp = *defOp;
    LDBG("Make tensor ptr op: " << makePointerOp);

    Operation::operand_range shape = makePointerOp.getShape();
    unsigned rank = shape.size();
    LDBG("Rank: " << rank);
    if (rank == 1)
      return;

    RankedTensorType tensorType = getRankedTensorType(makePointerOp);
    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    LDBG("elementWidth: " << elementWidth);

    if (!satisfies2DBlockReadAlignment(makePointerOp, elementWidth,
                                       memoryAccessOp, axisInfoAnalysis)) {
      LDBG("Alignment checks failed for: " << *memoryAccessOp);
      return;
    }

    Operation::operand_range strides = makePointerOp.getStrides();
    std::optional<unsigned> strideOneDim = getStrideOneDim(strides);
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
      std::optional<ttg::DotOperandEncodingAttr> dotLayout =
          getDotLayout(memoryAccessOp);
      if (dotLayout) {
        // Check if the load is being used by a tt.dot operation, and if so is
        // this the first operand and is it a transposed row major matrix. If
        // so, skip the block ptr attribute as performance is worse than if we
        // remove the tensor pointer.
        LDBG("dotLayout: " << *dotLayout);
        auto opIdx =
            static_cast<ttgi::DpasEncodingAttr::OpIdx>(dotLayout->getOpIdx());
        auto dotOrder =
            tt::gpu::getThreadOrder(*dotLayout, tensorType.getShape());
        const bool valueRowMajor =
            (dotOrder[rank - 2] == 1 && dotOrder[rank - 1] == 0);
        if (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA &&
            valueRowMajor ^ isRowMajor) {
          LDBG("Skipping block pointer attribute for transposed A matrix in "
               "dot operation");
          return;
        }
      }

      memoryAccessOp->setAttr(
          ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
          StringAttr::get(memoryAccessOp->getContext(),
                          isRowMajor ? "row_major" : "column_major"));
    }
  }

  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, tt::LoadOp, tt::StoreOp>::value>>
  void MaterializeTensorOfPointers(
      OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    if constexpr (std::is_same_v<OpType, tt::LoadOp>) {
#if 0
      if (op.getMask()) {
        LDBG("Load op has mask, skip block IO attribute");
        return;
      }
#endif
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
    auto isMajor = [rank](RankedTensorType tensorTy, unsigned fastChangeDim,
                          const tt::AxisInfo &axisInfo) {
      assert((fastChangeDim < rank) && "fastChangeDim must be less than rank");
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
      if (axisInfo.getStride(otherDim) < 0) {
        LDBG("Found unknown stride: " << axisInfo.getStride(otherDim));
        return false;
      }

      // Surface pitch is required to be 16 bytes aligned.
      Type elemTy =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned elemSizeInBytes = elemTy.getIntOrFloatBitWidth() / 8;
      if ((axisInfo.getStride(otherDim) * elemSizeInBytes) % 16 != 0) {
        LDBG("Found Non 16 bytes aligned stride: "
             << axisInfo.getStride(otherDim));
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
                  StringAttr::get(op.getContext(), "row_major"));

    const bool isColMajor =
        isMajor(tensorTy, rank - 2 /*fastChangeDim*/, *axisInfo);
    if (isColMajor)
      op->setAttr(blockIOAttrName,
                  StringAttr::get(op.getContext(), "column_major"));
  }

  // Return the load layout if it is a dot layout. If it is not, check if the
  // load result is converted to a dot layout. If so, return the dot layout,
  // otherwise return nullopt.
  std::optional<ttg::DotOperandEncodingAttr> getDotLayout(Operation *op) const {
    auto resultTypes = op->getResultTypes();
    if (resultTypes.size() == 0)
      return std::nullopt; // Store op;
    RankedTensorType tensorType = dyn_cast<RankedTensorType>(resultTypes[0]);
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

  std::optional<unsigned> static getStrideOneDim(
      const Operation::operand_range &strides) {
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
                OpType, tt::MakeTensorPtrOp, tt::MakeTensorDescOp>::value>>
  bool satisfies2DBlockReadAlignment(
      OpType makePointerOp, unsigned elementWidth, Operation *loadOp,
      tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {

    // Find the make tensor ptr operation that created the base ptr for the load
    // operation.
    Operation::operand_range shape = makePointerOp.getShape();
    if (shape.size() == 1)
      return false;

    Operation::operand_range strides = makePointerOp.getStrides();
    std::optional<unsigned> strideOneDim = getStrideOneDim(strides);
    if (!strideOneDim) {
      LDBG("Could not find stride one dimension in: " << makePointerOp);
      return false;
    }

    unsigned strideOneDimVal = strideOneDim.value();
    LDBG("strideOneDim: " << strideOneDimVal);

    // Ensure the base ptr is 4-byte aligned.
    // Note: the HW requires the address to be 64-byte aligned, however we will
    // compensate by imposing restrictions on the offsetX and baseWidth.
    const tt::AxisInfo *axisInfo = axisInfoAnalysis.getAxisInfo(makePointerOp);
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

    if constexpr (std::is_same_v<OpType, tt::MakeTensorPtrOp>) {
      // Analyze the initial offset corresponding to the stride one dimension to
      // ensure it satisfies HW constraints.
      Value offset =
          tt::intel::getFinalValue(makePointerOp.getOffsets()[strideOneDimVal]);

      if (!ttgi::isDivisible(offset, divisor)) {
        LLVM_DEBUG({
          llvm::dbgs() << "offset does not satisfies HW constraints: ";
          offset.printAsOperand(llvm::dbgs(), {});
          llvm::dbgs() << "\ndivisor: " << divisor << "\n";
        });
        return false;
      }
      LDBG("offset: " << offset);

      Region *loadRgn = loadOp->getParentRegion();
      Region *makeTensorPtrRgn = makePointerOp->getParentRegion();
      bool inSameRegion = (loadRgn == makeTensorPtrRgn);
      if (inSameRegion)
        return satisfies2DBlockReadAlignment(offset, divisor);

      // TODO: analyze tt.advance (issue #3762).
    }

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
