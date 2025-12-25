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
    mod.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<tt::LoadOp>(op))
        return visit(loadOp, axisInfoAnalysis, strideAnalysis, context);
      if (auto storeOp = dyn_cast<tt::StoreOp>(op))
        return visit(storeOp, axisInfoAnalysis, strideAnalysis, context);
      if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op))
        return visit(loadOp, axisInfoAnalysis, strideAnalysis, context);
      if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(op))
        return visit(storeOp, axisInfoAnalysis, strideAnalysis, context);
    });
  }

private:
  static RankedTensorType getMemoryAccessTensorType(Operation *memoryAccessOp) {
    Type resultType;
    auto resultTypes = memoryAccessOp->getResultTypes();
    if (resultTypes.empty()) {
      if (auto storeOp = dyn_cast<tt::StoreOp>(memoryAccessOp))
        resultType = storeOp.getValue().getType();
      else if (auto descStoreOp =
                   dyn_cast<tt::DescriptorStoreOp>(memoryAccessOp))
        resultType = descStoreOp.getSrc().getType();
    } else {
      resultType = resultTypes.front();
    }

    return ttgi::getRankedTensorType(resultType);
  }

  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, tt::LoadOp, tt::StoreOp, tt::DescriptorLoadOp,
                tt::DescriptorStoreOp>::value>>
  void visit(OpType op, tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
             tt::intel::ModuleStrideAnalysis &strideAnalysis,
             MLIRContext *context) const {
    LDBG("Considering op: " << *op);

    if constexpr (llvm::is_one_of<OpType, tt::LoadOp, tt::StoreOp>::value) {
      Value ptr = op.getPtr();
      if (!tt::isTensorPointerType(ptr.getType()))
        return MaterializeTensorOfPointers(op, axisInfoAnalysis,
                                           strideAnalysis);
      else {
        return MaterializeStructuredPointer(
            op, tt::intel::findDefiningOpOfType<tt::MakeTensorPtrOp>(ptr),
            axisInfoAnalysis);
      }
    }

    if constexpr (llvm::is_one_of<OpType, tt::DescriptorLoadOp,
                                  tt::DescriptorStoreOp>::value)
      return MaterializeStructuredPointer(
          op,
          tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(op.getDesc()),
          axisInfoAnalysis);
  }

  template <typename MakePtrOpType>
  void MaterializeStructuredPointer(
      Operation *memoryAccessOp, std::optional<MakePtrOpType> defOp,
      tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {
    // Find the make tensor pointer/descriptor operation that created the base
    // pointer.
    if (!defOp) {
      LDBG("Could not find make tensor pointer/descriptor op for: "
           << *memoryAccessOp);
      return;
    }

    MakePtrOpType makePointerOp = *defOp;
    LDBG("Make tensor pointer/descriptor op: " << makePointerOp);

    Operation::operand_range shape = makePointerOp.getShape();
    unsigned rank = shape.size();
    LDBG("Rank: " << rank);
    if (rank == 1)
      return;

    RankedTensorType memoryAccessTensorType =
        getMemoryAccessTensorType(memoryAccessOp);
    unsigned elementWidth = memoryAccessTensorType.getElementTypeBitWidth();
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
        auto dotOrder = tt::gpu::getThreadOrder(
            *dotLayout, memoryAccessTensorType.getShape());
        const bool valueRowMajor =
            (dotOrder[rank - 2] == 1 && dotOrder[rank - 1] == 0);
        if (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA &&
            valueRowMajor ^ isRowMajor) {
          LDBG("Skipping block pointer attribute for transposed A matrix in "
               "dot operation");
          return;
        }
      }

      if constexpr (std::is_same_v<MakePtrOpType, tt::MakeTensorDescOp>) {
        assert(isRowMajor && "tensor descriptor is expected to be row major");
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
                  StringAttr::get(op.getContext(), "row_major"));

    const bool isColMajor =
        isMajor(tensorTy, rank - 2 /*fastChangeDim*/, *axisInfo);
    if (isColMajor)
      op->setAttr(blockIOAttrName,
                  StringAttr::get(op.getContext(), "column_major"));
  }

  // Inspect the first result of the given operation: if its type has a dot
  // layout, return that layout. Otherwise, if the first result is converted
  // by all of its users to an identical dot layout, return that layout;
  // return nullopt for operations without results or without such a dot layout.
  std::optional<ttg::DotOperandEncodingAttr> getDotLayout(Operation *op) const {
    RankedTensorType tensorType = getMemoryAccessTensorType(op);
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
      OpType makePointerOp, unsigned elementWidth, Operation *memoryAccessOp,
      tt::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) const {

    // Find the make tensor ptr operation that created the base ptr for the load
    // operation.
    Operation::operand_range shape = makePointerOp.getShape();
    unsigned rank = shape.size();
    if (rank == 1)
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
    const tt::AxisInfo *axisInfo =
        axisInfoAnalysis.getAxisInfo(makePointerOp.getResult());
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

    Value offset;
    if constexpr (std::is_same_v<OpType, tt::MakeTensorPtrOp>) {
      offset =
          tt::intel::getFinalValue(makePointerOp.getOffsets()[strideOneDimVal]);
    }
    if constexpr (std::is_same_v<OpType, tt::MakeTensorDescOp>) {
      if (auto descLoadOp = dyn_cast<triton::DescriptorLoadOp>(memoryAccessOp))
        offset =
            tt::intel::getFinalValue(descLoadOp.getIndices()[strideOneDimVal]);
      if (auto descStoreOp =
              dyn_cast<triton::DescriptorStoreOp>(memoryAccessOp))
        offset =
            tt::intel::getFinalValue(descStoreOp.getIndices()[strideOneDimVal]);
    }
    assert(offset && "Expected to find offset operand");
    // Analyze the load/store-time index in the stride-one dimension to ensure
    // it satisfies HW constraints.
    if (!ttgi::isDivisible(offset, divisor)) {
      LLVM_DEBUG({
        llvm::dbgs() << "offset does not satisfies HW constraints: ";
        offset.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\ndivisor: " << divisor << "\n";
      });
      return false;
    }
    LDBG("offset: " << offset);

    if constexpr (std::is_same_v<OpType, tt::MakeTensorPtrOp>) {
      Region *loadRgn = memoryAccessOp->getParentRegion();
      Region *makeTensorPtrRgn = makePointerOp->getParentRegion();
      bool inSameRegion = (loadRgn == makeTensorPtrRgn);
      if (inSameRegion)
        return satisfies2DBlockReadAlignment(offset, divisor);
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
