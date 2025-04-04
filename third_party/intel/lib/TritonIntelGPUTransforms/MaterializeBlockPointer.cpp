#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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
#include "Analysis/AxisInfo.h"
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
            ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName()))
      return;

    mlir::triton::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    MLIRContext *context = &getContext();
    mod.walk([context, this, &axisInfoAnalysis](tt::LoadOp loadOp) {
      LDBG("Considering op: " << loadOp);

      Value ptr = loadOp.getPtr();
      if (!tt::isTensorPointerType(ptr.getType()))
        return MaterializeTensorOfPointers(loadOp, axisInfoAnalysis);

      assert(isa<RankedTensorType>(loadOp.getResult().getType()) &&
             "Expected 'loadOp' to load a tensor value.");

      tt::MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);
      LDBG("Found make tensor ptr op: " << makeTensorPtrOp);
      auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
      auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
      auto elementWidth = tensorType.getElementTypeBitWidth();
      LDBG("elementWidth: " << elementWidth);

      Operation::operand_range shape = makeTensorPtrOp.getShape();
      unsigned rank = shape.size();
      LDBG("Rank: " << rank);
      if (rank == 1)
        return;

      // We will compensate the offset of non-64 bytes aligned base to the
      // OffsetX and BaseWidth. The OffsetX and BaseWidth has extra restriction
      // that it has to be 4 bytes aligned.
      auto base = makeTensorPtrOp.getBase();
      if (!ttgi::isDivisible(base, 4)) {
        LDBG("Found Non 4 bytes aligned base: " << base);
        return;
      }

      Operation::operand_range strides = makeTensorPtrOp.getStrides();
      int fastChangeDim = -1;
      for (size_t i = 0; i < strides.size(); ++i) {
        if (tt::intel::isConstant(strides[i], 1)) {
          fastChangeDim = i;
          break;
        }
      }

      LDBG("Fast change dim: " << fastChangeDim);
      if (fastChangeDim < 0) {
        return;
      }

      // Check the BaseWidth.
      Value BaseWidth = shape[fastChangeDim];
      if (!ttgi::isDivisible(BaseWidth, std::ceil(32 / elementWidth))) {
        LDBG("Found Non 4 bytes aligned BaseWidth: " << BaseWidth);
        return;
      }

      // Check the OffsetX
      Operation::operand_range offsets = makeTensorPtrOp.getOffsets();
      Value OffsetX = offsets[fastChangeDim];
      if (!ttgi::isDivisible(OffsetX, std::ceil(32 / elementWidth))) {
        LDBG("Found Non 4 bytes aligned offsetX: " << OffsetX);
        return;
      }

      // TODO: Check the OffsetX from tl.advance

      if (fastChangeDim == rank - 2 && elementWidth == 8) {
        // TODO: column major layout w/ fp8 has performance regression
        return;
      }

      if (fastChangeDim >= (rank - 2)) {
        // HW 2D block read instruction only supports contiguous access.
        Value fastChangeStride = strides[fastChangeDim];
        LLVM_DEBUG({
          DBGS() << "fastChangeStride: ";
          fastChangeStride.print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        if (!tt::intel::isConstant(fastChangeStride, 1))
          return;

        // Across Intel platforms, the strictest pitch restriction is to be a
        // multiple of OWord(128 bits).
        Value pitch =
            strides[(fastChangeDim == rank - 1) ? rank - 2 : rank - 1];
        LDBG("Pitch: " << pitch);
        if (!ttgi::isDivisible(pitch, 128 / elementWidth))
          return;

        const bool isRowMajor = fastChangeDim == rank - 1;
        std::optional<ttg::DotOperandEncodingAttr> dotLayout =
            getDotLayout(loadOp);
        if (dotLayout) {
          // Check if the load is being used by a tt.dot operation, and if so is
          // this the first operand and is it a transposed row major matrix. If
          // so, skip the block ptr attribute as performance is worse than if we
          // remove the tensor pointer.
          LDBG("dotLayout: " << *dotLayout);
          auto opIdx =
              static_cast<ttgi::DpasEncodingAttr::OpIdx>(dotLayout->getOpIdx());
          auto dotOrder = mlir::triton::gpu::getThreadOrder(tensorType);
          const bool valueRowMajor = (dotOrder[0] == 1 && dotOrder[1] == 0);
          if (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA &&
              valueRowMajor ^ isRowMajor) {
            LDBG("Skipping block pointer attribute for transposed A matrix in "
                 "dot operation");
            return;
          }
        }

        loadOp->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                        StringAttr::get(context, isRowMajor ? "row_major"
                                                            : "column_major"));
      }
    });
  }

private:
  void MaterializeTensorOfPointers(
      tt::LoadOp loadOp,
      mlir::triton::intel::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    MLIRContext *context = loadOp.getContext();
    Value ptr = loadOp.getPtr();
    assert(!tt::isTensorPointerType(ptr.getType()) &&
           "Expected 'loadOp' to load a tensor value.");

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return;

    LDBG("Considering tensor of pointer load op: " << loadOp);

    if (loadOp.getMask()) {
      LDBG("Load op has mask, skip block IO attribute");
      return;
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
    auto axisInfo = axisInfoAnalysis.getAxisInfo(ptr);
    unsigned rank = axisInfo->getRank();
    if (rank != 2) {
      LDBG("Rank is not 2, skip block IO attribute");
      return;
    }

    const bool isRowMajor =
        axisInfo->getContiguity(1) == tensorTy.getDimSize(1) &&
        axisInfo->getStride(0) > 0 && axisInfo->getDivisibility(1) % 4 == 0;
    if (isRowMajor) {
      LDBG("Setting row_major attribute\n");
      loadOp->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                      StringAttr::get(context, "row_major"));
    }
  }

  // Return the load layout if it is a dot layout. If it is not, check if the
  // load result is converted to a dot layout. If so, return the dot layout,
  // otherwise return nullopt.
  std::optional<ttg::DotOperandEncodingAttr>
  getDotLayout(tt::LoadOp loadOp) const {
    Value ptr = loadOp.getPtr();
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

    Operation::user_range users = loadOp->getUsers();
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
};

} // anonymous namespace
