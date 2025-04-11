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

    MLIRContext *context = &getContext();
    mod.walk([context, this](tt::LoadOp loadOp) {
      LDBG("Considering op: " << loadOp);

      Value ptr = loadOp.getPtr();
      if (!tt::isTensorPointerType(ptr.getType()))
        return;

      assert(isa<RankedTensorType>(loadOp.getResult().getType()) &&
             "Expected 'loadOp' to load a tensor value.");

      tt::MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);
      if (!makeTensorPtrOp) {
        LDBG("Could not find make tensor ptr op.");
        return;
      }
      LDBG("Found make tensor ptr op: " << makeTensorPtrOp);

      Operation::operand_range shape = makeTensorPtrOp.getShape();
      unsigned rank = shape.size();
      LDBG("Rank: " << rank);
      if (rank == 1)
        return;

      // Note: we will compensate the offset of non-64 bytes aligned base to the
      // offsetX and baseWidth.
      if (!satisfies2DBlockReadAlignment(loadOp)) {
        LDBG("Alignment checks failed for: " << loadOp);
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
        if (!ttgi::isDivisible(pitch, 128 / elementWidth))
          return;

        const bool isRowMajor = (strideOneDimVal == rank - 1);
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

  bool satisfies2DBlockReadAlignment(tt::LoadOp loadOp) const {
    Value ptr = loadOp.getPtr();
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expected a ptr to a tensor of ptrs.");
    assert(isa<RankedTensorType>(loadOp.getResult().getType()) &&
           "Expected 'loadOp' to load a ranked tensor value.");

    // Find the make tensor ptr operation that created the base ptr for the load
    // operation.
    tt::MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);
    assert(makeTensorPtrOp && "Expected a make tensor ptr op.");

    Operation::operand_range shape = makeTensorPtrOp.getShape();
    if (shape.size() == 1)
      return false;

    // The 2D block read function we generate is restricted to 4-byte aligned
    // pointers.
    // Ensure the base ptr is 4-byte aligned.
    TypedValue<tt::PointerType> base = makeTensorPtrOp.getBase();
    if (!ttgi::isDivisible(base, 4)) {
      LDBG("Found non 4-bytes aligned base: " << base);
      return false;
    }

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

    // Analyze the shape of the stride one dimension to ensure it satisfies HW
    // constraints.
    Value baseWidth = shape[strideOneDimVal];
    unsigned divisor = std::ceil(32 / elementWidth);
    if (!ttgi::isDivisible(baseWidth, divisor)) {
      LDBG("baseWidth does not satisfies HW constraint: " << baseWidth);
      return false;
    }
    LDBG("baseWidth: " << baseWidth);

    // Analyze the initial offset corresponding to the stride one dimension to
    // ensure it satisfies HW constraints.
    Value offset = makeTensorPtrOp.getOffsets()[strideOneDimVal];
    if (!ttgi::isDivisible(offset, divisor)) {
      LDBG("offset does not satisfies HW constraints: " << offset);
      return false;
    }
    LDBG("offset: " << offset);

    // TODO: analyze tt.advance (issue #3762).

    return true;
  }
};

} // anonymous namespace
