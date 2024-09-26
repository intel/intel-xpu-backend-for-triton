#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonintelgpu-materialize-block-pointer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
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
    mod.walk([context](tt::LoadOp loadOp) {
      LDBG("Considering op: " << loadOp);

      Value ptr = loadOp.getPtr();
      if (!tt::isTensorPointerType(ptr.getType()))
        return;

      assert(isa<RankedTensorType>(loadOp.getResult().getType()) &&
             "Expected 'loadOp' to load a tensor value.");

      tt::MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);
      LDBG("Found make tensor ptr op: " << makeTensorPtrOp);
      auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
      auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
      Operation::operand_range shape = makeTensorPtrOp.getShape();
      unsigned rank = shape.size();
      LDBG("Rank: " << rank);
      if (rank == 1)
        return;

      Operation::operand_range strides = makeTensorPtrOp.getStrides();
      int fastChangeDim = -1;
      for (size_t i = 0; i < strides.size(); ++i) {
        if (mlir::triton::gpu::intel::isConstant(strides[i], 1)) {
          fastChangeDim = i;
          break;
        }
      }
      LDBG("Fast change dim: " << fastChangeDim);
      if (fastChangeDim < 0) {
        return;
      }
      ArrayRef<int32_t> order = makeTensorPtrOp.getOrder();

      if (fastChangeDim >= (rank - 2)) {
        // HW 2D block read instruction only supports contiguous access.
        Value fastChangeStride = strides[fastChangeDim];
        LLVM_DEBUG({
          DBGS() << "fastChangeStride: ";
          fastChangeStride.print(llvm::dbgs());
          llvm::dbgs() << "\n";
        });
        if (!mlir::triton::gpu::intel::isConstant(fastChangeStride, 1))
          return;

        // Across Intel platforms, the strictest pitch restriction is to be a
        // multiple of OWord(128 bits).
        Value pitch =
            strides[(fastChangeDim == rank - 1) ? rank - 2 : rank - 1];
        LDBG("Pitch: " << pitch);
        if (!ttgi::isDivisible(pitch,
                               128 / tensorType.getElementTypeBitWidth()))
          return;

        loadOp->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                        StringAttr::get(context, fastChangeDim == rank - 1
                                                     ? "row_major"
                                                     : "column_major"));
      }
    });
  }
};

} // anonymous namespace
