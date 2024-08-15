#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Visitors.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUMATERIALIZEBLOCKPOINTER
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

bool isDivisible(Value v, unsigned divisor) {
  if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>())
    return (cast<IntegerAttr>(op.getValue()).getValue().getZExtValue() %
            divisor) == 0;

  if (auto op = v.getDefiningOp<mlir::arith::ExtSIOp>())
    return isDivisible(op.getOperand(), divisor);

  if (v.getParentBlock()->isEntryBlock() && isa<BlockArgument>(v)) {
    BlockArgument blockArg = cast<BlockArgument>(v);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto func = dyn_cast<tt::FuncOp>(parentOp)) {
      auto attr = func.getArgAttrOfType<IntegerAttr>(blockArg.getArgNumber(),
                                                     "tt.divisibility");
      return attr && attr.getValue().getZExtValue() % divisor == 0;
    }
    return false;
  }

  // in entryblock but not BlockArgument
  if (v.getParentBlock()->isEntryBlock() && (!isa<BlockArgument>(v)))
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);

  // in non-entryblock
  if (!v.getParentBlock()->isEntryBlock())
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);

  llvm::report_fatal_error("Operand of `MakeTensorPtrOp` is not the "
                           "function's argument or a constant value.");
  return false;
}

bool isConstantExp(Value v, unsigned expected) {
  if (v.getDefiningOp() == nullptr)
    return false;

  if (auto stride = dyn_cast<arith::ConstantOp>(v.getDefiningOp())) {
    if (auto strideInt = dyn_cast<IntegerAttr>(stride.getValue()))
      if (strideInt.getInt() == expected)
        return true;
  }

  return false;
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
            ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName()))
      return;

    MLIRContext *context = &getContext();
    mod.walk([context](tt::LoadOp loadOp) {
      Value ptr = loadOp.getPtr();
      if (!tt::isTensorPointerType(ptr.getType()))
        return;

      // the 2D load only work for dot operands.
      if (auto resultTy =
              dyn_cast<RankedTensorType>(loadOp.getResult().getType()))
        if (auto dotLayout =
                dyn_cast<ttg::DotOperandEncodingAttr>(resultTy.getEncoding())) {
          assert(isa<ttgi::DpasEncodingAttr>(dotLayout.getParent()) &&
                 "Expecting DpasEncodingAttr");

          tt::MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(ptr);
          auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
          auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
          ArrayRef<int32_t> order = makeTensorPtrOp.getOrder();
          unsigned rank = order.size();
          if (rank == 1)
            return;

          unsigned fastChangeDim = order[0];
          if (fastChangeDim >= (rank - 2)) {
            Operation::operand_range strides = makeTensorPtrOp.getStrides();

            // HW 2D block read instruction only supports contiguous access.
            Value fastChangeStride = strides[fastChangeDim];
            if (!isConstantExp(fastChangeStride, 1))
              return;

            // PVC requires pitch to be a multiple of QWord(64 bits).
            Value pitch =
                strides[(fastChangeDim == rank - 1) ? rank - 2 : rank - 1];
            if (!isDivisible(pitch, 64 / tensorType.getElementTypeBitWidth()))
              return;

            loadOp->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                            StringAttr::get(context, fastChangeDim == rank - 1
                                                         ? "row_major"
                                                         : "column_major"));
          }
        }
    });
  }
};

} // anonymous namespace
