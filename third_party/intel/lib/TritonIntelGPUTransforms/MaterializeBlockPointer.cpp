#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <numeric>

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
  if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto attr = dyn_cast<IntegerAttr>(op.getValue());
    return attr.getValue().getZExtValue() % divisor == 0;
  } else if (auto op = v.getDefiningOp<mlir::arith::ExtSIOp>()) {
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else if (v.getParentBlock()->isEntryBlock() && isa<BlockArgument>(v)) {
    BlockArgument blockArg = cast<BlockArgument>(v);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto func = dyn_cast<tt::FuncOp>(parentOp)) {
      auto attr = func.getArgAttrOfType<IntegerAttr>(blockArg.getArgNumber(),
                                                     "tt.divisibility");
      return attr && attr.getValue().getZExtValue() % divisor == 0;
    }
    return false;
  } else if (v.getParentBlock()->isEntryBlock() && (!isa<BlockArgument>(v))) {
    // in entryblock but not BlockArgument
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else if (!v.getParentBlock()->isEntryBlock()) {
    // in non-entryblock
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else {
    llvm::report_fatal_error("Operand of `MakeTensorPtrOp` is not the "
                             "function's argument or a constant value.");
    return false;
  }
}

bool isConstantExp(Value v, unsigned expected) {

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
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    if (!m->hasAttr(ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName()))
      return;

    ModuleOp mod = getOperation();
    mod.walk([context](tt::LoadOp loadOp) {
      auto src = loadOp->getOperand(0);
      if (tt::isTensorPointerType(src.getType())) {
        // the 2D load only work for dot operands.
        if (auto resultTy =
                dyn_cast<RankedTensorType>(loadOp->getResult(0).getType()))
          if (auto dotLayout = dyn_cast<triton::gpu::DotOperandEncodingAttr>(
                  resultTy.getEncoding()))
            if (auto dpasLayout =
                    dyn_cast<triton::gpu::intel::DpasEncodingAttr>(
                        dotLayout.getParent())) {
              auto makeTensorPtrOp = getMakeTensorPtrOp(src);
              auto ptrType =
                  cast<triton::PointerType>(makeTensorPtrOp.getType());
              auto tensorType =
                  cast<RankedTensorType>(ptrType.getPointeeType());

              Operation::operand_range strides = makeTensorPtrOp.getStrides();
              ArrayRef<int32_t> order = makeTensorPtrOp.getOrder();

              unsigned rank = order.size();
              if (rank == 1)
                return;

              unsigned fastChangeDim = order[0];
              if (fastChangeDim >= (rank - 2)) {
                // HW 2D block read instruction only supports contiguous
                // accessing.
                auto fastChangeStride = strides[fastChangeDim];
                if (!isConstantExp(fastChangeStride, 1))
                  return;

                // PVC requires pitch to be a multiple of QWord(64 bits).
                Value pitch =
                    strides[(fastChangeDim == rank - 1) ? rank - 2 : rank - 1];
                if (!isDivisible(pitch,
                                 64 / tensorType.getElementTypeBitWidth()))
                  return;

                loadOp->setAttr(
                    ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                    StringAttr::get(context, fastChangeDim == rank - 1
                                                 ? "row_major"
                                                 : "column_major"));
              }
            }
      }
    });
  }
};

} // anonymous namespace
