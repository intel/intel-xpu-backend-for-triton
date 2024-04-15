#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <numeric>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

static void
getForwardSliceImpl(Operation *op, Operation *def,
                    SetVector<Operation *> *forwardSlice, bool inclusive,
                    const SliceOptions::TransitiveFilter &filter = nullptr) {
  if (!op)
    return;

  // Evaluate whether we should keep this use.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive forwardSlice in the current scope.
  if (filter && !filter(op)) {
    if (inclusive)
      forwardSlice->insert(op);
    return;
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    for (BlockArgument &iterArg : forOp.getRegionIterArgs()) {
      auto initArg = forOp.getInitArgs()[iterArg.getArgNumber() - 1];
      if (initArg.getDefiningOp() == def) {
        for (Operation *iterUserOp : iterArg.getUsers()) {
          if (forwardSlice->count(iterUserOp) == 0)
            getForwardSliceImpl(iterUserOp, iterArg.getDefiningOp(),
                                forwardSlice, inclusive, filter);
        }
      }
    }
  } else {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &blockOp : block)
          if (forwardSlice->count(&blockOp) == 0)
            getForwardSliceImpl(&blockOp, nullptr, forwardSlice, inclusive,
                                filter);
  }
  for (Value result : op->getResults())
    for (Operation *userOp : result.getUsers())
      if (forwardSlice->count(userOp) == 0)
        getForwardSliceImpl(userOp, op, forwardSlice, inclusive, filter);

  forwardSlice->insert(op);
}

// inclusive: include the frontier in the slice.
static void getForwardSlice(Operation *op, SetVector<Operation *> *forwardSlice,
                            bool inclusive,
                            const ForwardSliceOptions &options = {}) {
  getForwardSliceImpl(op, nullptr, forwardSlice, inclusive, options.filter);
  if (!options.inclusive) {
    // Don't insert the top level operation, we just queried on it and don't
    // want it in the results.
    forwardSlice->remove(op);
  }

  // Reverse to get back the actual topological order.
  // std::reverse does not work out of the box on SetVector and I want an
  // in-place swap based thing (the real std::reverse, not the LLVM adapter).
  SmallVector<Operation *, 0> v(forwardSlice->takeVector());
  forwardSlice->insert(v.rbegin(), v.rend());
}

static bool isDivisible(Value v, unsigned divisor) {
  if (auto op = v.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto attr = op.getValue().dyn_cast<IntegerAttr>();
    return attr.getValue().getZExtValue() % divisor == 0;
  } else if (auto op = v.getDefiningOp<mlir::arith::ExtSIOp>()) {
    return isDivisible(v.getDefiningOp()->getOperand(0), divisor);
  } else if (v.getParentBlock()->isEntryBlock() && v.isa<BlockArgument>()) {
    BlockArgument blockArg = v.cast<BlockArgument>();
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto func = dyn_cast<tt::FuncOp>(parentOp)) {
      auto attr = func.getArgAttrOfType<IntegerAttr>(blockArg.getArgNumber(),
                                                     "tt.divisibility");
      return attr && attr.getValue().getZExtValue() % divisor == 0;
    }
    return false;
  } else if (v.getParentBlock()->isEntryBlock() && (!v.isa<BlockArgument>())) {
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

namespace {

/// An additional struct to record the meta information of tensor pointer
struct TensorPointerInfo {
private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tensorShape;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;

public:
  TensorPointerInfo() = default;

  TensorPointerInfo(const TensorPointerInfo &other) = default;

  TensorPointerInfo(Value base, const SmallVector<Value> &shape,
                    const SmallVector<Value> &strides,
                    const SmallVector<Value> &offsets,
                    const ArrayRef<int64_t> &tensorShape)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tensorShape(tensorShape) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tensorShape.size());
  }
};

struct TritonIntelGPUMaterializeBlockPointerPass
    : public TritonIntelGPUMaterializeBlockPointerBase<
          TritonIntelGPUMaterializeBlockPointerPass> {

public:
  TritonIntelGPUMaterializeBlockPointerPass() = default;
  TritonIntelGPUMaterializeBlockPointerPass(ttgi::DeviceArch arch) {
    this->deviceArch = arch;
  }

  void runOnOperation() override {
    //  Only the PVC has the 2D memory accessing
    if (deviceArch != ttgi::DeviceArch::PVC)
      return;

    ModuleOp mod = getOperation();
    SmallVector<tt::LoadOp> loadOps;
    mod.walk([&loadOps, this](tt::LoadOp _op) {
      auto src = _op->getOperand(0);
      // Only materialize the 2D load for dot operands.
      if (tt::isTensorPointerType(src.getType())) {
        if (auto resultTy = _op->getResult(0)
                                .getType()
                                .dyn_cast_or_null<RankedTensorType>())
          if (auto dotLayout =
                  resultTy.getEncoding()
                      .dyn_cast<triton::gpu::DotOperandEncodingAttr>())
            if (auto dpasLayout =
                    dotLayout.getParent()
                        .dyn_cast<triton::gpu::intel::DpasEncodingAttr>()) {
              auto makeTensorPtrOp = getMakeTensorPtrOp(src);
              auto ptrType =
                  makeTensorPtrOp.getType().cast<triton::PointerType>();
              auto tensorType =
                  ptrType.getPointeeType().cast<RankedTensorType>();

              auto base = makeTensorPtrOp.getBase();
              auto shape = makeTensorPtrOp.getShape();
              auto strides = makeTensorPtrOp.getStrides();
              auto offsets = makeTensorPtrOp.getOffsets();
              auto order = makeTensorPtrOp.getOrder();
              auto tensorShape = tensorType.getShape();

              auto pitchDivisible = false;
              if (strides.size() == 2) {
                auto pitch = strides[order[1]];
                // PVC requires pitch to be a multiple of QWord(64 bits).
                if (deviceArch == ttgi::DeviceArch::PVC) {
                  pitchDivisible = isDivisible(
                      pitch, 64 / tensorType.getElementTypeBitWidth());
                }
              }

              // HW 2D block read instruction only supports stride[-1]=1.
              auto fastChangeStride = strides.back();
              auto isContiguous = false;
              if (auto stride = dyn_cast<arith::ConstantOp>(
                      fastChangeStride.getDefiningOp())) {
                if (auto strideInt =
                        stride.getValue().dyn_cast<IntegerAttr>()) {
                  if (strideInt.getInt() == 1) {
                    isContiguous = true;
                  }
                }
              }

              if (pitchDivisible && isContiguous) {
                loadOps.push_back(_op);
              }
            }
      }
    });

    for (auto &loadOp : loadOps) {
      OpBuilder builder(loadOp);
      auto newResult = builder.create<ttgi::Load2DOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), loadOp.getPtr(),
          triton::PaddingOptionAttr::get(loadOp.getContext(),
                                         triton::PaddingOption::PAD_ZERO),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      loadOp->getResult(0).replaceAllUsesWith(newResult);
      loadOp.erase();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass>
mlir::triton::gpu::intel::createTritonIntelGPUMaterializeBlockPointerPass(
    ttgi::DeviceArch arch) {
  return std::make_unique<TritonIntelGPUMaterializeBlockPointerPass>(arch);
}
