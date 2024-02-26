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
namespace ttig = mlir::triton::gpu::intel;

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

  void runOnOperation() override {

    SmallVector<mlir::triton::MakeTensorPtrOp> makeTensorPtrWorkList;
    getOperation()->walk([&](mlir::triton::MakeTensorPtrOp op) {
      auto ptrType = op.getType().cast<triton::PointerType>();
      auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();

      auto base = op.getBase();
      auto shape = op.getShape();
      auto strides = op.getStrides();
      auto offsets = op.getOffsets();
      auto tensorShape = tensorType.getShape();

      auto fastChangeStride = strides.back();
      if (auto stride =
              dyn_cast<arith::ConstantOp>(fastChangeStride.getDefiningOp())) {
        if (auto strideInt = stride.getValue().dyn_cast<IntegerAttr>()) {
          if (strideInt.getInt() == 1) {
            makeTensorPtrWorkList.push_back(op);
          }
        }
      }
    });

    // map from loadOp to layout information.
    DenseMap<tt::LoadOp, TensorPointerInfo> loadOpWorkSet;
    for (auto &makeTensorOp : makeTensorPtrWorkList) {
      SetVector<Operation *> forwardSlice;
      auto filter = [](Operation *op) {
        // stop on the load ops.
        return !isa<tt::LoadOp>(op);
      };
      // The customized getForwardSlice from the mlir::getForwardSlice
      ::getForwardSlice(makeTensorOp.getOperation(), &forwardSlice, true,
                        {filter});

      for (auto &op : forwardSlice) {
        if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
          if (loadOpWorkSet.count(loadOp) == 0) {
            // Save information
            loadOpWorkSet[loadOp] = TensorPointerInfo();
          } else {
            // multiple memory mapping defined for the same loadOp.
            assert(false && "johnlu todo.");
          }
        }
      }
    }

    for (auto &iter : loadOpWorkSet) {
      tt::LoadOp &loadOp = iter.getFirst();
      OpBuilder builder(loadOp);
      auto newResult = builder.create<ttig::Load2DOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), loadOp.getPtr(),
          triton::PaddingOptionAttr::get(loadOp.getContext(),
                                         triton::PaddingOption::PAD_ZERO),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      loadOp->getResult(0).replaceAllUsesWith(newResult);
      loadOp.erase();
    }

    SmallVector<mlir::triton::StoreOp> storeOpWorkList;
    getOperation()->walk([&](mlir::triton::StoreOp op) {
      auto ptr = op.getPtr();
      if (auto makeTensorOp =
              dyn_cast<tt::MakeTensorPtrOp>(ptr.getDefiningOp())) {
        auto ptrType = makeTensorOp.getType().cast<triton::PointerType>();
        auto tensorType = ptrType.getPointeeType().cast<RankedTensorType>();

        auto base = makeTensorOp.getBase();
        auto shape = makeTensorOp.getShape();
        auto strides = makeTensorOp.getStrides();
        auto offsets = makeTensorOp.getOffsets();
        auto tensorShape = tensorType.getShape();

        auto fastChangeStride = strides.back();
        if (auto stride =
                dyn_cast<arith::ConstantOp>(fastChangeStride.getDefiningOp())) {
          if (auto strideInt = stride.getValue().dyn_cast<IntegerAttr>()) {
            if (strideInt.getInt() == 1) {
              storeOpWorkList.push_back(op);
            }
          }
        }
      }
    });

    for (auto &storeOp : storeOpWorkList) {
      OpBuilder builder(storeOp);
      auto newResult = builder.create<ttig::Store2DOp>(
          storeOp.getLoc(), storeOp.getPtr(), storeOp.getValue(),
          storeOp.getCache(), storeOp.getEvict());
      storeOp.erase();
    }
  }

private:
};

} // anonymous namespace

std::unique_ptr<Pass>
mlir::triton::gpu::intel::createTritonIntelGPUMaterializeBlockPointerPass() {
  return std::make_unique<TritonIntelGPUMaterializeBlockPointerPass>();
}
