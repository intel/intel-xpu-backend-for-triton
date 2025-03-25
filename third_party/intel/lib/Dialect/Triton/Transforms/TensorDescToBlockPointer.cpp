#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-tdesc-to-block-pointer"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELTENSORDESCTOBLOCKPOINTER
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

constexpr unsigned offsetBitwidth = 32u;
constexpr unsigned shapeAndStridesBitwidth = 64u;

Value findOrCreateCast(Location loc, Value val, Type tgtType,
                       OpBuilder &builder) {
  if (val.getType() == tgtType)
    return val;

  Block *block = builder.getInsertionBlock();
  const Block::iterator insertPoint = builder.getInsertionPoint();

  auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
    if (auto castOp = dyn_cast<arith::ExtSIOp>(op))
      return castOp.getIn() == val && castOp.getType() == tgtType;
    return false;
  });

  return (it != insertPoint)
             ? cast<arith::ExtSIOp>(*it)
             : getValueOrCreateCastToIndexLike(builder, loc, tgtType, val);
}

struct TritonIntelTensorDescToBlockPointer
    : tt::intel::impl::TritonIntelTensorDescToBlockPointerBase<
          TritonIntelTensorDescToBlockPointer> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    // Version loops containing masked operation in canonical form.
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<tt::DescriptorLoadOp>([&](auto loadOp) {
            if (failed(rewriteDescriptorLoadOrStoreOp(loadOp)))
              loadOp->emitRemark(
                  "TritonIntelTensorDescToBlockPointer: Failed to rewrite "
                  "tensor descriptor load operation ");
            return WalkResult::advance();
          })
          .Case<tt::DescriptorStoreOp>([&](auto storeOp) {
            if (failed(rewriteDescriptorLoadOrStoreOp(storeOp)))
              storeOp->emitRemark(
                  "TritonIntelTensorDescToBlockPointer: Failed to "
                  "rewrite tensor descriptor store operation");
            return WalkResult::advance();
          })
          .Default([&](auto) { return WalkResult::advance(); });
    });

    finalize();
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  tt::MakeTensorDescOp getMakeTensorDescOp(Value base) const {
    assert(base && isa<tt::TensorDescType>(base.getType()) &&
           "Expecting tensor desc");

    Operation *defOp = base.getDefiningOp();
    if (!defOp) {
      BlockArgument blockArg = cast<BlockArgument>(base);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp)) {
        int numIVs = forOp.getNumInductionVars();
        int initArgIdx = blockArg.getArgNumber() - numIVs;
        auto initArgs = forOp.getInitArgs();
        assert(initArgIdx >= 0 && initArgIdx < initArgs.size() &&
               "Unexpected 'initArgIdx' value");
        return getMakeTensorDescOp(initArgs[initArgIdx]);
      }
      llvm_unreachable("TODO: Handle other operations with init arguments");
    }

    return cast<tt::MakeTensorDescOp>(defOp);
  }

  Value findOrCreateMakeTensorPtr(Location loc, Value base, ValueRange shape,
                                  ValueRange strides, ValueRange offsets,
                                  ArrayRef<int32_t> sizes, OpBuilder &builder) {
    Block *block = builder.getInsertionBlock();
    const Block::iterator insertPoint = builder.getInsertionPoint();

    auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
      if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        triton::PointerType resType = makeTensorPtrOp.getResult().getType();
        auto tensorType = cast<RankedTensorType>(resType.getPointeeType());
        auto sameShape = [](ArrayRef<int64_t> arr1, ArrayRef<int32_t> arr2) {
          for (auto [dim1, dim2] : llvm::zip(arr1, arr2)) {
            if (dim1 != dim2)
              return false;
          }
          return true;
        };
        return makeTensorPtrOp.getBase() == base &&
               makeTensorPtrOp.getShape() == shape &&
               makeTensorPtrOp.getStrides() == strides &&
               makeTensorPtrOp.getOffsets() == offsets &&
               sameShape(tensorType.getShape(), sizes);
      }
      return false;
    });

    return (it != insertPoint) ? cast<tt::MakeTensorPtrOp>(*it)
                               : builder.createOrFold<tt::MakeTensorPtrOp>(
                                     loc, base, shape, strides, offsets, sizes,
                                     builder.getDenseI32ArrayAttr({1, 0}));
  }

  void visitMakeTensorDescOp(tt::MakeTensorDescOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value ptr = op.getBase();
    OperandRange shape = op.getShape();
    OperandRange strides = op.getStrides();

    LLVM_DEBUG(llvm::dbgs() << "Visiting: " << *op << "\n");

    // Case 1: the ptr has been already been mapped.
    if (Value mappedV = ptrMap.lookupOrNull(ptr)) {
    }

    // Case 2: the ptr has not previously been mapped.

    // Create a new block pointer.
  }

  template <typename OpTy,
            std::enable_if_t<llvm::is_one_of<OpTy, tt::DescriptorLoadOp,
                                             tt::DescriptorStoreOp>::value,
                             bool> = true>
  LogicalResult rewriteDescriptorLoadOrStoreOp(OpTy op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    TypedValue<tt::TensorDescType> tDesc = op.getDesc();
    tt::TensorDescType tDescType = tDesc.getType();
    tt::MakeTensorDescOp makeTensorDescOp = getMakeTensorDescOp(tDesc);
    assert(makeTensorDescOp && "Expected a MakeTensorDescOp");

    LLVM_DEBUG({
      llvm::dbgs() << "Rewriting: " << op << "\n";
      llvm::dbgs() << "where tensor desc is: " << makeTensorDescOp << "\n";
    });

    // Create a new block pointer if a suitable one doesn't already exist.
    SmallVector<Value> shapes, strides, offsets;
    SmallVector<int32_t> sizes;
    for (const auto [shape, stride, offset, size] :
         llvm::zip(makeTensorDescOp.getShape(), makeTensorDescOp.getStrides(),
                   op.getIndices(), tDescType.getBlockType().getShape())) {
      shapes.push_back(findOrCreateCast(
          loc, shape, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      strides.push_back(findOrCreateCast(
          loc, stride, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      offsets.push_back(findOrCreateCast(
          loc, offset, builder.getIntegerType(offsetBitwidth), builder));
      sizes.push_back(static_cast<int32_t>(size));
    }

    Value makeTensorPtrOp =
        findOrCreateMakeTensorPtr(loc, makeTensorDescOp.getBase(), shapes,
                                  strides, offsets, sizes, builder);

    LLVM_DEBUG({
      llvm::dbgs() << "With:\n";
      llvm::dbgs().indent(2) << makeTensorPtrOp << "\n";
    });

    constexpr bool isLoad = std::is_same_v<OpTy, tt::DescriptorLoadOp>;
    if constexpr (isLoad) {
      auto newOp = builder.createOrFold<tt::LoadOp>(
          loc, makeTensorPtrOp, op.getCache(), op.getEvict(),
          /*volatile*/ false);
      LLVM_DEBUG(llvm::dbgs().indent(2) << newOp << "\n");
      op.replaceAllUsesWith(newOp);
    } else {
      [[maybe_unused]] auto storeOp = builder.createOrFold<tt::StoreOp>(
          loc, makeTensorPtrOp, op.getSrc(), tt::CacheModifier::NONE,
          tt::EvictionPolicy::NORMAL);
      LLVM_DEBUG(llvm::dbgs().indent(2) << storeOp << "\n");
    }

    cleanUp.insert(op);
    cleanUp.insert(makeTensorDescOp);

    return success();
  }

  void finalize() {
    // Cleanup unused operations.
    bool erasedOperation;
    do {
      erasedOperation = false;
      SmallPtrSet<Operation *, 8> erased;
      for (Operation *op : cleanUp) {
        if (!op->getUsers().empty() || !op->getRegions().empty())
          continue;

        erased.insert(op);
        op->erase();
        erasedOperation = true;
      }
      cleanUp.remove_if([&](Operation *op) { return erased.contains(op); });
    } while (erasedOperation);

    // Remove operations that contain a region.
    for (Operation *op : cleanUp) {
      if (!op->getUsers().empty())
        continue;
      op->erase();
    }
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
  IRMapping ptrMap;
};

} // namespace
