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
  assert(isa<IntegerType>(tgtType) && isa<IntegerType>(val.getType()) &&
         "Expecting integer types");
  assert(val.getType().getIntOrFloatBitWidth() <=
             tgtType.getIntOrFloatBitWidth() &&
         "Expecting smaller type");

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

    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<tt::DescriptorLoadOp, tt::DescriptorStoreOp>(
              [&](auto loadOrStoreOp) {
                if (failed(rewriteDescriptorLoadOrStoreOp(loadOrStoreOp)))
                  loadOrStoreOp->emitRemark(
                      "TritonIntelTensorDescToBlockPointer: Failed to rewrite");
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
        unsigned numIVs = forOp.getNumInductionVars();
        int initArgIdx = blockArg.getArgNumber() - numIVs;
        if (isModifiedInLoop(forOp, blockArg)) {
          LLVM_DEBUG(llvm::dbgs() << blockArg << "is loop variant");
          return nullptr;
        }
        Operation::operand_range initArgs = forOp.getInitArgs();
        assert(initArgIdx >= 0 && initArgIdx < initArgs.size() &&
               "Unexpected 'initArgIdx' value");
        return getMakeTensorDescOp(initArgs[initArgIdx]);
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "TODO: Unhandled non operation: " << base << "\n");
      return nullptr;
    }

    if (defOp->getNumRegions() != 0) {
      LLVM_DEBUG(llvm::dbgs() << "TODO: defOp with region: " << *defOp << "\n");
      return nullptr;
    }
    if (auto makeTensorDescOp = dyn_cast<tt::MakeTensorDescOp>(defOp))
      return makeTensorDescOp;

    llvm_unreachable("TODO: Unhandled defOp kind");
    return nullptr;
  }

  bool isModifiedInLoop(scf::ForOp forOp, BlockArgument &blockArg) const {
    unsigned argNo = blockArg.getArgNumber();
    unsigned numIVs = forOp.getNumInductionVars();
    int initArgIdx = blockArg.getArgNumber() - numIVs;
    Value yieldedVal = forOp.getYieldedValues()[initArgIdx];
    return (yieldedVal != blockArg);
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

  template <typename OpTy,
            std::enable_if_t<llvm::is_one_of<OpTy, tt::DescriptorLoadOp,
                                             tt::DescriptorStoreOp>::value,
                             bool> = true>
  LogicalResult rewriteDescriptorLoadOrStoreOp(OpTy op) {
    assert(op && "Expecting a valid operation");
    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    TypedValue<tt::TensorDescType> tDesc = op.getDesc();
    tt::TensorDescType tDescType = tDesc.getType();
    tt::MakeTensorDescOp makeTensorDescOp = getMakeTensorDescOp(tDesc);

    if (!makeTensorDescOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "could not find tt.make_tensor_descriptor defining: "
                 << tDesc << "\n");
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "which has tdesc: " << makeTensorDescOp << "\n");

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
      SmallVector<int32_t> boundaryCheck;
      for (size_t i = 0; i < makeTensorDescOp.getShape().size(); ++i)
        boundaryCheck.push_back(i);
      auto loadOp = builder.createOrFold<tt::LoadOp>(
          loc, makeTensorPtrOp, boundaryCheck, /*padding*/ std::nullopt,
          op.getCache(), op.getEvict(),
          /*volatile*/ false);
      LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n");
      op.replaceAllUsesWith(loadOp);
    } else {
      SmallVector<int32_t> boundaryCheck;
      for (size_t i = 0; i < makeTensorDescOp.getShape().size(); ++i)
        boundaryCheck.push_back(i);
      [[maybe_unused]] auto storeOp = builder.createOrFold<tt::StoreOp>(
          loc, makeTensorPtrOp, op.getSrc(), boundaryCheck,
          tt::CacheModifier::NONE, tt::EvictionPolicy::NORMAL);
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
};

} // namespace
