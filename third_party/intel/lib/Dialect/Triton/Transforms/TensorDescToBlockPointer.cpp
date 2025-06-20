#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
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

    WalkResult res = moduleOp->walk<WalkOrder::PreOrder>([](Operation *op) {
      if (isa<tt::DescriptorGatherOp>(op) || isa<tt::DescriptorScatterOp>(op) ||
          isa<tt::DescriptorReduceOp>(op)) {
        op->emitRemark(
            "TritonIntelTensorDescToBlockPointer: Failed to rewrite");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "TritonIntelTensorDescToBlockPointer: Skipping module - "
                    "contains unsupported operations\n");
      return;
    }

    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      return TypeSwitch<Operation *, WalkResult>(op)
          .Case<tt::MakeTensorDescOp>([&](auto makeTensorDescOp) {
            if (failed(rewriteMakeTensorDescriptorOp(makeTensorDescOp)))
              makeTensorDescOp->emitRemark(
                  "TritonIntelTensorDescToBlockPointer: Failed to rewrite");
            return WalkResult::advance();
          })
          .Case<tt::DescriptorLoadOp, tt::DescriptorStoreOp>(
              [&](auto loadOrStoreOp) {
                if (failed(rewriteDescriptorLoadOrStoreOp(loadOrStoreOp)))
                  loadOrStoreOp->emitRemark(
                      "TritonIntelTensorDescToBlockPointer: Failed to rewrite");
                return WalkResult::advance();
              })
          .Default([&](auto) { return WalkResult::advance(); });
    });

    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);

    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  // Create a new block pointer if a suitable one doesn't already exist.
  // Otherwise, return the existing one. The function takes the base, shape,
  // strides, offsets, sizes of the block pointer to create/lookup and its
  // tensor element type (to ensure the block pointer has the tensor layout).
  tt::MakeTensorPtrOp
  findOrCreateMakeTensorPtr(Location loc, Value base, ValueRange shape,
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

    auto makeTensorPtrOp = [&]() {
      auto makeTensorPtr = builder.create<tt::MakeTensorPtrOp>(
          loc, base, shape, strides, offsets, sizes,
          builder.getDenseI32ArrayAttr({1, 0}));
      return makeTensorPtr;
    };

    return (it != insertPoint) ? cast<tt::MakeTensorPtrOp>(*it)
                               : makeTensorPtrOp();
  }

  void propagateToLoops(Operation *op) {
    auto loopOp = dyn_cast<LoopLikeOpInterface>(op);
    if (!loopOp)
      return;

    bool updated = false;
    for (auto [initArg, rgnInitArg, yieldVal, loopRes] :
         llvm::zip(loopOp.getInits(), loopOp.getRegionIterArgs(),
                   loopOp.getYieldedValues(), loopOp->getResults())) {
      Type initArgType = initArg.getType();
      Type rgnInitArgType = rgnInitArg.getType();
      assert(rgnInitArgType == loopRes.getType() &&
             rgnInitArgType == yieldVal.getType() && "Type mismatch");
      if (rgnInitArgType != initArgType) {
        rgnInitArg.setType(initArgType);
        yieldVal.setType(initArgType);
        loopRes.setType(initArgType);
        updated = true;
      }
    }
    if (!updated)
      return;

    // For while loops we also need to update the "after" region arguments.
    if (auto loopOp = dyn_cast<scf::WhileOp>(op)) {
      for (auto [initArg, rgnAfterArg] :
           llvm::zip(loopOp.getInits(), loopOp.getAfterArguments())) {
        Type initArgType = initArg.getType();
        if (rgnAfterArg.getType() != initArgType)
          rgnAfterArg.setType(initArgType);
      }
    }

    // Propagate the loop results to their users.
    for (Operation *user : loopOp->getUsers())
      propagateToLoops(user);
  }

  LogicalResult rewriteMakeTensorDescriptorOp(tt::MakeTensorDescOp op) {
    assert(op && "Expecting a valid operation");
    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    tt::TensorDescType tDescType = op.getType();

    // Create a new block pointer if a suitable one doesn't already exist.
    SmallVector<Value> shapes, strides, offsets;
    SmallVector<int32_t> sizes;
    for (const auto [shape, stride, size] :
         llvm::zip(op.getShape(), op.getStrides(),
                   tDescType.getBlockType().getShape())) {
      shapes.push_back(findOrCreateCast(
          loc, shape, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      strides.push_back(findOrCreateCast(
          loc, stride, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      Value zero =
          tt::intel::findOrCreateIntConstant(loc, 0, offsetBitwidth, builder);
      offsets.push_back(zero);
      sizes.push_back(static_cast<int32_t>(size));
    }

    auto tensorPtr = findOrCreateMakeTensorPtr(
        loc, op.getBase(), shapes, strides, offsets, sizes, builder);
    LLVM_DEBUG({
      llvm::dbgs() << "With:\n";
      llvm::dbgs().indent(2) << tensorPtr << "\n";
    });

    op->replaceAllUsesWith(tensorPtr);
    cleanUp.insert(op);

    // Propagate the `tensorPtr` type to loops init args, yielded values,
    // results, ... (if necessary).
    for (Operation *user : tensorPtr->getUsers())
      propagateToLoops(user);

    return success();
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
    Value ptr = op.getOperand(0);
    assert(triton::isTensorPointerType(ptr.getType()) &&
           "Expecting a block ptr");
    auto ptrType = cast<tt::PointerType>(ptr.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    ptr =
        builder.create<tt::AdvanceOp>(loc, ptr.getType(), ptr, op.getIndices());

    SmallVector<int32_t> boundaryCheck;
    for (size_t i = 0; i < tensorType.getRank(); ++i)
      boundaryCheck.push_back(i);

    constexpr bool isLoad = std::is_same_v<OpTy, tt::DescriptorLoadOp>;
    if constexpr (isLoad) {
      auto loadOp = builder.createOrFold<tt::LoadOp>(
          loc, ptr, boundaryCheck,
          /*padding*/ std::nullopt, op.getCache(), op.getEvict(),
          /*volatile*/ false);
      LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n");
      op.replaceAllUsesWith(loadOp);
    } else {
      [[maybe_unused]] auto storeOp = builder.createOrFold<tt::StoreOp>(
          loc, ptr, op.getSrc(), boundaryCheck, tt::CacheModifier::NONE,
          tt::EvictionPolicy::NORMAL);
      LLVM_DEBUG(llvm::dbgs().indent(2) << storeOp << "\n");
    }

    cleanUp.insert(op);

    return success();
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
};

} // namespace
