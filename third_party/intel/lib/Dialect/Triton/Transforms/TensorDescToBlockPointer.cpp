#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-tdesc-to-block-pointer"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELTENSORDESCTOBLOCKPOINTER
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

constexpr unsigned offsetBitwidth = 32u;
constexpr unsigned shapeAndStridesBitwidth = 64u;

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

// Returns the default blocked encoding for the given shape.
// Returns nullptr if TensorDescToBlockPointer pass is run before
// TritonToTritonGPU pass.
Attribute maybeGetDefaultBlockedEncoding(Operation *op,
                                         ArrayRef<int64_t> shape) {
  // numWarps is unavailable before TritonToTritonGPUPass, so tensor has no
  // encoding yet.
  if (!ttg::maybeLookupNumWarps(op))
    return Attribute();

  OpBuilder builder(op);
  return ttg::getDefaultBlockedEncoding(
      builder.getContext(), shape, ttg::lookupNumWarps(op),
      ttg::lookupThreadsPerWarp(builder), ttg::lookupNumCTAs(builder));
}

// Recursively collect all encodings from DescriptorLoadOp/DescriptorStoreOp
// users, following through loop arguments and other passthrough operations.
void collectEncodingsFromUsers(Value value, SmallPtrSetImpl<Value> &visited,
                               llvm::SetVector<Attribute> &encodings) {
  if (!visited.insert(value).second)
    return;

  for (Operation *user : value.getUsers()) {
    // Direct load/store user.
    if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(user)) {
      if (auto encoding =
              cast<RankedTensorType>(loadOp.getType()).getEncoding())
        encodings.insert(encoding);
      continue;
    }
    if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(user)) {
      if (auto encoding =
              cast<RankedTensorType>(storeOp.getSrc().getType()).getEncoding())
        encodings.insert(encoding);
      continue;
    }

    // Follow through loop arguments.
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [initArg, regionArg] :
           llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
        if (initArg == value)
          collectEncodingsFromUsers(regionArg, visited, encodings);
      }
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      for (auto [initArg, beforeArg] :
           llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments())) {
        if (initArg == value)
          collectEncodingsFromUsers(beforeArg, visited, encodings);
      }
      continue;
    }

    // Follow through scf.condition to while's "after" region args.
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      if (auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp())) {
        for (auto [condArg, afterArg] :
             llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments())) {
          if (condArg == value)
            collectEncodingsFromUsers(afterArg, visited, encodings);
        }
      }
      continue;
    }

    // Follow through yield to loop results and back to region iter args.
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
        for (auto [yieldedVal, result, regionArg] :
             llvm::zip(yieldOp.getOperands(), forOp.getResults(),
                       forOp.getRegionIterArgs())) {
          if (yieldedVal == value) {
            collectEncodingsFromUsers(result, visited, encodings);
            collectEncodingsFromUsers(regionArg, visited, encodings);
          }
        }
      } else if (auto whileOp =
                     dyn_cast<scf::WhileOp>(yieldOp->getParentOp())) {
        // Yield in while's "after" region goes back to "before" region args
        // and to loop results.
        for (auto [yieldedVal, result, beforeArg] :
             llvm::zip(yieldOp.getOperands(), whileOp.getResults(),
                       whileOp.getBeforeArguments())) {
          if (yieldedVal == value) {
            collectEncodingsFromUsers(result, visited, encodings);
            collectEncodingsFromUsers(beforeArg, visited, encodings);
          }
        }
      } else if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
        // Yield in if's then/else region goes to the if's results.
        for (auto [yieldedVal, result] :
             llvm::zip(yieldOp.getOperands(), ifOp.getResults())) {
          if (yieldedVal == value)
            collectEncodingsFromUsers(result, visited, encodings);
        }
      }
      continue;
    }

    // Follow through select op.
    if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
      collectEncodingsFromUsers(selectOp.getResult(), visited, encodings);
      continue;
    }
  }
}

// Find the single encoding from all users of a value.
// Asserts if multiple different encodings are found.
Attribute findEncodingFromUsers(Value value) {
  SmallPtrSet<Value, 8> visited;
  llvm::SetVector<Attribute> encodings;
  collectEncodingsFromUsers(value, visited, encodings);

  assert(encodings.size() <= 1 &&
         "MakeTensorDescOp with multiple encodings is not yet supported");
  return encodings.empty() ? Attribute() : encodings.front();
}

// Find and adjust encoding for a tensor descriptor operation.
// The encoding is found from DescriptorLoad/Store users. If the layout rank
// differs from the tensor rank (e.g., rank-reducing loads), the encoding is
// adjusted by adding batch dimensions with size 1.
Attribute findEncodingForTensorDesc(tt::MakeTensorDescOp op) {
  auto layout = dyn_cast_or_null<ttg::LayoutEncodingTrait>(
      findEncodingFromUsers(op.getResult()));
  if (!layout)
    return layout;

  ArrayRef<int64_t> blockShape = op.getType().getBlockType().getShape();
  unsigned tensorRank = blockShape.size();
  unsigned layoutRank = layout.getRank();
  if (layoutRank == tensorRank)
    return layout;

  assert(tensorRank > layoutRank &&
         "Expected tensor rank to be greater than layout rank");

  ttg::BlockedEncodingAttr blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout);
  if (blocked) {
    // Adjust BlockedEncodingAttr to match the tensor rank by adding
    // dimensions to the front (batch dimensions) with size 1.
    unsigned addCount = tensorRank - layoutRank;

    SmallVector<unsigned> newSizePerThread(addCount, 1);
    ArrayRef<unsigned> sizePerThread = blocked.getSizePerThread();
    newSizePerThread.append(sizePerThread.begin(), sizePerThread.end());

    SmallVector<unsigned> newThreadsPerWarp(addCount, 1);
    ArrayRef<unsigned> threadsPerWarp = blocked.getThreadsPerWarp();
    newThreadsPerWarp.append(threadsPerWarp.begin(), threadsPerWarp.end());

    SmallVector<unsigned> newWarpsPerCTA(addCount, 1);
    ArrayRef<unsigned> warpsPerCTA = blocked.getWarpsPerCTA();
    newWarpsPerCTA.append(warpsPerCTA.begin(), warpsPerCTA.end());

    SmallVector<unsigned> newOrder;
    ArrayRef<unsigned> order = blocked.getOrder();
    for (unsigned idx : order)
      newOrder.push_back(idx + addCount);
    for (int i = addCount - 1; i >= 0; --i)
      newOrder.push_back(i);

    // Extend CGALayout to the new rank by prepending identity layouts for
    // the new batch dimensions.
    ttg::CGAEncodingAttr cgaLayout = blocked.getCGALayout();
    tt::LinearLayout ll = cgaLayout.getLinearLayout();
    StringAttr kBlock = *ll.getInDimNames().begin();
    SmallVector<StringAttr> standardOuts =
        tt::standardOutDimNames(op.getContext(), tensorRank);
    for (unsigned i = layoutRank; i < tensorRank; ++i)
      ll = tt::LinearLayout::identity1D(1, kBlock, standardOuts[i]) * ll;
    // Rename out dims to dim0..dimn-1
    SmallVector<std::pair<StringAttr, int32_t>> dimSizes = ll.getOutDims();
    for (auto [i, dim] : llvm::enumerate(standardOuts))
      dimSizes[i].first = dim;
    ll = tt::LinearLayout(ll.getBases(), dimSizes, false);
    ttg::CGAEncodingAttr newCGALayout =
        ttg::CGAEncodingAttr::get(op.getContext(), std::move(ll));

    return ttg::BlockedEncodingAttr::get(op.getContext(), newSizePerThread,
                                         newThreadsPerWarp, newWarpsPerCTA,
                                         newOrder, newCGALayout);
  }

  // For other encoding types, fall back to default blocked encoding.
  return maybeGetDefaultBlockedEncoding(op, blockShape);
}

struct TritonIntelTensorDescToBlockPointer
    : tt::intel::impl::TritonIntelTensorDescToBlockPointerBase<
          TritonIntelTensorDescToBlockPointer> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp->walk<WalkOrder::PreOrder>([&](tt::DescriptorLoadOp loadOp) {
      // Retrieve the padding option from the MakeTensorDescOp.
      std::optional<tt::MakeTensorDescOp> makeTensorDescOp =
          tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
              loadOp.getDesc());
      assert(makeTensorDescOp.has_value() &&
             "Expecting to find the defining MakeTensorDescOp");
      OpToPaddingMap[loadOp] = makeTensorDescOp->getPadding();
      return WalkResult::advance();
    });

    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      assert(!isa<tt::DescriptorGatherOp>(op) &&
             !isa<tt::DescriptorScatterOp>(op) &&
             !isa<tt::DescriptorReduceOp>(op) &&
             "Expecting no gather/scatter/reduce ops at this stage");
      TypeSwitch<Operation *>(op)
          .Case<tt::MakeTensorDescOp>([&](auto makeTensorDescOp) {
            [[maybe_unused]] LogicalResult res =
                rewriteMakeTensorDescriptorOp(makeTensorDescOp);
            assert(succeeded(res) &&
                   "Failed to rewrite make_tensor_descriptor op");
          })
          .Case<tt::DescriptorLoadOp, tt::DescriptorStoreOp>(
              [&](auto loadOrStoreOp) {
                rewriteDescriptorLoadOrStoreOp(loadOrStoreOp);
              })
          .Default([&](auto) {});
      return WalkResult::advance();
    });

    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);

    LLVM_DEBUG(llvm::dbgs()
                   << "After TDesc to block_ptr: " << moduleOp << "\n";);
    moduleOp->walk([&](Operation *op) {
      assert(!hasATensorDescriptorType(op->getOperandTypes()) &&
             !hasATensorDescriptorType(op->getResultTypes()) &&
             "Expecting no tensor descriptor types after conversion");
    });
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  // Create a new block pointer if a suitable one doesn't already exist.
  // Otherwise, return the existing one. The function takes the base, shape,
  // strides, offsets, sizes of the block pointer to create/lookup and its
  // tensor element type (to ensure the block pointer has the tensor layout).
  tt::intel::MakeTensorPtrOp
  findOrCreateMakeTensorPtr(Location loc, Value base, ValueRange shape,
                            ValueRange strides, ValueRange offsets,
                            ArrayRef<int64_t> sizes, Attribute layout,
                            OpBuilder &builder) {
    Block *block = builder.getInsertionBlock();
    const Block::iterator insertPoint = builder.getInsertionPoint();
    auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
      if (auto makeTensorPtrOp = dyn_cast<tt::intel::MakeTensorPtrOp>(op)) {
        triton::PointerType resType = makeTensorPtrOp.getResult().getType();
        auto tensorType = cast<RankedTensorType>(resType.getPointeeType());
        auto sameShape = [](ArrayRef<int64_t> arr1, ArrayRef<int64_t> arr2) {
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
               sameShape(tensorType.getShape(), sizes) &&
               tensorType.getEncoding() == layout;
      }
      return false;
    });

    auto makeTensorPtrOp = [&]() {
      auto pointerType = cast<tt::PointerType>(base.getType());
      auto tensorType =
          RankedTensorType::get(sizes, pointerType.getPointeeType(), layout);
      auto tensorPtrType =
          tt::PointerType::get(tensorType, pointerType.getAddressSpace());
      auto makeTensorPtr = tt::intel::MakeTensorPtrOp::create(
          builder, loc, tensorPtrType, base, shape, strides, offsets,
          builder.getDenseI32ArrayAttr({1, 0}));
      return makeTensorPtr;
    };

    return (it != insertPoint) ? cast<tt::intel::MakeTensorPtrOp>(*it)
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
    SmallVector<int64_t> sizes;
    for (const auto [shape, stride, size] :
         llvm::zip(op.getShape(), op.getStrides(),
                   tDescType.getBlockType().getShape())) {
      shapes.push_back(tt::intel::findOrCreateCastOp(
          shape, builder.getIntegerType(shapeAndStridesBitwidth)));
      strides.push_back(tt::intel::findOrCreateCastOp(
          stride, builder.getIntegerType(shapeAndStridesBitwidth)));
      Value zero =
          tt::intel::findOrCreateIntConstant(loc, 0, offsetBitwidth, builder);
      offsets.push_back(zero);
      sizes.push_back(size);
    }

    Attribute layout = findEncodingForTensorDesc(op);
    auto tensorPtr = findOrCreateMakeTensorPtr(
        loc, op.getBase(), shapes, strides, offsets, sizes, layout, builder);
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
  void rewriteDescriptorLoadOrStoreOp(OpTy op) {
    assert(op && "Expecting a valid operation");

    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value operand = op.getOperand(0);
    assert(triton::isTensorPointerType(operand.getType()) &&
           "Expecting a block ptr");
    auto ptrType = cast<tt::PointerType>(operand.getType());
    auto descTensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    constexpr bool isLoad = std::is_same_v<OpTy, tt::DescriptorLoadOp>;
    RankedTensorType opTensorType;
    if constexpr (isLoad)
      opTensorType = cast<RankedTensorType>(op.getType());
    else
      opTensorType = cast<RankedTensorType>(op.getSrc().getType());

    assert((opTensorType.getShape() != descTensorType.getShape() ||
            opTensorType.getEncoding() == descTensorType.getEncoding()) &&
           "Expecting the same encoding");

    Value ptr = tt::intel::AdvanceOp::create(builder, loc, ptrType, operand,
                         op.getIndices());

    SmallVector<int32_t> boundaryCheck;
    for (size_t i = 0; i < descTensorType.getRank(); ++i)
      boundaryCheck.push_back(i);

    if constexpr (isLoad) {
      // Default to PAD_ZERO as this is the expected padding behavior for
      // descriptor loads. It should be specified in the tt.make_tensor_desc if
      // it is retrieved.
      triton::PaddingOption padding = triton::PaddingOption::PAD_ZERO;
      if (OpToPaddingMap.contains(op))
        padding = OpToPaddingMap[op];

      auto loadOp = tt::LoadOp::create(builder, loc, ptr, boundaryCheck,
                                       padding, op.getCache(), op.getEvict(),
                                       /*isVolatile*/ false);
      for (auto attr : op->getDiscardableAttrs())
        loadOp->setDiscardableAttr(attr.getName(), attr.getValue());

      if (descTensorType == opTensorType) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n");
        op.replaceAllUsesWith(loadOp.getResult());
      } else {
        // Note: the Triton combine pass might 'combine' a reshape op with the
        // descriptor load op by changing the result yielded by the descriptor
        // load op (see RankedReduceDescriptorLoads). In this case we need to
        // insert a reshape op to ensure the load op result has the expected
        // shape for subsequent operations.
        ArrayRef<int64_t> resShape = opTensorType.getShape();
        assert(descTensorType.getShape() != resShape &&
               "Expecting different shapes");
        assert(descTensorType.getElementType() ==
                   opTensorType.getElementType() &&
               "Expecting the same element type");
        auto reshapeOp = builder.createOrFold<tt::ReshapeOp>(
            loc, resShape, loadOp.getResult());
        LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n";
                   llvm::dbgs().indent(2) << reshapeOp << "\n");
        op.replaceAllUsesWith(reshapeOp);
      }
    } else {
      auto storeOp = tt::StoreOp::create(builder, loc, ptr, op.getSrc(),
                                         boundaryCheck, tt::CacheModifier::NONE,
                                         tt::EvictionPolicy::NORMAL);
      for (auto attr : op->getDiscardableAttrs())
        storeOp->setDiscardableAttr(attr.getName(), attr.getValue());
      LLVM_DEBUG(llvm::dbgs().indent(2) << storeOp << "\n");
    }

    cleanUp.insert(op);
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
  llvm::SmallDenseMap<Operation *, tt::PaddingOption, 8> OpToPaddingMap;
};

} // namespace
