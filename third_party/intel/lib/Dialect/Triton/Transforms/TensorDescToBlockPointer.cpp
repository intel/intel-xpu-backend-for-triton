#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
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
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#define DEBUG_TYPE "triton-intel-tdesc-to-block-pointer"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

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

// Collect all descriptor memory operations reachable from
// `value`, and record whether each load/store is reached through if/select
// control flow.
//
// A descriptor use is marked `true` in `users` if at least one path from the
// source descriptor to that use goes through if/select.
void collectDescriptorLoadStoreUsers(
    Value value, SmallPtrSetImpl<Value> &visitedNoIfSelect,
    SmallPtrSetImpl<Value> &visitedWithIfSelect,
    llvm::MapVector<Operation *, bool> &users, bool throughIfSelect = false) {
  SmallPtrSetImpl<Value> &visited =
      throughIfSelect ? visitedWithIfSelect : visitedNoIfSelect;
  if (!visited.insert(value).second)
    return;

  for (Operation *user : value.getUsers()) {
    if (isa<tt::DescriptorLoadOp, tt::DescriptorStoreOp,
            ttgi::DescriptorPrefetchOp>(user)) {
      users[user] = users.lookup(user) || throughIfSelect;
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [initArg, regionArg] :
           llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs()))
        if (initArg == value)
          collectDescriptorLoadStoreUsers(regionArg, visitedNoIfSelect,
                                          visitedWithIfSelect, users,
                                          throughIfSelect);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      for (auto [initArg, beforeArg] :
           llvm::zip(whileOp.getInits(), whileOp.getBeforeArguments()))
        if (initArg == value)
          collectDescriptorLoadStoreUsers(beforeArg, visitedNoIfSelect,
                                          visitedWithIfSelect, users,
                                          throughIfSelect);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      if (auto whileOp = dyn_cast<scf::WhileOp>(conditionOp->getParentOp()))
        for (auto [condArg, afterArg] :
             llvm::zip(conditionOp.getArgs(), whileOp.getAfterArguments()))
          if (condArg == value)
            collectDescriptorLoadStoreUsers(afterArg, visitedNoIfSelect,
                                            visitedWithIfSelect, users,
                                            throughIfSelect);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
        for (auto [yieldedVal, result, regionArg] :
             llvm::zip(yieldOp.getOperands(), forOp.getResults(),
                       forOp.getRegionIterArgs()))
          if (yieldedVal == value) {
            collectDescriptorLoadStoreUsers(result, visitedNoIfSelect,
                                            visitedWithIfSelect, users,
                                            throughIfSelect);
            collectDescriptorLoadStoreUsers(regionArg, visitedNoIfSelect,
                                            visitedWithIfSelect, users,
                                            throughIfSelect);
          }
      } else if (auto whileOp =
                     dyn_cast<scf::WhileOp>(yieldOp->getParentOp())) {
        for (auto [yieldedVal, result, beforeArg] :
             llvm::zip(yieldOp.getOperands(), whileOp.getResults(),
                       whileOp.getBeforeArguments()))
          if (yieldedVal == value) {
            collectDescriptorLoadStoreUsers(result, visitedNoIfSelect,
                                            visitedWithIfSelect, users,
                                            throughIfSelect);
            collectDescriptorLoadStoreUsers(beforeArg, visitedNoIfSelect,
                                            visitedWithIfSelect, users,
                                            throughIfSelect);
          }
      } else if (isa<scf::IfOp>(yieldOp->getParentOp())) {
        auto ifOp = cast<scf::IfOp>(yieldOp->getParentOp());
        for (auto [yieldedVal, result] :
             llvm::zip(yieldOp.getOperands(), ifOp.getResults()))
          if (yieldedVal == value)
            collectDescriptorLoadStoreUsers(result, visitedNoIfSelect,
                                            visitedWithIfSelect, users,
                                            /*throughIfSelect=*/true);
      }
      continue;
    }
    if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
      collectDescriptorLoadStoreUsers(selectOp.getResult(), visitedNoIfSelect,
                                      visitedWithIfSelect, users,
                                      /*throughIfSelect=*/true);
      continue;
    }
  }
}

// Represents the three possible ttig.block_io attribute values.
enum class BlockIOMode { None, RowMajor, ColumnMajor };

struct DescriptorUserInfo {
  Attribute encoding;
  BlockIOMode blockIOMode = BlockIOMode::None;
};

// For each MakeTensorDescOp whose load/store users have more than one distinct
// (encoding, blockIOMode) combination, clone the descriptor once per extra
// combination and redirect those users' descriptor operand (operand 0) to the
// appropriate clone. Afterwards every MakeTensorDescOp has at most one encoding
// and one access pattern (None/RowMajor/ColumnMajor) across all its users.
//
// Returns a map from each descriptor (originals and clones) to its
// DescriptorUserInfo (encoding + blockIOMode), so callers avoid re-traversal.
//
// For uses that flow through loops/yields (but not if/select) we replace
// operand 0 of the load/store directly. This is valid only for direct uses and
// loop-threaded uses because TensorDescType is immutable – the base pointer,
// shape and strides recorded in the descriptor do not change across loop
// iterations.
//
// For users reached through if/select operations, the descriptor is not cloned
// to avoid breaking control flow semantics. Such descriptors can only have a
// single encoding across all their users.
DenseMap<tt::MakeTensorDescOp, DescriptorUserInfo>
splitDescriptorsByUserInfo(ModuleOp moduleOp) {
  DenseMap<tt::MakeTensorDescOp, DescriptorUserInfo> descToInfo;

  SmallVector<tt::MakeTensorDescOp> descOps;
  moduleOp->walk([&](tt::MakeTensorDescOp op) { descOps.push_back(op); });

  for (tt::MakeTensorDescOp descOp : descOps) {
    // Collect all reachable descriptor load/store users in one traversal and
    // record whether each user is reached through if/select.
    llvm::MapVector<Operation *, bool> loadStoreUsers;
    SmallPtrSet<Value, 8> visitedNoIfSelect;
    SmallPtrSet<Value, 8> visitedWithIfSelect;
    collectDescriptorLoadStoreUsers(descOp.getResult(), visitedNoIfSelect,
                                    visitedWithIfSelect, loadStoreUsers);

    auto getBlockIOMode = [](Operation *userOp) {
      if (auto attr = userOp->getAttrOfType<StringAttr>(
              ttgi::TritonIntelGPUDialect::getBlockIOAttrName())) {
        if (auto mode = ttgi::symbolizeBlockIOMode(attr.getValue())) {
          if (*mode == ttgi::BlockIOMode::ColumnMajor)
            return BlockIOMode::ColumnMajor;
          if (*mode == ttgi::BlockIOMode::RowMajor)
            return BlockIOMode::RowMajor;
        }
      }
      return BlockIOMode::None;
    };

    // Group users by (encoding, blockIOMode) so each clone has a uniform
    // layout and access pattern.
    using UserGroupKey = std::pair<Attribute, BlockIOMode>;
    llvm::MapVector<UserGroupKey, SmallVector<Operation *>> groupedUsers;
    for (auto &[userOp, reachedThroughIfSelect] : loadStoreUsers) {
      (void)reachedThroughIfSelect;
      Attribute encoding;
      if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(userOp))
        encoding = cast<RankedTensorType>(loadOp.getType()).getEncoding();
      else if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(userOp))
        encoding =
            cast<RankedTensorType>(storeOp.getSrc().getType()).getEncoding();
      else if (auto prefetchOp = dyn_cast<ttgi::DescriptorPrefetchOp>(userOp))
        encoding = cast<tt::TensorDescType>(prefetchOp.getDesc().getType())
                       .getBlockType()
                       .getEncoding();
      BlockIOMode blockIOMode = getBlockIOMode(userOp);
      groupedUsers[{encoding, blockIOMode}].push_back(userOp);
    }

    if (groupedUsers.size() <= 1) {
      descToInfo[descOp] =
          groupedUsers.empty()
              ? DescriptorUserInfo{}
              : DescriptorUserInfo{groupedUsers.front().first.first,
                                   groupedUsers.front().first.second};
      continue;
    }

    // Multiple groups: keep the first on the original, clone for the rest.
    OpBuilder builder(descOp.getContext());
    builder.setInsertionPointAfter(descOp);
    bool isFirst = true;
    for (auto &[groupKey, ops] : groupedUsers) {
      DescriptorUserInfo info{groupKey.first, groupKey.second};
      if (isFirst) {
        descToInfo[descOp] = info;
        isFirst = false;
        continue;
      }

      // Check if any of these ops are only reachable through if/select.
      // If so, we cannot safely clone the descriptor.
      bool hasIfSelectUsers = false;
      for (Operation *op : ops) {
        if (loadStoreUsers.lookup(op)) {
          hasIfSelectUsers = true;
          break;
        }
      }
      assert(!hasIfSelectUsers && "FIXME: Support TensorDescOp with multiple "
                                  "encodings that flow through if/select.");

      auto clone = cast<tt::MakeTensorDescOp>(builder.clone(*descOp));
      descToInfo[clone] = info;
      for (Operation *userOp : ops) {
        if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(userOp))
          loadOp.getDescMutable().assign(clone.getResult());
        else if (auto storeOp = dyn_cast<tt::DescriptorStoreOp>(userOp))
          storeOp.getDescMutable().assign(clone.getResult());
        else if (auto prefetchOp = dyn_cast<ttgi::DescriptorPrefetchOp>(userOp))
          prefetchOp.getDescMutable().assign(clone.getResult());
      }
    }
  }

  return descToInfo;
}

// Adjust the encoding for a tensor descriptor operation.
// If the layout rank differs from the tensor rank (e.g., rank-reducing loads),
// the encoding is adjusted by adding batch dimensions with size 1.
Attribute findEncodingForTensorDesc(tt::MakeTensorDescOp op,
                                    Attribute encoding) {
  auto layout = dyn_cast_or_null<ttg::LayoutEncodingTrait>(encoding);
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

    // Ensure every MakeTensorDescOp has a single descriptor user info across
    // all its load/store users before the main rewriting walk, and record each
    // descriptor's user info to avoid re-traversing during rewriting.
    descToUserInfo = splitDescriptorsByUserInfo(moduleOp);

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
          .Case<ttgi::DescriptorPrefetchOp>(
              [&](auto prefetchOp) { rewriteDescriptorPrefetchOp(prefetchOp); })
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
  // Maps each MakeTensorDescOp to descriptor user information.
  // Populated by splitDescriptorsByUserInfo at the start of runOnOperation.
  DenseMap<tt::MakeTensorDescOp, DescriptorUserInfo> descToUserInfo;

  // Create a new block pointer if a suitable one doesn't already exist.
  // Otherwise, return the existing one. The function takes the base, shape,
  // strides, offsets, sizes of the block pointer to create/lookup and its
  // tensor element type (to ensure the block pointer has the tensor layout).
  // For column_major descriptors, the shape and strides are swapped.
  tt::MakeTensorPtrOp
  findOrCreateMakeTensorPtr(Location loc, Value base, ValueRange shape,
                            ValueRange strides, ValueRange offsets,
                            ArrayRef<int64_t> sizes, Attribute layout,
                            OpBuilder &builder, bool isColumnMajor) {
    // For column_major, reverse the tensor shape to match the transposed
    // interpretation used by block pointer lowering.
    SmallVector<int64_t> tensorPtrSizes(sizes.begin(), sizes.end());
    if (isColumnMajor)
      std::reverse(tensorPtrSizes.begin(), tensorPtrSizes.end());

    auto pointerType = cast<tt::PointerType>(base.getType());
    auto tensorType = RankedTensorType::get(
        tensorPtrSizes, pointerType.getPointeeType(), layout);
    auto tensorPtrType =
        tt::PointerType::get(tensorType, pointerType.getAddressSpace());

    return findOrCreateMakeTensorPtrWithType(loc, base, shape, strides, offsets,
                                             tensorPtrType, builder,
                                             isColumnMajor);
  }

  // Create or reuse a block pointer with the exact result type.
  // This is used when a MakeTensorDescOp result was already retagged to a
  // tensor-pointer type while propagating loop-carried types.
  tt::MakeTensorPtrOp
  findOrCreateMakeTensorPtrWithType(Location loc, Value base, ValueRange shape,
                                    ValueRange strides, ValueRange offsets,
                                    tt::PointerType resultType,
                                    OpBuilder &builder, bool isColumnMajor) {
    Block *block = builder.getInsertionBlock();
    const Block::iterator insertPoint = builder.getInsertionPoint();

    SmallVector<Value> adjustedShape(shape.begin(), shape.end());
    SmallVector<Value> adjustedStrides(strides.begin(), strides.end());
    if (isColumnMajor) {
      std::reverse(adjustedShape.begin(), adjustedShape.end());
      std::reverse(adjustedStrides.begin(), adjustedStrides.end());
    }

    auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
      if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        return makeTensorPtrOp.getResult().getType() == resultType &&
               makeTensorPtrOp.getBase() == base &&
               makeTensorPtrOp.getShape() == adjustedShape &&
               makeTensorPtrOp.getStrides() == adjustedStrides &&
               makeTensorPtrOp.getOffsets() == offsets;
      }
      return false;
    });

    auto makeTensorPtrOp = [&]() {
      auto order = isColumnMajor ? builder.getDenseI32ArrayAttr({0, 1})
                                 : builder.getDenseI32ArrayAttr({1, 0});
      return tt::MakeTensorPtrOp::create(builder, loc, resultType, base,
                                         adjustedShape, adjustedStrides,
                                         offsets, order);
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

    // Create a new block pointer if a suitable one doesn't already exist.
    SmallVector<Value> shapes, strides, offsets;
    SmallVector<int64_t> descSizes;
    Type resultType = op->getResult(0).getType();
    auto tDescType = dyn_cast<tt::TensorDescType>(resultType);
    ArrayRef<int64_t> blockShape;
    if (tDescType)
      blockShape = tDescType.getBlockType().getShape();

    unsigned dim = 0;
    for (const auto [shape, stride] :
         llvm::zip(op.getShape(), op.getStrides())) {
      shapes.push_back(tt::intel::findOrCreateCastOp(
          shape, builder.getIntegerType(shapeAndStridesBitwidth)));
      strides.push_back(tt::intel::findOrCreateCastOp(
          stride, builder.getIntegerType(shapeAndStridesBitwidth)));
      Value zero =
          tt::intel::findOrCreateIntConstant(loc, 0, offsetBitwidth, builder);
      offsets.push_back(zero);
      if (tDescType)
        descSizes.push_back(blockShape[dim]);
      ++dim;
    }

    auto userInfoIt = descToUserInfo.find(op);
    assert(userInfoIt != descToUserInfo.end() &&
           "Expected descriptor user info for MakeTensorDescOp");
    DescriptorUserInfo userInfo = userInfoIt->second;
    bool isColumnMajor = userInfo.blockIOMode == BlockIOMode::ColumnMajor;
    tt::MakeTensorPtrOp tensorPtr;

    if (tDescType) {
      Attribute layout = findEncodingForTensorDesc(op, userInfo.encoding);
      tensorPtr =
          findOrCreateMakeTensorPtr(loc, op.getBase(), shapes, strides, offsets,
                                    descSizes, layout, builder, isColumnMajor);
    } else {
      auto ptrType = dyn_cast<tt::PointerType>(resultType);
      assert(ptrType && triton::isTensorPointerType(resultType) &&
             "Expected TensorDescType or tensor pointer type");
      tensorPtr = findOrCreateMakeTensorPtrWithType(loc, op.getBase(), shapes,
                                                    strides, offsets, ptrType,
                                                    builder, isColumnMajor);
    }

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

    // For column_major loads, reverse the indices to match the swapped
    // shape/strides ordering in the block pointer.
    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    if (StringAttr blockIOAttr = op->template getAttrOfType<StringAttr>(
            ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
        blockIOAttr && ttgi::symbolizeBlockIOMode(blockIOAttr.getValue()) ==
                           ttgi::BlockIOMode::ColumnMajor) {
      assert(isLoad && "Expecting column_major access pattern only on loads");
      std::reverse(indices.begin(), indices.end());
    }

    Value ptr = tt::AdvanceOp::create(builder, loc, ptrType, operand, indices);

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

  void rewriteDescriptorPrefetchOp(ttgi::DescriptorPrefetchOp op) {
    assert(op && "Expecting a valid operation");

    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value operand = op.getOperand(0);
    assert(triton::isTensorPointerType(operand.getType()) &&
           "Expecting a block ptr");
    auto ptrType = cast<tt::PointerType>(operand.getType());

    SmallVector<Value> indices(op.getIndices().begin(), op.getIndices().end());
    if (StringAttr blockIOAttr = op->getAttrOfType<StringAttr>(
            ttgi::TritonIntelGPUDialect::getBlockIOAttrName());
        blockIOAttr && blockIOAttr.getValue() == "column_major") {
      std::reverse(indices.begin(), indices.end());
    }

    Value ptr = tt::AdvanceOp::create(builder, loc, ptrType, operand, indices);
    auto prefetchOp = ttgi::PrefetchOp::create(builder, loc, ptr, op.getCache(),
                                               op.getEvict(),
                                               /*isVolatile=*/false);
    for (auto attr : op->getDiscardableAttrs())
      prefetchOp->setDiscardableAttr(attr.getName(), attr.getValue());

    LLVM_DEBUG(llvm::dbgs().indent(2) << prefetchOp << "\n");
    cleanUp.insert(op);
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
  llvm::SmallDenseMap<Operation *, tt::PaddingOption, 8> OpToPaddingMap;
};

} // namespace
