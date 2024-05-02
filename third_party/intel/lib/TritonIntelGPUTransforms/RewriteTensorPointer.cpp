//===- RewriteTensorPointer.cpp -----------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Utility.h"

#include <memory>
#include <stack>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir {
#define GEN_PASS_DEF_TRITONINTELGPUREWRITETENSORPOINTER
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir

namespace {

/// Check if given value is divisible by the divisor.
bool isDivisible(Value value, unsigned divisor) {
  // Case 1: Value is defined by a constant operation
  if (auto constantOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
    return integerAttr && integerAttr.getValue().getZExtValue() % divisor == 0;
  }

  // Case 2: Value is a block argument of the entry block
  if (value.getParentBlock()->isEntryBlock() && isa<BlockArgument>(value)) {
    BlockArgument blockArg = cast<BlockArgument>(value);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (auto funcOp = dyn_cast<tt::FuncOp>(parentOp)) {
      auto divisibilityAttr = funcOp.getArgAttrOfType<IntegerAttr>(
          blockArg.getArgNumber(), "tt.divisibility");
      return divisibilityAttr &&
             divisibilityAttr.getValue().getZExtValue() % divisor == 0;
    }
  }

  // Case 3: Value is defined by a sign extension operation
  if (auto extSIOp = value.getDefiningOp<mlir::arith::ExtSIOp>())
    return isDivisible(extSIOp->getOperand(0), divisor);

  return false;
}

/// Check if the tensor pointer should be removed. The tensor pointer should be
/// removed if:
///   - the device architecture is not PVC
///   - the tensor pointer does not have DotEncoding with DpasEncoding parent
///   - the tensor pointer pitch is not divisible by Qword bitwidth
///   - the tensor pointer is not contiguous on memory
bool shouldRemove(tt::MakeTensorPtrOp &op, ttgi::DeviceArch deviceArch) {
  // Non-PVC device should always remove the tensor pointer
  if (deviceArch != ttgi::DeviceArch::PVC)
    return true;

  auto ptrType = cast<tt::PointerType>(op.getType());
  auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

  if (!ttgi::hasDotDpasEncoding(tensorType))
    return true;

  TypedValue<triton::PointerType> base = op.getBase();
  Operation::operand_range shape = op.getShape();
  Operation::operand_range strides = op.getStrides();
  Operation::operand_range offsets = op.getOffsets();
  ArrayRef<int32_t> order = op.getOrder();
  ArrayRef<int64_t> tensorShape = tensorType.getShape();

  // TODO: support column-major tensor
  // HW 2D block read instruction has restriction on pitch divisibility
  if (strides.size() == 2) {
    auto pitch = strides[order[1]];
    // PVC requires pitch to be a multiple of QWord(64 bits).
    if (!isDivisible(pitch, 64 / tensorType.getElementTypeBitWidth()))
      return true;
  }

  // HW 2D block read instruction only supports contiguous accessing.
  auto fastChangeStride = strides[order[0]];
  if (auto stride =
          dyn_cast<arith::ConstantOp>(fastChangeStride.getDefiningOp())) {
    if (auto strideInt = dyn_cast<IntegerAttr>(stride.getValue()))
      return strideInt.getInt() != 1;
  }

  return true;
}

/// The `RewritedInfo` struct is used to store information about a rewritten
/// tensor pointer. It holds the base pointer, shape, strides, offsets, and
/// encoding of the tensor. This information is used later in the code to handle
/// the rewritten tensor pointer.
struct RewritedInfo {
  RewritedInfo() = default;

  RewritedInfo(Value base, const SmallVector<Value> &shape,
               const SmallVector<Value> &strides,
               const SmallVector<Value> &offsets,
               const ArrayRef<int64_t> &tensorShape, Attribute layout)
      : base(base), shape(shape), strides(strides), offsets(offsets),
        tensorShape(tensorShape), layout(layout) {
    assert(shape.size() == strides.size() && shape.size() == offsets.size() &&
           shape.size() == tensorShape.size() &&
           "Expecting tensor shape, offsets and strides have the same size");
  }

  unsigned int length() const { return shape.size(); }

  Value getOffset(unsigned i) const { return offsets[i]; }

  SmallVector<Value> getOffsets() const { return offsets; }

  void setOffset(unsigned i, Value newOffset) {
    offsets[i] = newOffset;
    cachedOffsetWithRange.clear();
  }

  void setOffsets(const SmallVector<Value> &newOffsets) {
    offsets = newOffsets;
    cachedOffsetWithRange.clear();
  }

  void setEncoding(Attribute newLayout) { layout = newLayout; }

  // Creates a tensor with the values [0, tensorShape[axis]) + offsets[axis]
  // broadcasted to N dimensions along axis (i.e. so that
  // result[.., <axis'th dim> i, ...] = offsets[axis] + i).
  Value getExpandedOffsetWithRange(OpBuilder &builder, Location loc,
                                   unsigned i) {
    if (cachedOffsetWithRange.count(i))
      return cachedOffsetWithRange.at(i);

    // Ultimately this will look like:
    //
    //   % base = create_range ... : tensor<N>
    //   %a0 = expand_dims %base   : tensor<M, 1>
    //   %a1 = broadcast %a0       : tensor<M, N>
    //   %b0 = expand_dims %a1     : tensor<M, N, 1>
    //   %b1 = broadcast %b1       : tensor<M, N, K>
    //   ...
    //
    // The final result has layout this->layout.  When we subtract a dim,
    // that's equivalent to taking a sliced layout, so e.g. the layout of
    // %a0/%a1 is a slice of %b0/%b1's layout.
    size_t rank = tensorShape.size();
    MLIRContext *ctx = loc.getContext();

    // This code is creating a vector of layout attributes for a tensor. If a
    // layout is provided, it sets the layout of each axis based on the layout
    // of the previous axis, starting from the last axis and moving towards the
    // first. If the current axis is the one to remove, it skips it and moves to
    // the previous axis.
    SmallVector<Attribute, 4> layouts(rank);
    if (layout) {
      layouts[rank - 1] = layout;
      size_t axisToRemove = rank - 1;
      for (int64_t k = rank - 2; k >= 0; --k) {
        if (axisToRemove == i)
          --axisToRemove;
        layouts[k] =
            ttg::SliceEncodingAttr::get(ctx, axisToRemove, layouts[k + 1]);
        --axisToRemove;
      }
    }

    // Add range
    auto indexI32RowType = RankedTensorType::get(
        {tensorShape[i]}, builder.getI32Type(), layouts[0]);
    auto indexRowType = RankedTensorType::get({tensorShape[i]},
                                              builder.getI64Type(), layouts[0]);
    Value splatOffset =
        builder.create<tt::SplatOp>(loc, indexRowType, offsets[i]);
    Value range = builder.create<tt::MakeRangeOp>(loc, indexI32RowType, 0,
                                                  tensorShape[i]);
    Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

    // Expand dimensions
    Value expandedResult =
        builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
    for (int j = 0; j < tensorShape.size(); ++j) {
      if (j == i)
        continue;
      expandedResult = builder.create<tt::ExpandDimsOp>(loc, expandedResult, j);
    }

    return cachedOffsetWithRange[i] = expandedResult;
  }

  Value generatePtr(OpBuilder &builder, const Location &loc) {
    assert(tensorShape.size() == offsets.size() &&
           tensorShape.size() == strides.size() &&
           "Expecting tensor shape, offsets and strides have the same size");
    auto indexTensorType =
        RankedTensorType::get(tensorShape, builder.getI64Type(), layout);
    auto ptrType = cast<tt::PointerType>(base.getType());
    auto ptrTensorType = RankedTensorType::get(tensorShape, ptrType, layout);

    // Generate offsets per dimension
    Value ptr = builder.create<tt::SplatOp>(loc, ptrTensorType, base);
    for (unsigned i = 0; i < tensorShape.size(); ++i) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // We must splat strides into the expanded shape not a row for retaining
      // the divisibility information given by strides
      Value splatStride = builder.create<tt::SplatOp>(
          loc, offsetWithRange.getType(), strides[i]);
      Value offsetWithStride =
          builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
      Value broadcasted = builder.create<tt::BroadcastOp>(loc, indexTensorType,
                                                          offsetWithStride);

      // Add to the pointer
      ptr = builder.create<tt::AddPtrOp>(loc, ptrTensorType, ptr, broadcasted);
    }

    return ptr;
  }

  Value generateMask(OpBuilder &builder, const Location &loc,
                     const std::optional<ArrayRef<int32_t>> &boundaryCheck) {
    if (!boundaryCheck.has_value())
      return {};

    // Generate mask per dimension
    auto maskTensorType =
        RankedTensorType::get(tensorShape, builder.getI1Type(), layout);
    Value mask;
    for (auto i : boundaryCheck.value()) {
      auto offsetWithRange = getExpandedOffsetWithRange(builder, loc, i);

      // Compare with lower bound
      Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
          loc, 0, builder.getI64Type());
      Value splatLowerBound = builder.create<tt::SplatOp>(
          loc, offsetWithRange.getType(), lowerBound);
      Value cmpLower = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

      // Compare with upper bound
      Value splatUpperBound =
          builder.create<tt::SplatOp>(loc, offsetWithRange.getType(), shape[i]);
      Value cmpUpper = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, offsetWithRange, splatUpperBound);

      // And and broadcast
      Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
      Value broadcasted =
          builder.create<tt::BroadcastOp>(loc, maskTensorType, andResult);

      // And up all results
      if (!mask) {
        mask = broadcasted;
      } else {
        mask = builder.create<arith::AndIOp>(loc, mask, broadcasted);
      }
    }

    return mask;
  }

  Value generateOther(OpBuilder &builder, const Location &loc,
                      const std::optional<tt::PaddingOption> &padding) const {
    if (!padding.has_value())
      return Value();

    // Create element attribute
    auto elementType = cast<tt::PointerType>(base.getType()).getPointeeType();
    auto otherTensorType =
        RankedTensorType::get(tensorShape, elementType, layout);

    // Set zero padding value
    TypedAttr attr =
        elementType.isIntOrIndex()
            ? cast<TypedAttr>(builder.getIntegerAttr(elementType, 0))
            : cast<TypedAttr>(builder.getFloatAttr(elementType, 0));

    // Float NaN padding case
    if (padding.value() == tt::PaddingOption::PAD_NAN) {
      assert(!elementType.isIntOrIndex() &&
             "Expect element type to be non-integer type");
      auto apNaN = llvm::APFloat::getNaN(
          cast<FloatAttr>(attr).getValue().getSemantics());
      attr = builder.getFloatAttr(elementType, apNaN);
    }

    // Create tensor
    Value constant = builder.create<arith::ConstantOp>(loc, attr);
    return builder.create<tt::SplatOp>(loc, otherTensorType, constant);
  }

private:
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  SmallVector<Value> offsets;
  ArrayRef<int64_t> tensorShape;
  Attribute layout;

  // A cache to avoid generating the same offset with range
  DenseMap<unsigned, Value> cachedOffsetWithRange;
};
} // namespace

// TODO: this pass relies on assumptions of how block pointers are created and
// on pattern matches that walks the SSA links to find the base/strides. This is
// very fragile and to solve we should expose convert Ptr of tensor to a
// structure contains all values and not only offsets.
class TritonIntelGPURewriteTensorPointerPass
    : public impl::TritonIntelGPURewriteTensorPointerBase<
          TritonIntelGPURewriteTensorPointerPass> {
private:
  DenseMap<Value, RewritedInfo> rewritedInfo;
  DenseSet<Value> valueToRemove;

public:
  using impl::TritonIntelGPURewriteTensorPointerBase<
      TritonIntelGPURewriteTensorPointerPass>::
      TritonIntelGPURewriteTensorPointerBase;

  static bool needRewrite(Operation *op, const DenseSet<Value> &valueToRemove) {
    return std::any_of(op->getOperands().begin(), op->getOperands().end(),
                       [&valueToRemove](Value operand) {
                         return tt::isTensorPointerType(operand.getType()) &&
                                valueToRemove.count(operand);
                       });
  }

  static SmallVector<Value>
  generateNewOperands(const SmallVector<Value> &oldOperands, unsigned index,
                      const SmallVector<Value> &newValues) {
    assert(index < oldOperands.size() && "Index out of range");
    SmallVector<Value> newOperands;
    for (int i = 0; i < index; ++i)
      newOperands.push_back(oldOperands[i]);
    for (auto value : newValues)
      newOperands.push_back(value);
    for (auto i = index + 1; i < oldOperands.size(); ++i)
      newOperands.push_back(oldOperands[i]);
    return newOperands;
  }

  Operation *rewriteMakeTensorPtrOp(OpBuilder &builder, tt::MakeTensorPtrOp op,
                                    std::stack<Operation *> &eraser) {
    if (!valueToRemove.count(op.getResult()))
      return nullptr;
    // Save info for later use
    auto ptrType = cast<tt::PointerType>(op.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    // Cast I32 offsets into I64
    SmallVector<Value> i64Offsets;
    for (auto offset : op.getOffsets()) {
      auto i64Offset = builder.create<arith::ExtSIOp>(
          op.getLoc(), builder.getI64Type(), offset);
      i64Offsets.push_back(i64Offset);
    }

    // Save information
    rewritedInfo[op.getResult()] =
        RewritedInfo(op.getBase(), op.getShape(), op.getStrides(), i64Offsets,
                     tensorType.getShape(), tensorType.getEncoding());

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteAdvanceOp(OpBuilder &builder, tt::AdvanceOp op,
                              std::stack<Operation *> &eraser) {
    if (!valueToRemove.count(op.getResult()))
      return nullptr;

    // Get info from previous results
    assert(rewritedInfo.count(op.getPtr()) &&
           "Expecting AdvanceOp ptr in rewritedInfo");
    auto info = rewritedInfo[op.getPtr()];

    // Calculate new offsets
    assert(info.length() == op.getOffsets().size() &&
           "Expecting AdvanceOp ptr shape and offsets have the same size");
    SmallVector<Value> newOffsets;
    for (int i = 0; i < info.length(); ++i) {
      Value i64Offset = builder.create<arith::ExtSIOp>(
          op.getLoc(), builder.getI64Type(), op.getOffsets()[i]);
      Value newOffset = builder.create<arith::AddIOp>(
          op.getLoc(), info.getOffset(i), i64Offset);
      newOffsets.push_back(newOffset);
    }

    // Save info for later use
    info.setOffsets(newOffsets);
    rewritedInfo[op.getResult()] = info;

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteLoadStoreOp(OpBuilder &builder, Operation *op,
                                std::stack<Operation *> &eraser) {
    assert(isa<tt::LoadOp>(op) ||
           isa<tt::StoreOp>(op) && "Expecting LoadOp or StoreOp");
    if (!valueToRemove.count(op->getOperand(0)))
      return nullptr;

    // Get info from previous results
    auto ptr = op->getOperand(0);
    assert(rewritedInfo.count(ptr) &&
           "Expecting LoadOp/StoreOp ptr in rewritedInfo");
    auto info = rewritedInfo[ptr];

    // Load/store with tensor pointers implicitly will check the bound while
    // accessing memory, so we should set `mask` and `other` (according to the
    // padding). Also note that load with tensor pointers do not have `mask` and
    // `other` while building IR from Python AST
    std::optional<ArrayRef<int>> boundaryCheck;
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(!loadOp.getMask() && !loadOp.getOther() &&
             "LoadOp with tensor pointer should not have mask and other");
      boundaryCheck = loadOp.getBoundaryCheck();
      if (auto valueType =
              dyn_cast<RankedTensorType>(loadOp.getResult().getType()))
        info.setEncoding(valueType.getEncoding());
    } else {
      auto storeOp = cast<tt::StoreOp>(op);
      assert(!storeOp.getMask() &&
             "StoreOp with tensor pointer should not have mask");
      boundaryCheck = storeOp.getBoundaryCheck();
      if (auto valueType =
              dyn_cast<RankedTensorType>(storeOp.getValue().getType()))
        info.setEncoding(valueType.getEncoding());
    }

    // Generate new `ptr`, `mask` and `other`
    auto newPtr = info.generatePtr(builder, op->getLoc());
    auto newMask = info.generateMask(builder, op->getLoc(), boundaryCheck);
    Value newOther;
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      newOther = info.generateOther(builder, op->getLoc(), loadOp.getPadding());

    // Create a new operation
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      auto newResult = builder.create<tt::LoadOp>(
          loadOp.getLoc(), newPtr, newMask, newOther, loadOp.getCache(),
          loadOp.getEvict(), loadOp.getIsVolatile());
      op->getResult(0).replaceAllUsesWith(newResult);
    } else if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
      builder.create<tt::StoreOp>(storeOp.getLoc(), newPtr, storeOp.getValue(),
                                  newMask, storeOp.getCache(),
                                  storeOp.getEvict());
    }

    // Erase the original operation
    eraser.push(op);
    return nullptr;
  }

  Operation *rewriteIfOp(OpBuilder &builder, scf::IfOp op,
                         std::stack<Operation *> &eraser) {
    auto thenYieldOp = op.thenYield();
    assert(op.getNumResults() == thenYieldOp.getNumOperands() &&
           "Expecting IfOp results and its thenYieldOp operands have the same "
           "number");
    SmallVector<Value> results = thenYieldOp.getOperands();

    // get new result types
    SmallVector<Type> newRetTypes;
    bool needRewrite = false;
    for (unsigned i = 0; i < results.size(); ++i) {
      if (!tt::isTensorPointerType(results[i].getType()) ||
          !valueToRemove.count(results[i])) {
        newRetTypes.push_back(results[i].getType());
        continue;
      }
      needRewrite = true;
      auto makeTensorPtrOp = getMakeTensorPtrOp(results[i]);
      assert(rewritedInfo.count(makeTensorPtrOp.getResult()) &&
             "Expecting MakeTensorPtrOp of IfOp result in rewritedInfo");
      auto info = rewritedInfo[makeTensorPtrOp.getResult()];
      for (unsigned j = 0; j < info.length(); ++j) {
        newRetTypes.push_back(builder.getI64Type());
      }
    }
    if (!needRewrite)
      return op;
    // create and clone new IfOp
    bool hasElse = !op.getElseRegion().empty();
    scf::IfOp newOp = builder.create<scf::IfOp>(op.getLoc(), newRetTypes,
                                                op.getCondition(), hasElse);
    IRMapping mapping;
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      mapping.map(op->getOperand(i), newOp->getOperand(i));
    }
    auto rematerialize = [&](Block *block) {
      for (Operation &opInIf : block->getOperations()) {
        auto newOp = builder.clone(opInIf, mapping);
      }
    };
    builder.setInsertionPointToStart(newOp.thenBlock());
    rematerialize(op.thenBlock());
    if (hasElse) {
      builder.setInsertionPointToStart(newOp.elseBlock());
      rematerialize(op.elseBlock());
    }

    // supported nested ops
    for (auto &[k, v] : mapping.getValueMap())
      if (valueToRemove.find(k) != valueToRemove.end())
        valueToRemove.insert(v);

    // update rewritedInfo
    unsigned oldResIdx = 0, newResIdx = 0;
    while (oldResIdx < results.size()) {
      if (!tt::isTensorPointerType(results[oldResIdx].getType()) ||
          !valueToRemove.count(results[oldResIdx])) {
        oldResIdx++;
        newResIdx++;
      } else {
        auto makeTensorPtrOp = getMakeTensorPtrOp(results[oldResIdx]);
        assert(rewritedInfo.count(makeTensorPtrOp.getResult()) &&
               "Expecting MakeTensorPtrOp of IfOp result in rewritedInfo");
        auto info = rewritedInfo[makeTensorPtrOp.getResult()];
        for (unsigned j = 0; j < info.length(); ++j) {
          info.setOffset(j, newOp->getResult(newResIdx++));
        }
        rewritedInfo[op.getResult(oldResIdx)] = info;
        oldResIdx++;
      }
    }

    eraser.push(op);
    return newOp;
  }

  Operation *rewriteForOp(OpBuilder &builder, scf::ForOp op,
                          std::stack<Operation *> &eraser) {
    // Generate new iteration operands and set rewrited information
    SmallVector<Value> oldIterOperands = llvm::to_vector(op.getInitArgs());
    SmallVector<Value> newIterOperands = llvm::to_vector(op.getInitArgs());
    for (unsigned i = 0, oldI = 0, size = op.getInitArgs().size(); i < size;
         ++i, ++oldI) {
      if (!tt::isTensorPointerType(newIterOperands[i].getType()))
        continue;
      if (!valueToRemove.count(newIterOperands[i]))
        continue;

      // Expand the tensor pointer into offsets
      assert(rewritedInfo.count(newIterOperands[i]) &&
             "Expecting ForOp operands in rewritedInfo");
      auto info = rewritedInfo[newIterOperands[i]];
      newIterOperands =
          generateNewOperands(newIterOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }

    // Rebuild the loop type
    auto newForOp = builder.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                               op.getUpperBound(), op.getStep(),
                                               newIterOperands);

    // Create value mapping. Note that for tensor pointers, we use identity
    // mapping. It may refer to a value in the old loop, but we will rewrite it
    // later
    IRMapping mapping;
    for (unsigned i = 0, oldI = 0, sz = op.getInitArgs().size(); oldI < sz;
         ++i, ++oldI) {
      auto oldRegionIterArg = op.getRegionIterArg(oldI);
      if (tt::isTensorPointerType(oldRegionIterArg.getType()) &&
          valueToRemove.count(oldIterOperands[oldI])) {
        // Pass rewrited info inside
        assert(rewritedInfo.count(oldIterOperands[oldI]) &&
               "Expecting ForOp operands in rewritedInfo");
        auto info = rewritedInfo[oldIterOperands[oldI]];
        mapping.map(oldRegionIterArg, oldRegionIterArg);
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getRegionIterArg(i + j));
        rewritedInfo[oldRegionIterArg] = info;
        i += info.length() - 1;
      } else {
        mapping.map(oldRegionIterArg, newForOp.getRegionIterArg(i));
      }
    }
    mapping.map(op.getInductionVar(), newForOp.getInductionVar());

    // Clone body
    builder.setInsertionPointToStart(newForOp.getBody());
    for (auto &opInFor : *op.getBody()) {
      auto *newOp = builder.clone(opInFor, mapping);
      for (unsigned i = 0; i < opInFor.getNumResults(); ++i) {
        if (valueToRemove.count(opInFor.getResult(i)))
          valueToRemove.insert(newOp->getResult(i));
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }

    // supported nested scf.for ops
    for (auto &[k, v] : mapping.getValueMap())
      if (valueToRemove.find(k) != valueToRemove.end())
        valueToRemove.insert(v);

    // Replace later usages
    assert(op.getNumResults() == op.getInitArgs().size() &&
           "Expecting ForOp results and operands have the same number");
    for (unsigned i = 0, oldI = 0; oldI < op.getNumResults(); ++i, ++oldI) {
      auto oldResult = op.getResult(oldI);
      if (tt::isTensorPointerType(oldResult.getType()) &&
          valueToRemove.count(oldIterOperands[oldI])) {
        // Pack new offsets into rewrited info
        assert(rewritedInfo.count(oldIterOperands[oldI]) &&
               "Expecting ForOp operands in rewritedInfo");
        auto info = rewritedInfo[oldIterOperands[oldI]];
        for (unsigned j = 0; j < info.length(); ++j)
          info.setOffset(j, newForOp.getResult(i + j));
        i += info.length() - 1;
        rewritedInfo[oldResult] = info;
      } else {
        oldResult.replaceAllUsesWith(newForOp.getResult(i));
      }
    }

    // Erase later
    eraser.push(op);
    return newForOp;
  }

  Operation *rewriteYieldOp(OpBuilder &builder, scf::YieldOp op,
                            std::stack<Operation *> &eraser) {
    // Replace tensor pointers with offsets
    SmallVector<Value> newOperands = op->getOperands();
    for (unsigned i = 0, size = op.getNumOperands(); i < size; ++i) {
      if (!tt::isTensorPointerType(newOperands[i].getType()))
        continue;
      if (!valueToRemove.count(newOperands[i]))
        continue;

      assert(rewritedInfo.count(newOperands[i]) &&
             "Expecting YieldOp operands in rewritedInfo");
      auto info = rewritedInfo[newOperands[i]];
      newOperands = generateNewOperands(newOperands, i, info.getOffsets());
      i += info.length() - 1;
      size += info.length() - 1;
    }
    op->setOperands(newOperands);

    // No need to erase
    return nullptr;
  }

  Operation *rewriteOp(Operation *op, std::stack<Operation *> &eraser) {
    OpBuilder builder(op);

    // Rewrite `make_tensor_ptr` and `advance` and make a tensor of pointers
    // Rewriting functions return the next operation to visit, if there is no
    // next one, simply return `nullptr`
    if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
      return rewriteMakeTensorPtrOp(builder, makeTensorPtrOp, eraser);
    } else if (auto advanceOp = dyn_cast<tt::AdvanceOp>(op)) {
      return rewriteAdvanceOp(builder, advanceOp, eraser);
    } else if (isa<tt::LoadOp>(op) || isa<tt::StoreOp>(op)) {
      return rewriteLoadStoreOp(builder, op, eraser);
    } else if (op->getDialect()->getNamespace() == "scf" ||
               op->getDialect()->getNamespace() == "cf") {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        return rewriteIfOp(builder, ifOp, eraser);
      }
      if (!needRewrite(op, valueToRemove))
        return op;

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        return rewriteForOp(builder, forOp, eraser);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        return rewriteYieldOp(builder, yieldOp, eraser);
      } else {
        llvm_unreachable("Currently we only support tensor pointer usages "
                         "inside a `scf::ForOp` or `scf::IfOp`, others such as "
                         "`scf::WhileOp`, `cf::BranchOp` or `cf::CondBranchOp` "
                         "are not supported yet");
      }
    }

    // Otherwise return the original one
    return op;
  }

  void visitOperation(Operation *op, std::stack<Operation *> &eraser) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        // We need an extra copy because erasing operations may break the
        // iterator behavior
        SmallVector<Operation *> blockCopy;
        for (auto &nestedOp : block)
          blockCopy.push_back(&nestedOp);

        // Rewrite and recursively visit
        for (auto &nestedOp : blockCopy) {
          if (auto newOp = rewriteOp(nestedOp, eraser))
            visitOperation(newOp, eraser);
        }
      }
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto markTensorPointerForRemoval = [this](Value val) {
      if (tt::isTensorPointerType(val.getType())) {
        tt::MakeTensorPtrOp makeTensorPtrOp = getMakeTensorPtrOp(val);
        if (shouldRemove(makeTensorPtrOp, deviceArch))
          valueToRemove.insert(val);
      }
    };

    mod.walk([&](Operation *op) {
      if (llvm::isa<tt::MakeTensorPtrOp>(op)) {
        markTensorPointerForRemoval(op->getResult(0));
      } else if (llvm::isa<tt::AdvanceOp, tt::LoadOp>(op)) {
        markTensorPointerForRemoval(op->getOperand(0));
      } else if (llvm::isa<tt::StoreOp>(op)) {
        // TODO: Block store should not be removed when 2d store is enabled
        auto src = op->getOperand(0);
        if (tt::isTensorPointerType(src.getType()))
          valueToRemove.insert(src);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        for (auto arg : forOp.getInitArgs())
          markTensorPointerForRemoval(arg);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        for (auto operand : yieldOp.getOperands())
          markTensorPointerForRemoval(operand);
      }
    });

    // NOTES(Chenggang): we don't use `ConversionPatternRewriter`, because
    // MLIR does not support one-multiple value mapping. For example, if we use
    // `ConversionPatternRewriter`, we can not make a type converter, which
    // converts `ptr<tensor>` into multiple types `ptr<>, int64, int64, ...`
    // (containing the base/offsets/strides...). What we can do is to convert
    // `ptr<tensor>` into a single type `Tuple<ptr<>, int64, int64, ...>`. But
    // in this way, we also have to define `PackTuple` and `UnpackTuple`
    // operations and make a canonicalization pass to optimize, which is much
    // So here we recursively build the IR, to be specific, we have to rewrite
    // `tt.make_tensor_ptr`, `tt.advance`, `tt.load`, `tt.store`,
    // `scf.for` (tensor pointer usages may be in a loop fashion)
    std::stack<Operation *> eraser;
    visitOperation(getOperation(), eraser);

    // The operation could not be erased during visit, because they may have
    // later usages, so we erase after visit
    rewritedInfo.clear();
    valueToRemove.clear();
    while (!eraser.empty()) {
      auto op = eraser.top();
      eraser.pop();
      op->erase();
    }
  }
};
