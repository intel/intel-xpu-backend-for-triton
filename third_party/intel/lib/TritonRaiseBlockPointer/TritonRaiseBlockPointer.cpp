//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "intel/include/TritonRaiseBlockPointer/Passes.h"

#include "mlir/IR/Matchers.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-raise-block-pointer"

using namespace mlir;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONRAISEBLOCKPOINTER
#include "intel/include/TritonRaiseBlockPointer/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {
constexpr unsigned offsetBitwidth = 32;
constexpr unsigned shapeAndStridesBitwidth = 64;

// Data structure used to decode pointer arithmetics. Offsets, sizes, and
// strides are in unit of elements in a linearly laid-out memory, which is the
// same as pointer arithmetic operations in Triton language. Scalar is a
// shortcut used when the entire state describes a single scalar value. Source
// is the base pointer. If order is present, PtrState describes block pointer;
// otherwise it describes non-block pointers. When it describes block pointer,
// shape field means the same field as tt.make_tensor_ptr; when it describes a
// non-block pointer, shape field indicates how address wraps around (i.e.,
// modulo); a constant 0 indicates no modulo for the dimension.
struct PtrState {

  SmallVector<Value> offsets;
  SmallVector<Value> strides;
  SmallVector<Value> shape;
  SmallVector<int32_t> sizes;
  SmallVector<int32_t> order;

  Value source;
  Value scalar;

  int32_t getRank() const {
    assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
           offsets.size() == strides.size());
    return offsets.size();
  }

  bool isEmpty() const { return getRank() == 0 && !source && !scalar; }

  // Process addition of two PtrStates.
  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder) {
    assert(isEmpty() && lhsState.getRank() == rhsState.getRank());
    Location loc = op->getLoc();

    if (lhsState.source && rhsState.source) {
      op->emitRemark("TritonRaiseBlockPointer: do not support adding two "
                     "pointer states that both have base pointers");
      return failure();
    }

    source = lhsState.source ? lhsState.source : rhsState.source;

    ArithBuilder abuilder(builder, loc);
    for (uint64_t i = 0; i < lhsState.getRank(); ++i) {
      Value newOffset = abuilder.add(lhsState.offsets[i], rhsState.offsets[i]);
      offsets.push_back(newOffset);

      Value newStride = abuilder.add(lhsState.strides[i], rhsState.strides[i]);
      strides.push_back(newStride);

      sizes.push_back(lhsState.sizes[i]);
    }

    const PtrState *lhs = &lhsState;
    const PtrState *rhs = &rhsState;

    for (uint64_t i = 0; i < lhs->getRank(); ++i) {
      shape.push_back(lhs->shape[i]);
    }

    return success();
  }

  triton::MakeTensorPtrOp createTTMakeTensorPtrOp(OpBuilder &builder,
                                                  Location loc) {
    auto op = builder.create<triton::MakeTensorPtrOp>(
        loc, source, shape, strides, offsets, sizes, order);
    LLVM_DEBUG(llvm::dbgs() << "creating tt.make_tensor_ptr:\n" << op << "\n";);
    return op;
  }
};

#ifndef NDEBUG
template <typename T>
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const SmallVector<T> &v) {
  os << "{";
  if (!v.empty()) {
    os << v.front();
    llvm::for_each(ArrayRef<T>(v).drop_front(),
                   [&os](const T &el) { os << ", " << el; });
  }
  return os << "}";
}

static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const PtrState &state) {
  return os << "<offsets=" << state.offsets << "> <sizes=" << state.sizes
            << "> <strides=" << state.strides << "> <shape=" << state.shape
            << "> <order=" << state.order << ">";
}
#endif

struct TritonRaiseBlockPointer
    : triton::intel::impl::TritonRaiseBlockPointerBase<
          TritonRaiseBlockPointer> {
  using Base::Base;

  void runOnOperation() final {
    getOperation()->walk([this](triton::AddPtrOp addptr) {
      if (failed(rewriteAddPtrOp(addptr)))
        addptr->emitRemark("TritonRaiseToBlockPointer: Failed to rewrite");
    });
  }

  LogicalResult rewriteAddPtrOp(triton::AddPtrOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();

    PtrState state;
    if (failed(visitOperandAddptr(op, state, loc, builder)))
      return failure();

    knownPtrs[op.getResult()] = state;

    Value result = op.getResult();
    Value mapped = result;
    if (isa<RankedTensorType>(result.getType())) {
      Value maketptrOp = state.createTTMakeTensorPtrOp(builder, loc);
      mapped = maketptrOp;
    }

    ptrMap.map(result, mapped);

    return success();
  }

  LogicalResult visitOperandAddptr(triton::AddPtrOp addptrOp, PtrState &state,
                                   Location loc, OpBuilder &builder) {
    assert(state.isEmpty());

    PtrState ptrState;
    if (failed(visitOperand(addptrOp.getPtr(), ptrState, addptrOp.getLoc(),
                            builder))) {
      return failure();
    }

    PtrState offsetState;
    if (failed(visitOperand(addptrOp.getOffset(), offsetState,
                            addptrOp.getLoc(), builder))) {
      return failure();
    }

    assert(ptrState.source && "ptr field should provide source / base pointer");

    assert(ptrState.getRank() == offsetState.getRank() &&
           "ptr and offset field should have the same rank");

    LLVM_DEBUG(llvm::dbgs() << "Base: " << ptrState << "\n"
                            << "Offset: " << offsetState << "\n";);

    return state.addState(ptrState, offsetState, addptrOp, builder);
  }

  LogicalResult visitOperand(Value operand, PtrState &state, const Location loc,
                             OpBuilder &builder) {
    if (knownPtrs.find(operand) != knownPtrs.end()) {
      state = knownPtrs.lookup(operand);
      return success();
    }

    if (isa<IntegerType>(operand.getType())) {
      OpBuilder::InsertionGuard guard(builder);
      if (Operation *definingOp = operand.getDefiningOp())
        builder.setInsertionPointAfter(definingOp);
      auto castOp = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), operand);
      state.scalar = castOp.getResult();
      return success();
    }

    if (isa<IndexType>(operand.getType())) {
      state.scalar = operand;
      return success();
    }

    if (isa<triton::PointerType>(operand.getType())) {
      // A scalar pointer can either be produced by AddPtrOp or a block
      // argument
      if (Operation *op = operand.getDefiningOp()) {
        if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op))
          return visitOperandAddptr(addPtrOp, state, loc, builder);
        if (isa<triton::MakeTensorPtrOp>(op))
          llvm_unreachable(
              "Unexpected operand defining operation tt.make_tensor_ptr");
        llvm_unreachable("Unexpected operand defining operation");
      }
      state.source = operand;
      return success();
    }

    if (Operation *definingOp = operand.getDefiningOp()) {
      if (auto op = dyn_cast<triton::MakeRangeOp>(definingOp))
        return visitOperandMakeRange(op, state, loc, builder);
      if (auto op = dyn_cast<triton::SplatOp>(definingOp)) {
        return visitOperandSplat(op, state, loc, builder);
      }
    }

    llvm::errs() << "TritonRaiseBlockPointer: encountered addptr operand "
                    "produced by an unsupported operation\n"
                 << operand << "\n";

    return failure();
  }

  LogicalResult visitOperandMakeRange(triton::MakeRangeOp rangeOp,
                                      PtrState &state, Location loc,
                                      OpBuilder &builder) {
    assert(state.isEmpty());

    ArrayRef<int64_t> shape = cast<ShapedType>(rangeOp.getType()).getShape();

    uint32_t start = rangeOp.getStart();
    uint32_t end = rangeOp.getEnd();
    uint32_t stride = (end - start + shape[0] - 1) / shape[0];
    assert(stride == 1 &&
           "Expect make_range op to always return tensor of stride 1");

    state.offsets.push_back(
        builder.create<arith::ConstantIntOp>(loc, start, offsetBitwidth));
    state.strides.push_back(builder.create<arith::ConstantIntOp>(
        loc, stride, shapeAndStridesBitwidth));
    state.shape.push_back(
        builder.create<arith::ConstantIntOp>(loc, 0, shapeAndStridesBitwidth));
    state.sizes.push_back(shape[0]);

    LLVM_DEBUG(llvm::dbgs() << "MakeRange state: " << state << "\n";);

    return success();
  }

  LogicalResult visitOperandSplat(triton::SplatOp splatOp, PtrState &state,
                                  Location loc, OpBuilder &builder) {
    assert(state.isEmpty());

    Value src = splatOp.getSrc();
    Value dst = splatOp.getResult();
    ArrayRef<int64_t> dstShape = cast<ShapedType>(dst.getType()).getShape();

    if (failed(visitOperand(src, state, loc, builder)))
      return failure();

    if (!isa<IntegerType, IndexType, triton::PointerType>(src.getType())) {
      splatOp->emitRemark("TritonRaiseBlockPointer: unsupported splat pattern");
      return failure();
    }

    for (int64_t s : dstShape) {
      Value c0i32 =
          builder.create<arith::ConstantIntOp>(loc, 0, offsetBitwidth);
      Value c0i64 =
          builder.create<arith::ConstantIntOp>(loc, 0, shapeAndStridesBitwidth);
      state.offsets.push_back(c0i32);
      state.strides.push_back(c0i64);
      state.shape.push_back(c0i64);
      state.sizes.push_back(s);
    }

    // If we splat a integer value, scalar should become the offset of the
    // outer most dimension
    if (state.scalar) {
      state.offsets[0] = getValueOrCreateCastToIndexLike(
          builder, loc, builder.getIntegerType(offsetBitwidth), state.scalar);
    }

    LLVM_DEBUG(llvm::dbgs() << "Splat state: " << state << "\n";);

    return success();
  }

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  IRMapping ptrMap;
};
} // namespace
