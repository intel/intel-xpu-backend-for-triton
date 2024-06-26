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

  LogicalResult mulState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder) {
    assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

    Location loc = op->getLoc();

    assert(!lhsState.source && !rhsState.source &&
           "Multiplying base pointer does not make sense");

    assert(!(lhsState.scalar && rhsState.scalar) &&
           "do not expect to see both lhs and rhs are scalars");

    // currently do not support both tensors are effectively non-scalar
    if (!lhsState.scalar && !rhsState.scalar) {
      op->emitRemark("TritonRaiseBlockPointer: only support multiplying "
                     "pointer states when one of them represent a scalar");
      return failure();
    }

    PtrState const *lhs = &lhsState;
    PtrState const *rhs = &rhsState;

    if (!rhs->scalar && lhs->scalar)
      std::swap(lhs, rhs);

    Value i32Scalar = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getI32Type(), rhs->scalar);
    Value i64Scalar = getValueOrCreateCastToIndexLike(
        builder, loc, builder.getI64Type(), rhs->scalar);
    ArithBuilder abuilder(builder, loc);
    for (const auto &[offset, stride, dim, size] :
         llvm::zip(lhs->offsets, lhs->strides, lhs->shape, lhs->sizes)) {
      Value newOffset = abuilder.mul(offset, i32Scalar);
      Value newStride = abuilder.mul(stride, i64Scalar);
      Value newDim = abuilder.mul(dim, i64Scalar);

      offsets.push_back(newOffset);
      strides.push_back(newStride);
      shape.push_back(newDim);
      sizes.push_back(size);
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
    getOperation()->walk([this](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case([this](triton::AddPtrOp addptr) {
            if (failed(rewriteAddPtrOp(addptr)))
              addptr->emitRemark(
                  "TritonRaiseToBlockPointer: Failed to rewrite");
          })
          .Case<triton::LoadOp, triton::StoreOp>([this](auto loadstore) {
            if (failed(rewriteLoadStoreOp(loadstore)))
              loadstore->emitRemark(
                  "TritonRaiseToBlockPointer: Failed to rewrite");
          });
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
    assert(state.isEmpty() && "state is a return argument");

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

    Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      llvm::errs() << "TritonRaiseBlockPointer: encountered addptr block "
                      "argument operand\n"
                   << operand << "\n";
    }

    return TypeSwitch<Operation *, LogicalResult>(definingOp)
        .Case<arith::AddIOp, arith::ConstantOp, arith::MulIOp,
              triton::BroadcastOp, triton::MakeRangeOp, triton::SplatOp>(
            [this, &state, loc, &builder](auto op) {
              return visitAddPointerOperand(op, state, loc, builder);
            })
        .Default([](Operation *op) {
          llvm::dbgs() << "TritonRaiseBlockPointer: encountered addptr operand "
                          "produced by an unsupported operation\n"
                       << op << "\n";
          return failure();
        });
  }

  template <typename OpTy>
  LogicalResult visitAddPointerOperand(OpTy op, PtrState &state, Location loc,
                                       OpBuilder &builder);

  template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                               OpTy, triton::LoadOp, triton::StoreOp>::value>>
  LogicalResult rewriteLoadStoreOp(OpTy op) {
    constexpr bool isLoad = std::is_same_v<OpTy, triton::LoadOp>;
    constexpr StringLiteral opName =
        isLoad ? StringLiteral("loadOp") : StringLiteral("storeOp");

    Value ptr = ptrMap.lookupOrNull(op.getPtr());

    if (!ptr) {
      op->emitRemark("TritonRaiseBlockPointer: pointer is not replaced with "
                     "tt.make_tensor_ptr so ")
          << opName << " cannot be rewritten";
      return failure();
    }

    auto ptrType = dyn_cast<triton::PointerType>(ptr.getType());
    if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
      op->emitRemark("TritonRaiseBlockPointer: scalar ")
          << opName << " will not be rewritten";
      return failure();
    }

    OpBuilder builder(op);
    if constexpr (isLoad) {
      auto loadOp = builder.create<triton::LoadOp>(
          op.getLoc(), ptr, op.getMask(), op.getOther(), op.getBoundaryCheck(),
          op.getPadding(), op.getCache(), op.getEvict(), op.getIsVolatile());

      LLVM_DEBUG(llvm::dbgs() << "creating tt.load: " << loadOp << "\n";);

      op.replaceAllUsesWith(loadOp.getResult());
    } else {
      [[maybe_unused]] auto storeOp = builder.create<triton::StoreOp>(
          op.getLoc(), ptr, op.getValue(), op.getMask(), op.getBoundaryCheck(),
          op.getCache(), op.getEvict());

      LLVM_DEBUG(llvm::dbgs() << "creating tt.store: " << storeOp << "\n";);
    }

    op->erase();
    return success();
  }

  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  IRMapping ptrMap;
};

template <>
LogicalResult
TritonRaiseBlockPointer::visitAddPointerOperand(triton::MakeRangeOp rangeOp,
                                                PtrState &state, Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

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

template <>
LogicalResult
TritonRaiseBlockPointer::visitAddPointerOperand(triton::SplatOp splatOp,
                                                PtrState &state, Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

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
    Value c0i32 = builder.create<arith::ConstantIntOp>(loc, 0, offsetBitwidth);
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

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::AddIOp addOp, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  PtrState lhsState;
  if (failed(visitOperand(addOp.getLhs(), lhsState, loc, builder)))
    return failure();

  PtrState rhsState;
  if (failed(visitOperand(addOp.getRhs(), rhsState, loc, builder)))
    return failure();

  if (failed(state.addState(lhsState, rhsState, addOp, builder)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Add state: " << state << "\n";);

  return success();
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::MulIOp mulOp, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  PtrState lhsState;
  if (failed(visitOperand(mulOp.getLhs(), lhsState, loc, builder)))
    return failure();

  PtrState rhsState;
  if (failed(visitOperand(mulOp.getRhs(), rhsState, loc, builder)))
    return failure();

  if (failed(state.mulState(lhsState, rhsState, mulOp, builder)))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Mul state: " << state << "\n";);

  return success();
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::ConstantOp op, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  auto attr = cast<DenseElementsAttr>(op.getValue());
  Type elementType = attr.getElementType();
  assert(attr.isSplat() && isa<IntegerType>(elementType) &&
         "Expecting constant tensor");

  state.scalar = builder.create<arith::ConstantIndexOp>(
      loc, attr.getValues<IntegerAttr>()[0].getValue().getSExtValue());

  Type offsetType = builder.getIntegerType(offsetBitwidth);
  auto resultType = cast<ShapedType>(op.getResult().getType());
  Value offset = convertScalarToDtype(builder, loc, state.scalar, offsetType,
                                      /*isUnsignedCast=*/true);
  for (int32_t dim : resultType.getShape()) {
    state.offsets.push_back(offset);
    state.sizes.push_back(dim);
    state.strides.push_back(
        builder.create<arith::ConstantIntOp>(loc, 0, shapeAndStridesBitwidth));
    state.shape.push_back(
        builder.create<arith::ConstantIntOp>(loc, 0, shapeAndStridesBitwidth));
  }

  return success();
}

template <>
LogicalResult
TritonRaiseBlockPointer::visitAddPointerOperand(triton::BroadcastOp broadcastOp,
                                                PtrState &state, Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  Value src = broadcastOp.getSrc();
  Value dst = broadcastOp.getResult();

  if (!isa<ShapedType>(src.getType())) {
    broadcastOp->emitRemark(
        "TritonRaiseBlockPointer: Unsupported broadcast source type");
    return failure();
  }

  ArrayRef<int64_t> srcShape = cast<ShapedType>(src.getType()).getShape();
  ArrayRef<int64_t> dstShape = cast<ShapedType>(dst.getType()).getShape();

  // TODO: Implement srcShape.size() < dstShape.size() case
  assert(srcShape.size() == dstShape.size() &&
         "rank of source cannot be different to rank of destination");

  if (failed(visitOperand(src, state, loc, builder))) {
    return failure();
  }

  llvm::copy(dstShape, state.sizes.begin());

  LLVM_DEBUG(llvm::dbgs() << "Broadcast state: " << state << "\n";);

  return success();
}
} // namespace
