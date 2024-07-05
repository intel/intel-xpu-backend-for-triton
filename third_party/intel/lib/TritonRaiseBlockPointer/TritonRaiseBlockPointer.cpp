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

std::optional<int64_t> getIntAttr(const OpFoldResult ofr) {
  if (ofr.is<Attribute>() && isa<IntegerAttr>(ofr.get<Attribute>()))
    return dyn_cast<IntegerAttr>(ofr.get<Attribute>()).getInt();
  return std::nullopt;
}

// This function folds the `op` operation and returns the constant value if it
// has successfully folded to a constant. Otherwise, it returns `std::nullopt`.
std::optional<int64_t> getFoldedConstantValue(Operation *op) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(results))) {
    return std::nullopt;
  }

  // If fold succeeded but `results` is empty, we give a second try, after the
  // operands have been switched during the first call to `fold()`.
  if (results.empty()) {
    if (failed(op->fold(results))) {
      return std::nullopt;
    }
  }

  if (results.size() != 1) {
    return std::nullopt;
  }

  auto intAttr = getIntAttr(results[0]);
  if (intAttr.has_value()) {
    return intAttr.value();
  }

  auto val = dyn_cast<Value>(results[0]);
  assert(val);
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return std::nullopt;

  intAttr = getIntAttr(constOp.getValue());
  if (intAttr.has_value()) {
    return intAttr.value();
  }
  return std::nullopt;
}

// return true if the `val` value is a constant containing a value equal to zero
bool hasConstZero(const Value val) {
  auto intVal = getFoldedConstantValue(val.getDefiningOp());
  if (intVal.has_value()) {
    return (intVal.value() == 0);
  }
  return false;
}

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

  // @return true if the `PtrState` structure describes a block pointer,
  // otherwise it describes a non-block pointer.
  bool isBlockPtr() const { return !order.empty(); }

  // This function checks whether the pointer addresses wraps around on the
  // dimention `dim`.
  // @return true if the address wraps around, (i.e. has modulo).
  // Note that this function should only be called when PtrState describes a
  // non-block pointer.
  bool dimHasModulo(uint32_t dim) const {
    assert(
        !isBlockPtr() &&
        "Analysis should not check modulo if PtrState describes block pointer");

    assert(dim < getRank() && "Dim cannot be higher than the tensor rank.");

    // When PtrState describes a non-block pointer, shape field indicates how
    // address wraps around. As a result, a constant 0 indicates no wrap around
    // (i.e. modulo) for the dimension.
    SmallVector<OpFoldResult> results;
    Operation *op = shape[dim].getDefiningOp();
    auto intVal = getFoldedConstantValue(op);
    if (intVal.has_value()) {
      return (intVal.value() != 0);
    }

    return true;
  }

  // @return true if addresses wrap around in any of the pointer dimension.
  bool hasModulo() const {
    for (int32_t i = 0; i < getRank(); i++) {
      if (dimHasModulo(i)) {
        return true;
      }
    }
    return false;
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

    // AddPtr where both lhs and rhs containing modulo operators not supported
    if (lhsState.hasModulo() && rhsState.hasModulo()) {
      op->emitRemark(
          "TritonRaiseBlockPointer: do not support adding two pointer states "
          "that both have modulo");
      return failure();
    }

    if (lhsState.hasModulo() || rhsState.hasModulo()) {
      // visitOperandSplat and visitOperandExpandDims should enforce below
      assert(lhsState.getRank() <= 2);
    }

    // dealing with modulo:
    // - If lhs has no modulo, skip
    // - If rhs has zero offset on dim i, we can just use lhs's modulo
    // - If i == 0 and rhs is the result of a splat, we will allow the add. This
    // is because the user may be trying to express adding a constant offset to
    // increment dim1, but pointer analysis cannot differentiate dim1 vs dim0 in
    // this case.
    // - Else, the analysis fails

    // An example for the 3rd condition above can look like:
    // %0 = tt.splat %scalar
    // %1 = tt.splat %ptr
    // %2 = tt.arange
    // %3 = arith.remsi %2, %size
    // %4 = tt.addptr %1, %3
    // %5 = tt.addptr %4, %0
    // %5 may also occur in a loop to increment %4 every iteration.

    // Note that this is not bullet-proof. E.g., broken IR can actually
    // increment dim0 while dim0 already has modulo, since Triton offsets are
    // element-wise and not in unit of lower dimensions. However, this is highly
    // unlikely but the analysis will provide wrong result. Hence we provide a
    // warning in this case.
    const PtrState *lhs = &lhsState;
    const PtrState *rhs = &rhsState;

    if (rhs->hasModulo()) {
      std::swap(lhs, rhs);
    }

    for (uint64_t i = 0; i < lhs->getRank(); i++) {
      if (!lhs->dimHasModulo(i)) {
        shape.push_back(lhs->shape[i]);
      } else if (hasConstZero(rhs->offsets[i])) {
        shape.push_back(lhs->shape[i]);
      } else if (i == 0 && lhs->getRank() == 2 && rhs->scalar) {
        shape.push_back(lhs->shape[1]);
        shape.push_back(lhs->shape[0]);
        op->emitWarning(
            "TritonRaiseBlockPointer: allowing adding pointer state with "
            "modulo in dim 0 to "
            "another pointer state with offset in dim 0.\nPlease verify the "
            "operand that contains a scalar is meant to increment pointers in "
            "dim1. If that is not the case it WILL LEAD TO WRONG COMPILATION "
            "RESULTS.\n\nTo avoid this warning, use expand_dims (instead of "
            "splat) to explicitly specify which dimension contains the "
            "scalar.");
        break;
      } else {
        op->emitRemark("TritonRaiseBlockPointer: do not support adding to "
                       "operand with modulo");
        return failure();
      }
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

      Value newOffset =
          abuilder.mul(getValueOrCreateCastToIndexLike(
                           builder, loc, builder.getI32Type(), offset),
                       i32Scalar);
      Value newStride =
          abuilder.mul(getValueOrCreateCastToIndexLike(
                           builder, loc, builder.getI64Type(), stride),
                       i64Scalar);
      Value newDim = abuilder.mul(getValueOrCreateCastToIndexLike(
                                      builder, loc, builder.getI64Type(), dim),
                                  i64Scalar);

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
        .Case<arith::AddIOp, arith::ConstantOp, arith::MulIOp, arith::RemUIOp,
              arith::RemSIOp, triton::BroadcastOp, triton::MakeRangeOp,
              triton::SplatOp, triton::ExpandDimsOp>(
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
                               OpTy, arith::RemSIOp, arith::RemUIOp>::value>>
  LogicalResult visitAddPointerRemOperand(OpTy remOp, PtrState &state,
                                          Location loc, OpBuilder &builder);

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

template <typename OpTy, typename = std::enable_if_t<llvm::is_one_of<
                             OpTy, arith::RemSIOp, arith::RemUIOp>::value>>
LogicalResult TritonRaiseBlockPointer::visitAddPointerRemOperand(
    OpTy remOp, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  PtrState rhsState;
  if (visitOperand(remOp.getRhs(), rhsState, loc, builder).failed()) {
    return failure();
  }

  if (!rhsState.scalar) {
    remOp->emitRemark(
        "TritonRaiseBlockPointer: only support cases when rhs of remainder "
        "contains scalar");
    return failure();
  }

  if (visitOperand(remOp.getLhs(), state, loc, builder).failed()) {
    return failure();
  }

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  if (state.hasModulo()) {
    remOp->emitRemark("TritonRaiseBlockPointer: do not support multiple modulo "
                      "within an expression");
    return failure();
  }

  if (state.getRank() == 1) {
    // Apply the modulo before expanding shape, the common pattern is
    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
    // stride_ak)
    state.shape.back() = rhsState.scalar;
  } else if (state.getRank() == 2) {
    // torch inductor expands the tensor shape before applying the modulo.
    //
    // We only support either:
    // - (tl.arange(0, end)[:, None] % mod), or
    // - (tl.arange(0, end)[None, :] % mod)
    //
    // In both cases, we apply the modulo to the non-singleton dimension.
    auto shape = cast<TensorType>(remOp.getResult().getType()).getShape();
    if (shape[0] == 1) {
      state.shape[1] = rhsState.scalar;
    } else if (shape[1] == 1) {
      state.shape[0] = rhsState.scalar;
    } else {
      remOp->emitRemark("TritonRaiseBlockPointer: taking modulo on a 2D tensor "
                        "with no singleton "
                        "dimension not supported");
      return failure();
    }
  } else {
    remOp->emitRemark("TritonRaiseBlockPointer: unsupported modulo pattern");
    return failure();
  }

  return success();
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::RemSIOp remOp, PtrState &state, Location loc, OpBuilder &builder) {
  return visitAddPointerRemOperand(remOp, state, loc, builder);
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::RemUIOp remOp, PtrState &state, Location loc, OpBuilder &builder) {
  return visitAddPointerRemOperand(remOp, state, loc, builder);
}

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
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    triton::ExpandDimsOp expandDimsOp, PtrState &state, Location loc,
    OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  if (failed(visitOperand(expandDimsOp.getSrc(), state, loc, builder))) {
    return failure();
  }

  ArrayRef<int64_t> dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  auto axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  Value c0i32 = builder.create<arith::ConstantIntOp>(loc, 0, offsetBitwidth);
  Value c0i64 =
      builder.create<arith::ConstantIntOp>(loc, 0, shapeAndStridesBitwidth);
  state.offsets.insert(state.offsets.begin() + axis, c0i32);
  state.sizes.insert(state.sizes.begin() + axis, 1);
  state.strides.insert(state.strides.begin() + axis, c0i64);
  state.shape.insert(state.shape.begin() + axis, c0i64);

  if (state.hasModulo() && state.getRank() > 2) {
    expandDimsOp->emitRemark("TritonRaiseBlockPointer: unsupported scenario "
                             "where expand_dims result "
                             "has modulo and rank > 2");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs() << "ExpandDims state: " << state << "\n";);

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

  assert(srcShape.size() <= dstShape.size() &&
         "rank of source cannot be greater than the rank of destination");

  if (failed(visitOperand(src, state, loc, builder))) {
    return failure();
  }

  if (srcShape.size() == dstShape.size()) {
    llvm::copy(dstShape, state.sizes.begin());
  } else {
    // Offset must be equal, otherwise we don.t know which offset should be
    // propagated to the new axis.
    for (int i = 1; i < state.offsets.size(); ++i) {
      if (state.offsets[0] != state.offsets[i]) {
        broadcastOp->emitRemark(
            "TritonRaiseBlockPointer: Unsupported broadcast with different "
            "offsets while source rank and destination rank differ.");
        return failure();
      }
    }

    // Create the new axis.
    // The positions of the new axis are determined based and the shape values.
    // If shape are the same, the new axis are added at the end.
    size_t srcAxis = 0;
    for (size_t axis = 0; axis < dstShape.size(); ++axis) {
      if ((srcAxis < srcShape.size()) &&
          (srcShape[srcAxis] == dstShape[axis])) {
        ++srcAxis;
        continue;
      }
      Value c0i32 =
          builder.create<arith::ConstantIntOp>(loc, 0, offsetBitwidth);
      Value c0i64 =
          builder.create<arith::ConstantIntOp>(loc, 0, shapeAndStridesBitwidth);
      state.offsets.insert(state.offsets.begin() + axis,
                           getValueOrCreateCastToIndexLike(
                               builder, loc,
                               builder.getIntegerType(offsetBitwidth),
                               state.offsets[0]));
      state.sizes.insert(state.sizes.begin() + axis, dstShape[axis]);
      state.strides.insert(state.strides.begin() + axis, c0i64);
      state.shape.insert(state.shape.begin() + axis, c0i64);
    }

    // The following condition has been duplicated from the expand_dim support
    // TODO : Verify if we need still need it given that triton `make_block_ptr`
    // op differs from triton-shared `make_block_ptr` op regarding how address
    // wrap around are handled.
    if (state.hasModulo() && state.getRank() > 2) {
      broadcastOp->emitRemark("TritonRaiseBlockPointer: unsupported scenario "
                              "where broadcast result "
                              "has modulo and rank > 2");
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Broadcast state: " << state << "\n";);

  return success();
}
} // namespace
