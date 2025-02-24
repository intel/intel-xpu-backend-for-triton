//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/TritonRaiseBlockPointer/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <set>

#define DEBUG_TYPE "triton-raise-block-pointer"

// This pass does manage to raise tensor of pointers into block pointers for
// simple cases (e.g. 03 matmul tutorial). However, this pass has several know
// limitations:
//   - Masks and modulos are not correctly handled by this pass. Issue #1784
//   (https://github.com/intel/intel-xpu-backend-for-triton/issues/1784) has
//   been created to address this limitation.
//   - The pattern matching method used in this pass makes it prone to fail
//   raising memory accesses. For the moment, the most fragile part of the pass
//   is probably the support for fixing the axis of the offsets
//   (see comment l.867).

using namespace mlir;
namespace tt = mlir::triton;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONRAISEBLOCKPOINTER
#include "intel/include/TritonRaiseBlockPointer/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

constexpr unsigned offsetBitwidth = 32u;
constexpr unsigned shapeAndStridesBitwidth = 64u;

// Lookup for a constant with the given value and bitwidth in the current block
// (before the builder insertion point). Return it a suitable constant is found,
// otherwise create a new one.
Value findOrCreateConstant(Location loc, int val, unsigned bitWidth,
                           OpBuilder &builder) {
  Block *block = builder.getInsertionBlock();
  const Block::iterator insertPoint = builder.getInsertionPoint();

  auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
    if (auto cstOp = dyn_cast<arith::ConstantIntOp>(op))
      return cstOp.value() == val &&
             cstOp.getType().getIntOrFloatBitWidth() == bitWidth;
    return false;
  });

  return (it != insertPoint)
             ? cast<arith::ConstantIntOp>(*it)
             : builder.createOrFold<arith::ConstantIntOp>(loc, val, bitWidth);
}

Value findOrCreateCast(Location loc, Value val, Type tgtType,
                       OpBuilder &builder) {
  Block *block = builder.getInsertionBlock();
  const Block::iterator insertPoint = builder.getInsertionPoint();

  auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
    if (auto castOp = dyn_cast<arith::IndexCastOp>(op))
      return castOp.getIn() == val && castOp.getType() == tgtType;
    return false;
  });

  return (it != insertPoint)
             ? cast<arith::IndexCastOp>(*it)
             : getValueOrCreateCastToIndexLike(builder, loc, tgtType, val);
}

Value findOrCreateMakeTensorPtr(Location loc, Value source, ValueRange shape,
                                ValueRange strides, ValueRange offsets,
                                ArrayRef<int> order, ArrayRef<int> sizes,
                                OpBuilder &builder) {
  Block *block = builder.getInsertionBlock();
  const Block::iterator insertPoint = builder.getInsertionPoint();

  auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
    if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
      return makeTensorPtrOp.getBase() == source &&
             makeTensorPtrOp.getShape() == shape &&
             makeTensorPtrOp.getStrides() == strides &&
             makeTensorPtrOp.getOffsets() == offsets &&
             makeTensorPtrOp.getOrder() == order;
    }
    return false;
  });

  // Note: We are forcing the shape to be unknown to pointer increments that may
  // wrap around (via the tt.advance operation).
  Value zero = findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder);
  SmallVector<Value> zeros;
  for (int i = 0; i < shape.size(); ++i)
    zeros.push_back(zero);

  return (it != insertPoint)
             ? cast<tt::MakeTensorPtrOp>(*it)
             : builder.createOrFold<tt::MakeTensorPtrOp>(
                   loc, source, zeros, strides, offsets, sizes, order);
}

Value getFinalValue(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    // look init values outside the loop
    BlockArgument blockArg = dyn_cast<BlockArgument>(value);
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp))
      return getFinalValue(forOp.getInitArgs()[blockArg.getArgNumber() - 1]);

    return value;
  }

  if (isa<tt::ExpandDimsOp, tt::BroadcastOp, tt::SplatOp, arith::IndexCastOp>(
          defOp))
    return getFinalValue(defOp->getOperand(0));

  if (auto addOp = dyn_cast<arith::AddIOp>(defOp)) {
    if (ttgi::isConstant(addOp.getLhs(), 0))
      return getFinalValue(addOp.getRhs());
    if (ttgi::isConstant(addOp.getRhs(), 0))
      return getFinalValue(addOp.getLhs());
    return addOp.getResult();
  }

  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
    if (ttgi::isConstant(mulOp.getLhs(), 1) ||
        ttgi::isConstant(mulOp.getRhs(), 0))
      return getFinalValue(mulOp.getRhs());
    if (ttgi::isConstant(mulOp.getRhs(), 1) ||
        ttgi::isConstant(mulOp.getLhs(), 0))
      return getFinalValue(mulOp.getLhs());
    return mulOp.getResult();
  }

  if (auto divOp = dyn_cast<arith::DivUIOp>(defOp)) {
    if (ttgi::isConstant(divOp.getRhs(), 1) ||
        ttgi::isConstant(divOp.getLhs(), 0))
      return getFinalValue(divOp.getLhs());
    return divOp.getResult();
  }

  return value;
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
  SmallVector<int> sizes;
  SmallVector<int> order;
  Value source;
  Value scalar;

  int getRank() const {
    assert(offsets.size() == sizes.size() && offsets.size() == strides.size() &&
           offsets.size() == strides.size());
    return offsets.size();
  }

  // @return true if the `PtrState` structure describes a block pointer,
  // otherwise it describes a non-block pointer.
  bool isBlockPtr() const { return !order.empty(); }

  // This function checks whether the pointer addresses wraps around on the
  // dimension `dim`.
  // @return true if the address wraps around, (i.e. has modulo).
  // Note that this function should only be called when PtrState describes a
  // non-block pointer.
  bool dimHasModulo(unsigned dim) const {
    assert(!isBlockPtr() && "Analysis should not check modulo if PtrState "
                            "describes block pointer");
    assert(dim < getRank() && "Dim cannot be higher than the tensor rank.");

    // When PtrState describes a non-block pointer, shape field indicates how
    // address wraps around. As a result, a constant 0 indicates no wrap
    // around (i.e. modulo) for the dimension.
    return !ttgi::isConstant(shape[dim], 0);
  }

  // @return true if addresses wrap around in any of the pointer dimension.
  bool hasModulo() const {
    for (int i = 0; i < getRank(); i++) {
      if (dimHasModulo(i))
        return true;
    }
    return false;
  }

  bool isEmpty() const { return getRank() == 0 && !source && !scalar; }

  // Process addition of two PtrStates.
  LogicalResult addState(const PtrState &lhsState, const PtrState &rhsState,
                         Operation *op, OpBuilder &builder) {
    assert(isEmpty() && lhsState.getRank() == rhsState.getRank());

    if (lhsState.source && rhsState.source) {
      op->emitRemark("TritonRaiseBlockPointer: do not support adding two "
                     "pointer states that both have base pointers");
      return failure();
    }

    source = lhsState.source ? lhsState.source : rhsState.source;
    Location loc = op->getLoc();

    if (lhsState.scalar && rhsState.scalar) {
      scalar =
          builder.create<arith::AddIOp>(loc, lhsState.scalar, rhsState.scalar);
      scalar = findOrCreateCast(loc, getFinalValue(scalar),
                                lhsState.scalar.getType(), builder);

    } else if (lhsState.getRank() == 0)
      scalar = lhsState.scalar ? lhsState.scalar : rhsState.scalar;

    for (unsigned i = 0; i < lhsState.getRank(); ++i) {
      Value newOffset = builder.create<arith::AddIOp>(loc, lhsState.offsets[i],
                                                      rhsState.offsets[i]);
      offsets.push_back(findOrCreateCast(loc, getFinalValue(newOffset),
                                         lhsState.offsets[i].getType(),
                                         builder));

      Value newStride = builder.create<arith::AddIOp>(loc, lhsState.strides[i],
                                                      rhsState.strides[i]);
      strides.push_back(findOrCreateCast(loc, getFinalValue(newStride),
                                         lhsState.strides[i].getType(),
                                         builder));

      sizes.push_back(lhsState.sizes[i]);
    }

    // AddPtr where both lhs and rhs containing modulo operators not supported
    if (lhsState.hasModulo() && rhsState.hasModulo()) {
      op->emitRemark(
          "TritonRaiseBlockPointer: do not support adding two pointer states "
          "that both have modulo");
      return failure();
    }

    assert(!(lhsState.hasModulo() || rhsState.hasModulo()) ||
           (lhsState.getRank() <= 2) && "cannot have rank > 2 if operand one "
                                        "of the operands has a modulo");

    // dealing with modulo:
    // - If lhs has no modulo, skip
    // - If rhs has zero offset on dim i, we can just use lhs's modulo
    // - Else, the analysis fails

    // An example for the 3rd condition above can look like:
    // %0 = tt.splat %scalar
    // %1 = tt.splat %ptr
    // %2 = tt.arange
    // %3 = arith.remsi %2, %size
    // %4 = tt.addptr %1, %3
    // %5 = tt.addptr %4, %0
    // %5 may also occur in a loop to increment %4 every iteration.

    const PtrState *lhs = &lhsState;
    const PtrState *rhs = &rhsState;
    if (rhs->hasModulo())
      std::swap(lhs, rhs);

    for (unsigned i = 0; i < lhs->getRank(); ++i) {
      if (!lhs->dimHasModulo(i) || ttgi::isConstant(rhs->offsets[i], 0)) {
        shape.push_back(lhs->shape[i]);
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

    const PtrState *lhs = &lhsState;
    const PtrState *rhs = &rhsState;
    if (!rhs->scalar && lhs->scalar)
      std::swap(lhs, rhs);

    Location loc = op->getLoc();

    for (const auto &[offset, stride, dim, size] :
         llvm::zip(lhs->offsets, lhs->strides, lhs->shape, lhs->sizes)) {
      Value newOffset = builder.create<arith::MulIOp>(
          loc,
          findOrCreateCast(loc, offset, builder.getIntegerType(offsetBitwidth),
                           builder),
          findOrCreateCast(loc, rhs->scalar,
                           builder.getIntegerType(offsetBitwidth), builder));
      newOffset =
          findOrCreateCast(loc, getFinalValue(newOffset),
                           builder.getIntegerType(offsetBitwidth), builder);

      Value newStride = builder.create<arith::MulIOp>(
          loc,
          findOrCreateCast(loc, stride,
                           builder.getIntegerType(shapeAndStridesBitwidth),
                           builder),
          findOrCreateCast(loc, rhs->scalar,
                           builder.getIntegerType(shapeAndStridesBitwidth),
                           builder));
      newStride = findOrCreateCast(
          loc, getFinalValue(newStride),
          builder.getIntegerType(shapeAndStridesBitwidth), builder);

      Value newDim = builder.create<arith::MulIOp>(
          loc,
          findOrCreateCast(loc, dim,
                           builder.getIntegerType(shapeAndStridesBitwidth),
                           builder),
          findOrCreateCast(loc, rhs->scalar,
                           builder.getIntegerType(shapeAndStridesBitwidth),
                           builder));
      newDim = findOrCreateCast(loc, getFinalValue(newDim),
                                builder.getIntegerType(shapeAndStridesBitwidth),
                                builder);

      offsets.push_back(newOffset);
      strides.push_back(newStride);
      shape.push_back(newDim);
      sizes.push_back(size);
    }

    return success();
  }

  Value createTTMakeTensorPtrOp(OpBuilder &builder, Location loc) const {
    SmallVector<Value> newOffsets, newStrides, newShape;

    for (const auto &[offset, stride, dim] :
         llvm::zip(offsets, strides, shape)) {
      newOffsets.push_back(computeOffset(offset, stride, builder, loc));
      newStrides.push_back(findOrCreateCast(
          loc, stride, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      newShape.push_back(findOrCreateCast(
          loc, dim, builder.getIntegerType(shapeAndStridesBitwidth), builder));
    }

    return findOrCreateMakeTensorPtr(loc, source, newShape, newStrides,
                                     newOffsets, order, sizes, builder);
  }

  std::optional<Value> createTTAdvanceOp(Value ptr,
                                         tt::MakeTensorPtrOp makeTPtrOp,
                                         OpBuilder &builder,
                                         Location loc) const {
    assert(triton::isTensorPointerType(ptr.getType()) &&
           "Expecting a block ptr");
    SmallVector<Value> newOffsets;

    // We need to generate a `tt.advance` operation as follows:
    //   tt.advance ptr, (x0, x1)
    // where:
    //   x0 = off0 / (stride0 * stride0)
    //   x1 = off1 / stride1
    // The integer the divisions above are correct only if `num % denom == 0`,
    // therefore we give up if none of the strides is one.

    bool noStrideIsOne = llvm::all_of(makeTPtrOp.getStrides(), [&](Value str) {
      return !ttgi::isConstant(getFinalValue(str), 1);
    });
    if (noStrideIsOne)
      return std::nullopt;

    // We can generate a tt.advance operation as follow:
    //   Case 1: all offsets are non-zero ==> all strides must be one
    //   Case 2: one offset is zero
    //     2a) offsets: (0, off1) strides: (*, 1) ==> tt.advance ptr, (0, off1)
    //     2b) offsets: (off0, 0) strides: (*, 1) ==> tt.advance ptr, (0, off0)

    bool allOffsetsNotZero = llvm::all_of(offsets, [&](Value offset) {
      return !ttgi::isConstant(getFinalValue(offset), 0);
    });

    // Case 1: all offsets are non-zero.
    if (allOffsetsNotZero) {
      assert(offsets.size() == 1 &&
             "TODO: can we generate tt.advance ptr, (0, off0*str0 + off1) ?");

      if (llvm::any_of(makeTPtrOp.getStrides(), [&](Value stride) {
            return !ttgi::isConstant(getFinalValue(stride), 1);
          }))
        return std::nullopt;

      for (Value offset : offsets)
        newOffsets.push_back(offset);

      return builder.createOrFold<tt::AdvanceOp>(loc, ptr.getType(), ptr,
                                                 newOffsets);
    }

    // Case 2: at least one offset is zero.
    assert(offsets.size() == 2 && "Expecting two offsets");
    bool zeroIdx = !ttgi::isConstant(getFinalValue(offsets[0]), 0);
    Value nonZeroOffset = offsets[!zeroIdx];
    Value zeroOffset = offsets[zeroIdx];

    if (ttgi::isConstant(getFinalValue(makeTPtrOp.getStrides()[0]), 1))
      newOffsets = {nonZeroOffset, zeroOffset};
    else
      newOffsets = {zeroOffset, nonZeroOffset};

    return builder.createOrFold<tt::AdvanceOp>(loc, ptr.getType(), ptr,
                                               newOffsets);
  }

private:
  Value computeOffset(Value offset, Value stride, OpBuilder &builder,
                      Location loc) const {
    if (ttgi::isConstant(stride, 0))
      return findOrCreateCast(loc, offset,
                              builder.getIntegerType(offsetBitwidth), builder);

    Value divOffset = builder.create<arith::DivUIOp>(
        loc, builder.getIntegerType(offsetBitwidth),
        findOrCreateCast(loc, offset, builder.getIntegerType(offsetBitwidth),
                         builder),
        findOrCreateCast(loc, stride, builder.getIntegerType(offsetBitwidth),
                         builder));
    return findOrCreateCast(loc, getFinalValue(divOffset),
                            builder.getIntegerType(offsetBitwidth), builder);
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
  if (state.source)
    os << "<source=" << state.source << "> ";
  if (state.scalar)
    os << " <scalar=" << state.scalar << "> ";

  return os << "<offsets=" << state.offsets << "> <sizes=" << state.sizes
            << "> <strides=" << state.strides << "> <shape=" << state.shape
            << "> <order=" << state.order << ">";
}
#endif

// Utility class aggregating information required to create a versioning
// condition.
class VersioningCondition {
public:
  VersioningCondition(Value S, Value BS) : S(S), BS(BS) {
    assert(isValid() && "Invalid values supplied");
  }

  // Create the condition: (S % BS == 0 && S > BS)
  Value materialize(OpBuilder &builder, Location loc) const {
    assert(S && BS && "Expecting valid values");
    Value zero =
        builder.createOrFold<arith::ConstantIntOp>(loc, 0, S.getType());
    Value cmp1 = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq,
        builder.create<arith::RemSIOp>(loc, S, BS), zero);
    Value cmp2 =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, S, BS);
    return builder.create<arith::AndIOp>(loc, cmp1, cmp2);
  }

private:
  bool isValid() const {
    Type SType = S.getType(), BSType = BS.getType();
    if (!isa<IntegerType>(SType) || !isa<IntegerType>(BSType))
      return false;

    return cast<IntegerType>(SType).getWidth() ==
           cast<IntegerType>(BSType).getWidth();
  }

  Value S;  // The length of a row/column.
  Value BS; // The block size.
};

// Utility class responsible for collecting masked operation in a loop that are
// amenable to having their mask dropped when the loop is versioned.
class MaskedOpsCollector {
  friend class LoopVersioner;

public:
  bool collectMaskedOps(scf::ForOp &forOp) {
    // Nested loop aren't currently handled.
    if (forOp->template getParentOfType<scf::ForOp>())
      return false;

    // Ensure the loop upper bound is in canonical form (N+END-1)/END.
    if (!hasValidUpperBound(forOp))
      return false;

    assert(versioningCond && "Expecting a valid versioning condition");

    // Collect masked loads in the loop if they have canonical mask.
    for (auto op : forOp.getOps<tt::LoadOp>()) {
      Value mask = op.getMask();
      if (mask && isValidMask(getFinalValue(mask)))
        maskedOps.insert(op);
    }

    // TODO: collect masked stores in the loop if they have canonical mask.

    return maskedOps.size();
  }

private:
  // Check whether the loop UB is in canonical form: (N+END-1)/END and create
  // the versioning condition to use for the loop if so.
  bool hasValidUpperBound(scf::ForOp &forOp) {
    Value ub = getFinalValue(forOp.getUpperBound());
    Operation *defOp = ub.getDefiningOp();
    if (!defOp || !isa<arith::DivSIOp>(defOp))
      return false;

    auto divOp = cast<arith::DivSIOp>(defOp);
    Operation *divLhsOp = divOp.getLhs().getDefiningOp();
    Operation *divRhsOp = divOp.getRhs().getDefiningOp();
    if (!divLhsOp || !divRhsOp || !isa<arith::AddIOp>(divLhsOp) ||
        !isa<arith::ConstantOp>(divRhsOp))
      return false;

    auto divNumOp = cast<arith::AddIOp>(divLhsOp);
    auto divDenOp = cast<arith::ConstantIntOp>(divRhsOp);
    Operation *addLhsOp = divNumOp.getLhs().getDefiningOp();
    Operation *addRhsOp = divNumOp.getRhs().getDefiningOp();
    if (addLhsOp || !isa<arith::ConstantIntOp>(addRhsOp) ||
        (divDenOp.value() != cast<arith::ConstantIntOp>(addRhsOp).value() + 1))
      return false;

    versioningCond = std::make_unique<VersioningCondition>(divNumOp.getLhs(),
                                                           divOp.getRhs());
    return true;
  }

  // Check whether a mask is in canonical form: (0..END) < N - i*END
  bool isValidMask(Value mask) const {
    assert(mask.getDefiningOp() && "Expected a valid mask operation");
    auto cmpOp = cast<arith::CmpIOp>(mask.getDefiningOp());
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (pred != arith::CmpIPredicate::slt)
      return false;

    Operation *lhs = getFinalValue(cmpOp.getLhs()).getDefiningOp();
    Operation *rhs = getFinalValue(cmpOp.getRhs()).getDefiningOp();
    if (!isa<tt::MakeRangeOp>(lhs) || !isa<arith::SubIOp>(rhs))
      return false;

    auto rangeOp = cast<tt::MakeRangeOp>(lhs);
    unsigned end = rangeOp.getEnd();
    assert(end > rangeOp.getStart() && "Invalid range");

    auto subOp = cast<arith::SubIOp>(rhs);
    Operation *subLhs = subOp.getLhs().getDefiningOp();
    Operation *subRhs = subOp.getRhs().getDefiningOp();
    if (subLhs || !isa<arith::MulIOp>(subRhs))
      return false;

    auto mulOp = cast<arith::MulIOp>(subRhs);
    Operation *mulLhs = mulOp.getLhs().getDefiningOp();
    Operation *mulRhs = mulOp.getRhs().getDefiningOp();
    if (mulLhs && mulRhs)
      return false;

    if (!mulLhs && isa<arith::ConstantIntOp>(mulRhs))
      return cast<arith::ConstantIntOp>(mulRhs).value() == end;
    if (!mulRhs && isa<arith::ConstantIntOp>(mulLhs))
      return cast<arith::ConstantIntOp>(mulLhs).value() == end;

    return false;
  }

private:
  using MaskedOperations = SmallPtrSet<Operation *, 8>;
  // Masked operations in the loop that can be have their mask dropped when the
  // loop is versioned using the condition builder associated with this class.
  MaskedOperations maskedOps;
  std::unique_ptr<VersioningCondition> versioningCond = nullptr;
};

class LoopVersioner {
public:
  // TODO: Extend the versioning region to encompass the downward exposed uses
  // of the return values.
  bool version(scf::ForOp &forOp, MaskedOpsCollector &collector) const {
    if (!canVersion(forOp))
      return false;

    // Collect loop results that are downward exposed.
    auto getUsedResults = [](const scf::ForOp &forOp) {
      SmallVector<Type> resTypes;
      for (Value res : forOp->getResults()) {
        if (!res.getUsers().empty())
          resTypes.push_back(res.getType());
      }
      return resTypes;
    };

    // Create the versioning condition.
    OpBuilder builder(forOp);
    Location loc = forOp.getLoc();
    Value versioningCond = collector.versioningCond->materialize(builder, loc);
    auto ifOp =
        builder.create<scf::IfOp>(loc, getUsedResults(forOp), versioningCond,
                                  /*withThenRegion=*/true,
                                  /*withElseRegion=*/true);

    // Clone the original loop into the 2 if branches.
    OpBuilder thenB = ifOp.getThenBodyBuilder();
    OpBuilder elseB = ifOp.getElseBodyBuilder();

    IRMapping map;
    Operation *thenForLoop = thenB.clone(*forOp.getOperation(), map);
    Operation *elseForLoop = elseB.clone(*forOp.getOperation());

    // Collect results in 'clonedLoop' corresponding to downward exposed results
    // 'forOp'.
    auto pruneUnusedResults = [&](const scf::ForOp &forOp,
                                  Operation *clonedLoop) {
      SmallVector<Value> prunedResults;
      for (auto [idx, val] : llvm::enumerate(forOp->getResults())) {
        if (!val.getUsers().empty())
          prunedResults.push_back(clonedLoop->getResult(idx));
      }
      return prunedResults;
    };

    // Create the yield operations for the two if branches.
    thenB.create<scf::YieldOp>(loc, pruneUnusedResults(forOp, thenForLoop));
    elseB.create<scf::YieldOp>(loc, pruneUnusedResults(forOp, elseForLoop));

    // Drop the mask from candidate masked operations in the "then" region's
    // cloned loop.
    for (Operation *maskedOp : collector.maskedOps) {
      Operation *mappedOp = map.lookup(maskedOp);
      if (auto loadOp = dyn_cast<tt::LoadOp>(mappedOp)) {
        OpBuilder builder(mappedOp);
        auto newLoad = builder.create<tt::LoadOp>(
            loadOp.getLoc(), loadOp.getPtr(), loadOp.getCache(),
            loadOp.getEvict(), loadOp.getIsVolatile());
        mappedOp->replaceAllUsesWith(newLoad);
        mappedOp->erase();
      }
      // TODO: stores
    }

    // Replace the uses of the original loop results.
    unsigned idx = 0;
    for (Value res : forOp.getResults()) {
      if (!res.getUsers().empty())
        res.replaceAllUsesWith(ifOp->getResult(idx++));
    }

    forOp.erase();

    return true;
  }

private:
  // Currently we can version the loop only is it doesn't have downward
  // exposed uses of return values that are a tensor of pointers.
  // Note: this is due to the fact the results yielded by the 2 versioning
  // branches have different types for ptr (only in one versioned loop tensor of
  // ptrs are changed to block ptrs) 'then' part of the versioning branch and
  // leave them as is in the 'else' branch).
  bool canVersion(scf::ForOp &forOp) const {
    return llvm::any_of(forOp.getResults(), [](Value res) {
      return !tt::isTensorPointerType(res.getType()) || res.getUsers().empty();
    });
  }
};

struct TritonRaiseBlockPointer
    : tt::intel::impl::TritonRaiseBlockPointerBase<TritonRaiseBlockPointer> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Drop the mask or version loops containing masked operations.
    if (IgnoreMasks)
      dropMasks(moduleOp);
    else {
      // Collect masked operations amenable to versioning in each loop.
      moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
        MaskedOpsCollector collector;
        LoopVersioner loopVersioner;
        if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op)) {
          if (collector.collectMaskedOps(forOp)) {
            [[maybe_unused]] bool loopVersioned =
                loopVersioner.version(forOp, collector);
            if (loopVersioned)
              LLVM_DEBUG(llvm::dbgs() << "Loop versioned\n");
          }
        }
        return WalkResult::advance();
      });

      LLVM_DEBUG(llvm::dbgs() << "After versioning:\n" << moduleOp << "\n");
      assert(succeeded(verify(moduleOp)) && "Module verification failed");
    }

    // Perform the transformation.
    if (failed(rewriteOp(moduleOp)))
      moduleOp->emitWarning("TritonRaiseToBlockPointer failed");

    // Cleanup unused operations.
    for (Operation *op : cleanUp) {
      if (op->getUsers().empty())
        op->erase();
    }

    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  LogicalResult rewriteOp(Operation *rootOp, bool isNested = false) {
    assert(rootOp && "Expected a valid operation");

    bool fail = false;
    rootOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op == rootOp)
        return WalkResult::advance();

      return TypeSwitch<Operation *, WalkResult>(op)
          .Case([&](tt::AddPtrOp addptr) {
            if (failed(rewriteAddPtrOp(addptr))) {
              addptr->emitRemark(
                  "TritonRaiseToBlockPointer: Failed to rewrite AddPtrOp");
              if (isNested)
                fail = true;
            }
            return WalkResult::advance();
          })
          .Case<tt::LoadOp, tt::StoreOp>([&](auto loadstore) {
            if (failed(rewriteLoadStoreOp(loadstore))) {
              loadstore->emitRemark(
                  "TritonRaiseToBlockPointer: Failed to rewrite load/store");
              if (isNested)
                fail = true;
              return WalkResult::advance();
            }
            return WalkResult::skip();
          })
          .Case<scf::ForOp>([&](auto forOp) {
            if (failed(rewriteForOp(forOp))) {
              forOp->emitRemark(
                  "TritonRaiseToBlockPointer: Failed to rewrite ForOp");
              if (isNested)
                fail = true;
              return WalkResult::advance();
            }
            return WalkResult::skip();
          })
          .Default([&](auto) { return WalkResult::advance(); });
    });

    if (fail)
      return failure();

    return success();
  }

  LogicalResult rewriteForOp(scf::ForOp op) {
    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << *op << "\n");

    for (auto &bodyOp : op.getRegion().getOps()) {
      if (auto forOp = dyn_cast<scf::ForOp>(bodyOp)) {
        op->emitRemark("TritonRaiseToBlockPointer: nested loops currently "
                       "not supported");
        return failure();
      }
    }

    SmallVector<Value> newInitArgs;
    SmallVector<std::pair<int, Value>> initArgIndex;
    OpBuilder builder(op);

    auto canBeRewrittenUsingBlockPtr = [&](Operation *op) {
      return TypeSwitch<Operation *, bool>(op)
          .Case<tt::AddPtrOp>([](auto) { return true; })
          .Case<tt::LoadOp>(
              [this](auto loadOp) { return IgnoreMasks || !loadOp.getMask(); })
          .Case<tt::StoreOp>([this](auto storeOp) {
            return IgnoreMasks || !storeOp.getMask();
          })
          .Default([](auto) { return false; });
    };

    // Create a new list of init args
    for (auto [i, arg] : llvm::enumerate(op.getInitArgs())) {
      if (Value mappedV = ptrMap.lookupOrNull(arg)) {
        if (auto makeTensorPtrOp =
                mappedV.getDefiningOp<tt::MakeTensorPtrOp>()) {
          if (llvm::any_of(op.getRegionIterArgs()[i].getUsers(),
                           [&](Operation *user) {
                             return !canBeRewrittenUsingBlockPtr(user);
                           })) {
            op->emitRemark("TritonRaiseToBlockPointer: Loop contains ops that "
                           "cannot be rewritten using a block ptr");
            return failure();
          }

          // replace the argument with the mapped value, and register the new
          // pointer
          newInitArgs.push_back(mappedV);
          initArgIndex.push_back(std::make_pair(i, mappedV));

          continue;
        } else {
          llvm::errs() << "mappedV: " << mappedV << "\n";
          llvm_unreachable("Unexpected mapped value");
        }
      }

      // If any of the analysis failed, or init arg is not pointer related or
      // prior rewrite has failed. Pass as is
      newInitArgs.push_back(arg);
    }

    // Create a new scf::ForOp that uses updated init args and same loop body
    auto newOp = builder.create<scf::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep(),
        newInitArgs,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
          IRMapping cloneMap;
          cloneMap.map(op.getInductionVar(), iv);
          cloneMap.map(op.getInitArgs(), newInitArgs);
          cloneMap.map(op.getRegionIterArgs(), args);
          for (auto &bodyOp : op.getRegion().getOps())
            b.clone(bodyOp, cloneMap);
        });

    for (auto [i, mappedV] : initArgIndex)
      ptrMap.map(newOp.getRegionIterArgs()[i], mappedV);

    // Update the loop body.
    constexpr bool isNested = true;
    if (failed(rewriteOp(newOp, isNested))) {
      newOp->erase();
      op->emitRemark("TritonRaiseToBlockPointer: update loop body failed when "
                     "rewriting for op");
      return failure();
    }

    // Rewrite the yield operation.
    if (op.getNumRegionIterArgs()) {
      auto yieldOp = cast<scf::YieldOp>(newOp.getBody()->getTerminator());
      for (auto [i, v] : llvm::enumerate(yieldOp->getOperands())) {
        if (Value mappedV = ptrMap.lookupOrNull(v))
          yieldOp->replaceUsesOfWith(v, mappedV);
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "After updating the loop body\n";
      llvm::dbgs() << "new for:\n";
      newOp->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";

      llvm::dbgs() << "old for:\n";
      op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
      llvm::dbgs() << "\n";
    });

    // Replace the results that correspond to the original scf.for
    ResultRange resultsToReplaceWith(newOp.result_begin(),
                                     newOp.result_begin() + op.getNumResults());
    op->replaceAllUsesWith(resultsToReplaceWith);
    op->erase();

    LLVM_DEBUG({
      auto modOp =
          builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
      llvm::dbgs() << "Module:\n" << modOp << "\n";
    });

    return success();
  }

  bool lookForMultiplyingValueInDefiningPath(Value &val, Value &ref) const {
    if (Operation *defOp = getFinalValue(val).getDefiningOp()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(defOp)) {
        if ((mulOp.getLhs() == ref) || (mulOp.getRhs() == ref))
          return true;
      }
    }
    return false;
  }

  bool areValuesEqual(Value val1, Value val2) const {
    if (val1 == val2)
      return true;

    Operation *op1 = val1.getDefiningOp();
    Operation *op2 = val2.getDefiningOp();
    if (op1 && op2) {
      std::optional<int64_t> intVal1 = ttgi::getFoldedConstantValue(op1);
      std::optional<int64_t> intVal2 = ttgi::getFoldedConstantValue(op2);
      if (intVal1.has_value() && intVal2.has_value())
        return intVal1.value() == intVal2.value();
    }
    return false;
  }

  std::optional<unsigned>
  checkIfOffsetMultipliedByStride(Value operand,
                                  SmallVector<Value> &strides) const {
    Operation *defOp = operand.getDefiningOp();

    SmallVector<Value> finalStrides;
    // check whether all strides are different, if not => skip
    for (auto stride : strides) {
      Value currentVal = getFinalValue(stride);
      if (llvm::any_of(finalStrides, [&](Value val) {
            return areValuesEqual(val, currentVal);
          }))
        return std::nullopt;
      finalStrides.push_back(currentVal);
    }

    unsigned axis = 0u;
    for (auto finalStride : finalStrides) {
      // search for a mul to finalStride in the predecessors
      if (lookForMultiplyingValueInDefiningPath(operand, finalStride))
        return axis;
      if (ttgi::isConstant(finalStride, 1))
        return axis;
      ++axis;
    }
    return std::nullopt;
  }

  // Return true if a `tt::ExpandOp` has been found is the defining path.
  bool hasExpandOpInDefiningPath(Value value) const {
    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
      // look init values outside the loop
      BlockArgument blockArg = dyn_cast<BlockArgument>(value);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp);
      return forOp ? hasExpandOpInDefiningPath(
                         forOp.getInitArgs()[blockArg.getArgNumber() - 1])
                   : false;
    }

    if (isa<tt::ExpandDimsOp>(defOp))
      return true;
    if (isa<arith::ConstantOp, tt::MakeRangeOp>(defOp))
      return false;
    if (isa<tt::BroadcastOp>(defOp) || isa<tt::SplatOp>(defOp) ||
        isa<arith::IndexCastOp>(defOp) || isa<arith::RemUIOp>(defOp) ||
        isa<arith::RemSIOp>(defOp))
      return hasExpandOpInDefiningPath(defOp->getOperand(0));
    if (isa<arith::AddIOp>(defOp) || isa<arith::MulIOp>(defOp))
      return hasExpandOpInDefiningPath(defOp->getOperand(0)) ||
             hasExpandOpInDefiningPath(defOp->getOperand(1));

    return true;
  }

  LogicalResult rewriteAddPtrOp(tt::AddPtrOp op) {
    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value ptr = op.getPtr();

    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << *op << "\n");

    // Case 1: the ptr has been already been mapped.
    if (Value mappedV = ptrMap.lookupOrNull(ptr)) {
      // Case 1a: the ptr has been mapped to a make_tensor_ptr operation.
      if (auto makeTPtrOp = mappedV.getDefiningOp<tt::MakeTensorPtrOp>()) {
        PtrState state;
        if (failed(visitOperand(op.getOffset(), state, loc, builder)))
          return failure();

        Value basePtr = tt::isTensorPointerType(ptr.getType()) ? ptr : mappedV;
        std::optional<Value> advanceOp =
            state.createTTAdvanceOp(basePtr, makeTPtrOp, builder, loc);
        if (!advanceOp.has_value()) {
          LLVM_DEBUG(llvm::dbgs() << "Rewriting failed for: " << *op << "\n");
          return failure();
        }

        cleanUp.insert(op);
        ptrMap.map(op.getResult(), *advanceOp);

        LLVM_DEBUG({
          llvm::dbgs() << "Rewrote:\n\t" << op << "\nto:\n\t" << advanceOp
                       << "\n";
          auto modOp =
              builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
          llvm::dbgs() << "Module:\n" << modOp << "\n";
        });

        return success();
      }

      // Case 1b: the ptr has been mapped to a tt.advance operation.
      if (auto advanceOp = mappedV.getDefiningOp<tt::AdvanceOp>()) {
        PtrState state;
        if (failed(visitOperand(op.getOffset(), state, loc, builder)))
          return failure();

        // Skip through a chain of tt.advance operations...
        Value ptr = advanceOp.getPtr();
        while (auto advanceOp = ptr.getDefiningOp<tt::AdvanceOp>())
          ptr = advanceOp.getPtr();

        // ... until we find the make_tensor_ptr operation defining the block
        // ptr feeding the first tt.advance operation.
        auto makeTPtrOp = ptr.getDefiningOp<tt::MakeTensorPtrOp>();
        assert(makeTPtrOp && "Expected a MakeTensorPtrOp");

        std::optional<Value> newAdvanceOp = state.createTTAdvanceOp(
            advanceOp.getResult(), makeTPtrOp, builder, loc);
        if (!newAdvanceOp.has_value()) {
          LLVM_DEBUG(llvm::dbgs() << "Rewriting failed for: " << *op << "\n");
          return failure();
        }

        cleanUp.insert(op);
        ptrMap.map(op.getResult(), *newAdvanceOp);

        LLVM_DEBUG({
          llvm::dbgs() << "Rewrote:\n\t" << op << "\nto:\n\t" << newAdvanceOp
                       << "\n";
          auto modOp =
              builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
          llvm::dbgs() << "Module:\n" << modOp << "\n";
        });

        return success();
      }

      llvm_unreachable("Unexpected mappedV defining operation");
    }

    // Case 2: the ptr has not previously been mapped.
    // If the addptr operation increments a scalar pointer, give up.
    Value result = op.getResult();
    if (!isa<RankedTensorType>(result.getType())) {
      LLVM_DEBUG(llvm::dbgs() << "Rewriting failed for: " << *op << "\n");
      return failure();
    }

    // Otherwise, attempt to rewrite the AddPtrOp into a MakeTensorPtrOp.
    PtrState state;
    if (failed(visitOperandAddptr(op, state, loc, builder))) {
      LLVM_DEBUG(llvm::dbgs() << "Rewriting failed for: " << *op << "\n");
      return failure();
    }

    assert(!state.isBlockPtr() && "Expected tensor of pointers");

    knownPtrs[result] = state;

    Value makePtrOp = state.createTTMakeTensorPtrOp(builder, loc);
    knownPtrs[makePtrOp] = std::move(state);

    cleanUp.insert(op);
    ptrMap.map(result, makePtrOp);

    LLVM_DEBUG({
      llvm::dbgs() << "Rewrote:\n\t" << op << "\nto:\n\t" << makePtrOp << "\n";
      auto modOp =
          builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
      llvm::dbgs() << "Module:\n" << modOp << "\n";
    });

    return success();
  }

  LogicalResult visitOperandMakeTensorPtr(tt::MakeTensorPtrOp makeTPtrOp,
                                          PtrState &state, const Location loc,
                                          OpBuilder &builder,
                                          bool addedByPass = false) {
    assert(state.isEmpty() && "state is a return argument");

    if (auto iter = knownPtrs.find(makeTPtrOp.getResult());
        iter != knownPtrs.end()) {
      state = iter->second;
      return success();
    }

    state.source = makeTPtrOp.getBase();

    auto resType = cast<tt::PointerType>(makeTPtrOp.getResult().getType());
    auto pointeeType = cast<ShapedType>(resType.getPointeeType());
    ArrayRef<int64_t> shape = pointeeType.getShape();

    for (int i = 0; i < pointeeType.getRank(); i++) {
      state.sizes.push_back(shape[i]);

      auto strideCst = builder.createOrFold<arith::IndexCastOp>(
          loc, builder.getIndexType(), makeTPtrOp.getStrides()[i]);
      auto offsetCst = builder.createOrFold<arith::IndexCastOp>(
          loc, builder.getIndexType(), makeTPtrOp.getOffsets()[i]);
      auto scaledOffset =
          builder.createOrFold<arith::MulIOp>(loc, offsetCst, strideCst);
      state.offsets.push_back(
          findOrCreateCast(loc, getFinalValue(scaledOffset),
                           builder.getIntegerType(offsetBitwidth), builder));
    }
    state.strides = makeTPtrOp.getStrides();
    state.shape = makeTPtrOp.getShape();
    state.order = SmallVector<int>(makeTPtrOp.getOrder());

    return success();
  }

  LogicalResult visitOperandAddptr(tt::AddPtrOp addptrOp, PtrState &state,
                                   Location loc, OpBuilder &builder) {
    assert(state.isEmpty() && "state is a return argument");

    PtrState ptrState;
    if (failed(visitOperand(addptrOp.getPtr(), ptrState, loc, builder)))
      return failure();

    PtrState offsetState;
    if (failed(visitOperand(addptrOp.getOffset(), offsetState, loc, builder)))
      return failure();

    // The axis to which the offset must be applied need to be known.
    // However, in some cases, the pass fails to detect whether an offset
    // should be applied to an axis other than the first. We, therefore, try
    // to find out if the offset is multiplied by a known stride. Example:
    //    off += BLOCK_SIZE_K * stride_ak
    // Indeed, as the axis of the stride is known with certainty, we can
    // assume that if the offset is multiplied by a known stride, the axis of
    // offset should correspond to the axis of the stride axis. In the
    // previous example, suppose we have strides = [stride_am, stride_ak] but
    // offsets = [off, 0] As we found that `off` is multiplied by `stride_ak`,
    // we correct the axis of the offsets to align the axis of `off` with axis
    // of `stride_ak`. The corrected offsets then become: [0, off]
    // Limitations:
    //     - this approach based on pattern matching + user code assumptions
    //     is (very) fragile.
    //       if user code does not directly multiply the offset by the stride
    //       value identified by the pass, the analysis will fail.
    //     - in theory, this correction support should fail if the analysis
    //     cannot reach a certain level of certainty.
    //       Typically, if stride values are the same (e.g. [512, 512]), the
    //       support is unable to determine the right axis and will not
    //       correct anything. That said, we do not guarantee the current
    //       support does not give rise to false positive detections.
    Operation *parentOp = addptrOp->getParentOp();
    if (isa<scf::ForOp>(parentOp)) {
      // ExpandOp directly sets offset to the expected axis.
      // So if an ExpandOp has been found in defining path, the analysis is
      // skipped.
      if (!hasExpandOpInDefiningPath(addptrOp.getOffset())) {
        std::optional<unsigned> axis = checkIfOffsetMultipliedByStride(
            addptrOp.getOffset(), ptrState.strides);
        if (axis && *axis >= 1)
          std::swap(offsetState.offsets[0], offsetState.offsets[*axis]);
      }
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

    if (isa<IndexType>(operand.getType())) {
      state.scalar = operand;
      return success();
    }

    if (isa<IntegerType>(operand.getType())) {
      OpBuilder::InsertionGuard guard(builder);
      if (Operation *definingOp = operand.getDefiningOp())
        builder.setInsertionPointAfter(definingOp);
      state.scalar = builder.createOrFold<arith::IndexCastOp>(
          loc, builder.getIndexType(), operand);
      return success();
    }

    if (isa<tt::PointerType>(operand.getType())) {
      // A scalar pointer can either be produced by AddPtrOp or a block
      // argument
      if (Operation *op = operand.getDefiningOp()) {
        if (auto addPtrOp = dyn_cast<tt::AddPtrOp>(op))
          return visitOperandAddptr(addPtrOp, state, loc, builder);
        if (isa<tt::MakeTensorPtrOp>(op))
          llvm_unreachable(
              "Unexpected operand defining operation tt.make_tensor_ptr");
        llvm_unreachable("Unexpected operand defining operation");
      } else {
        // If the operand is an iter-arg of an for loop, give up.
        if (isa<scf::ForOp>(operand.getParentBlock()->getParentOp()))
          return failure();

        state.source = operand;
        return success();
      }
    }

    Operation *definingOp = operand.getDefiningOp();
    if (!definingOp) {
      if (!knownPtrs.contains(operand)) {
        llvm::errs() << "TritonRaiseBlockPointer: encountered addptr block "
                        "argument operand\n"
                     << operand << "\n";
        return failure();
      }

      // This operand must be an iter-arg of an inner-loop in a multiple-level
      // nested loop, which means its PtrState must have already been
      // populated during rewriteForOp of the parent loop.
      state = knownPtrs[operand];
      return success();
    }

    return TypeSwitch<Operation *, LogicalResult>(definingOp)
        .Case<arith::AddIOp, arith::ConstantOp, arith::MulIOp, arith::RemUIOp,
              arith::RemSIOp, arith::ExtSIOp, arith::ExtUIOp, tt::BroadcastOp,
              tt::MakeRangeOp, tt::SplatOp, tt::ExpandDimsOp>(
            [this, &state, loc, &builder](auto op) {
              return visitAddPointerOperand(op, state, loc, builder);
            })
        .Default([](Operation *op) {
          llvm::errs() << "TritonRaiseBlockPointer: encountered addptr operand "
                          "produced by unsupported operation: "
                       << *op << "\n";
          return failure();
        });
  }

  template <typename OpTy>
  LogicalResult visitAddPointerOperand(OpTy op, PtrState &state, Location loc,
                                       OpBuilder &builder);

  template <typename OpTy,
            std::enable_if_t<
                llvm::is_one_of<OpTy, arith::RemSIOp, arith::RemUIOp>::value,
                bool> = true>
  LogicalResult visitAddPointerRemOperand(OpTy remOp, PtrState &state,
                                          Location loc, OpBuilder &builder);

  template <typename OpTy,
            std::enable_if_t<
                llvm::is_one_of<OpTy, arith::ExtSIOp, arith::ExtUIOp>::value,
                bool> = true>
  LogicalResult visitAddPointerExtOperand(OpTy extOp, PtrState &state,
                                          Location loc, OpBuilder &builder);

  template <
      typename OpTy,
      std::enable_if_t<llvm::is_one_of<OpTy, tt::LoadOp, tt::StoreOp>::value,
                       bool> = true>
  LogicalResult rewriteLoadStoreOp(OpTy op) {
    // If the pointer is already a block pointer, there is nothing to do.
    if (tt::isTensorPointerType(op.getPtr().getType()))
      return success();

    // If the pointer doesn't have a corresponding block pointer, there is
    // nothing to do.
    Value ptr = ptrMap.lookupOrNull(op.getPtr());
    if (!ptr)
      return success();

    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << *op << "\n");

    constexpr bool isLoad = std::is_same_v<OpTy, tt::LoadOp>;
    constexpr StringLiteral opName =
        isLoad ? StringLiteral("loadOp") : StringLiteral("storeOp");

    auto ptrType = dyn_cast<tt::PointerType>(ptr.getType());
    if (ptrType && !isa<ShapedType>(ptrType.getPointeeType())) {
      op->emitRemark("TritonRaiseBlockPointer: scalar ")
          << opName << " will not be rewritten";
      return failure();
    }

    // As masks are incompatible with block pointer load/store ops
    // Masks must be handled before the operation can be rewritten.
    // This will be done in a future PR (Issue #1784).
    // In the meantime, operations with a mask are not rewritten.
    if (op.getMask())
      return success();

    SmallVector<int> boundary;
    if (auto iter = knownPtrs.find(ptr); iter != knownPtrs.end()) {
      PtrState state = iter->second;
      for (int axis = 0; axis < state.shape.size(); ++axis) {
        if (!ttgi::isConstant(state.shape[axis], 0))
          boundary.push_back(axis);
      }
    }
    ArrayRef<int> newBoundaryCheck(boundary);

    OpBuilder builder(op);
    if constexpr (isLoad) {
      auto loadOp = builder.createOrFold<tt::LoadOp>(
          op.getLoc(), ptr, newBoundaryCheck, op.getPadding(), op.getCache(),
          op.getEvict(), op.getIsVolatile());
      LLVM_DEBUG(llvm::dbgs() << "Created: " << loadOp << "\n";);
      op.replaceAllUsesWith(loadOp);
    } else {
      [[maybe_unused]] auto storeOp = builder.createOrFold<tt::StoreOp>(
          op.getLoc(), ptr, op.getValue(), newBoundaryCheck, op.getCache(),
          op.getEvict());
      LLVM_DEBUG(llvm::dbgs() << "Created: " << storeOp << "\n";);
    }

    op->erase();

    LLVM_DEBUG({
      auto modOp =
          builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
      llvm::dbgs() << "Module:\n" << modOp << "\n";
    });

    return success();
  }

  void dropMasks(ModuleOp moduleOp) const {
    assert(IgnoreMasks && "Expecting 'IgnoreMask' flag to be set");

    SmallVector<Operation *> opsWithMask;
    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<tt::LoadOp, tt::StoreOp>([&](auto opWithMask) {
            if (opWithMask.getMask()) {
              opsWithMask.push_back(opWithMask);
            }
            return WalkResult::advance();
          })
          .Default([&](auto) { return WalkResult::advance(); });
    });

    for (Operation *op : opsWithMask) {
      TypeSwitch<Operation *>(op)
          .Case<tt::LoadOp>([&](auto loadOp) {
            loadOp->emitWarning("TritonRaiseBlockPointer: ignoring mask");
            OpBuilder builder(loadOp);
            auto newLoadOp = builder.create<tt::LoadOp>(
                loadOp.getLoc(), loadOp.getPtr(), loadOp.getBoundaryCheck(),
                loadOp.getPadding(), loadOp.getCache(), loadOp.getEvict(),
                loadOp.getIsVolatile());
            loadOp->replaceAllUsesWith(newLoadOp);
            loadOp->erase();
          })
          .Case<tt::StoreOp>([&](auto storeOp) {
            storeOp->emitWarning("TritonRaiseBlockPointer: ignoring mask");
            OpBuilder builder(storeOp);
            auto newStoreOp = builder.createOrFold<tt::StoreOp>(
                storeOp.getLoc(), storeOp.getPtr(), storeOp.getValue(),
                storeOp.getBoundaryCheck(), storeOp.getCache(),
                storeOp.getEvict());

            Operation *maskOpToErase = nullptr;
            if (storeOp.getMask().hasOneUse())
              maskOpToErase = storeOp.getMask().getDefiningOp();

            storeOp->erase();
            if (maskOpToErase)
              maskOpToErase->erase();
          });
    }
  }

  static void dump(const IRMapping &map) {
    for (auto [key, val] : map.getValueMap()) {
      llvm::dbgs() << "key: " << key << "(0x" << &key << "), value: " << val
                   << "\n";
    }
  }

  static void dump(const llvm::SmallDenseMap<Value, PtrState> &knownPtrs) {
    for (auto [key, state] : knownPtrs) {
      llvm::dbgs() << "key: " << key << " state: " << state << "\n";
    }
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
  llvm::SmallDenseMap<Value, PtrState> knownPtrs;
  IRMapping ptrMap;
};

template <
    typename OpTy,
    std::enable_if_t<
        llvm::is_one_of<OpTy, arith::RemSIOp, arith::RemUIOp>::value, bool>>
LogicalResult TritonRaiseBlockPointer::visitAddPointerRemOperand(
    OpTy remOp, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  PtrState rhsState;
  if (failed(visitOperand(remOp.getRhs(), rhsState, loc, builder)))
    return failure();

  if (!rhsState.scalar) {
    remOp->emitRemark(
        "TritonRaiseBlockPointer: only support cases when rhs of remainder "
        "contains scalar");
    return failure();
  }

  if (failed(visitOperand(remOp.getLhs(), state, loc, builder)))
    return failure();

  // If there are multiple modulo ops on an expression (e.g.: (a % b) % c), we
  // would have already populated the modulo states after visiting the lhs.
  // Assert that all the modulo states are empty.
  if (state.hasModulo()) {
    remOp->emitRemark("TritonRaiseBlockPointer: do not support multiple modulo "
                      "within an expression");
    return failure();
  }

  switch (state.getRank()) {
  case 1:
    // Apply the modulo before expanding shape, the common pattern is
    // offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    // a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] *
    // stride_ak)
    state.shape.back() = rhsState.scalar;
    break;
  case 2: {
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
                        "with no singleton dimension not supported");
      return failure();
    }
    break;
  }
  default:
    remOp->emitRemark("TritonRaiseBlockPointer: unsupported modulo pattern");
    return failure();
  }

  LLVM_DEBUG(llvm::dbgs().indent(2) << "RemOp state: " << state << "\n";);
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

template <
    typename OpTy,
    std::enable_if_t<
        llvm::is_one_of<OpTy, arith::ExtSIOp, arith::ExtUIOp>::value, bool>>
LogicalResult TritonRaiseBlockPointer::visitAddPointerExtOperand(
    OpTy extOp, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");
  return visitOperand(extOp.getIn(), state, loc, builder);
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::ExtSIOp extOp, PtrState &state, Location loc, OpBuilder &builder) {
  return visitAddPointerExtOperand(extOp, state, loc, builder);
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::ExtUIOp extOp, PtrState &state, Location loc, OpBuilder &builder) {
  return visitAddPointerExtOperand(extOp, state, loc, builder);
}

template <>
LogicalResult
TritonRaiseBlockPointer::visitAddPointerOperand(tt::MakeRangeOp rangeOp,
                                                PtrState &state, Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  ArrayRef<int64_t> shape = cast<ShapedType>(rangeOp.getType()).getShape();
  unsigned start = rangeOp.getStart();
  unsigned end = rangeOp.getEnd();
  unsigned stride = (end - start + shape[0] - 1) / shape[0];
  assert(stride == 1 &&
         "Expect make_range op to always return tensor of stride 1");

  state.offsets.push_back(
      findOrCreateConstant(loc, start, offsetBitwidth, builder));
  state.strides.push_back(
      findOrCreateConstant(loc, stride, shapeAndStridesBitwidth, builder));
  state.shape.push_back(
      findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder));
  state.sizes.push_back(shape[0]);

  LLVM_DEBUG(llvm::dbgs().indent(2) << "MakeRange state: " << state << "\n";);
  return success();
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    tt::SplatOp splatOp, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  Value src = splatOp.getSrc();
  Value dst = splatOp.getResult();
  ArrayRef<int64_t> dstShape = cast<ShapedType>(dst.getType()).getShape();

  if (failed(visitOperand(src, state, loc, builder)))
    return failure();

  if (!isa<IntegerType, IndexType, tt::PointerType>(src.getType())) {
    splatOp->emitRemark("TritonRaiseBlockPointer: unsupported splat pattern");
    return failure();
  }

  Value c0i32 = findOrCreateConstant(loc, 0, offsetBitwidth, builder);
  Value c0i64 = findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder);

  for (int64_t s : dstShape) {
    state.offsets.push_back(c0i32);
    state.strides.push_back(c0i64);
    state.shape.push_back(c0i64);
    state.sizes.push_back(s);
  }

  // If we splat a integer value, scalar should become the offset of the outer
  // most dimension.
  if (state.scalar)
    state.offsets[0] = findOrCreateCast(
        loc, state.scalar, builder.getIntegerType(offsetBitwidth), builder);

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Splat state: " << state << "\n";);
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

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Add state: " << state << "\n";);
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

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Mul state: " << state << "\n";);
  return success();
}

template <>
LogicalResult TritonRaiseBlockPointer::visitAddPointerOperand(
    arith::ConstantOp op, PtrState &state, Location loc, OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  auto attr = cast<DenseElementsAttr>(op.getValue());
  assert(attr.isSplat() && isa<IntegerType>(attr.getElementType()) &&
         "Expecting constant tensor");

  state.scalar = builder.createOrFold<arith::ConstantIndexOp>(
      loc, attr.getValues<IntegerAttr>()[0].getValue().getSExtValue());

  Type offsetType = builder.getIntegerType(offsetBitwidth);
  auto resultType = cast<ShapedType>(op.getResult().getType());
  Value offset = convertScalarToDtype(builder, loc, state.scalar, offsetType,
                                      /*isUnsignedCast=*/true);
  state.offsets.push_back(offset);
  state.offsets.insert(state.offsets.end(), resultType.getShape().size() - 1,
                       findOrCreateConstant(loc, 0, offsetBitwidth, builder));
  state.strides.insert(
      state.strides.end(), resultType.getShape().size(),
      findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder));
  state.shape.insert(
      state.shape.end(), resultType.getShape().size(),
      findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder));

  for (int dim : resultType.getShape())
    state.sizes.push_back(dim);

  LLVM_DEBUG(llvm::dbgs().indent(2) << "ConstantOp state: " << state << "\n";);
  return success();
}

template <>
LogicalResult
TritonRaiseBlockPointer::visitAddPointerOperand(tt::ExpandDimsOp expandDimsOp,
                                                PtrState &state, Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  if (failed(visitOperand(expandDimsOp.getSrc(), state, loc, builder)))
    return failure();

  ArrayRef<int64_t> dstShape =
      cast<ShapedType>(expandDimsOp.getResult().getType()).getShape();
  unsigned axis = expandDimsOp.getAxis();

  assert(dstShape[axis] == 1 &&
         "expect changed dimension to be 1 in expand_dims");

  // insert dimension info
  Value c0i32 = findOrCreateConstant(loc, 0, offsetBitwidth, builder);
  Value c0i64 = findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder);
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

  LLVM_DEBUG(llvm::dbgs().indent(2) << "ExpandDims state: " << state << "\n";);
  return success();
}

template <>
LogicalResult
TritonRaiseBlockPointer::visitAddPointerOperand(tt::BroadcastOp broadcastOp,
                                                PtrState &state, Location loc,
                                                OpBuilder &builder) {
  assert(state.isEmpty() && "state is a return argument");

  Value src = broadcastOp.getSrc();
  if (!isa<ShapedType>(src.getType())) {
    broadcastOp->emitRemark(
        "TritonRaiseBlockPointer: Unsupported broadcast source type");
    return failure();
  }

  Value dst = broadcastOp.getResult();
  ArrayRef<int64_t> srcShape = cast<ShapedType>(src.getType()).getShape();
  ArrayRef<int64_t> dstShape = cast<ShapedType>(dst.getType()).getShape();

  assert(srcShape.size() <= dstShape.size() &&
         "rank of source cannot be greater than the rank of destination");

  if (failed(visitOperand(src, state, loc, builder)))
    return failure();

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
    // The positions of the new axis are determined based and the shape
    // values. If shape are the same, the new axis are added at the end.
    size_t srcAxis = 0;
    for (size_t axis = 0; axis < dstShape.size(); ++axis) {
      if ((srcAxis < srcShape.size()) &&
          (srcShape[srcAxis] == dstShape[axis])) {
        ++srcAxis;
        continue;
      }
      Value c0i32 = findOrCreateConstant(loc, 0, offsetBitwidth, builder);
      Value c0i64 =
          findOrCreateConstant(loc, 0, shapeAndStridesBitwidth, builder);
      state.offsets.insert(
          state.offsets.begin() + axis,
          findOrCreateCast(loc, state.offsets[0],
                           builder.getIntegerType(offsetBitwidth), builder));
      state.sizes.insert(state.sizes.begin() + axis, dstShape[axis]);
      state.strides.insert(state.strides.begin() + axis, c0i64);
      state.shape.insert(state.shape.begin() + axis, c0i64);
    }

    // The following condition has been duplicated from the expand_dim support
    // TODO : Verify if we need still need it given that triton
    // `make_block_ptr` op differs from triton-shared `make_block_ptr` op
    // regarding how address wrap around are handled.
    if (state.hasModulo() && state.getRank() > 2) {
      broadcastOp->emitRemark("TritonRaiseBlockPointer: unsupported scenario "
                              "where broadcast result "
                              "has modulo and rank > 2");
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs().indent(2) << "Broadcast state: " << state << "\n";);
  return success();
}

} // namespace
