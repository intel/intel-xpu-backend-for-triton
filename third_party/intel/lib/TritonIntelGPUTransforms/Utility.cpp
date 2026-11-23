//===- Utility.cpp - Triton Intel GPU utilities -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/Utility.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {

template <typename OpType,
          typename = std::enable_if_t<llvm::is_one_of<
              OpType, tt::DescriptorLoadOp, tt::DescriptorStoreOp>::value>>
RankedTensorType getRankedTensorType(OpType op) {
  if constexpr (std::is_same_v<OpType, tt::DescriptorLoadOp>)
    return op.getType();
  if constexpr (std::is_same_v<OpType, tt::DescriptorStoreOp>)
    return op.getSrc().getType();
}

RankedTensorType getRankedTensorType(Type ptrTy) {
  return dyn_cast<RankedTensorType>(ptrTy);
}

static bool isSingleValue(Value value) {
  // Don't consider load as expensive if it is loading a scalar.
  if (auto tensorTy = getRankedTensorType(value.getType()))
    return tensorTy.getNumElements() == 1;
  // TODO: Handle other cases.
  // For example, when ptr is a tensor of single value.
  // It means that ptr is a resultant of broadcast or generated through
  // a chain of broadcast and other operations.
  // Rematerialize it without considering contiguous memory access pattern is
  // fine.
  return true;
}

bool isDivisible(Value value, int64_t divisor) {
  // Nothing is provably divisible by zero, and a negative divisor has no
  // meaning for the extents, strides and offsets callers pass. Callers must
  // reject such divisors themselves; release builds answer conservatively.
  assert(divisor > 0 && "Expecting a positive divisor");
  if (divisor <= 0)
    return false;

  // Every integer is divisible by 1, regardless of how `value` is defined.
  if (divisor == 1)
    return true;

  // Case 1: Value is defined by a constant operation. Integer constants are
  // interpreted as signed, as the index arithmetic producing them is.
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
    if (!integerAttr)
      return false;
    std::optional<int64_t> intVal = integerAttr.getValue().trySExtValue();
    return intVal && *intVal % divisor == 0;
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
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // An iteration argument advanced by a loop-invariant amount,
      // `x = init; x += c`, holds `init + i * c` on iteration i, so it is
      // divisible on every iteration if both `init` and `c` are. `c` must be
      // defined outside the loop, which keeps it independent of `x` and the
      // recursion finite.
      if (blockArg != forOp.getInductionVar()) {
        Value step = tt::intel::getIterArgIncrement(forOp, blockArg);
        if (!step || !forOp.isDefinedOutsideOfLoop(step))
          return false;
        return isDivisible(forOp.getTiedLoopInit(blockArg)->get(), divisor) &&
               isDivisible(step, divisor);
      }
      // Nested loops aren't currently handled.
      if (forOp->template getParentOfType<scf::ForOp>())
        return false;
      if (!forOp.getSingleInductionVar())
        return false;
      return isDivisible(forOp.getLowerBound(), divisor) &&
             isDivisible(forOp.getStep(), divisor);
    }
  }

  // Case 3: Value is defined by a muli operation.
  if (auto mulIOp = value.getDefiningOp<arith::MulIOp>()) {
    return isDivisible(mulIOp->getOperand(0), divisor) ||
           isDivisible(mulIOp->getOperand(1), divisor);
  }

  // Case 4: Value is defined by arith::ExtSIOp, arith::TruncIOp,
  // tt::AddPtrOp, arith::AddIOp, arith::SubIOp, arith::MinSIOp or
  // arith::MaxSIOp operation. The result of min/max is one of its operands.
  if (auto *op = value.getDefiningOp()) {
    if (isa<arith::ExtSIOp, arith::TruncIOp, tt::AddPtrOp, arith::AddIOp,
            arith::SubIOp, arith::MinSIOp, arith::MaxSIOp>(op)) {
      return llvm::all_of(op->getOperands(), [&](Value operand) {
        return isDivisible(operand, divisor);
      });
    }
  }

  // Case 5: Value is defined by an arith.select operation. The condition is
  // irrelevant as long as both candidate values are divisible.
  if (auto selectOp = value.getDefiningOp<arith::SelectOp>()) {
    return isDivisible(selectOp.getTrueValue(), divisor) &&
           isDivisible(selectOp.getFalseValue(), divisor);
  }

  return false;
}

// Only the x offsets take an encoding: the slice of the result along dim 1.
static Attribute inferGatherSrcEncoding(MLIRContext *ctx, Attribute dstEnc) {
  return SliceEncodingAttr::get(ctx, 1, cast<DistributedEncodingTrait>(dstEnc));
}

Attribute inferSrcEncoding(Operation *op, Attribute encoding) {
  if (auto dotEnc = dyn_cast<DotOperandEncodingAttr>(encoding)) {
    if (auto parentEnc = dyn_cast<DpasEncodingAttr>(dotEnc.getParent())) {
      if (auto fp4ToFpOp = dyn_cast<gpu::Fp4ToFpOp>(op)) {
        // Dispatch DotEncoding + DPASEncoding to the
        // TritonIntelGPUInferLayoutInterface
        Attribute srcEnc;
        llvm::ArrayRef<int64_t> shape = fp4ToFpOp.getSrc().getType().getShape();
        if (succeeded(parentEnc.getDialect()
                          .getRegisteredInterface<DialectInferLayoutInterface>()
                          ->inferFp4ToFpOpEncoding(
                              shape, fp4ToFpOp.getAxis(), parentEnc, srcEnc,
                              /*fwdInference*/ false, std::nullopt)))
          return srcEnc;
        return {};
      }
    }
  }

  if (isa<tt::DescriptorGatherOp, ttgi::DescriptorGatherOp>(op))
    return inferGatherSrcEncoding(op->getContext(), encoding);

  return mlir::inferSrcEncoding(op, encoding);
}

Value buildConvertedXOffsets(OpBuilder &builder, Location loc, Value xOffsets,
                             RankedTensorType valueType) {
  auto xOffsetsType = cast<RankedTensorType>(xOffsets.getType());
  Attribute enc =
      inferGatherSrcEncoding(builder.getContext(), valueType.getEncoding());
  return ConvertLayoutOp::create(builder, loc,
                                 xOffsetsType.cloneWithEncoding(enc), xOffsets);
}

CachePolicy getCachePolicy(Attribute cachePolicy) {
  auto policy = dyn_cast_if_present<tt::CachePolicyAttr>(cachePolicy);
  if (!policy)
    return {};
  return {policy.getCacheModifier(), policy.getEvictionPolicy()};
}

bool isVolatileOrCVLoad(Operation *op) {
  auto load = dyn_cast<tt::LoadOp>(op);
  return load && (load.getIsVolatile() ||
                  getCachePolicy(load.getCachePolicyAttr()).cacheModifier ==
                      tt::CacheModifier::CV);
}

/// True for a masked load whose result does not reach a dot. `block_io` is what
/// tells the layout propagation the load is free in any encoding, so the load
/// is rematerialized into its consumer's layout. That is the right trade when
/// the consumer is a dot: the 2D block message delivers the DPAS operand
/// directly. With any other consumer the load gives up its coalesced encoding
/// for nothing, and the mask bounds the resulting tile besides. Keep such a
/// load anchored -- it can still lower to a 2D block message, just in the
/// encoding the memory access itself wants. A broadcast load whose result is
/// combined with a DPAS value (e.g. a GEMM bias) is the exception: it is never
/// an anchor, see feedsDpasValue.
static bool isMaskedLoadWithoutDotConsumer(Operation *op) {
  auto loadOp = dyn_cast<tt::LoadOp>(op);
  if (!loadOp)
    return false;
  Value mask = loadOp.getMask();
  if (!mask || matchPattern(mask, m_One()))
    return false;

  SmallVector<Value> worklist{loadOp.getResult()};
  SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation *user : val.getUsers()) {
      if (isa<tt::DotOp, tt::DotScaledOp>(user))
        return false;
      // Only look through layout-preserving and elementwise producers: a dot
      // reached past a reduction is not fed by this tile.
      if (!isa<ttg::ConvertLayoutOp, tt::FpToFpOp, tt::TransOp, tt::BroadcastOp,
               tt::ReshapeOp, tt::SplatOp>(user) &&
          !user->hasTrait<OpTrait::Elementwise>())
        continue;
      if (!visited.insert(user).second)
        continue;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return true;
}

/// Per-query results of getUniformDims, keyed by (value, depth), so that a
/// value reached through several operands (e.g. `x = x * x` chains) is analysed
/// once per depth: the walk stays linear in the number of values instead of
/// exponential in the depth, and its answer does not depend on visit order.
using UniformDimsCache =
    DenseMap<std::pair<Value, unsigned>, SmallVector<bool>>;

static SmallVector<bool> getUniformDims(Value value, UniformDimsCache &cache,
                                        unsigned depth);

/// For each dimension of the tensor \p value, true if every element along that
/// dimension provably has the same value, e.g. a pointer tensor built from a
/// broadcast row such as an addmm bias with stride (0, 1). Looks through the
/// ops that build such tensors and answers false for anything else.
static SmallVector<bool>
computeUniformDims(Value value, UniformDimsCache &cache, unsigned depth) {
  auto tensorTy = cast<RankedTensorType>(value.getType());
  ArrayRef<int64_t> shape = tensorTy.getShape();
  // A dimension of size 1 is trivially uniform.
  SmallVector<bool> uniform =
      llvm::map_to_vector(shape, [](int64_t dim) { return dim == 1; });
  constexpr unsigned maxDepth = 16;
  Operation *def = value.getDefiningOp();
  if (!def || depth >= maxDepth)
    return uniform;

  auto allUniform = [&]() { return SmallVector<bool>(shape.size(), true); };
  auto fromOperand = [&](Value operand) -> std::optional<SmallVector<bool>> {
    auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandTy || operandTy.getShape() != shape)
      return std::nullopt;
    return getUniformDims(operand, cache, depth + 1);
  };

  if (isa<tt::SplatOp>(def))
    return allUniform();
  if (auto constantOp = dyn_cast<arith::ConstantOp>(def))
    return isa<SplatElementsAttr>(constantOp.getValue()) ? allUniform()
                                                         : uniform;
  if (auto broadcastOp = dyn_cast<tt::BroadcastOp>(def)) {
    Value src = broadcastOp.getSrc();
    SmallVector<bool> srcUniform = getUniformDims(src, cache, depth + 1);
    ArrayRef<int64_t> srcShape =
        cast<RankedTensorType>(src.getType()).getShape();
    for (auto [dim, srcSize] : llvm::enumerate(srcShape))
      uniform[dim] = uniform[dim] || srcSize == 1 || srcUniform[dim];
    return uniform;
  }
  if (auto expandDimsOp = dyn_cast<tt::ExpandDimsOp>(def)) {
    SmallVector<bool> srcUniform =
        getUniformDims(expandDimsOp.getSrc(), cache, depth + 1);
    srcUniform.insert(srcUniform.begin() + expandDimsOp.getAxis(), true);
    return srcUniform;
  }
  if (isa<ttg::ConvertLayoutOp>(def)) {
    if (std::optional<SmallVector<bool>> srcUniform =
            fromOperand(def->getOperand(0)))
      return *srcUniform;
    return uniform;
  }
  // An elementwise result is uniform along a dimension when every operand is.
  // Inline asm is excluded: it receives packed elements, so an output element
  // can depend on its neighbours and uniform operands prove nothing.
  if (def->hasTrait<OpTrait::Elementwise>() && isPure(def) &&
      def->getNumResults() == 1 && !isa<tt::ElementwiseInlineAsmOp>(def)) {
    SmallVector<bool> result = allUniform();
    for (Value operand : def->getOperands()) {
      std::optional<SmallVector<bool>> operandUniform = fromOperand(operand);
      if (!operandUniform)
        return uniform;
      for (auto [dim, isUniform] : llvm::enumerate(*operandUniform))
        result[dim] = result[dim] && isUniform;
    }
    for (auto [dim, isUniform] : llvm::enumerate(result))
      uniform[dim] = uniform[dim] || isUniform;
    return uniform;
  }
  return uniform;
}

static SmallVector<bool> getUniformDims(Value value, UniformDimsCache &cache,
                                        unsigned depth) {
  auto key = std::make_pair(value, depth);
  auto it = cache.find(key);
  if (it != cache.end())
    return it->second;
  SmallVector<bool> uniform = computeUniformDims(value, cache, depth);
  cache.try_emplace(key, uniform);
  return uniform;
}

/// Number of distinct addresses a tensor of pointers can hold when it is
/// broadcast along some, but not all, of its dimensions: the product of the
/// extents of the dimensions along which the pointer varies. Otherwise the
/// logical element count. A fully uniform tile (e.g. a load of a splat pointer)
/// keeps the logical count: it is not a broadcast row or column, and whether
/// such a load should anchor a layout is a separate question. A dimension the
/// walk cannot prove uniform (an unhandled op, or the depth limit) counts as
/// varying: that is safe for the uniformity proof, but it can make a fully
/// uniform tile look like a broadcast, and such a load that feeds DPAS is then
/// never an anchor (see isExpensiveLoadOrStore).
static int64_t getNumDistinctElements(Value ptr) {
  auto ptrTy = cast<RankedTensorType>(ptr.getType());
  UniformDimsCache cache;
  SmallVector<bool> uniformDims = getUniformDims(ptr, cache, /*depth=*/0);
  int64_t numDistinct = 1;
  bool anyUniform = false, anyVarying = false;
  for (auto [size, isUniform] :
       llvm::zip_equal(ptrTy.getShape(), uniformDims)) {
    if (size == 1)
      continue;
    if (isUniform) {
      anyUniform = true;
    } else {
      anyVarying = true;
      numDistinct *= size;
    }
  }
  return anyUniform && anyVarying ? numDistinct : ptrTy.getNumElements();
}

/// True if the result of \p loadOp is converted to a DPAS encoding, or is
/// combined through layout-only and elementwise ops with a value that is in, or
/// is converted from, a DPAS encoding: e.g. a bias added to a dot accumulator
/// in a GEMM epilogue. Such a load is rematerialized into the DPAS encoding
/// instead of converting the whole tile through shared memory.
static bool feedsDpasValue(tt::LoadOp loadOp) {
  auto isDpasValue = [](Value value) {
    auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
    if (tensorTy && hasDpasEncoding(tensorTy))
      return true;
    auto convertOp = value.getDefiningOp<ttg::ConvertLayoutOp>();
    return convertOp && hasDpasEncoding(convertOp.getSrc().getType());
  };

  SmallVector<Value> worklist{loadOp.getResult()};
  SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    for (Operation *user : val.getUsers()) {
      if (auto convertOp = dyn_cast<ttg::ConvertLayoutOp>(user))
        if (hasDpasEncoding(convertOp.getType()))
          return true;
      bool elementwise = user->hasTrait<OpTrait::Elementwise>();
      if (!elementwise && !isa<ttg::ConvertLayoutOp, tt::FpToFpOp,
                               tt::BroadcastOp, tt::ReshapeOp>(user))
        continue;
      if (!visited.insert(user).second)
        continue;
      if (elementwise && llvm::any_of(user->getOperands(), isDpasValue))
        return true;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return false;
}

bool isExpensiveLoadOrStore(Operation *op) {
  assert(
      (isa<tt::LoadOp, tt::StoreOp, tt::DescriptorLoadOp, tt::DescriptorStoreOp,
           tt::DescriptorGatherOp, ttgi::DescriptorGatherOp>(op)) &&
      "Expecting Triton LoadOp, StoreOp, DescriptorLoadOp, "
      "DescriptorStoreOp or DescriptorGatherOp");
  Value base = op->getOperand(0);

  if (isa<tt::LoadOp, tt::StoreOp>(op)) {
    // A size 1 tensor is not expensive since all threads will load the same
    // value.
    if (isSingleValue(base))
      return false;
  }

  // Loads or stores that use a block pointer are expensive if they cannot be
  // lowered to 2D block read/write operations. Temporarily leverage the
  // "ttig.block_io" attribute to filter out inexpensive loads.
  // Exceptions: 1D-reshaped loads and stores (indicated by
  // ttig.block_io_stride) have a specific encoding that matches HW delivery
  // order, and descriptor loads marked ttig.block_io_anchor were relaid out
  // for 2D block IO. Both must be anchored.
  Attribute blockIOAttr =
      op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
  if (blockIOAttr &&
      !op->getAttr(TritonIntelGPUDialect::getBlockIOStrideAttrName()) &&
      !op->hasAttr(TritonIntelGPUDialect::getBlockIOAnchorAttrName()) &&
      !isMaskedLoadWithoutDotConsumer(op))
    return false;

  // Loads or stores that use more threads than elements can be presumed to have
  // a high hit-rate that makes them cheap.
  RankedTensorType ptrType;
  if (auto descLoadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
    ptrType = getRankedTensorType(descLoadOp);
  } else if (auto descStoreOp = dyn_cast<tt::DescriptorStoreOp>(op)) {
    ptrType = getRankedTensorType(descStoreOp);
  } else if (isa<tt::DescriptorGatherOp, ttgi::DescriptorGatherOp>(op)) {
    ptrType = cast<RankedTensorType>(op->getResult(0).getType());
  } else {
    ptrType = getRankedTensorType(base.getType());
  }

  if (ptrType) {
    int numWarps = ttg::lookupNumWarps(op);
    auto mod = op->getParentOfType<ModuleOp>();
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    // A load whose pointer is broadcast along some dimensions, such as an
    // addmm bias with stride (0, 1), reads each distinct address repeatedly;
    // when its result feeds a DPAS value, anchoring it keeps a convert of the
    // whole tile to DPAS (and usually one of the output back) through shared
    // memory, whose live values can also push the kernel into spilling.
    // - A broadcast row that becomes a row-major 2D block load is never an
    //   anchor: rematerialized in the DPAS encoding, the lowering issues one
    //   message per distinct column block and shares it across the
    //   replicated rows.
    // - Any other such load (a column broadcast, or a row whose mask makes it
    //   a gather, e.g. an M that is not a multiple of the message height) is
    //   not shared, so it is costed by its distinct addresses: it stops being
    //   an anchor only with fewer of them than threads.
    // Other loads, and stores, keep the logical element count.
    int64_t numElements = ptrType.getNumElements();
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      int64_t numDistinct = getNumDistinctElements(base);
      if (numDistinct < numElements && feedsDpasValue(loadOp)) {
        auto blockIO = loadOp->getAttrOfType<StringAttr>(
            TritonIntelGPUDialect::getBlockIOAttrName());
        if (blockIO &&
            blockIO.getValue() == stringifyBlockIOMode(BlockIOMode::RowMajor))
          return false;
        numElements = numDistinct;
      }
    }
    return numElements >= numWarps * threadsPerWarp;
  }

  return false;
}

bool hasDotDpasEncoding(RankedTensorType tensorType) {
  if (!tensorType.getEncoding())
    return false;

  auto dotLayout =
      dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  if (!dotLayout)
    return false;

  return isa<ttgi::DpasEncodingAttr>(dotLayout.getParent());
}

bool hasDpasEncoding(RankedTensorType tensorType) {
  return isa_and_nonnull<ttgi::DpasEncodingAttr>(tensorType.getEncoding());
}

std::optional<DotOperandEncodingAttr>
getDotEncoding(RankedTensorType tensorType) {
  if (!tensorType.getEncoding())
    return std::nullopt;

  auto dotLayout =
      dyn_cast<ttg::DotOperandEncodingAttr>(tensorType.getEncoding());
  if (!dotLayout)
    return std::nullopt;

  return dotLayout;
}

// Check if the convert will be performed by reordering registers.
static bool isFreeConvert(Operation *op) {
  auto convertOp = dyn_cast<ttg::ConvertLayoutOp>(op);
  if (!convertOp)
    return false;
  return cvtReordersRegisters(convertOp.getSrc().getType(),
                              convertOp.getType());
}

LogicalResult getConvertBackwardSlice(
    OpOperand &root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation,
    std::function<Value(OpOperand &, Attribute)> getExistingConversion) {
  DenseSet<std::pair<OpOperand *, Attribute>> seen;
  SmallVector<std::pair<OpOperand *, Attribute>> queue;

  bool enableForLoopSupport = mlir::triton::tools::getBoolEnv(
      "TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP");

  auto enqueue = [&](OpOperand &operand, Attribute encoding) {
    auto x = std::make_pair(&operand, encoding);
    if (!seen.insert(x).second) {
      return; // Already enqueued, skip
    }
    queue.push_back(x);
  };
  enqueue(root, rootEncoding);

  auto updateLayout = [&](Value value, Attribute encoding) {
    assert(isa<RankedTensorType>(value.getType()));
    Attribute &existing = layout[value];
    if (existing && existing != encoding)
      return failure();
    existing = encoding;
    return success();
  };

  while (!queue.empty()) {
    auto [currentValueUse, encoding] = queue.back();
    Value currentValue = currentValueUse->get();
    queue.pop_back();
    if (!isa<RankedTensorType>(currentValue.getType()))
      continue;

    // Skip propagating through for op results for now.
    // TODO: enable this based on needs.
    if (!enableForLoopSupport && currentValue.getDefiningOp<scf::ForOp>())
      return failure();

    if (failed(updateLayout(currentValue, encoding)))
      return failure();

    // If the value already has the desired encoding, we can stop here without
    // adding it to the slice.
    auto currentValueType = cast<RankedTensorType>(currentValue.getType());
    if (currentValueType.getEncoding() == encoding)
      continue;
    // Backward propagation is only supported for region ops whose results
    // survive rewriteSlice's generic clone+setType path. ForOp and IfOp have
    // explicit handling below; everything else implementing
    // RegionBranchOpInterface is rejected, because retyping a result tied to a
    // region terminator's operands leaves the terminator at the old encoding
    // (see WarpYieldOp::verify). Note this is a conservative proxy for
    // structured control flow, not an exact test: region ops that do not
    // implement the interface (tt.reduce, tt.scan, tt.map_elementwise) pass
    // through here and rely on their regions being scalar-level, so retyping
    // the tensor result keeps them valid.
    if (Operation *defOp = currentValue.getDefiningOp())
      if (isa<RegionBranchOpInterface>(defOp) &&
          !isa<scf::ForOp, scf::IfOp>(defOp))
        return failure();
    slice.insert(currentValue);

    Value existing;
    if (getExistingConversion &&
        (existing = getExistingConversion(*currentValueUse, encoding))) {
      if (failed(updateLayout(existing, encoding)))
        return failure();
      currentValue = existing;
    }
    if (auto forOp = currentValue.getDefiningOp<scf::ForOp>()) {
      if (stopPropagation && stopPropagation(forOp))
        continue;

      auto loopRes = cast<OpResult>(currentValue);
      OpOperand *initOperand = forOp.getTiedLoopInit(loopRes);
      BlockArgument blockArg = forOp.getTiedLoopRegionIterArg(loopRes);
      OpOperand *yieldOperand = forOp.getTiedLoopYieldedValue(blockArg);

      enqueue(*initOperand, encoding);
      enqueue(*yieldOperand, encoding);

      continue;
    }
    if (auto ifOp = currentValue.getDefiningOp<scf::IfOp>()) {
      if (stopPropagation && stopPropagation(ifOp))
        continue;
      unsigned argIdx = mlir::cast<OpResult>(currentValue).getResultNumber();

      OpOperand &thenValue = ifOp.thenYield()->getOpOperand(argIdx);
      OpOperand &elseValue = ifOp.elseYield()->getOpOperand(argIdx);

      enqueue(thenValue, encoding);
      enqueue(elseValue, encoding);

      continue;
    }
    if (auto *definingOp = currentValue.getDefiningOp()) {
      // If the op has multiple results we need to update all results layout.
      for (Value result : definingOp->getResults()) {
        if (result == currentValue || !isa<RankedTensorType>(result.getType()))
          continue;
        if (failed(updateLayout(result, encoding)))
          return failure();
        slice.insert(result);
      }
      if (isFreeConvert(definingOp)) {
        enqueue(definingOp->getOpOperand(0), encoding);
        continue;
      }
      if (auto fixedOperands = canUseResultEncoding(definingOp, encoding)) {
        // Another path through the slice must preserve these operand layouts.
        for (OpOperand *operand : *fixedOperands) {
          Value src = operand->get();
          auto srcEncoding =
              cast<RankedTensorType>(src.getType()).getEncoding();
          if (failed(updateLayout(src, srcEncoding)))
            return failure();
        }
        continue;
      }
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (auto gather = dyn_cast<GatherOp>(definingOp)) {
        // Specially handle gather since its transfer function only applies
        // between its index operand and result.
        auto srcEncoding = ttgi::inferSrcEncoding(gather, encoding);
        if (!srcEncoding)
          return failure();
        enqueue(gather.getIndicesMutable(), srcEncoding);
        continue;
      }
      // Cannot remat across tt.call: callee signature wouldn't update.
      if (isa<tt::CallOp>(definingOp))
        return failure();
      for (auto [i, operand] : llvm::enumerate(definingOp->getOpOperands())) {
        if (isa<RankedTensorType>(operand.get().getType())) {
          Attribute srcEncoding;
          if (auto castOp = dyn_cast<gpu::CastFpOpInterface>(definingOp))
            srcEncoding = castOp.inferSrcEncoding(i, encoding);
          else
            srcEncoding = ttgi::inferSrcEncoding(definingOp, encoding);
          if (!srcEncoding)
            return failure();
          enqueue(operand, srcEncoding);
        }
      }
      continue;
    }
    auto blockArg = cast<BlockArgument>(currentValue);
    Block *block = blockArg.getOwner();
    Operation *parentOp = block->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      OpOperand *initOperand = forOp.getTiedLoopInit(blockArg);
      OpOperand &yieldOperand = forOp.getBody()->getTerminator()->getOpOperand(
          blockArg.getArgNumber() - forOp.getNumInductionVars());
      enqueue(*initOperand, encoding);
      enqueue(yieldOperand, encoding);
      continue;
    }
    // TODO: add support for WhileOp and other region types.
    return failure();
  }
  return success();
}

LLVM::LLVMFuncOp lookupOrCreateSPIRVFn(Operation *symbolTable, StringRef name,
                                       ArrayRef<Type> paramTypes,
                                       Type resultType) {
  auto func = dyn_cast_or_null<LLVM::LLVMFuncOp>(
      SymbolTable::lookupSymbolIn(symbolTable, name));
  if (!func) {
    OpBuilder b(symbolTable->getRegion(0));
    func = LLVM::LLVMFuncOp::create(
        b, symbolTable->getLoc(), name,
        LLVM::LLVMFunctionType::get(resultType, paramTypes));
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  }
  return func;
}

LLVM::CallOp createSPIRVBuiltinCall(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    LLVM::LLVMFuncOp func, ValueRange args) {
  auto call = LLVM::CallOp::create(rewriter, loc, func, args);
  call.setCConv(func.getCConv());
  return call;
}

SmallVector<unsigned> calculateDPASInstShapeA(unsigned repeatCount,
                                              unsigned systolicDepth,
                                              unsigned opsPerChannel) {
  return {repeatCount, systolicDepth * opsPerChannel};
}

SmallVector<unsigned> calculateDPASInstShapeB(unsigned systolicDepth,
                                              unsigned opsPerChannel,
                                              unsigned executionSize) {
  return {systolicDepth * opsPerChannel, executionSize};
}

SmallVector<unsigned> calculateDPASInstShapeC(unsigned repeatCount,
                                              unsigned executionSize) {
  return {repeatCount, executionSize};
}

SmallVector<unsigned> calculateShapeA(unsigned repeatCount,
                                      unsigned systolicDepth,
                                      unsigned opsPerChannel,
                                      ArrayRef<unsigned> repCluster) {
  SmallVector<unsigned> instShapeA =
      calculateDPASInstShapeA(repeatCount, systolicDepth, opsPerChannel);
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeA[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeA[1];
  return resShape;
}

SmallVector<unsigned> calculateShapeB(unsigned systolicDepth,
                                      unsigned opsPerChannel,
                                      unsigned executionSize,
                                      ArrayRef<unsigned> repCluster) {
  SmallVector<unsigned> instShapeB =
      calculateDPASInstShapeB(systolicDepth, opsPerChannel, executionSize);
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeB[0];
  resShape[rank - 1] = instShapeB[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> calculateShapeC(unsigned repeatCount,
                                      unsigned executionSize,
                                      ArrayRef<unsigned> repCluster) {
  SmallVector<unsigned> instShapeC =
      calculateDPASInstShapeC(repeatCount, executionSize);
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeC[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeC[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> calculateWarpsPerTile(unsigned capRepeatCount,
                                            unsigned capExecutionSize,
                                            const ArrayRef<int64_t> shape,
                                            unsigned numWarps) {
  size_t rank = shape.size();
  SmallVector<unsigned> ret(rank, 1);

  if (rank == 3) {
    int batchWarp = numWarps;
    while (batchWarp > shape[0])
      batchWarp /= 2;
    ret[0] = batchWarp;
    numWarps /= batchWarp;
  }

  // Try to find a proper tiling shape for the dot operation.
  // It doubles the warp number in col or row in each time based on column to
  // width ratio.
  // By this, we can minimize the duplication of the dot operands A and B.
  SmallVector<int64_t> shapePerWarp{capRepeatCount, capExecutionSize};
  uint32_t rowColRatio = llvm::divideCeil(capRepeatCount, capExecutionSize);
  uint32_t colRowRatio = llvm::divideCeil(capExecutionSize, capRepeatCount);

  int rowDim = rank - 2, colDim = rank - 1;
  do {
    if (ret[rowDim] * ret[colDim] >= numWarps)
      break;
    if (shape[rowDim] / (shapePerWarp[0] * colRowRatio) / ret[rowDim] >=
        shape[colDim] / (shapePerWarp[1] * rowColRatio) / ret[colDim]) {
      if (ret[rowDim] < shape[rowDim] / shapePerWarp[0])
        ret[rowDim] *= 2;
      else
        ret[colDim] *= 2;
    } else {
      ret[colDim] *= 2;
    }
  } while (true);

  return ret;
}

SmallVector<unsigned>
calculateRepCluster(const DpasEncodingAttr::DPASCapability &dpasCap,
                    unsigned opsPerChan, ArrayRef<int64_t> retShape,
                    unsigned threadsPerWarp, unsigned a_bitwidth, bool is_FP8,
                    ArrayRef<int64_t> a_shape, ArrayRef<int64_t> b_shape,
                    ArrayRef<unsigned> warpsPerTile) {
  size_t rank = retShape.size();
  SmallVector<unsigned> repCluster(rank, 1);

  unsigned repeatCount = std::min(
      dpasCap.repeatCount, static_cast<unsigned>(retShape[rank - 2]) /*M*/);
  unsigned numElemsPerRowForA =
      opsPerChan == 1 ? dpasCap.systolicDepth
                      : dpasCap.systolicDepth * 2; // A is packed to i16 or i32.
  unsigned minM = llvm::divideCeil(threadsPerWarp, numElemsPerRowForA);
  repeatCount = std::max(repeatCount, minM);

  if (dpasCap.executionSize == 16) {
    unsigned dpasElemBitWidths = a_bitwidth;

    // Upcast FP8 to FP16 is the DPAS engine doesn't support FP8 natively.
    if (!dpasCap.supportsFP8 && is_FP8)
      dpasElemBitWidths = 2 * dpasElemBitWidths;

    // Enlarge the repCluster size to use the large 2D load for A and B
    // operands.
    constexpr unsigned PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS = 32;
    constexpr unsigned PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS = 64;

    unsigned maxRepClusterM = PVC_2D_LOAD_MAXIMUM_NUMBER_OF_ROWS / repeatCount;
    SmallVector<int64_t> repA = calculateDPASRepetitions(
        a_shape, static_cast<ttgi::DpasEncodingAttr::OpIdx>(0), warpsPerTile,
        repCluster, repeatCount, dpasCap.systolicDepth, dpasCap.executionSize,
        opsPerChan);

    unsigned repClusterDimM =
        std::min(maxRepClusterM, static_cast<unsigned>(repA[1]));

    unsigned maxRepClusterN = PVC_2D_LOAD_MAXIMUM_BYTES_OF_COLS /
                              ((dpasElemBitWidths / 8) * dpasCap.executionSize);
    SmallVector<int64_t> repB = calculateDPASRepetitions(
        b_shape, static_cast<ttgi::DpasEncodingAttr::OpIdx>(1), warpsPerTile,
        repCluster, repeatCount, dpasCap.systolicDepth, dpasCap.executionSize,
        opsPerChan);

    unsigned repClusterDimN =
        std::min(maxRepClusterN, static_cast<unsigned>(repB[2]));
    repCluster[rank - 2] = repClusterDimM;
    repCluster[rank - 1] = repClusterDimN;
  }

  return repCluster;
}

SmallVector<int64_t>
calculateDPASRepetitions(ArrayRef<int64_t> shape, DpasEncodingAttr::OpIdx opIdx,
                         ArrayRef<unsigned> warpsPerCTA,
                         ArrayRef<unsigned> repCluster, unsigned repeatCount,
                         unsigned systolicDepth, unsigned executionSize,
                         unsigned opsPerChannel) {
  // Always return a 3D shape repetitions for the ease of value handling, same
  // to mma.
  size_t rank = shape.size();
  SmallVector<int64_t> rep(3, 1);

  switch (opIdx) {
  case DpasEncodingAttr::OpIdx::OperandA: {
    SmallVector<unsigned> shapePerWarp =
        calculateShapeA(repeatCount, systolicDepth, opsPerChannel, repCluster);

    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / shapePerWarp[rank - 1])};
  } break;
  case DpasEncodingAttr::OpIdx::OperandB: {
    SmallVector<unsigned> shapePerWarp = calculateShapeB(
        systolicDepth, opsPerChannel, executionSize, repCluster);

    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / shapePerWarp[rank - 2]),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  } break;
  case DpasEncodingAttr::OpIdx::OperandC: {
    SmallVector<unsigned> shapePerWarp =
        calculateShapeC(repeatCount, executionSize, repCluster);

    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  } break;
  }

  llvm_unreachable("unexpected opIdx");
}

bool mayWriteMemory(Operation *op, bool ignoreSharedMemory) {
  std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
      getEffectsRecursively(op);
  if (!effects)
    return true;
  return llvm::any_of(*effects, [&](const MemoryEffects::EffectInstance &e) {
    if (!isa<MemoryEffects::Write>(e.getEffect()))
      return false;
    SideEffects::Resource *resource = e.getResource();
    if (ignoreSharedMemory && isa<SharedMemory>(resource))
      return false;
    return !isa<L2Cache>(resource);
  });
}

} // namespace mlir::triton::gpu::intel
