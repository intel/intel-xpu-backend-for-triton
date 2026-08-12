//===- Utility.cpp - Triton Intel GPU utilities -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton/Analysis/Utility.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"

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

bool isDivisible(Value value, unsigned divisor) {
  // Every integer is divisible by 1, regardless of how `value` is defined.
  if (divisor == 1)
    return true;

  // Case 1: Value is defined by a constant operation
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
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
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // Nested loops aren't currently handled.
      if (forOp->template getParentOfType<scf::ForOp>())
        return false;
      if (!forOp.getSingleInductionVar())
        return false;
      // Check only if the block arg is the loop-var.
      if (blockArg != forOp.getInductionVar())
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
  // tt::AddPtrOp or arith::AddIOp operation.
  if (auto *op = value.getDefiningOp()) {
    if (isa<arith::ExtSIOp, arith::TruncIOp, tt::AddPtrOp, arith::AddIOp>(op)) {
      return llvm::all_of(op->getOperands(), [&](Value operand) {
        return isDivisible(operand, divisor);
      });
    }
  }

  return false;
}

bool isNonNegative(Value value) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;

  // tt.get_program_id always returns [0, 2^31-1].
  if (isa<tt::GetProgramIdOp>(defOp))
    return true;

  // tt.get_num_programs returns [1, 2^31-1].
  if (isa<tt::GetNumProgramsOp>(defOp))
    return true;

  // tt.make_range with non-negative start.
  if (auto makeRange = dyn_cast<tt::MakeRangeOp>(defOp))
    return makeRange.getStartAttr().getInt() >= 0;

  // Non-negative constant (scalar or tensor).
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getValue().isNonNegative();
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.getElementType().isSignlessInteger()) {
        return llvm::all_of(denseAttr.getValues<APInt>(),
                            [](const APInt &v) { return v.isNonNegative(); });
      }
    }
  }

  // arith.addi / arith.muli of two non-negative values. This assumes no signed
  // overflow, which holds for the descriptor offsets this helper is applied to:
  // they are `programId * blockSize (+ ...)` expressions bounded by the tensor
  // shape, well within the i32 positive range. (Even a false positive here is
  // memory-safe: the `< shape` bounds check is always retained; only the
  // redundant `>= 0` sign check is elided.)
  if (auto addOp = dyn_cast<arith::AddIOp>(defOp))
    return isNonNegative(addOp.getLhs()) && isNonNegative(addOp.getRhs());
  if (auto mulOp = dyn_cast<arith::MulIOp>(defOp))
    return isNonNegative(mulOp.getLhs()) && isNonNegative(mulOp.getRhs());

  // arith.remui / arith.divui / arith.extui always produce non-negative
  // results (unsigned ops and zero-extension keep the MSB clear).
  if (isa<arith::RemUIOp, arith::DivUIOp, arith::ExtUIOp>(defOp))
    return true;

  // arith.divsi: non-negative iff both dividend and divisor are non-negative.
  if (auto divOp = dyn_cast<arith::DivSIOp>(defOp))
    return isNonNegative(divOp.getLhs()) && isNonNegative(divOp.getRhs());

  // arith.remsi: result has the same sign as the dividend (truncation toward
  // zero), so a non-negative dividend guarantees a non-negative remainder.
  if (auto remOp = dyn_cast<arith::RemSIOp>(defOp))
    return isNonNegative(remOp.getLhs());

  // arith.extsi preserves the signed value; non-negative iff source is.
  // (arith.trunci is intentionally NOT handled: truncating a non-negative
  // value can set the sign bit of the narrower type, e.g. i32 128 -> i8 -128.)
  if (auto extOp = dyn_cast<arith::ExtSIOp>(defOp))
    return isNonNegative(extOp.getIn());

  // arith.maxsi: non-negative if either operand is non-negative.
  if (auto maxOp = dyn_cast<arith::MaxSIOp>(defOp))
    return isNonNegative(maxOp.getLhs()) || isNonNegative(maxOp.getRhs());

  // arith.minsi: non-negative iff both operands are non-negative.
  if (auto minOp = dyn_cast<arith::MinSIOp>(defOp))
    return isNonNegative(minOp.getLhs()) && isNonNegative(minOp.getRhs());

  // tt.splat / tt.expand_dims / tt.broadcast: propagate from source.
  if (auto splatOp = dyn_cast<tt::SplatOp>(defOp))
    return isNonNegative(splatOp.getSrc());
  if (auto expandOp = dyn_cast<tt::ExpandDimsOp>(defOp))
    return isNonNegative(expandOp.getSrc());
  if (auto broadcastOp = dyn_cast<tt::BroadcastOp>(defOp))
    return isNonNegative(broadcastOp.getSrc());

  return false;
}

static Attribute inferSrcEncoding(ttgi::DescriptorGatherOp op,
                                  Attribute dstEnc) {
  // only the offsets require the slice encoding, the base pointer is a scalar
  // and does not require any encoding.
  return SliceEncodingAttr::get(op->getContext(), 1,
                                cast<DistributedEncodingTrait>(dstEnc));
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

  if (auto gatherOp = dyn_cast<ttgi::DescriptorGatherOp>(op))
    return inferSrcEncoding(gatherOp, encoding);

  return mlir::inferSrcEncoding(op, encoding);
}

// A block_io load that does not validate as a 2D block load in its current
// (coalesced) layout is treated as expensive so it anchors that layout (see
// isExpensiveLoadOrStore). That is correct when the only alternative is a
// worse layout -- e.g. a store back-propagating across tt.trans, de-coalescing
// the load into a vec1 gather (issue #7090). It is wrong, however, when the
// load value flows into a DPAS dot operand: hoistConvertDotOperand would
// relabel the load to that dot-operand layout, and for narrow (e.g. f8)
// operands that dot-operand layout is a genuine 2D block load even though the
// coalesced layout is not. Anchoring would block that strictly-better hoist.
//
// Return true when a forward walk from the load result reaches a DPAS
// dot-operand encoding under which the load validates as a 2D block load,
// traversing only the layout- and width-preserving ops the hoist itself
// crosses (elementwise same-width, broadcast, views, convert_layout). A
// width-changing op (e.g. tt.fp_to_fp) is a barrier: past it the dot operand
// is a different element type and hoistConvertDotOperand would not relabel
// this load (its leaf-load bitwidth guard, issue #6737).
static bool loadHoistsToBlock2DDotOperand(Operation *loadOp,
                                          RankedTensorType loadTy) {
  Type loadElemTy = loadTy.getElementType();
  if (!loadElemTy.isIntOrFloat())
    return false;
  unsigned loadBitWidth = loadElemTy.getIntOrFloatBitWidth();

  // Does the load validate as a 2D block load once relabeled to `enc` (keeping
  // its own shape and element type)?
  auto validatesAsDotOperand = [&](Attribute enc) {
    auto dotEnc = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(enc);
    if (!dotEnc || !isa<ttgi::DpasEncodingAttr>(dotEnc.getParent()))
      return false;
    return blockIOLoadValidatesAs2DBlock(loadOp, loadTy.cloneWithEncoding(enc));
  };

  // Only cross ops that move no data between threads and preserve element
  // width. A width-changing op or one whose result is not an int/float tensor
  // (e.g. a loaded index feeding tt.addptr, which yields a pointer tensor) is a
  // barrier: the dot-operand value chain never runs through it.
  auto isWidthPreservingElementwise = [&](Operation *op) {
    if (!(op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)))
      return false;
    for (Value res : op->getResults()) {
      auto rt = dyn_cast<RankedTensorType>(res.getType());
      if (!rt)
        continue;
      Type et = rt.getElementType();
      if (!et.isIntOrFloat() || et.getIntOrFloatBitWidth() != loadBitWidth)
        return false;
    }
    return true;
  };

  SmallVector<Value> worklist{loadOp->getResult(0)};
  SmallPtrSet<Value, 16> visited;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!visited.insert(v).second)
      continue;
    for (Operation *user : v.getUsers()) {
      // A convert to a DPAS dot operand is the layout hoistConvertDotOperand
      // would push onto the load; check whether the load validates there.
      if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user)) {
        if (validatesAsDotOperand(cvt.getType().getEncoding()))
          return true;
        worklist.push_back(cvt.getResult());
        continue;
      }
      if (isWidthPreservingElementwise(user) || isView(user) ||
          isa<tt::BroadcastOp>(user))
        llvm::append_range(worklist, user->getResults());
      // Anything else (dot, store, width-changing op, ...) is a barrier.
    }
  }
  return false;
}

bool isExpensiveLoadOrStore(Operation *op) {
  assert((isa<tt::LoadOp, tt::StoreOp, tt::DescriptorLoadOp,
              tt::DescriptorStoreOp>(op)) &&
         "Expecting Triton LoadOp, StoreOp, DescriptorLoadOp or "
         "DescriptorStoreOp");
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
  // Exception: 1D-reshaped loads and stores (indicated by
  // ttig.block_io_stride) have a specific encoding that matches HW delivery
  // order and must be anchored.
  Attribute blockIOAttr =
      op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
  if (blockIOAttr &&
      !op->getAttr(TritonIntelGPUDialect::getBlockIOStrideAttrName())) {
    // A block_io load is only cheap if it genuinely validates as a 2D block
    // load. If it would fall back to a per-element gather, treat it as
    // expensive so it anchors its layout (and is not rematerialized into a
    // de-coalesced gather). Exception: if the load feeds a DPAS dot operand
    // whose layout *is* a valid 2D block load, keep it cheap so
    // hoistConvertDotOperand can relabel it to that strictly-better layout
    // instead of anchoring the coalesced gather (issue #7090 fp8 chains).
    if (isa<tt::DescriptorLoadOp, tt::LoadOp>(op)) {
      auto loadTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (loadTy && !ttgi::blockIOLoadValidatesAs2DBlock(op, loadTy) &&
          !loadHoistsToBlock2DDotOperand(op, loadTy))
        return true;
    }
    return false;
  }

  // Loads or stores that use more threads than elements can be presumed to have
  // a high hit-rate that makes them cheap.
  RankedTensorType ptrType;
  if (auto descLoadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
    ptrType = getRankedTensorType(descLoadOp);
  } else if (auto descStoreOp = dyn_cast<tt::DescriptorStoreOp>(op)) {
    ptrType = getRankedTensorType(descStoreOp);
  } else {
    ptrType = getRankedTensorType(base.getType());
  }

  if (ptrType) {
    int numWarps = ttg::lookupNumWarps(op);
    auto mod = op->getParentOfType<ModuleOp>();
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    return ptrType.getNumElements() >= numWarps * threadsPerWarp;
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

  std::optional<bool> enableForLoopSupport =
      mlir::triton::tools::isEnvValueBool(mlir::triton::tools::getStrEnv(
          "TRITON_INTEL_REMOVELAYOUTCONVERSION_SUPPORT_FOR_LOOP"));

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
      if (canUseResultEncoding(definingOp, encoding))
        continue;
      if (stopPropagation && stopPropagation(definingOp))
        continue;
      if (isa<triton::CatOp>(definingOp))
        return failure();
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
          if (auto upcast = dyn_cast<gpu::UpcastFpOpInterface>(definingOp))
            srcEncoding = upcast.inferSrcEncoding(i, encoding);
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

} // namespace mlir::triton::gpu::intel
