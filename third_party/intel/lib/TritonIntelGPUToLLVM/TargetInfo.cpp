#include "TargetInfo.h"
#include "Utility.h"
#include "intel/include/TritonIntelGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

using namespace mlir;
namespace mlir::triton::intel {
// Check if the reduction can use a redux op and return the kind.
static std::optional<NVVM::ReduxKind> matchReduxKind(triton::ReduceOp op,
                                                     int computeCapability) {
  if (computeCapability < 80)
    return std::nullopt;
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return std::nullopt;
  Block *block = &(*op.getCombineOp().begin());
  Operation *yield = block->getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return std::nullopt;
  auto intType = reduceOp->getResultTypes()[0].dyn_cast<IntegerType>();
  if (!intType || intType.getWidth() > 32)
    return std::nullopt;
  if (reduceOp->getOperand(0) != block->getArgument(0) ||
      reduceOp->getOperand(1) != block->getArgument(1))
    return std::nullopt;
  if (isa<arith::AddIOp>(reduceOp))
    return NVVM::ReduxKind::ADD;
  if (isa<arith::AndIOp>(reduceOp))
    return NVVM::ReduxKind::AND;
  if (isa<arith::OrIOp>(reduceOp))
    return NVVM::ReduxKind::OR;
  if (isa<arith::XOrIOp>(reduceOp))
    return NVVM::ReduxKind::XOR;
  if (isa<arith::MinSIOp>(reduceOp))
    return NVVM::ReduxKind::MIN;
  if (isa<arith::MinUIOp>(reduceOp))
    return NVVM::ReduxKind::UMIN;
  if (isa<arith::MaxSIOp>(reduceOp))
    return NVVM::ReduxKind::MAX;
  if (isa<arith::MaxUIOp>(reduceOp))
    return NVVM::ReduxKind::UMAX;
  return std::nullopt;
}

bool TargetInfo::supportMaximumMinimum() const {
  return computeCapability >= 80;
}
Value TargetInfo::ballot(ConversionPatternRewriter &rewriter, Location loc,
                         Type type, Value cmp) const {
  Value threadMask = int_val(type.getIntOrFloatBitWidth(), -1);
  return rewriter.create<NVVM::VoteBallotOp>(loc, type, threadMask, cmp);
}
Value TargetInfo::storeShared(ConversionPatternRewriter &rewriter, Location loc,
                              Value ptr, Value val, Value pred) const {
  mlir::LLVM::utils::createPredicatedBlock(rewriter, loc, pred, [&] {
    store(val, ptr);
    return ArrayRef<Value>();
  });
  return Value();
}

Value TargetInfo::loadShared(ConversionPatternRewriter &rewriter, Location loc,
                             Value ptr, Type elemTy, Value pred) const {
  assert(ptr.getType().cast<mlir::LLVM::LLVMPointerType>().getAddressSpace() ==
             3 &&
         "Invalid addr space for loadShared");
  Value undef = undef(elemTy);
  Block &endBlock = mlir::LLVM::utils::createPredicatedBlock(
      rewriter, loc, pred, SmallVector<Value, 1>{undef}, [&] {
        Value ret = load(elemTy, ptr);
        return SmallVector<Value, 1>{ret};
      });
  return *endBlock.args_begin();
}

static TritonGEN::ShflKind toGenShuffleMode(NVVM::ShflKind mode) {
  switch (mode) {
  case NVVM::ShflKind::bfly:
    return TritonGEN::ShflKind::XOR;
  case NVVM::ShflKind::up:
    return TritonGEN::ShflKind::UP;
  case NVVM::ShflKind::down:
    return TritonGEN::ShflKind::DOWN;
  case NVVM::ShflKind::idx:
    return TritonGEN::ShflKind::IDX;
  }
  llvm_unreachable("unsupported NVVM::ShflKind");
}

static Value commonShflSync(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, Value i, NVVM::ShflKind mode,
                            Value clamp) {
  unsigned bits = val.getType().getIntOrFloatBitWidth();

  if (bits == 64) {
    Type vecTy = vec_ty(f32_ty, 2);
    Value vec = bitcast(val, vecTy);
    Value val0 = extract_element(f32_ty, vec, i32_val(0));
    Value val1 = extract_element(f32_ty, vec, i32_val(1));
    val0 = commonShflSync(loc, rewriter, val0, i, mode, clamp);
    val1 = commonShflSync(loc, rewriter, val1, i, mode, clamp);
    vec = undef(vecTy);
    vec = insert_element(vecTy, vec, val0, i32_val(0));
    vec = insert_element(vecTy, vec, val1, i32_val(1));
    return bitcast(vec, val.getType());
  }
  Type type = val.getType();
  return rewriter.create<TritonGEN::SubGroupShuffleOp>(loc, type, val, i,
                                                       toGenShuffleMode(mode));
}

Value TargetInfo::shuffleXor(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), NVVM::ShflKind::bfly,
                        i32_val(0x1f));
}

Value TargetInfo::shuffleUp(Location loc, ConversionPatternRewriter &rewriter,
                            Value val, int i) const {
  return commonShflSync(loc, rewriter, val, i32_val(i), NVVM::ShflKind::up,
                        i32_val(0x0));
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, int i) const {
  return mlir::LLVM::utils::shflIdxSync(loc, rewriter, val, i32_val(i));
}

Value TargetInfo::shuffleIdx(Location loc, ConversionPatternRewriter &rewriter,
                             Value val, Value i) const {
  return commonShflSync(loc, rewriter, val, i, NVVM::ShflKind::idx,
                        i32_val(0x1f));
}

bool TargetInfo::warpReduce(ConversionPatternRewriter &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce) const {
  if (auto kind = matchReduxKind(op, computeCapability)) {
    // Based on benchmarking on A100 redux op gives a speed up only when doing
    // a single reduction (not partitioned) and when the mask is static.
    // Therefore we currently only enable it to reduce across all the lanes.
    if (numLaneToReduce == 32) {
      assert(acc.size() == 1);
      Value mask = i32_val(0xFFFFFFFF);
      // Even though we currently don't use redux for partitioned reduction
      // the code below supports it in case we want to tweak the heuristic.
      if (numLaneToReduce < 32) {
        // For partitioned reduction we need to calculate the mask so that
        // each group of numLaneToReduce threads has the correct mask.
        unsigned bitmask = (1 << numLaneToReduce) - 1;
        Value threadId = getThreadId(rewriter, loc);
        Value laneId = urem(threadId, i32_val(32));
        mask = shl(i32_val(bitmask),
                   and_(laneId, i32_val(~(numLaneToReduce - 1))));
      }
      for (unsigned i = 0; i < acc.size(); ++i) {
        unsigned bitwidth = acc[i].getType().cast<IntegerType>().getWidth();
        if (bitwidth < 32) {
          if (*kind == NVVM::ReduxKind::MIN || *kind == NVVM::ReduxKind::MAX)
            acc[i] = sext(i32_ty, acc[i]);
          else
            acc[i] = zext(i32_ty, acc[i]);
        }
        acc[i] = rewriter.create<NVVM::ReduxOp>(loc, acc[i].getType(), acc[0],
                                                *kind, mask);
        if (bitwidth < 32)
          acc[i] = trunc(int_ty(bitwidth), acc[i]);
      }
      return true;
    }
  }
  return false;
}

bool TargetInfo::processReplicaUsingStMatrix(
    ConversionPatternRewriter &rewriter, Location loc, Value smemBase,
    SmallVector<Value> &vals, RankedTensorType srcTy, Type elemTy,
    ArrayRef<unsigned> paddedRepShape, ArrayRef<unsigned> origRepShape,
    ArrayRef<unsigned> outOrd, unsigned accumNumReplicates) const {
  return false;
}

} // namespace mlir::triton::intel
