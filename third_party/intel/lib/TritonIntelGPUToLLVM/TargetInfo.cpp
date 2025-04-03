//===- TargetInfo.cpp - Target dependent information ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo.h"
#include "intel/include/TritonIntelGPUToLLVM/XeAsmFormat.h"
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/FormatVariadic.h>

#include "Dialect/TritonIntelGPU/IR/Utils.h"
#include "SPIRVTargetInfo.h"
#include "Utility.h"

using namespace mlir;

namespace mlir::triton::intel {

struct XeSIMDReduceInstr : public XeVISAInstr {

  XeSIMDReduceInstr(XeBuilder *builder, std::string binOp, unsigned warpSize,
                    unsigned numLaneToReduce, unsigned accSize, Type elemTy,
                    XeArch arch)
      : XeVISAInstr(builder, simdReduceAsm(binOp, warpSize, numLaneToReduce,
                                           accSize, elemTy, arch)) {};
};

bool TargetInfo::supportMaximumMinimum() const { return false; }
Value TargetInfo::ballot(RewriterBase &rewriter, Location loc, Type type,
                         Value cmp) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Emulate vote.ballot.sync behavior using shift, shuffle, and or.
  // TODO: check for more efficient solution.
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Value threadId = getThreadId(rewriter, loc);
  int numThreadPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
  Value laneId = b.and_(threadId, b.i32_val(numThreadPerWarp - 1));
  Value reduced_val = b.shl(b.select(cmp, b.i32_val(1), b.i32_val(0)), laneId);
  for (int offs = 1; offs < numThreadPerWarp; offs = offs << 1) {
    Value other_val = LLVM::intel::shuffleXor(loc, rewriter, reduced_val, offs);
    reduced_val = b.or_(reduced_val, other_val);
  }
  return reduced_val;
}

void TargetInfo::barrier(Location loc, RewriterBase &rewriter,
                         triton::gpu::AddrSpace targets) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(targets);
}

void TargetInfo::warpSync(Location loc, RewriterBase &rewriter) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(triton::gpu::AddrSpace::Local);
}

Value TargetInfo::getClusterCTAId(RewriterBase &rewriter, Location loc) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  // Clusters of thread blocks aren't supported.
  return b.i32_val(0);
}

void TargetInfo::storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Value val,
                              Value pred) const {
  LLVM::intel::createPredicatedBlock(rewriter, loc, pred, [&] {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    b.store(val, ptr);
    return ArrayRef<Value>();
  });
}

Value TargetInfo::loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                              std::optional<Value> ctaId, Type elemTy,
                              Value pred, Operation *localLoadOp) const {
  assert(cast<mlir::LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() ==
             3 &&
         "Invalid addr space for loadShared");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value undef = b.undef(elemTy);
  Block &endBlock = LLVM::intel::createPredicatedBlock(
      rewriter, loc, pred, SmallVector<Value, 1>{undef}, [&] {
        Value ret = b.load(elemTy, ptr);
        return SmallVector<Value, 1>{ret};
      });
  return *endBlock.args_begin();
}

Value TargetInfo::shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::intel::shuffleXor(loc, rewriter, val, i);
}

Value TargetInfo::shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                            int i) const {
  return LLVM::intel::shuffleUp(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             int i) const {
  return LLVM::intel::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                             Value i) const {
  return LLVM::intel::shuffleIdx(loc, rewriter, val, i);
}

Value TargetInfo::permute(RewriterBase &rewriter, Location loc, Value a,
                          Value b, Value selector) const {
  return LLVM::intel::permute(loc, rewriter, a, b, selector);
}

Value TargetInfo::programId(RewriterBase &rewriter, Location loc,
                            ModuleOp moduleOp, ProgramIDDim axis) const {
  Value blockId =
      ::mlir::gpu::BlockIdOp::create(rewriter, loc, mlir::gpu::Dimension(axis));
  return arith::IndexCastOp::create(rewriter, loc, i32_ty, blockId);
}

static SmallVector<ArrayRef<Value>> splitInBatches(ArrayRef<Value> srcValues,
                                                   size_t batchSize) {
  SmallVector<ArrayRef<Value>> batches;
  for (; !srcValues.empty(); srcValues = srcValues.drop_front(batchSize))
    batches.push_back(srcValues.take_front(batchSize));
  return batches;
}

bool TargetInfo::warpBatchReduce(
    RewriterBase &rewriter, Location loc,
    std::map<SmallVector<unsigned>, SmallVector<Value>> &acc,
    triton::ReduceOp op, unsigned numLaneToReduce, unsigned interleave) const {
  // No horizontal reduce required.
  if (numLaneToReduce == 1)
    return false;
  // Horizontal reduce with interleave stride not supported.
  if (interleave > 1)
    return false;
  // Check if it is a simple reduce operation supported by
  // TritonGEN::SubGroupReduceOp.
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return false;
  Region &combineOp = op.getCombineOp();
  if (combineOp.getBlocks().size() > 1)
    return false;
  Block &block = *combineOp.begin();
  Operation *yield = block.getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return false;
  if (reduceOp->getOperand(0) != block.getArgument(0) ||
      reduceOp->getOperand(1) != block.getArgument(1))
    return false;

  auto mod = op->getParentOfType<ModuleOp>();
  unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  // TODO: support clustered reduce.
  // if (numLaneToReduce != warpSize)
  //   return false;

  if (!isSupportedWarpReduceOp(reduceOp, numLaneToReduce, warpSize))
    return false;

  unsigned minSGSize =
      mod->getAttrOfType<IntegerAttr>(
             gpu::intel::TritonIntelGPUDialect::getMinSGSizeAttrName())
          .getInt();

  if (isa<arith::AddFOp, arith::MaxNumFOp>(reduceOp)) {
    // So we have to align the number of simd reduce results to the warp size.
    unsigned accSize = acc.size();
    if (accSize == 1)
      return false;
    if ((accSize - 1) & accSize)
      return false;
    unsigned numResultPerRow = warpSize / numLaneToReduce;
    accSize = std::min(accSize, warpSize / numResultPerRow);
    Type elemType = acc.begin()->second[0].getType();
    VectorType reduceTy = vec_ty(elemType, accSize);

    // Group the acc in batch.
    SmallVector<Value> inputAccs;
    for (auto it : acc) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &val = acc[key];
      assert(val.size() == 1 && "acc size has to be 1 for ungrouped input");
      inputAccs.push_back(val[0]);
    }
    SmallVector<SmallVector<Value>> resultAccs(inputAccs.size() / accSize);

    std::string batchedHorizontalReduce;
    // TODO: support all possible reduction modes
    TypeSwitch<Operation *>(reduceOp)
        .Case<arith::AddFOp>([&](auto) { batchedHorizontalReduce = "add"; })
        .Case<arith::MaxNumFOp>([&](auto) { batchedHorizontalReduce = "max"; })
        .Default(
            [&](auto) { llvm_unreachable("Unhandled batched reduce kind"); });

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto laneId = getLaneId(rewriter, loc);
    auto reducePartial = b.udiv(laneId, b.i32_val(numLaneToReduce));
    llvm::transform(
        splitInBatches(inputAccs, accSize), std::begin(resultAccs),
        [&](ArrayRef<Value> inputs) {
          auto inputRange = llvm::enumerate(inputs);
          Value batchedReduceVal = std::accumulate(
              std::begin(inputRange), std::end(inputRange),
              LLVM::PoisonOp::create(rewriter, loc, reduceTy).getRes(),
              [reduceTy, loc, &rewriter](Value acc, auto entry) -> Value {
                auto [index, src] = entry;
                auto b = TritonLLVMOpBuilder(loc, rewriter);
                return b.insert_element(reduceTy, acc, src, b.i32_val(index));
              });
          XeBuilder xeBuilder;
          XeSIMDReduceInstr &bReduceOp = *xeBuilder.create<XeSIMDReduceInstr>(
              batchedHorizontalReduce, warpSize, numLaneToReduce, accSize,
              elemType, minSGSize == 8 ? Xe : Xe2);
          // The VISA inline asm doesn't support uniform result type. "=rw.u"
          //    auto res = vISABuilder.newOperand("=rw.u");
          XeBuilder::Operand *res = xeBuilder.newOperand("=rw");
          XeBuilder::Operand *in = xeBuilder.newOperand(batchedReduceVal, "rw");
          bReduceOp({res, in}, /*onlyAttachMLIRArgs=*/true);
          // Type resultTy = reduceTy.getElementType();
          Value result = xeBuilder.launch(rewriter, loc, elemType, false);
          SmallVector<Value> ret(accSize);
          for (unsigned j = 0; j < accSize; ++j) {
            if (numResultPerRow > 1) {
              ret[j] = LLVM::intel::shuffleIdx(
                  loc, rewriter, result,
                  b.add(b.i32_val(j * numResultPerRow), reducePartial));
            } else {
              ret[j] = LLVM::intel::shuffleIdx(loc, rewriter, result, j);
            }
          }
          return ret;
        });

    unsigned grouped_iter = 0;
    for (unsigned i = 0; i < resultAccs.size(); ++i) {
      // The output of the inline vISA has to be the non-uniform value.
      // Have to shuffle the result to get the reduce value.
      SmallVector<Value> ret = resultAccs[i];
      for (unsigned j = 0; j < accSize; ++j) {
        inputAccs[grouped_iter++] = ret[j];
      }
    }
    grouped_iter = 0;
    for (auto it : acc) {
      const SmallVector<unsigned> &key = it.first;
      SmallVector<Value> &val = acc[key];
      val[0] = inputAccs[grouped_iter++];
    }
#if 0
    auto res = vISABuilder.newOperand("=rw.u");
    auto in = vISABuilder.newOperand(batchedReduceVal, "rw");
    bReduceOp({res, in}, /*onlyAttachMLIRArgs=*/true);
    Value ret = vISABuilder.launch(rewriter, loc, reduceTy, false);
    Type resultTy = reduceTy.getElementType();
    for (unsigned i = 0; i < grouped_accs.size(); ++i) {
      grouped_accs[i] = b.extract_element(resultTy, ret, b.i32_val(i));
    }
#endif

    return true;
  }

  return false;
}

bool TargetInfo::warpReduce(RewriterBase &rewriter, Location loc,
                            SmallVector<Value> &acc, triton::ReduceOp op,
                            unsigned numLaneToReduce,
                            unsigned interleave) const {
  // No horizontal reduce required.
  if (numLaneToReduce == 1)
    return false;
  // Horizontal reduce with interleave stride not supported.
  if (interleave > 1)
    return false;
  // Check if it is a simple reduce operation supported by
  // TritonGEN::SubGroupReduceOp.
  if (op.getNumOperands() != 1 || op.getNumResults() != 1)
    return false;
  Region &combineOp = op.getCombineOp();
  if (combineOp.getBlocks().size() > 1)
    return false;
  Block &block = *combineOp.begin();
  Operation *yield = block.getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return false;
  if (reduceOp->getOperand(0) != block.getArgument(0) ||
      reduceOp->getOperand(1) != block.getArgument(1))
    return false;

  auto mod = op->getParentOfType<ModuleOp>();
  unsigned warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  if (!isSupportedWarpReduceOp(reduceOp, numLaneToReduce, warpSize))
    return false;

  for (unsigned i = 0; i < acc.size(); ++i) {
    acc[i] = genWarpReduce(rewriter, loc, acc[i], reduceOp, numLaneToReduce,
                           warpSize);
  }

  return true;
}

std::string TargetInfo::getMulhiFuncName(Type resultElementTy) const {
  std::string funcName =
      resultElementTy.isInteger(32) ? "__imf_umulhi" : "__imf_umul64hi";
  return funcName;
}

void TargetInfo::printf(RewriterBase &rewriter, Value formatStrStart,
                        int /*formatStrByteCount*/, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  emitter.printf(rewriter, formatStrStart, /*formatStrByteCount*/ 0, args,
                 isSigned);
}

void TargetInfo::printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
                        ArrayRef<bool> isSigned) const {
  emitter.printf(rewriter, msg, args, isSigned);
}

void TargetInfo::assertFail(RewriterBase &rewriter, Location loc,
                            StringRef message, StringRef file, StringRef func,
                            int line) const {
  return emitter.assertFail(rewriter, loc, message, file, func, line);
}

int TargetInfo::getSharedAddressSpace() const {
  return TritonGEN::TritonGENMemorySpace::kWorkgroup;
}

bool TargetInfo::supportVectorizedAtomics() const {
  // Note: not currently tested or used, but AMD generally supports vectorized
  // atomics.
  return true;
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = 3;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace for now");
  }
  return spaceId;
}

Value TargetInfo::getGlobalStringStart(Location loc, RewriterBase &rewriter,
                                       StringRef name, StringRef value,
                                       unsigned addressSpace) const {
  return emitter.getGlobalStringStart(loc, rewriter, name, value, addressSpace);
}

std::unique_ptr<TargetInfo> createTargetInfo(ModuleOp mod) {
  if (triton::gpu::intel::hasSpirvTargetArch(mod))
    return std::unique_ptr<TargetInfo>(new SPIRVTargetInfo());
  llvm_unreachable("createTargetInfo: unsupported target arch");
}

} // namespace mlir::triton::intel
