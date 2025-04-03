//===- TargetInfo.h - Target dependent information ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOINTEL_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOINTEL_H

#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir::triton::intel {
class TargetInfo : public mlir::triton::TargetInfoBase {
public:
  TargetInfo() = default;

  bool supportMaximumMinimum() const override;

  Value getClusterCTAId(RewriterBase &rewriter, Location loc) const override;

  Value ballot(RewriterBase &rewriter, Location loc, Type type,
               Value cmp) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Value val,
                    Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    std::optional<Value> ctaId, Type elemTy,
                    Value pred) const override;
  bool canUseStMatrix(RankedTensorType tensorTy, ArrayRef<unsigned> repShape,
                      ArrayRef<unsigned> paddedRepShape,
                      ArrayRef<unsigned> order,
                      int swizzleByteSize) const override;

  void storeMatrixShared(RewriterBase &rewriter, Location loc, Value ptr,
                         Value val) const override;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  int axis) const override;

  bool warpBatchReduce(RewriterBase &rewriter, Location loc,
                       std::map<SmallVector<unsigned>, SmallVector<Value>> &acc,
                       triton::ReduceOp op, unsigned numLaneToReduce,
                       unsigned interleave) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned numLaneToReduce,
                  unsigned interleave) const override;

  std::string getMulhiFuncName(Type resultElementTy) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;
  int getSharedAddressSpace() const override;

  bool supportVectorizedAtomics() const override;

  int getAddressSpace(Attribute addressSpace) const override;

  Value getGlobalStringStart(Location loc, RewriterBase &rewriter,
                             StringRef name, StringRef value,
                             unsigned addressSpace) const;

protected:
  virtual bool isSupportedWarpReduceOp(Operation *op, unsigned numLanesToReduce,
                                       unsigned warpSize) const = 0;
  virtual Value genWarpReduce(RewriterBase &rewriter, Location loc, Value acc,
                              Operation *reduceOp, unsigned numLanesToReduce,
                              unsigned warpSize) const = 0;

private:
  LLVM::GlobalOp getGlobalString(Location loc, RewriterBase &rewriter,
                                 StringRef name, StringRef value,
                                 unsigned addressSpace) const;

  mutable llvm::DenseMap<std::pair<unsigned, StringAttr>, LLVM::GlobalOp>
      globals;
};

std::unique_ptr<TargetInfo> createTargetInfo(ModuleOp mod);

} // namespace mlir::triton::intel
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOINTEL_H
