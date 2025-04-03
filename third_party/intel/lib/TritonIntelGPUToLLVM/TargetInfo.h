//===- TargetInfo.h - Target dependent information ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOINTEL_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOINTEL_H

#include "Utils/LibCallEmitter.h"
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

  Value getGlobalTimer(RewriterBase &rewriter, Location loc) const override;

  StringRef getAtomicSyncScope(MemSyncScope scope) const override;

  Value loadRelaxed(RewriterBase &rewriter, Location loc, Value ptr,
                    Type valueTy, Value pred,
                    MemSyncScope scope) const override;

  void storeRelaxed(RewriterBase &rewriter, Location loc, Value ptr,
                    Value value, Value pred, MemSyncScope scope) const override;

  void barrier(Location loc, RewriterBase &rewriter,
               triton::gpu::AddrSpace targets) const override;
  void clusterBarrier(Location loc, RewriterBase &rewriter,
                      Operation *sourceOp) const override;

  void warpSync(Location loc, RewriterBase &rewriter) const override;

  void storeDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Value val, Value pred) const override;
  Value loadDShared(RewriterBase &rewriter, Location loc, Value ptr,
                    Value ctaId, Type elemTy, Value pred,
                    Operation *localLoadOp = nullptr) const override;

  Value shuffleXor(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleUp(RewriterBase &rewriter, Location loc, Value val,
                  int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   int i) const override;
  Value shuffleIdx(RewriterBase &rewriter, Location loc, Value val,
                   Value i) const override;

  Value permute(RewriterBase &rewriter, Location loc, Value a, Value b,
                Value selector) const override;

  Value programId(RewriterBase &rewriter, Location loc, ModuleOp moduleOp,
                  ProgramIDDim axis) const override;

  bool warpBatchReduce(RewriterBase &rewriter, Location loc,
                       SmallVector<SmallVector<Value>> &acc,
                       triton::ReduceOp op,
                       unsigned reduceLaneIdMask) const override;

  bool warpReduce(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                  triton::ReduceOp op, unsigned reduceLaneIdMask,
                  unsigned broadcastLaneIdMask) const override;

  /// Replace the in-warp phase of a scan with a single hardware sub-group scan,
  /// updating `acc` in place. Returns false when the scan is not eligible, in
  /// which case the caller must emit the generic shuffle chain instead.
  ///
  /// `scanDim` is the number of lanes holding unique data along the scan axis;
  /// it must equal `warpSize` because the builtin scans the whole sub-group
  /// all-or-nothing (SPIR-V has no portable clustered scan).
  bool warpScan(RewriterBase &rewriter, Location loc, SmallVector<Value> &acc,
                triton::ScanOp op, unsigned scanDim, unsigned warpSize) const;

  unsigned getReductionTreeArity(Operation *combinerOp) const override;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const override;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const override;
  void assertTrap(RewriterBase &rewriter, Location loc) const override;
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

  /// Unlike `isSupportedWarpReduceOp`, which is handed the combine operation,
  /// this matches it out of `op` and returns it, so the whole eligibility
  /// decision lives in one place.
  virtual FailureOr<Operation *>
  matchSupportedWarpScanOp(triton::ScanOp op) const = 0;
  virtual Value genWarpScan(RewriterBase &rewriter, Location loc, Value acc,
                            Operation *combineOp, unsigned warpSize) const = 0;

private:
  LLVM::GlobalOp getGlobalString(Location loc, RewriterBase &rewriter,
                                 StringRef name, StringRef value,
                                 unsigned addressSpace) const;

  const mlir::triton::gpu::intel::LibCallEmitter emitter;
};

std::unique_ptr<TargetInfo> createTargetInfo(ModuleOp mod);

} // namespace mlir::triton::intel
#endif // TRITON_CONVERSION_TRITONGPU_TO_LLVM_TARGETINFOINTEL_H
