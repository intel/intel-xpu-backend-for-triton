//===- LibCallEmitter.h - Emit library calls for Intel backend --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_UTILS_LIBCALLEMITTER_H
#define TRITON_INTEL_UTILS_LIBCALLEMITTER_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::gpu::intel {

class LibCallEmitter {
public:
  LibCallEmitter() = default;

  void printf(RewriterBase &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args,
              ArrayRef<bool> isSigned = {}) const;

  void printf(RewriterBase &rewriter, StringRef msg, ValueRange args,
              ArrayRef<bool> isSigned = {}) const;

  void assertFail(RewriterBase &rewriter, Location loc, StringRef message,
                  StringRef file, StringRef func, int line) const;

  Value getGlobalStringStart(Location loc, RewriterBase &rewriter,
                             StringRef name, StringRef value,
                             unsigned addressSpace) const;

private:
  LLVM::GlobalOp getGlobalString(Location loc, RewriterBase &rewriter,
                                 StringRef name, StringRef value,
                                 unsigned addressSpace) const;

  mutable llvm::DenseMap<std::pair<unsigned, StringAttr>, LLVM::GlobalOp>
      globals;
};

} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_UTILS_LIBCALLEMITTER_H
