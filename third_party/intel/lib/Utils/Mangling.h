//===- Mangling.h - Function name mangling utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_UTILS_MANGLING_H
#define TRITON_INTEL_UTILS_MANGLING_H

#include "mlir/IR/Types.h"

#include <string>

namespace mlir::triton::gpu::intel {
std::string getTypeMangling(mlir::Type type, bool isUnsigned = false);
std::string mangle(llvm::StringRef baseName, llvm::ArrayRef<mlir::Type> types,
                   ArrayRef<bool> isUnsigned = {});
} // namespace mlir::triton::gpu::intel

#endif // TRITON_INTEL_UTILS_MANGLING_H
