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

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {
std::string getTypeMangling(mlir::Type type);
std::string mangle(llvm::StringRef baseName, llvm::ArrayRef<mlir::Type> types);
} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_INTEL_UTILS_MANGLING_H
