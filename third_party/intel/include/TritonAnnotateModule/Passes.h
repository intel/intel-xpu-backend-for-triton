//===- Passes.h - Triton Annotate Module Pass -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANNOTATE_MODULE_PASSES_H
#define TRITON_ANNOTATE_MODULE_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::intel {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "intel/include/TritonAnnotateModule/Passes.h.inc"
} // namespace mlir::triton::intel

#endif // TRITON_ANNOTATE_MODULE_PASSES_H
