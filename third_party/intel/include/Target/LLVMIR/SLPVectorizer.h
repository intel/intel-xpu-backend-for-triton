//===- SLPVectorizer.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This pass implements the Bottom Up SLP vectorizer. It detects consecutive
// stores that can be put together into vector-stores. Next, it attempts to
// construct vectorizable tree using the use-def chains. If a profitable tree
// was found, the SLP vectorizer performs vectorization on the tree.
//
// The pass is inspired by the work described in the paper:
//  "Loop-Aware SLP in GCC" by Ira Rosen, Dorit Nuzman, Ayal Zaks.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TARGET_LLVMIR_SLPVECTORIZER_H
#define TRITON_TARGET_LLVMIR_SLPVECTORIZER_H

namespace llvm {
class Module;
} // namespace llvm

namespace mlir::triton::intel {
void SLPVectorizer(llvm::Module &module, bool trace);
} // namespace mlir::triton::intel

#endif // TRITON_TARGET_LLVMIR_SLPVECTORIZER_H
