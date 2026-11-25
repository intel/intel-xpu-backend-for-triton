//===-- TritonGENDialect.td - dialect op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_DIALECT
#define TRITONGEN_DIALECT

include "mlir/IR/OpBase.td"

def TritonGEN_Dialect : Dialect {
  let name = "triton_gen";
  let cppNamespace = "::mlir::triton::TritonGEN";
  let summary = "The TritonGEN dialect in Triton.";
  let useDefaultAttributePrinterParser = 1;

  let description = [{
    TritonGEN is a dialect for representing operations on Intel GPUs.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];

  let extraClassDeclaration = [{
    /// Get the name for the attribute used to specify cache control
    /// decorations.
    static constexpr ::llvm::StringRef getCacheControlsAttrName() {
      return ::llvm::StringLiteral("triton_gen.DecorationCacheControlINTEL");
    }

    /// Get the name for the attribute used to specify FPRoundingMode
    /// decorations (SPV_INTEL_rounded_divide_sqrt).
    static constexpr ::llvm::StringRef getFPRoundingModeAttrName() {
      return ::llvm::StringLiteral("triton_gen.FPRoundingMode");
    }

    /// Get the name for the attribute that keeps a floating-point operation
    /// out of fast math (TRITON_INTEL_FAST_MATH). It becomes metadata of the
    /// same name on the LLVM instruction, which set_fast_math gives the flags
    /// of the default math instead.
    static constexpr ::llvm::StringRef getKeepFPMathAttrName() {
      return ::llvm::StringLiteral("triton_gen.keep_fp_math");
    }
  }];
}

#endif // TRITONGEN_DIALECT
