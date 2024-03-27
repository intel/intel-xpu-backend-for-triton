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

  let description = [{
    TritonGEN is a dialect for representing operations on Intel GPUs.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to annotate max work group size
    /// required for kernels.
    static constexpr ::llvm::StringLiteral getMaxWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("gen.max_work_group_size");
    }

    /// Get the name of the attribute used to annotate exact work group size
    /// required for kernels.
    static constexpr ::llvm::StringLiteral getReqdWorkGroupSizeAttrName() {
      return ::llvm::StringLiteral("gen.reqd_work_group_size");
    }

    /// Get the name for the attribute used to annotate the exact sub group
    /// size required for kernels.
    static constexpr ::llvm::StringLiteral getReqdSubGroupSizeAttrName() {
      return ::llvm::StringLiteral("gen.intel_reqd_sub_group_size");
    }
  }];
}

#endif // TRITONGEN_DIALECT
