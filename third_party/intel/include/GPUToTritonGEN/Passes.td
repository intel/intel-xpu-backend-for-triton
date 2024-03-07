//===-- Passes.td - GPU to TritonGEN Conversion Passes -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GPU_TO_TRITONGEN_CONVERSION_PASSES
#define GPU_TO_TRITONGEN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertGPUToTritonGEN : Pass<"convert-gpu-to-tritongen", "mlir::gpu::GPUModuleOp"> {
  let summary = "Generate TritonGEN operations for gpu operations";
  let constructor = "mlir::createLowerGPUToTritonGENPass()";
  // TODO: attempt to remove CF and memref dialect.
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect",
    "TritonGEN::TritonGENDialect",
  ];
  // TODO: remove options
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"hasRedux", "has-redux", "bool", /*default=*/"false",
           "Target gpu supports redux">,
    Option<"useOpaquePointers", "use-opaque-pointers", "bool",
           /*default=*/"true", "Generate LLVM IR using opaque pointers "
           "instead of typed pointers">,
  ];
}

#endif // GPU_TO_TRITONGEN_CONVERSION_PASSES
