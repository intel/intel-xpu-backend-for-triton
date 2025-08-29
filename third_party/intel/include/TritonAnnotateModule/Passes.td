//===-- Passes.td - Triton Annotate Module Passes ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ANNOTATE_MODULE_PASSES
#define TRITON_ANNOTATE_MODULE_PASSES

include "mlir/Pass/PassBase.td"

def TritonAnnotateModule: Pass<"triton-annotate-module", "mlir::ModuleOp"> {
  let summary = "Annotate the module with attributes used to steer code generation";
  let description = [{
    This pass analyzes the kernel's workload pattern and annotates the module with
    attributes that can be used to steer exploitation of HW features of the target
    device by subsequent transformations.
    For example, this pass can override the number of threads per warp (aka subgroup
    size) provided by the driver in order to enable lowering 'tt.dot' operations to
    DPAS instructions.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];

  let options = [
    Option<"minSGSize", "min-sg-size", "unsigned", /*default*/"32",
           "minimum subgroup size supported by the target device">,
    Option<"supportSG2DBlock", "support-sg-2d-block", "bool", /*default*/"false",
           "whether subgroup 2D block operations (e.g., 2D block read/write) are available">,
    Option<"supportDPAS", "support-dpas", "bool", /*default*/"false",
           "whether DPAS instruction is available">,
    Option<"supportBlockScaleDPAS", "support-block-scale-dpas", "bool", /*default*/"false",
           "whether Block Scale DPAS instruction is available">,
    Option<"supportBF16Conversion", "support-bf16-conversion", "bool", /*default*/"false",
           "whether BF16 conversion instruction is available">,
    Option<"support16BitAtomics", "support-16bit-atomics", "bool", /*default*/"false",
           "whether 16bit atomic operations are available">,
    Option<"threadsPerWarp", "threads-per-warp",
           "unsigned", /*default*/"32",
           "number of threads per warp (aka subgroup size)">,
    Option<"targetArch", "target-arch", "std::string", /*default*/"\"spir64\"",
           "target architecture name">
  ];
}

#endif // TRITON_ANNOTATE_MODULE_PASSES
