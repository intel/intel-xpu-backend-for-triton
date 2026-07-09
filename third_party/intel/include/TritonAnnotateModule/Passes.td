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
    Option<"support2DBlockIO", "support-2d-block-io", "bool", /*default*/"false",
           "whether subgroup 2D block operations (e.g., 2D block read/write) are available">,
    Option<"supportDPAS", "support-dpas", "bool", /*default*/"false",
           "whether DPAS instruction is available">,
    Option<"supportDPASWithBF8", "support-dpas-bf8", "bool", /*default*/"false",
           "whether DPAS instruction with BF8 support is available">,
    Option<"supportBlockScaleDPAS", "support-block-scale-dpas", "bool", /*default*/"false",
           "whether Block Scale DPAS instruction is available">,
    Option<"supportBF16Conversion", "support-bf16-conversion", "bool", /*default*/"false",
           "whether BF16 conversion instruction is available">,
    Option<"supportF4Conversion", "support-f4-conversion", "bool", /*default*/"false",
           "whether F4 conversion instruction is available">,
    Option<"supportF8Conversion", "support-f8-conversion", "bool", /*default*/"false",
           "whether B/HF8 <-> HF conversion instruction is available">,
    Option<"support16BitAtomics", "support-16bit-atomics", "bool", /*default*/"false",
           "whether 16bit atomic operations are available">,
    Option<"supportSigmoid", "support-sigmoid", "bool", /*default*/"false",
           "whether sigmoid operation is available">,
    Option<"supportBfloat16Arithmetic", "support-bfloat16-arithmetic", "bool", /*default*/"false",
           "whether bfloat16 arithmetic is supported">,
    Option<"supportPredicatedIO", "support-predicated-io", "bool", /*default*/"false",
           "whether predicated io is supported">,
    Option<"threadsPerWarp", "threads-per-warp",
           "unsigned", /*default*/"32",
           "number of threads per warp (aka subgroup size)">,
    Option<"targetArch", "target-arch", "std::string", /*default*/"\"spir64\"",
           "target architecture name">,
    Option<"isLTS", "is-lts", "bool", /*default*/"false",
           "whether driver is LTS">,
    Option<"supportPrefetch256Bytes", "support-prefetch-256b", "bool", /*default*/"false",
           "whether prefetch 256 bytes is supported">,
    Option<"support256bLoadStore", "support-256b-load-store", "bool", /*default*/"false",
           "whether 256-bit per-thread elementwise load/store is supported">,
    Option<"supportRoundedDivideSqrt", "support-rounded-divide-sqrt", "bool", /*default*/"false",
           "whether rounded divide and sqrt operations are available">,
    Option<"useClRoundedDivideSqrt", "use-cl-rounded-divide-sqrt", "bool", /*default*/"false",
           "whether to emit plain LLVM div/sqrt ops relying on the -cl-fp32-correctly-rounded-divide-sqrt build flag">,
    Option<"isFastMath", "is-fast-math", "bool", /*default*/"false",
           "whether fast-math approximations are allowed (set by TRITON_INTEL_FAST_MATH=1)">,
    Option<"sub32DPAS", "sub-32-dpas", "bool", /*default*/"false",
           "whether to enable DPAS for 32 subgroup size kernels">
  ];
}

#endif // TRITON_ANNOTATE_MODULE_PASSES
