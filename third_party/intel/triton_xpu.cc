#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

using namespace mlir::triton;
using namespace mlir::triton::gpu;

// Macros to create a pass that takes pass options.
#define ADD_PASS_WRAPPER_OPT_1(name, builder, ty0)                             \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })

void init_triton_intel_passes_ttir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_convert_to_ttgpuir_warp",
                     mlir::triton::createConvertTritonToTritonGPUWarpPass,
                     unsigned);
}

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_to_llvmir", intel::createConvertTritonIntelGPUToLLVM);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul",
                     intel::createTritonIntelGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_decompose_unsupported_conversions",
                     intel::createIntelDecomposeUnsupportedConversions);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     intel::createIntelAllocateSharedMemory);
  ADD_PASS_WRAPPER_OPT_1("add_pipeline", intel::createTritonIntelGPUPipeline,
                         int);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     intel::createTritonIntelGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     intel::createTritonIntelGPURewriteTensorPointer);
  ADD_PASS_WRAPPER_0("add_prefetch_block",
                     intel::createTritonIntelGPUPrefetchBlock);
  ADD_PASS_WRAPPER_0("add_distribute_to_warps",
                     intel::createTritonIntelGPUDistributeToWarps);
  ADD_PASS_WRAPPER_0("add_match_target_size",
                     intel::createTritonIntelGPUMatchTargetSize);
}

void init_triton_intel(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_intel_passes_ttir(passes.def_submodule("ttir"));
  init_triton_intel_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry
        .insert<TritonGEN::TritonGENDialect, intel::TritonIntelGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
