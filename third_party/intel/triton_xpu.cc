#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Target/LLVMIR/LICM.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

using namespace mlir::triton;

// Macros to create a pass that takes pass options.
#define ADD_PASS_WRAPPER_OPT_1(name, builder, ty0)                             \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })

#define ADD_PASS_WRAPPER_OPT_2(name, builder, ty0, ty1)                        \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder({val0, val1}));                                         \
  })

void init_triton_intel_passes_ttir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_convert_to_ttgpuir_warp",
                     mlir::triton::createConvertTritonToTritonGPUWarpPass,
                     unsigned);
}

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  py::enum_<gpu::intel::DeviceArch>(m, "DEVICE_ARCH", py::module_local())
      .value("UNKNOWN", gpu::intel::DeviceArch::UNKNOWN)
      .value("ATS", gpu::intel::DeviceArch::ATS)
      .value("PVC", gpu::intel::DeviceArch::PVC)
      .export_values();

  ADD_PASS_WRAPPER_0("add_to_llvmir",
                     gpu::intel::createConvertTritonIntelGPUToLLVM);
  ADD_PASS_WRAPPER_OPT_1("add_accelerate_matmul",
                         gpu::intel::createTritonIntelGPUAccelerateMatmul,
                         gpu::intel::DeviceArch);
  ADD_PASS_WRAPPER_0("add_decompose_unsupported_conversions",
                     gpu::intel::createIntelDecomposeUnsupportedConversions);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     gpu::intel::createIntelAllocateSharedMemory);
  ADD_PASS_WRAPPER_OPT_2("add_pipeline",
                         gpu::intel::createTritonIntelGPUPipeline, int,
                         gpu::intel::DeviceArch);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     gpu::intel::createTritonIntelGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_OPT_1("add_rewrite_tensor_pointer",
                         gpu::intel::createTritonIntelGPURewriteTensorPointer,
                         gpu::intel::DeviceArch);
  ADD_PASS_WRAPPER_OPT_2("add_prefetch_block",
                         gpu::intel::createTritonIntelGPUPrefetchBlock, int,
                         bool);
  ADD_PASS_WRAPPER_0("add_distribute_to_warps",
                     gpu::intel::createTritonIntelGPUDistributeToWarps);
  ADD_PASS_WRAPPER_0("add_match_target_size",
                     gpu::intel::createTritonIntelGPUMatchTargetSize);
}

void init_triton_intel(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_intel_passes_ttir(passes.def_submodule("ttir"));
  init_triton_intel_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<TritonGEN::TritonGENDialect,
                    gpu::intel::TritonIntelGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("post_process_llir", [](llvm::Module *mod) { intel::LICM(*mod); });
}
