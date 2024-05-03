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

void init_triton_intel_passes_ttir(py::module &&m) {
  m.def("add_convert_to_ttgpuir_warp", [](mlir::PassManager &pm, int numWarps) {
    pm.addPass(mlir::triton::createConvertTritonToTritonGPUWarpPass(numWarps));
  });
}

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton::gpu;

  // Device arch
  py::enum_<intel::DeviceArch>(m, "DEVICE_ARCH", py::module_local())
      .value("UNKNOWN", intel::DeviceArch::UNKNOWN)
      .value("ATS", intel::DeviceArch::ATS)
      .value("PVC", intel::DeviceArch::PVC)
      .export_values();

  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(intel::createConvertTritonIntelGPUToLLVM());
  });
  m.def(
      "add_accelerate_matmul",
      [](mlir::PassManager &pm, intel::DeviceArch arch) {
        pm.addPass(intel::createTritonIntelGPUAccelerateMatmul({arch}));
      },
      py::arg("pm"), py::arg("arch") = intel::DeviceArch::UNKNOWN);
  m.def("add_decompose_unsupported_conversions", [](mlir::PassManager &pm) {
    pm.addPass(intel::createIntelDecomposeUnsupportedConversions());
  });
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm) {
    pm.addPass(intel::createIntelAllocateSharedMemory());
  });
  m.def(
      "add_pipe_line_pass",
      [](mlir::PassManager &pm, int numStages, intel::DeviceArch arch) {
        pm.addPass(intel::createTritonIntelGPUPipeline({numStages, arch}));
      },
      py::arg("pm"), py::arg("numStages"),
      py::arg("arch") = intel::DeviceArch::UNKNOWN);
  m.def("add_remove_layout_conversions", [](mlir::PassManager &pm) {
    pm.addPass(intel::createTritonIntelGPURemoveLayoutConversions());
  });
  m.def(
      "add_rewrite_tensor_pointer",
      [](mlir::PassManager &pm, intel::DeviceArch arch) {
        pm.addPass(intel::createTritonIntelGPURewriteTensorPointer({arch}));
      },
      py::arg("pm"), py::arg("arch") = intel::DeviceArch::UNKNOWN);
  m.def("add_prefetch_block", [](mlir::PassManager &pm) {
    pm.addPass(intel::createTritonIntelGPUPrefetchBlock());
  });
  m.def("add_distribute_to_warps", [](mlir::PassManager &pm) {
    pm.addPass(intel::createTritonIntelGPUDistributeToWarps());
  });
  m.def("add_match_target_size", [](mlir::PassManager &pm) {
    pm.addPass(intel::createTritonIntelGPUMatchTargetSize());
  });
}

void init_triton_intel(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_intel_passes_ttir(passes.def_submodule("ttir"));
  init_triton_intel_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonGEN::TritonGENDialect,
                    mlir::triton::gpu::intel::TritonIntelGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
