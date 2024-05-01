#include "TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"
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
  py::enum_<mlir::triton::gpu::intel::DeviceArch>(m, "DEVICE_ARCH",
                                                  py::module_local())
      .value("UNKNOWN", mlir::triton::gpu::intel::DeviceArch::UNKNOWN)
      .value("ATS", mlir::triton::gpu::intel::DeviceArch::ATS)
      .value("PVC", mlir::triton::gpu::intel::DeviceArch::PVC)
      .export_values();

  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonIntelGPUToLLVMPass());
  });
  m.def(
      "add_accelerate_matmul",
      [](mlir::PassManager &pm, mlir::triton::gpu::intel::DeviceArch arch) {
        pm.addPass(
            mlir::triton::gpu::intel::createTritonIntelGPUAccelerateMatmulPass(
                arch));
      },
      py::arg("pm"),
      py::arg("arch") = mlir::triton::gpu::intel::DeviceArch::UNKNOWN);
  m.def("add_decompose_unsupported_conversions", [](mlir::PassManager &pm) {
    pm.addPass(createIntelDecomposeUnsupportedConversionsPass());
  });
  m.def("add_allocate_shared_memory", [](mlir::PassManager &pm) {
    pm.addPass(createIntelAllocateSharedMemoryPass());
  });
  m.def("add_remove_layout_conversions", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::gpu::intel::
                   createTritonIntelGPURemoveLayoutConversionsPass());
  });
  m.def(
      "add_rewrite_tensor_pointer",
      [](mlir::PassManager &pm, mlir::triton::gpu::intel::DeviceArch arch) {
        pm.addPass(mlir::triton::gpu::intel::
                       createTritonIntelGPURewriteTensorPointerPass(arch));
      },
      py::arg("pm"),
      py::arg("arch") = mlir::triton::gpu::intel::DeviceArch::UNKNOWN);
  m.def("add_prefetch_block", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::gpu::intel::createPrefetchBlockPass());
  });
  m.def("add_distribute_to_warps", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::gpu::intel::createTritonIntelGPUDistributeToWarpsPass());
  });
  m.def("add_match_target_size", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::gpu::intel::createMatchTargetSizePass());
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
