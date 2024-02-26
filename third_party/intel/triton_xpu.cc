#include "TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
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

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  using namespace mlir::triton::gpu;

  // Device arch
  py::enum_<mlir::triton::gpu::intel::DeviceArch>(m, "DEVICE_ARCH",
                                                  py::module_local())
      .value("UNKNOWN", mlir::triton::gpu::intel::DeviceArch::UNKNOWN)
      .value("ATS", mlir::triton::gpu::intel::DeviceArch::ATS)
      .value("PVC", mlir::triton::gpu::intel::DeviceArch::PVC)
      .export_values();

  m.def("add_to_llvmir", [](mlir::PassManager &pm, int32_t capability) {
    pm.addPass(mlir::triton::createConvertTritonIntelGPUToLLVMPass(capability));
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
   }).def("add_materialize_block_pointer", [](mlir::PassManager &self) {
    self.addPass(mlir::triton::gpu::intel::
                     createTritonIntelGPUMaterializeBlockPointerPass());
  });
}

void init_triton_intel_passes_ttnvgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_1("add_plan_cta", mlir::createTritonNvidiaGPUPlanCTAPass,
                     mlir::triton::nvidia_gpu::ClusterInfo *);
}

void init_triton_intel(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_intel_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  init_triton_intel_passes_ttnvgpuir(passes.def_submodule("ttnvgpuir"));

  // cluster info
  py::class_<mlir::triton::nvidia_gpu::ClusterInfo>(m, "ClusterInfo")
      .def(py::init<>())
      .def_readwrite("clusterDimX",
                     &mlir::triton::nvidia_gpu::ClusterInfo::clusterDimX)
      .def_readwrite("clusterDimY",
                     &mlir::triton::nvidia_gpu::ClusterInfo::clusterDimY)
      .def_readwrite("clusterDimZ",
                     &mlir::triton::nvidia_gpu::ClusterInfo::clusterDimZ)
      .def("__repr__", [](mlir::triton::nvidia_gpu::ClusterInfo &self) {
        std::ostringstream oss;
        oss << "(" << self.clusterDimX << ", " << self.clusterDimY << ", "
            << self.clusterDimZ << ")";
        return oss.str();
      });

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::TritonGEN::TritonGENDialect,
                    mlir::triton::gpu::intel::TritonIntelGPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}
