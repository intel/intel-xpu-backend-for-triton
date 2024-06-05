#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Target/LLVMIR/Dialect/TritonGEN/TritonGENToLLVMIRTranslation.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "intel/include/TritonToTritonGPUWarp/Passes.h"
#include "intel/include/TritonToTritonGPUWarp/TritonToTritonGPUWarpPass.h"

#include "triton/Target/SPIRV/SPIRVTranslation.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ret = py::return_value_policy;

// Macros to create a pass that takes pass options.
#define ADD_PASS_WRAPPER_OPT_1(name, builder, ty0)                             \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })
#define ADD_PASS_WRAPPER_OPT_2(name, builder, ty0, ty1)                        \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder({val0, val1}));                                         \
  })

static uint32_t findKernels(llvm::Module &M,
                            std::set<llvm::Function *> &functions) {
  assert(functions.empty() && "Expecting an empty set");
  uint32_t numKernels = 0;
  for (llvm::Function &function : M.functions())
    if (function.getCallingConv() == llvm::CallingConv::SPIR_KERNEL) {
      functions.insert(&function);
      ++numKernels;
    }
  return numKernels;
}

void init_triton_intel_passes_ttir(py::module &&m) {
  ADD_PASS_WRAPPER_OPT_1("add_convert_to_ttgpuir_warp",
                         mlir::triton::createConvertTritonToTritonGPUWarp,
                         unsigned);
}

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  py::enum_<intel::DeviceArch>(m, "DEVICE_ARCH", py::module_local())
      .value("UNKNOWN", intel::DeviceArch::UNKNOWN)
      .value("ATS", intel::DeviceArch::ATS)
      .value("PVC", intel::DeviceArch::PVC)
      .export_values();

  ADD_PASS_WRAPPER_0("add_to_llvmir", intel::createConvertTritonIntelGPUToLLVM);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul",
                     intel::createTritonIntelGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_decompose_unsupported_conversions",
                     intel::createIntelDecomposeUnsupportedConversions);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     intel::createIntelAllocateSharedMemory);
  ADD_PASS_WRAPPER_OPT_2("add_pipeline", intel::createTritonIntelGPUPipeline,
                         int, bool);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     intel::createTritonIntelGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     intel::createTritonIntelGPURewriteTensorPointer);
  ADD_PASS_WRAPPER_OPT_2("add_prefetch_block",
                         intel::createTritonIntelGPUPrefetchBlock, int, bool);
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
    mlir::registerTritonGENDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("set_spv_target_triple", [](llvm::Module *mod) {
    // FIXME: Change triple back to spir64-unknown-unknown, when missing
    // SPIR-V 1.4 features are backported.
    std::string triple = "spirv64v1.3-unknown-unknown";
    std::string layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                         "256-v256:256-v512:512-v1024:1024-n8:16:32:64";
    mod->setTargetTriple(triple);
    mod->setDataLayout(layout);
  });

  m.def(
      "translate_to_spirv",
      [](const std::string llvmIR) -> std::tuple<py::object, std::string> {
        std::string name;
        std::string spirvBitcode;
        {
          py::gil_scoped_release allow_threads;
          // create LLVM module from C++
          llvm::LLVMContext context;
          std::unique_ptr<llvm::MemoryBuffer> buffer =
              llvm::MemoryBuffer::getMemBuffer(llvmIR.c_str());
          llvm::SMDiagnostic error;
          std::unique_ptr<llvm::Module> module =
              llvm::parseIR(buffer->getMemBufferRef(), error, context);
          if (!module) {
            llvm::report_fatal_error(
                "failed to parse IR: " + error.getMessage() +
                "lineno: " + std::to_string(error.getLineNo()));
          }
          // Get name of kernel in the module
          std::set<llvm::Function *> kernels;
          uint32_t numKernels = findKernels(*module, kernels);
          assert(numKernels == 1 && "Expecting a single SPIR kernel");
          name = (*kernels.begin())->getName().str();
          spirvBitcode = triton::translateLLVMIRToSPIRV(*module);
        }
        return std::make_tuple(py::bytes(spirvBitcode), name);
      },
      ret::take_ownership);
}
