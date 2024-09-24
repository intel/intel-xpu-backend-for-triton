#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Target/LLVMIR/Dialect/TritonGEN/TritonGENToLLVMIRTranslation.h"
#include "intel/include/Target/LLVMIR/PostProcess.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "intel/include/TritonToTritonGPUWarp/Passes.h"

#include "triton/Target/SPIRV/SPIRVTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace mlir::triton;
using ret = py::return_value_policy;

// Macros to create a pass that takes pass options.
#define ADD_PASS_WRAPPER_OPT_1(name, builder, ty0)                             \
  m.def(name,                                                                  \
        [](mlir::PassManager &pm, ty0 val0) { pm.addPass(builder({val0})); })
#define ADD_PASS_WRAPPER_OPT_2(name, builder, ty0, ty1)                        \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1) {                  \
    pm.addPass(builder({val0, val1}));                                         \
  })
#define ADD_PASS_WRAPPER_OPT_5(name, builder, ty0, ty1, ty2, ty3, ty4)         \
  m.def(name, [](mlir::PassManager &pm, ty0 val0, ty1 val1, ty2 val2,          \
                 ty3 val3, ty4 val4) {                                         \
    pm.addPass(builder({val0, val1, val2, val3, val4}));                       \
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
                         intel::createConvertTritonToTritonGPUWarp, unsigned);
}

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_to_llvmir",
                     gpu::intel::createConvertTritonIntelGPUToLLVM);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul",
                     gpu::intel::createTritonIntelGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_decompose_unsupported_conversions",
                     gpu::intel::createIntelDecomposeUnsupportedConversions);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     gpu::intel::createIntelAllocateSharedMemory);
  ADD_PASS_WRAPPER_OPT_2("add_pipeline",
                         gpu::intel::createTritonIntelGPUPipeline, int, bool);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     gpu::intel::createTritonIntelGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_rewrite_tensor_pointer",
                     gpu::intel::createTritonIntelGPURewriteTensorPointer);
  ADD_PASS_WRAPPER_OPT_2("add_prefetch_block",
                         gpu::intel::createTritonIntelGPUPrefetchBlock, int,
                         bool);
  ADD_PASS_WRAPPER_0("add_distribute_to_warps",
                     gpu::intel::createTritonIntelGPUDistributeToWarps);
  ADD_PASS_WRAPPER_0("add_match_target_size",
                     gpu::intel::createTritonIntelGPUMatchTargetSize);
  ADD_PASS_WRAPPER_0("add_schedule_load",
                     gpu::intel::createTritonIntelGPUScheduleLoad);
  ADD_PASS_WRAPPER_OPT_5("add_triton_annotate_module",
                         gpu::intel::createTritonAnnotateModule, unsigned, bool,
                         bool, bool, unsigned);
  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     gpu::intel::createTritonIntelGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_materialize_block_pointer",
                     gpu::intel::createTritonIntelGPUMaterializeBlockPointer);
  ADD_PASS_WRAPPER_0("add_schedule_ops",
                     gpu::intel::createTritonIntelGPUScheduleOps);
}

void init_triton_intel(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_intel_passes_ttir(passes.def_submodule("ttir"));
  init_triton_intel_passes_ttgpuir(passes.def_submodule("ttgpuir"));

  // cluster info
  py::class_<gpu::intel::ClusterInfo>(m, "ClusterInfo")
      .def(py::init<>())
      .def_readwrite("clusterDimX", &gpu::intel::ClusterInfo::clusterDimX)
      .def_readwrite("clusterDimY", &gpu::intel::ClusterInfo::clusterDimY)
      .def_readwrite("clusterDimZ", &gpu::intel::ClusterInfo::clusterDimZ)
      .def("__repr__", [](gpu::intel::ClusterInfo &self) {
        std::ostringstream oss;
        oss << "(" << self.clusterDimX << ", " << self.clusterDimY << ", "
            << self.clusterDimZ << ")";
        return oss.str();
      });

  m.def("optimize_module", [](llvm::Module *mod,
                              const llvm::OptimizationLevel &opt) {
    if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
      return;
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    using namespace llvm;
    if (!flagList.empty()) {
      auto options = llvm::cl::getRegisteredOptions();
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    ModuleAnalysisManager mam;

    PassInstrumentationCallbacks *instrCbPtr = nullptr;
    PassInstrumentationCallbacks passInstrCb;
    StandardInstrumentations standardInstr(mod->getContext(),
                                           /*DebugLogging*/ true);
    if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
      auto optMap = llvm::cl::getRegisteredOptions();
      auto optIt = optMap.find("print-after-all");
      if (optIt != optMap.end()) {
        auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
        *optPtr = true;
      }
      standardInstr.registerCallbacks(passInstrCb, &mam);
      instrCbPtr = &passInstrCb;
    }

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    // SLPVectorizer causes test_core.py::test_dot_mulbroadcasted to fail.
    // It vectorizes @llvm.fmuladd.f32 with @llvm.fmuladd.v32f32. We can
    // consider to reenable SLP vectorization when the failure is
    // investigated.
    tuningOptions.SLPVectorization = false;

    PassBuilder pb(nullptr /*targetMachine*/, tuningOptions, std::nullopt,
                   instrCbPtr);

    std::string pluginFile =
        mlir::triton::tools::getStrEnv("LLVM_PASS_PLUGIN_PATH");

    if (!pluginFile.empty()) {
      // TODO: Add some logging here that we inserted a pass into the LLVM
      // pass pipeline
      auto passPlugin = llvm::PassPlugin::Load(pluginFile);
      if (!passPlugin) {
        llvm::Error Err = passPlugin.takeError();
        std::string ErrMsg =
            "Pass Plugin Error: " + llvm::toString(std::move(Err));
        throw std::runtime_error(ErrMsg);
      }
      passPlugin->registerPassBuilderCallbacks(pb);
    }

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          // Triton generates large structure of scalars which may pessimise
          // optimizations, we run a pass to break up phi of struct to make
          // sure all the struct are removed for the following passes.
          fpm.addPass(BreakStructPhiNodesPass());
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
    mpm.run(*mod, mam);
  });

  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<TritonGEN::TritonGENDialect,
                    gpu::intel::TritonIntelGPUDialect>();
    mlir::registerTritonGENDialectTranslation(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("set_spv_target_triple", [](llvm::Module *mod) {
    std::string triple = "spir64-unknown-unknown";
    std::string layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                         "256-v256:256-v512:512-v1024:1024-n8:16:32:64";
    mod->setTargetTriple(triple);
    mod->setDataLayout(layout);
  });

  m.def("post_process_llir",
        [](llvm::Module *mod) { intel::postProcessLLVMIR(*mod); });

  m.def(
      "translate_to_spirv",
      [](const std::string &llvmIR) -> std::tuple<py::object, std::string> {
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
          const uint32_t numKernels = findKernels(*module, kernels);
          assert(numKernels == 1 && "Expecting a single SPIR kernel");
          name = (*kernels.begin())->getName().str();
          spirvBitcode = triton::translateLLVMIRToSPIRV(*module);
        }
        return std::make_tuple(py::bytes(spirvBitcode), name);
      },
      ret::take_ownership);
}
