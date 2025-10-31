#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Operator.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"

#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Target/LLVMIR/Dialect/TritonGEN/TritonGENToLLVMIRTranslation.h"
#include "intel/include/Target/LLVMIR/PostProcess.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "intel/include/TritonGENToLLVM/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "intel/lib/Target/LLVMIR/LLVMPasses.h"

#include "intel/include/Target/SPIRV/SPIRVTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <chrono>

namespace py = pybind11;

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace mlir::triton;
using ret = py::return_value_policy;

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
  ADD_PASS_WRAPPER_0("add_convert_tdesc_to_block_pointer",
                     intel::createTritonIntelTensorDescToBlockPointer);
  ADD_PASS_WRAPPER_0("add_remove_masks", intel::createTritonIntelRemoveMasks);
  ADD_PASS_WRAPPER_0("add_fuse_reshape", intel::createTritonIntelFuseReshape);
}

void init_triton_intel_passes_ttgpuir(py::module &&m) {
  ADD_PASS_WRAPPER_0("add_to_llvmir",
                     gpu::intel::createConvertTritonIntelGPUToLLVM);
  ADD_PASS_WRAPPER_0("add_gen_to_llvm", createConvertTritonGENToLLVM);
  ADD_PASS_WRAPPER_0("add_accelerate_matmul",
                     gpu::intel::createTritonIntelGPUAccelerateMatmul);
  ADD_PASS_WRAPPER_0("add_rewrite_stack_ptr",
                     gpu::intel::createTritonIntelGPURewriteStackPtr);
  ADD_PASS_OPTION_WRAPPER_2("add_pipeline",
                            gpu::intel::createTritonIntelGPUPipeline, int,
                            enum gpu::intel::SplitBarrierScope);
  ADD_PASS_WRAPPER_0("add_allocate_shared_memory",
                     gpu::intel::createIntelAllocateSharedMemory);
  ADD_PASS_WRAPPER_0("add_remove_layout_conversions",
                     gpu::intel::createTritonIntelGPURemoveLayoutConversions);
  ADD_PASS_WRAPPER_0("add_optimize_dot_operands",
                     gpu::intel::createTritonIntelGPUOptimizeDotOperands);
  ADD_PASS_WRAPPER_0("add_coalesce", gpu::intel::createTritonIntelGPUCoalesce);

  py::class_<gpu::intel::TritonAnnotateModuleOptions>(m,
                                                      "AnnotateModuleOptions")
      .def(py::init<>())
      .def_readwrite("min_sg_size",
                     &gpu::intel::TritonAnnotateModuleOptions::minSGSize)
      .def_readwrite("support_sg_2d_block",
                     &gpu::intel::TritonAnnotateModuleOptions::supportSG2DBlock)
      .def_readwrite("support_dpas",
                     &gpu::intel::TritonAnnotateModuleOptions::supportDPAS)
      .def_readwrite(
          "support_bf16_conversion",
          &gpu::intel::TritonAnnotateModuleOptions::supportBF16Conversion)
      .def_readwrite(
          "support_16bit_atomics",
          &gpu::intel::TritonAnnotateModuleOptions::support16BitAtomics)
      .def_readwrite("threads_per_warp",
                     &gpu::intel::TritonAnnotateModuleOptions::threadsPerWarp)
      .def_readwrite("target_arch",
                     &gpu::intel::TritonAnnotateModuleOptions::targetArch);
  ADD_PASS_OPTION_WRAPPER_1("add_triton_annotate_module",
                            gpu::intel::createTritonAnnotateModule,
                            gpu::intel::TritonAnnotateModuleOptions);

  ADD_PASS_WRAPPER_0("add_reduce_data_duplication",
                     gpu::intel::createTritonIntelGPUReduceDataDuplication);
  ADD_PASS_WRAPPER_0("add_materialize_block_pointer",
                     gpu::intel::createTritonIntelGPUMaterializeBlockPointer);
  ADD_PASS_WRAPPER_0("add_optimize_reduction_locality",
                     gpu::intel::createTritonIntelGPUOptimizeReductionLocality);
  ADD_PASS_WRAPPER_0("add_reduce_variable_liveness",
                     gpu::intel::createTritonIntelGPUReduceVariableLiveness);
}

void init_triton_intel_passes_arith(py::module &&m) {
  m.def("add_arith_emulate_unsupported_floats",
        [](mlir::PassManager &pm,
           const std::vector<std::string> &sourceTypeStrs,
           const std::string &targetTypeStr) {
          pm.addPass(mlir::arith::createArithEmulateUnsupportedFloats(
              {llvm::SmallVector<std::string>{sourceTypeStrs.begin(),
                                              sourceTypeStrs.end()},
               targetTypeStr}));
        });
}

void init_triton_intel(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_intel_passes_ttir(passes.def_submodule("ttir"));
  init_triton_intel_passes_ttgpuir(passes.def_submodule("ttgpuir"));
  init_triton_intel_passes_arith(passes.def_submodule("arith"));

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

  // Split barrier scope enum
  py::enum_<gpu::intel::SplitBarrierScope>(m, "SplitBarrierScope")
      .value("none", gpu::intel::SplitBarrierScope::None)
      .value("Workgroup", gpu::intel::SplitBarrierScope::Workgroup)
      .value("Subgroup", gpu::intel::SplitBarrierScope::Subgroup);

  m.def("optimize_module", [](llvm::Module *mod,
                              const llvm::OptimizationLevel &opt,
                              std::optional<py::function> pyCb = std::nullopt) {
    if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
      return;

    py::gil_scoped_release gil_release;

    // Check to see if we are passing a list of flags to disable
    // optimizations.
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
    llvm::DenseMap<llvm::StringRef,
                   std::chrono::high_resolution_clock::time_point>
        passStartTimes;

    if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
      auto optMap = llvm::cl::getRegisteredOptions();
      auto optIt = optMap.find("print-after-all");
      if (optIt != optMap.end()) {
        auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
        *optPtr = true;
      }
      standardInstr.registerCallbacks(passInstrCb, &mam);
      instrCbPtr = &passInstrCb;
    } else if (pyCb) {
      instrCbPtr = &passInstrCb;
      passInstrCb.registerBeforeNonSkippedPassCallback(
          [&passStartTimes](llvm::StringRef id, llvm::Any) {
            passStartTimes[id] = std::chrono::high_resolution_clock::now();
          });
      passInstrCb.registerAfterPassCallback(
          [&passStartTimes, &pyCb](llvm::StringRef id, llvm::Any,
                                   const llvm::PreservedAnalyses &) {
            auto start = passStartTimes.find(id);
            if (start != passStartTimes.end()) {
              auto end = std::chrono::high_resolution_clock::now();
              auto time =
                  std::chrono::duration_cast<std::chrono::duration<float>>(
                      end - start->second)
                      .count();
              passStartTimes.erase(start);
              py::gil_scoped_acquire gil_acquire;
              pyCb->operator()(id.str(), time, 0);
            }
          });
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
    pb.registerPeepholeEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          // The Triton masked load pattern can generate instances where the
          // mask value causes undefined behavior in sdiv/srem instructions.
          // The language allows this UB as the result of those arithmetic
          // instructions is never used, and control flow to avoid
          // computation of these instructions would negatively affect
          // performance. But, LLVM SimplifyCFG aggressively marks code
          // paths with undefined behavior as dead. This can result in
          // removal of the mask path and incorrect results from legal
          // Triton kernels due to masked elements being used in
          // computation. Run a pass to add a freeze instruction between
          // masked loads and sdiv/srem to signal to LLVM we consider the
          // sdiv/srem operands to be well defined.
          fpm.addPass(FreezeMaskedDivRemPass());
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

  m.def("get_threads_per_warp", [](mlir::ModuleOp &mod) -> py::object {
    auto ret = mlir::triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    return py::int_(ret);
  });

  m.def("has_precise_divide_sqrt", [](mlir::ModuleOp &mod) -> bool {
    using namespace mlir;
    WalkResult result = mod.walk([&](Operation *op) {
      if (isa<mlir::triton::PreciseDivFOp, mlir::triton::PreciseSqrtOp>(op))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return result.wasInterrupted();
  });

  // FIXME: This is for internal experimentation. In the end we will need a
  // producer flag (e.g. PyTorch flag) to allow the Triton compiler to use the
  // fast math semantics on all arithmetic operations.
  // https://github.com/intel/intel-xpu-backend-for-triton/issues/3862
  m.def("set_fast_math", [](llvm::Module *mod) {
    std::optional<bool> fastMath = mlir::triton::tools::isEnvValueBool(
        mlir::triton::tools::getStrEnv("TRITON_INTEL_FAST_MATH"));
    std::optional<bool> enableFpFusion = mlir::triton::tools::isEnvValueBool(
        mlir::triton::tools::getStrEnv("TRITON_DEFAULT_FP_FUSION"));
    if (fastMath.has_value() && !fastMath.value())
      return;

    using namespace llvm;
    for (Function &func : *mod) {
      for (Instruction &inst : instructions(func)) {
        if (auto *op = dyn_cast<FPMathOperator>(&inst)) {
          FastMathFlags FMF;
          // Default to allow contract when default fp fusion is not disabled.
          if ((!enableFpFusion.has_value() || enableFpFusion.value()) &&
              !fastMath.has_value()) {
            if (op->getOpcode() == Instruction::FAdd ||
                op->getOpcode() == Instruction::FMul)
              FMF.setAllowContract(true);
          } else if (fastMath.has_value() && fastMath.value())
            FMF.setFast(true);
          inst.setFastMathFlags(FMF);
        }
      }
    }
  });

  m.def("set_spv_target_triple", [](llvm::Module *mod) {
    std::string triple = "spir64-unknown-unknown";
    std::string layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                         "256-v256:256-v512:512-v1024:1024-n8:16:32:64";
    mod->setTargetTriple(llvm::Triple(triple));
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
