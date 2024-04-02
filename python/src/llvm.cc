#include "mlir/IR/BuiltinOps.h" // mlir::ModuleOp
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Target/SPIRV/SPIRVTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace py = pybind11;

namespace llvm {
struct BreakStructPhiNodesPass : PassInfoMixin<BreakStructPhiNodesPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static StringRef name() { return "BreakStructPhiNodesPass"; }
};
} // namespace llvm

using namespace llvm;

std::string translateLLVMIRToASM(llvm::Module &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enable_fp_fusion, bool isObject) {
  using namespace mlir;
  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }
  if (mlir::triton::tools::getBoolEnv("LLVM_IR_ENABLE_DUMP")) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);

  // create machine
  module.setTargetTriple(triple);
  std::string error;
  auto target =
      llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
  llvm::TargetOptions opt;
  if (enable_fp_fusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module.getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive)};
  // set data layout
  module.setDataLayout(machine->createDataLayout());
  // emit machine code
  std::string result;
  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    for (llvm::Function &f : module.functions())
      f.addFnAttr(llvm::Attribute::AlwaysInline);
    llvm::legacy::PassManager pass;
    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);
    pass.run(module);
  }
  return result;
}

using ret = py::return_value_policy;

static uint32_t findKernels(llvm::Module &M,
                            std::set<llvm::Function *> &functions) {
  assert(functions.empty() && "Expecting an empty set");
  uint32_t numKernels = 0;
  for (llvm::Function &function : M.functions())
    if (function.getCallingConv() == CallingConv::SPIR_KERNEL) {
      functions.insert(&function);
      ++numKernels;
    }
  return numKernels;
}

/// Amend SPIR kernels in the given LLVM module by translating GEN passthrough
/// attributes into LLVM metadata.
static void amendLLVMIR(llvm::Module &llvmMod, llvm::LLVMContext &ctx) {
  // Collect SPIR kernels.
  std::set<llvm::Function *> kernels;
  uint32_t numKernels = findKernels(llvmMod, kernels);
  assert(numKernels == 1 && "Expecting a single SPIR kernel");
  llvm::Function *kernel = *kernels.begin();

  // Given a string \p str of the form "n1,n2,...", parse it as a
  // vector of integers (n1,n2,...).
  auto extractFromString = [](StringRef str) -> SmallVector<int64_t> {
    auto parseAsInt = [](StringRef str, int64_t &intVal) {
      bool failed = str.getAsInteger(10, intVal);
      return !failed;
    };

    SmallVector<int64_t> result;
    std::pair<StringRef, StringRef> pair;
    do {
      pair = str.split(',');
      str = pair.second;
      int64_t intVal;
      if (!parseAsInt(pair.first, intVal))
        break;

      result.push_back(intVal);
    } while (true);

    return result;
  };

  // Attach metadata to \p func given its name \p attrName and value \p attrVal.
  auto attachMetadata = [&](StringRef attrName, StringRef attrVal,
                            llvm::Function *func) {
    SmallVector<llvm::Metadata *, 3> metadata;
    llvm::Type *i64 = llvm::IntegerType::get(ctx, 64);
    for (int64_t val : extractFromString(attrVal))
      metadata.push_back(
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(i64, val)));

    llvm::MDNode *node = llvm::MDNode::get(ctx, metadata);
    func->setMetadata(attrName, node);
  };

  // Attach required metadata to the kernel.
  using namespace mlir::triton;
  SmallVector<llvm::StringLiteral> genAttrs{
      TritonGEN::TritonGENDialect::getMaxWorkGroupSizeAttrName(),
      TritonGEN::TritonGENDialect::getReqdWorkGroupSizeAttrName(),
      TritonGEN::TritonGENDialect::getReqdSubGroupSizeAttrName()};

  for (llvm::StringLiteral genAttr : genAttrs) {
    if (!kernel->hasFnAttribute(genAttr))
      continue;

    Attribute fnAttr = kernel->getFnAttribute(genAttr);
    assert(fnAttr.isStringAttribute() && "Expecting a string attribute");
    attachMetadata(fnAttr.getKindAsString().split('.').second,
                   fnAttr.getValueAsString(), kernel);
    kernel->removeFnAttr(genAttr);
  }
}

void init_triton_llvm(py::module &&m) {

  py::class_<llvm::LLVMContext>(m, "context", py::module_local())
      .def(py::init<>());

  py::class_<llvm::Module::FunctionListType>(m, "function_list")
      .def(
          "__iter__",
          [](llvm::Module::FunctionListType &s) {
            return py::make_iterator(s.begin(), s.end());
          },
          py::keep_alive<0, 1>());

  py::class_<llvm::Module>(m, "module", py::module_local())
      .def(
          "__str__",
          [](llvm::Module *self) {
            std::string str;
            llvm::raw_string_ostream os(str);
            os << *self;
            return os.str();
          },
          ret::take_ownership)
      .def(
          "get_functions",
          [](llvm::Module *mod) -> llvm::Module::FunctionListType & {
            return mod->getFunctionList();
          },
          ret::reference_internal);

  py::class_<llvm::Function>(m, "function", py::module_local())
      .def("set_calling_conv", &llvm::Function::setCallingConv)
      .def("add_fn_attr", [](llvm::Function *fn, std::string &name,
                             std::string &val) { fn->addFnAttr(name, val); })
      .def("has_public_visibility",
           [](llvm::Function *fn) {
             return fn->getVisibility() == llvm::GlobalValue::DefaultVisibility;
           })
      .def("is_declaration", &llvm::Function::isDeclaration);

  // optimization levels
  py::class_<llvm::OptimizationLevel>(m, "optimization_level",
                                      py::module_local());
  m.attr("OPTIMIZE_O0") = (llvm::OptimizationLevel::O0);
  m.attr("OPTIMIZE_O1") = (llvm::OptimizationLevel::O1);
  m.attr("OPTIMIZE_O2") = (llvm::OptimizationLevel::O2);
  m.attr("OPTIMIZE_O3") = (llvm::OptimizationLevel::O3);
  m.attr("OPTIMIZE_Os") = (llvm::OptimizationLevel::Os);
  m.attr("OPTIMIZE_Oz") = (llvm::OptimizationLevel::Oz);

  m.def(
      "to_module",
      [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx) {
        std::unique_ptr<llvm::Module> llvmMod =
            mlir::translateModuleToLLVMIR(mod, ctx);
        amendLLVMIR(*llvmMod, ctx);
        return llvmMod;
      },
      py::keep_alive<0, 2>());

  m.def("optimize_module", [](llvm::Module *mod,
                              const llvm::OptimizationLevel &opt) {
    if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
      return;
    using namespace llvm;
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
    // SLPVectorizer causes test_core.py::test_dot_mulbroadcastred to fail.
    // It vectorizes @llvm.fmuladd.f32 with @llvm.fmuladd.v32f32. We can
    // consider to reenable SLP vectorization when the failure is investigated.
    tuningOptions.SLPVectorization = false;

    PassBuilder pb(nullptr /*targetMachine*/, tuningOptions, std::nullopt,
                   instrCbPtr);

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          // Triton generates large structure of scalars which may pessimise
          // optimizations, we run a pass to break up phi of struct to make sure
          // all the struct are removed for the following passes.
          fpm.addPass(BreakStructPhiNodesPass());
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
    mpm.run(*mod, mam);
  });

  m.def(
      "translate_to_spirv",
      [](const std::string llvmIR) -> std::tuple<py::object, std::string> {
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
        std::string name = (*kernels.begin())->getName().str();
        std::string spirvBitcode = triton::translateLLVMIRToSPIRV(*module);
        return std::make_tuple(py::bytes(spirvBitcode), name);
      },
      ret::take_ownership);

  m.def(
      "translate_to_asm",
      [](std::string llvmIR, std::string triple, std::string proc,
         std::string features, std::vector<std::string> flags,
         bool enable_fp_fusion, bool isObject) -> py::object {
        std::string obj;
        {
          // when allow_threads goes out of scope, gil will be released
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
          obj = translateLLVMIRToASM(*module, triple, proc, features, flags,
                                     enable_fp_fusion, isObject);
        }
        if (isObject)
          return py::bytes(obj);
        else
          return py::str(obj);
      },
      ret::take_ownership);

  m.def("set_spv_target_triple", [](llvm::Module *mod) {
    std::string triple = "spir64-unknown-unknown";
    std::string layout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:"
                         "256-v256:256-v512:512-v1024:1024-n8:16:32:64";
    mod->setTargetTriple(triple);
    mod->setDataLayout(layout);
  });

  m.def("init_targets", []() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
      llvm::InitializeAllTargetInfos();
      llvm::InitializeAllTargets();
      llvm::InitializeAllTargetMCs();
      llvm::InitializeAllAsmParsers();
      llvm::InitializeAllAsmPrinters();
    });
  });

  m.def("link_extern_libs", [](llvm::Module *dstMod,
                               const std::vector<std::string> &paths) {
    if (paths.empty())
      return;

    LLVMContext &ctx = dstMod->getContext();
    llvm::Linker linker(*dstMod);
    for (const std::string &path : paths) {
      llvm::SMDiagnostic err;
      std::unique_ptr<llvm::Module> libMod = llvm::parseIRFile(path, err, ctx);
      if (!libMod) {
        std::string message = "Failed to parse library at " + path;
        throw std::invalid_argument(message);
      }
      libMod->setTargetTriple(dstMod->getTargetTriple());
      libMod->setDataLayout(dstMod->getDataLayout());
      if (linker.linkInModule(std::move(libMod),
                              llvm::Linker::Flags::LinkOnlyNeeded)) {
        std::string message = "Failed to link library at " + path;
        throw std::invalid_argument(message);
      }
    }
  });
}
