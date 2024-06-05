#include "mlir/Support/LLVM.h"
// #include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "triton/Target/SPIRV/SPIRVTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/ScopedNoAliasAA.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
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
  bool disableLLVMOpt = mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT");
  if (!disableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
    if (!flagList.empty()) {
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
  }

  // inline everything
  for (llvm::Function &f : module.functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);
  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  const bool enabledTiming =
      mlir::triton::tools::getBoolEnv("LLVM_ENABLE_TIMING");
  if (enabledTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  pm.run(module);

  SmallString<0> timePassesStr;
  raw_svector_ostream reportStream(timePassesStr);

  if (enabledTiming) {
    reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }
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
      std::nullopt,
      disableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};
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

    if (enabledTiming) {
      reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
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

class LICMPass : public PassInfoMixin<LICMPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);
    auto &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();

    auto inSubLoop = [](BasicBlock *BB, Loop *L, LoopInfo &LI) {
      return LI.getLoopFor(BB) != L;
    };

    for (auto *L : LI) {
      llvm::errs() << "Perform LICM on Loop with header at block "
                   << L->getHeader()->getNameOrAsOperand() << "\n";
      llvm::errs() << "Loop: " << L->getName() << "\n";
      L->dump();

      LoopBlocksRPO workList(L);
      workList.perform(&LI);

      BasicBlock *Preheader = L->getLoopPreheader();
      assert(Preheader && "Loop does not have a preheader");

      for (BasicBlock *BB : workList) {
        for (Instruction &I : llvm::make_early_inc_range(*BB)) {
          if (inSubLoop(BB, L, LI))
            continue;

          if (L->hasLoopInvariantOperands(&I))
            llvm::errs() << "operands are loop invariant: " << I << "\n";

          if (L->hasLoopInvariantOperands(&I) && canHoist(I, AA, DT, MSSA, L)) {
            llvm::errs() << "Hoisting: " << I << "\n";
          }
        }
      }
    }

    return PreservedAnalyses::all();
  }

private:
  bool isHoistable(Instruction &I) const {
    // Only these instructions are hoistable/sinkable.
    return (isa<CallInst>(I) || isa<CastInst>(I) || isa<UnaryOperator>(I) ||
            isa<BinaryOperator>(I) || isa<SelectInst>(I) ||
            isa<GetElementPtrInst>(I) || isa<CmpInst>(I) ||
            isa<InsertElementInst>(I) || isa<ExtractElementInst>(I) ||
            isa<ShuffleVectorInst>(I) || isa<ExtractValueInst>(I) ||
            isa<InsertValueInst>(I) || isa<FreezeInst>(I));
  }

  // Return true if LI is invariant within scope of the loop. LI is invariant if
  // the loop is dominated by an invariant.start representing the same memory
  // location and size as the memory location LI loads from, and also the
  // invariant.start has no uses.
  bool isLoadInvariantInLoop(LoadInst *LI, DominatorTree &DT, Loop *L) const {
    Value *Addr = LI->getPointerOperand();
    const DataLayout &DL = LI->getModule()->getDataLayout();
    const TypeSize LocSizeInBits = DL.getTypeSizeInBits(LI->getType());

    // It is not currently possible for clang to generate an invariant.start
    // intrinsic with scalable vector types because we don't support thread
    // local sizeless types and we don't permit sizeless types in structs or
    // classes. Furthermore, even if support is added for this in future the
    // intrinsic itself is defined to have a size of -1 for variable sized
    // objects. This makes it impossible to verify if the intrinsic envelops our
    // region of interest. For example, both <vscale x 32 x i8> and <vscale x 16
    // x i8> types would have a -1 parameter, but the former is clearly double
    // the size of the latter.
    if (LocSizeInBits.isScalable())
      return false;

    // If we've ended up at a global/constant, bail. We shouldn't be looking at
    // use lists for non-local Values in a loop pass.
    if (isa<Constant>(Addr))
      return false;

    unsigned UsesVisited = 0;
    // Traverse all uses of the load operand value, to see if invariant.start is
    // one of the uses, and whether it dominates the load instruction.
    for (auto *U : Addr->users()) {
      IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
      // If there are escaping uses of invariant.start instruction, the load
      // maybe non-invariant.
      if (!II || II->getIntrinsicID() != Intrinsic::invariant_start ||
          !II->use_empty())
        continue;
      ConstantInt *InvariantSize = cast<ConstantInt>(II->getArgOperand(0));
      // The intrinsic supports having a -1 argument for variable sized objects
      // so we should check for that here.
      if (InvariantSize->isNegative())
        continue;
      uint64_t InvariantSizeInBits = InvariantSize->getSExtValue() * 8;
      // Confirm the invariant.start location size contains the load operand
      // size in bits. Also, the invariant.start should dominate the load, and
      // we should not hoist the load out of a loop that contains this
      // dominating invariant.start.
      if (LocSizeInBits.getFixedValue() <= InvariantSizeInBits &&
          DT.properlyDominates(II->getParent(), L->getHeader()))
        return true;
    }
    return false;
  }

  bool canHoist(Instruction &I, AAResults &AA, DominatorTree &DT,
                MemorySSA &MSSA, Loop *L) {
    if (!isHoistable(I))
      return false;

    // Return true if MSSA knows there are no MemoryDefs in the loop.
    auto isReadOnly = [](MemorySSA &MSSA, const Loop *L) {
      for (auto *BB : L->getBlocks())
        if (MSSA.getBlockDefs(BB))
          return false;
      return true;
    };

    auto pointerInvalidatedByLoop = [&](MemorySSA &MSSA, MemoryUse *MU, Loop *L,
                                        Instruction *I) {
      BatchAAResults BAA(MSSA.getAA());
      auto getClobberingMemoryAccess = [](MemorySSA &MSSA, BatchAAResults &BAA,
                                          MemoryUseOrDef *MA) {
        return MSSA.getSkipSelfWalker()->getClobberingMemoryAccess(MA, BAA);
      };
      MemoryAccess *Source = getClobberingMemoryAccess(MSSA, BAA, MU);
      return !MSSA.isLiveOnEntryDef(Source) && L->contains(Source->getBlock());
    };

#if 0
    // Loads have extra constraints we have to verify before we can hoist them.
    if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
      if (!LI->isUnordered())
        return false; // Don't sink/hoist volatile or ordered atomic loads!

      // Loads from constant memory are always safe to move, even if they end up
      // in the same alias set as something that ends up being modified.
      if (!isModSet(AA.getModRefInfoMask(LI->getOperand(0))))
        return true;
      if (LI->hasMetadata(LLVMContext::MD_invariant_load))
        return true;

      // This checks for an invariant.start dominating the load.
      //      if (isLoadInvariantInLoop(LI, DT, L))
      //      return true;

      //    auto MU = cast<MemoryUse>(MSSA->getMemoryAccess(LI));

      bool InvariantGroup = LI->hasMetadata(LLVMContext::MD_invariant_group);
    } else
#endif

    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      // Only allow hoisting builtin calls.
      if (!CI->getCalledFunction()->getName().starts_with(
              "__builtin_IB_subgroup"))
        return false;

      if (CI->mayThrow()) {
        llvm::errs() << "CallInst may throw: " << *CI << "\n";
        return false;
      }
      if (CI->isConvergent()) {
        llvm::errs() << "CallInst is convergent: " << *CI << "\n";
        return false;
      }

      MemoryEffects Behavior = AA.getMemoryEffects(CI);
      llvm::errs() << "Behavior: " << Behavior << "\n";

      if (Behavior.doesNotAccessMemory())
        return true;
      if (Behavior.onlyReadsMemory()) {
        llvm::errs() << "only reads memory: " << *CI << "\n";
        // A readonly argmemonly function only reads from memory pointed to by
        // it's arguments with arbitrary offsets.  If we can prove there are no
        // writes to this memory in the loop, we can hoist it.
        if (Behavior.onlyAccessesArgPointees()) {
          llvm::errs() << "onlyAccessesArgPointees: " << *CI << "\n";
          // TODO: expand to writeable arguments
          for (Value *Op : CI->args())
            if (Op->getType()->isPointerTy() &&
                pointerInvalidatedByLoop(
                    MSSA, cast<MemoryUse>(MSSA.getMemoryAccess(CI)), L, &I))
              return false;
          return true;
        }

        // If this call only reads from memory and there are no writes to
        // memory in the loop, we can hoist or sink the call as appropriate.
        if (isReadOnly(MSSA, L))
          return true;
      }
      llvm::errs() << "not hoistable, at line: " << __LINE__ << "\n";
      return false;
    }

    assert(!I.mayReadOrWriteMemory() && "unhandled aliasing");

    return true;
  }
};

/// Attempt to hoist loop invariant calls (for address payloads)
/// FIXME: This is a temporary workaround (should be done by IGC). We should
/// remove it once that feature is implemented.
static void hostInvariantCalls(llvm::Module &llvmMod) {
  std::set<llvm::Function *> kernels;
  uint32_t numKernels = findKernels(llvmMod, kernels);
  assert(numKernels == 1 && "Expecting a single SPIR kernel");
  llvm::Function *kernel = *kernels.begin();

  llvm::errs() << "llvmMod: " << llvmMod << "\n";
  llvm::errs() << "Found kernel: " << kernel->getName() << "\n";

  PassBuilder PB;
  FunctionAnalysisManager FAM;

  FAM.registerPass([&] { return AssumptionAnalysis(); });
  FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  FAM.registerPass([&] { return DominatorTreeAnalysis(); });
  FAM.registerPass([&] { return LoopAnalysis(); });
  FAM.registerPass([&] { return TargetLibraryAnalysis(); });
  FAM.registerPass([&] { return TargetIRAnalysis(); });
  FAM.registerPass([&] { return BasicAA(); });
  FAM.registerPass([&] { return ScopedNoAliasAA(); });
  FAM.registerPass([&] { return TypeBasedAA(); });
  FAM.registerPass([&] { return MemorySSAAnalysis(); });
  FAM.registerPass([&] {
    AAManager AA;
    AA.registerFunctionAnalysis<BasicAA>();
    AA.registerFunctionAnalysis<ScopedNoAliasAA>();
    AA.registerFunctionAnalysis<TypeBasedAA>();
    return AA;
  });

  FunctionPassManager FPM;
  FPM.addPass(LICMPass());
  FPM.run(*kernel, FAM);

  llvm::errs() << "kernel: " << *kernel << "\n";
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

  // Module Flag behavior. See
  // https://llvm.org/doxygen/classllvm_1_1Module.html#a0a5c55e12c97b80021330fe82b642293
  // for details.
  py::class_<llvm::Module::ModFlagBehavior>(m, "module_flag_behavior",
                                            py::module_local());
  m.attr("MODULE_FLAG_BEHAVIOR_ERROR") = llvm::Module::Error;
  m.attr("MODULE_FLAG_BEHAVIOR_WARNING") = llvm::Module::Warning;
  m.attr("MODULE_FLAG_BEHAVIOR_REQUIRE") = llvm::Module::Require;
  m.attr("MODULE_FLAG_BEHAVIOR_OVERRIDE") = llvm::Module::Override;
  m.attr("MODULE_FLAG_BEHAVIOR_APPEND") = llvm::Module::Append;
  m.attr("MODULE_FLAG_BEHAVIOR_APPEND_UNIQUE") = llvm::Module::AppendUnique;
  m.attr("MODULE_FLAG_BEHAVIOR_MAX") = llvm::Module::Max;
  m.attr("MODULE_FLAG_BEHAVIOR_MIN") = llvm::Module::Min;

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
            // Note: Backends assume that we are compiling exactly one kernel
            // (i.e. one function that's that's called by the CPU) and that
            // it's the first function in this list.
            return mod->getFunctionList();
          },
          ret::reference_internal)
      .def("add_flag",
           [](llvm::Module *mod, llvm::Module::ModFlagBehavior behavior,
              std::string &key, uint32_t value) {
             return mod->addModuleFlag(behavior, key, value);
           });

  py::class_<llvm::Function>(m, "function", py::module_local())
      .def_property_readonly(
          "name", [](llvm::Function *fn) { return fn->getName().str(); })
      .def("set_calling_conv", &llvm::Function::setCallingConv)
      .def("add_fn_attr", [](llvm::Function *fn, std::string &name,
                             std::string &val) { fn->addFnAttr(name, val); })

      // Sets the nvvm.maxreg property on the given function.
      .def("set_nvvm_maxnreg",
           [](llvm::Function *fn, int maxnreg) {
             auto op = MDNode::get(
                 fn->getContext(),
                 {
                     ValueAsMetadata::get(fn),
                     MDString::get(fn->getContext(), "maxnreg"),
                     ConstantAsMetadata::get(ConstantInt::get(
                         Type::getInt32Ty(fn->getContext()), maxnreg)),
                 });
             fn->getParent()
                 ->getOrInsertNamedMetadata("nvvm.annotations")
                 ->addOperand(op);
           })
      // External functions that are definitions (i.e. not declarations) are
      // kernel functions.
      .def("is_declaration", &llvm::Function::isDeclaration)
      .def("is_external_linkage", [](llvm::Function *fn) {
        return fn->getLinkage() == llvm::GlobalValue::ExternalLinkage;
      });

  // optimization levels
  py::class_<llvm::OptimizationLevel>(m, "optimization_level",
                                      py::module_local());
  m.attr("OPTIMIZE_O0") = llvm::OptimizationLevel::O0;
  m.attr("OPTIMIZE_O1") = llvm::OptimizationLevel::O1;
  m.attr("OPTIMIZE_O2") = llvm::OptimizationLevel::O2;
  m.attr("OPTIMIZE_O3") = llvm::OptimizationLevel::O3;
  m.attr("OPTIMIZE_Os") = llvm::OptimizationLevel::Os;
  m.attr("OPTIMIZE_Oz") = llvm::OptimizationLevel::Oz;

  m.def(
      "to_module",
      [](mlir::ModuleOp &mod, llvm::LLVMContext &ctx) {
        std::unique_ptr<llvm::Module> llvmMod =
            mlir::translateModuleToLLVMIR(mod, ctx);
        return llvmMod;
      },
      py::keep_alive<0, 2>());

  m.def(
      "optimize_module",
      [](llvm::Module *mod, const llvm::OptimizationLevel &opt,
         const std::string triple) {
        if (mlir::triton::tools::getBoolEnv("DISABLE_LLVM_OPT"))
          return;
        // Check to see if we are passing a list of flags to disable
        // optimizations.
        auto flagList = mlir::triton::tools::getStrEnv("DISABLE_LLVM_OPT");
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
        // SLPVectorizer causes test_core.py::test_dot_mulbroadcasted to fail.
        // It vectorizes @llvm.fmuladd.f32 with @llvm.fmuladd.v32f32. We can
        // consider to reenable SLP vectorization when the failure is
        // investigated.
        tuningOptions.SLPVectorization = false;

        if (!triple.empty())
          mod->setTargetTriple(triple.c_str());

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
              // Triton generates large structure of scalars which may
              // pessimise optimizations, we run a pass to break up phi of
              // struct to make sure all the struct are removed for the
              // following passes.
              fpm.addPass(BreakStructPhiNodesPass());
              fpm.addPass(InstCombinePass());
            });
        mpm.addPass(pb.buildPerModuleDefaultPipeline(opt));
        mpm.run(*mod, mam);

        hostInvariantCalls(*mod);
      },
      py::arg("mod"), py::arg("opt"), py::arg("triple") = "");

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
    // FIXME: Change triple back to spir64-unknown-unknown, when missing
    // SPIR-V 1.4 features are backported.
    std::string triple = "spirv64v1.3-unknown-unknown";
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

      std::unordered_set<std::string> externalFns;
      for (llvm::Function &fn : libMod->functions()) {
        if (!fn.isDeclaration())
          externalFns.insert(fn.getName().str());
      }

      if (linker.linkInModule(std::move(libMod),
                              llvm::Linker::Flags::LinkOnlyNeeded)) {
        std::string message = "Failed to link library at " + path;
        throw std::invalid_argument(message);
      }

      // Mark linked-in functions as internal because backends use
      // external linkage as a signifier of kernel functions.
      for (llvm::Function &fn : dstMod->functions()) {
        if (externalFns.count(fn.getName().str())) {
          // FIXME: Temporary workaround to avoid marking SPIR_FUNC
          // functions with InternalLinkage, which causes
          // test_subprocess.py::test_assert to fail.
          if (fn.getCallingConv() == CallingConv::SPIR_FUNC)
            continue;
          fn.setLinkage(llvm::GlobalValue::InternalLinkage);
        }
      }
    }
  });
}
