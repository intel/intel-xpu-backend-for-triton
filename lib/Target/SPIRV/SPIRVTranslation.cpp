#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonGPUToSPIRV/TritonGPUToSPIRVPass.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "SPIRV-Tools/tools/io.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/linker.hpp"
#include "spirv-tools/optimizer.hpp"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <dlfcn.h>
#include <filesystem>

namespace mlir {
namespace triton {

static spv_target_env defaultSPIRVTargetEnv = SPV_ENV_OPENCL_2_2;

LogicalResult assembleSPIRV(std::string spirvCode, raw_ostream &output) {
  auto DisMessagePrinter = [](spv_message_level_t Level, const char *source,
                              const spv_position_t &position,
                              const char *message) -> void {
    llvm::errs() << " spirv assemble error: " << message << "\n";
  };
  spvtools::SpirvTools SpvTool(defaultSPIRVTargetEnv);
  SpvTool.SetMessageConsumer(DisMessagePrinter);

  std::vector<uint32_t> binary;
  if (!SpvTool.Assemble(spirvCode, &binary, SPV_TEXT_TO_BINARY_OPTION_NONE)) {
    return failure("SPIRV: Failed to assemble the code");
  }
  std::stringstream is;
  is.rdbuf()->pubsetbuf(reinterpret_cast<char *>(&binary[0]),
                        binary.size() * sizeof(uint32_t));
  output << is.str();
  return mlir::success();
}

LogicalResult disassembleSPIRV(uint32_t *binary_ptr, size_t binary_size,
                               raw_ostream &output) {
  auto DisMessagePrinter = [](spv_message_level_t Level, const char *source,
                              const spv_position_t &position,
                              const char *message) -> void {
    llvm::errs() << " spirv disassemble error: " << message << "\n";
  };
  spvtools::SpirvTools SpvTool(defaultSPIRVTargetEnv);
  SpvTool.SetMessageConsumer(DisMessagePrinter);

  std::string spriv_code;
  if (!SpvTool.Disassemble(binary_ptr, binary_size, &spriv_code)) {
    return failure("SPIRV: Failed to generate textual assembly");
  }
  output << spriv_code;
  return mlir::success();
}

static LogicalResult
getInterfaceVariables(spirv::FuncOp funcOp,
                      SmallVectorImpl<Attribute> &interfaceVars) {
  auto module = funcOp->getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return failure();
  }
  SetVector<Operation *> interfaceVarSet;

  // TODO: This should in reality traverse the entry function
  // call graph and collect all the interfaces. For now, just traverse the
  // instructions in this function.
  funcOp.walk([&](spirv::AddressOfOp addressOfOp) {
    auto var =
        module.lookupSymbol<spirv::GlobalVariableOp>(addressOfOp.getVariable());
    // TODO: Per SPIR-V spec: "Before version 1.4, the interface’s
    // storage classes are limited to the Input and Output storage classes.
    // Starting with version 1.4, the interface’s storage classes are all
    // storage classes used in declaring all global variables referenced by the
    // entry point’s call tree." We should consider the target environment here.
    switch (var.getType().cast<spirv::PointerType>().getStorageClass()) {
    case spirv::StorageClass::Input:
    case spirv::StorageClass::Output:
      interfaceVarSet.insert(var.getOperation());
      break;
    default:
      break;
    }
  });
  for (auto &var : interfaceVarSet) {
    interfaceVars.push_back(SymbolRefAttr::get(
        funcOp.getContext(), cast<spirv::GlobalVariableOp>(var).getSymName()));
  }
  return success();
}

static bool linkExternLib(std::vector<uint32_t> &binary,
                          std::map<std::string, std::string> &externLibPaths) {
  if (externLibPaths.empty())
    return true;

  spvtools::Context ctx(defaultSPIRVTargetEnv);
  auto print_msg_to_stderr = [](spv_message_level_t, const char *,
                                const spv_position_t &, const char *m) {
    llvm::errs() << " spirv link error: " << m << "\n";
  };
  ctx.SetMessageConsumer(print_msg_to_stderr);

  spvtools::LinkerOptions link_option;
  link_option.SetAllowPartialLinkage(false);
  link_option.SetCreateLibrary(false);

  std::vector<std::vector<uint32_t>> libs{binary};
  for (auto &path : externLibPaths) {
    // Read the library binary.
    std::vector<uint32_t> lib;
    if (!ReadBinaryFile<uint32_t>(path.second.c_str(), &lib)) {
      llvm::errs() << "Failed to load library: " << path.second;
      return false;
    }
    libs.push_back(lib);
  }

  std::vector<uint32_t> linked_binary;
  if (SPV_SUCCESS != spvtools::Link(ctx, libs, &linked_binary, link_option)) {
    llvm::errs() << "Failed to link libs:";
    for (auto &path : externLibPaths) {
      llvm::errs() << " " << path.first;
    }
    return false;
  }

  binary.swap(linked_binary);

  return true;
}

static bool optimizeSPIRVModule(std::vector<uint32_t> &binary) {
  spvtools::Optimizer optimizer(defaultSPIRVTargetEnv);
  auto print_msg_to_stderr = [](spv_message_level_t, const char *,
                                const spv_position_t &, const char *m) {
    llvm::errs() << " spirv opt error: " << m << "\n";
  };
  optimizer.SetMessageConsumer(print_msg_to_stderr);
  //  optimizer.SetTimeReport(&std::cout);

  spvtools::ValidatorOptions validator_options;
  spvtools::OptimizerOptions optimizer_options;
  optimizer_options.set_validator_options(validator_options);
  optimizer_options.set_run_validator(false);

  auto runOptimizer = [&](const std::vector<std::string> &flags,
                          std::vector<uint32_t> &binary) {
    if (!optimizer.RegisterPassesFromFlags(flags)) {
      llvm::errs() << " spirv opt error: pass register failed"
                   << "\n";
      return false;
    }

    std::vector<uint32_t> optimized;
    bool ok = optimizer.Run(binary.data(), binary.size(), &optimized,
                            optimizer_options);

    binary.swap(optimized);
    return ok;
  };

  // There is no recursive aggresive opt in SPIRV optimizer. We run the opt in
  // two stage.
  if (runOptimizer({"--eliminate-dead-functions", "--eliminate-dead-inserts",
                    "--eliminate-dead-variables", "--eliminate-dead-members",
                    "--eliminate-dead-code-aggressive",
                    "--eliminate-dead-input-components"},
                   binary) &&
      runOptimizer({"--eliminate-dead-const"}, binary)) {
    return true;
  }
  return false;
}

static std::map<std::string, std::string>
getExternLibs(spirv::ModuleOp module) {
  std::map<std::string, std::string> externLibs;
  SmallVector<spirv::FuncOp> funcs;
  module.walk([&](spirv::FuncOp func) {
    if (func.isExternal())
      funcs.push_back(func);
  });

  for (auto &func : funcs) {
    if (func.getOperation()->hasAttr("libname")) {
      auto name =
          func.getOperation()->getAttr("libname").dyn_cast<StringAttr>();
      auto path =
          func.getOperation()->getAttr("libpath").dyn_cast<StringAttr>();
      if (name) {
        std::string libName = name.str();
        // Note: skip the libdevice path. Use the Intel IMF lib.
        if (libName.compare("libdevice") != 0)
          externLibs[libName] = path.str();
      }
    }
  }

  if (module.getOperation()->hasAttr("triton_gpu.externs")) {
    auto dict = module.getOperation()
                    ->getAttr("triton_gpu.externs")
                    .dyn_cast<DictionaryAttr>();
    for (auto &attr : dict) {
      auto libName = attr.getName().strref().trim().str();
      // Note: skip the libdevice path. Use the Intel IMF lib.
      if (libName.compare("libdevice") != 0) {
        externLibs[libName] =
            attr.getValue().dyn_cast<StringAttr>().strref().trim().str();
      }
    }
  }

  if (!funcs.empty()) {
    std::vector<std::string> lib_names = {
        "libsycl-fallback-imf.spv", "libsycl-fallback-imf-fp64.spv",
        "libsycl-fallback-imf-bf16.spv", "libsycl-imf-dl.spv",
        "libsycl-imf-dl-fp64.spv"};
    // first search for environmental path
    std::string env_path = ::triton::tools::getenv("TRITON_LIBDEVICE_PATH");
    if (!env_path.empty()) {
      for (auto &lib_name : lib_names) {
        externLibs.try_emplace(lib_name, env_path + "/" + lib_name);
      }
      return externLibs;
    }
    namespace fs = std::filesystem;
    // Search for math lib relative to its library path if used from Python
    // Then native code is in `triton/_C/libtriton.so` and libdevice in
    // `triton/third_party/sycl/lib/libsycl-fallback-imf.spv`
    static const auto this_library_path = [] {
      Dl_info fileinfo;
      if (dladdr(reinterpret_cast<void *>(&getExternLibs), &fileinfo) == 0) {
        return std::filesystem::path();
      }
      return std::filesystem::path(fileinfo.dli_fname);
    }();
    static const auto runtime_path =
        this_library_path.parent_path().parent_path() / "third_party" / "xpu" /
        "lib";
    if (fs::exists(runtime_path)) {
      for (auto &lib_name : lib_names) {
        externLibs.try_emplace(lib_name, (runtime_path / lib_name).string());
      }
    } else {
      static const auto this_file_path = std::filesystem::path(__FILE__);
      static const auto compiletime_path = this_file_path.parent_path()
                                               .parent_path()
                                               .parent_path()
                                               .parent_path() /
                                           "python" / "triton" / "third_party" /
                                           "xpu" / "lib";
      if (!fs::exists(compiletime_path)) {
        std::string error_msg = "Can't find libdevice at neither " +
                                runtime_path.string() + " nor " +
                                compiletime_path.string();
        llvm::report_fatal_error(error_msg.c_str());
      }
      for (auto &lib_name : lib_names) {
        externLibs.try_emplace(lib_name,
                               (compiletime_path / lib_name).string());
      }
    }
  }

  return externLibs;
}

static LogicalResult translateTritonSPIRVToSPIRVIR(ModuleOp module,
                                                   raw_ostream &output) {
  if (!module)
    return failure();

  SmallVector<uint32_t, 0> binary;

  SmallVector<spirv::ModuleOp, 1> spirvModules;
  OpBuilder builder(module->getContext());

  module.walk([&](ModuleOp op) {
    auto newModuleOp =
        builder.create<spirv::ModuleOp>(op.getLoc(), op.getName());

    unsigned threadsPerWarp =
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(op);

    auto &region = op.getRegion();
    auto &parent = *newModuleOp.getBody()->getParent();
    auto iter = newModuleOp.getBody()->getIterator();

    parent.getBlocks().splice(iter, region.getBlocks());

    // Remove the terminator block that was automatically added by builder
    auto &last_block = newModuleOp.getBodyRegion().back();
    last_block.getParent()->getBlocks().remove(last_block);

    // copy the attributes
    newModuleOp->setAttrs(op->getAttrDictionary());

    // Set the spirv module attributes
    newModuleOp->setAttr(
        triton::gpu::TritonGPUDialect::getThreadsPerWarpAttrName(),
        IntegerAttr::get(mlir::IntegerType::get(builder.getContext(), 32),
                         llvm::APInt(32, threadsPerWarp)));

    newModuleOp->setAttr("addressing_model",
                         builder.getAttr<spirv::AddressingModelAttr>(
                             spirv::AddressingModel::Physical64));
    newModuleOp->setAttr(
        "memory_model",
        builder.getAttr<spirv::MemoryModelAttr>(spirv::MemoryModel::OpenCL));
    spirv::Capability caps_opencl[] = {
        // clang-format off
            spirv::Capability::Addresses,
            spirv::Capability::Float16Buffer,
            spirv::Capability::Int64,
            spirv::Capability::Int16,
            spirv::Capability::Int8,
            spirv::Capability::Kernel,
            spirv::Capability::Linkage,
            spirv::Capability::Vector16,
            spirv::Capability::GenericPointer,
            spirv::Capability::Groups,
            spirv::Capability::Float16,
            spirv::Capability::Float64,
            spirv::Capability::AtomicFloat32AddEXT,
            spirv::Capability::ExpectAssumeKHR,
            spirv::Capability::SubgroupDispatch,
            spirv::Capability::VectorComputeINTEL,
            spirv::Capability::VectorAnyINTEL,
        // clang-format on
    };
    spirv::Extension exts_opencl[] = {
        spirv::Extension::SPV_EXT_shader_atomic_float_add,
        spirv::Extension::SPV_KHR_expect_assume,
        spirv::Extension::SPV_INTEL_vector_compute};
    newModuleOp->setAttr("vce_triple", spirv::VerCapExtAttr::get(
                                           spirv::Version::V_1_4, caps_opencl,
                                           exts_opencl, builder.getContext()));

    spirvModules.push_back(newModuleOp);
  });

  if (spirvModules.empty())
    return module.emitError("found no 'spv.module' op");

  if (spirvModules.size() != 1)
    return module.emitError("found more than one 'spv.module' op");

  for (auto &sprivModule : spirvModules) {
    int threadsPerWarp = sprivModule->getAttr("triton_gpu.threads-per-warp")
                             .cast<IntegerAttr>()
                             .getInt();
    sprivModule.walk([&](spirv::FuncOp op) {
      auto entryPointAttrName = spirv::getEntryPointABIAttrName();
      auto entryPointAttr =
          op->getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName);
      if (!entryPointAttr) {
        return;
      }

      OpBuilder::InsertionGuard moduleInsertionGuard(builder);
      auto spirvModule = op->getParentOfType<spirv::ModuleOp>();
      builder.setInsertionPointToEnd(spirvModule.getBody());

      // Adds the spv.EntryPointOp after collecting all the interface variables
      // needed.
      SmallVector<Attribute, 1> interfaceVars;
      if (failed(getInterfaceVariables(op, interfaceVars))) {
        return;
      }

      builder.create<spirv::EntryPointOp>(
          op.getLoc(), spirv::ExecutionModel::Kernel, op, interfaceVars);

      builder.create<spirv::ExecutionModeOp>(
          op.getLoc(), op, spirv::ExecutionMode::SubgroupSize, threadsPerWarp);

      op->removeAttr(entryPointAttrName);
      op->removeAttr("sym_visibility");
    });
    // Clean up the Triton attribute.
    sprivModule.walk([&](mlir::Operation *op) {
      op->removeAttr("tt.contiguity");
      op->removeAttr("tt.divisibility");
    });
  }

  if (failed(spirv::serialize(spirvModules[0], binary)))
    return failure();

  // Link external libraries before perform optimizations.
  // This allows the optimizers to inline and perform
  // analyses on the used library functions, and eliminate any unused functions
  // as dead code.
  auto externLibs = getExternLibs(spirvModules[0]);
  std::vector<uint32_t> linked_binary(binary.data(),
                                      binary.data() + binary.size());
  if (!linkExternLib(linked_binary, externLibs))
    return failure();

  if (!optimizeSPIRVModule(linked_binary))
    return failure();

  if (failed(
          disassembleSPIRV(linked_binary.data(), linked_binary.size(), output)))
    return failure();

  return mlir::success();
}

std::string
translateTritonGPUToSPIRVIR(mlir::ModuleOp module,
                            std::map<std::string, int> computeCapability) {
  mlir::PassManager pm(module->getContext());
  mlir::registerPassManagerCLOptions();
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "failed to apply pass manager CL options\n";
    return nullptr;
  }
  auto printingFlags = mlir::OpPrintingFlags();
  printingFlags.elideLargeElementsAttrs(16);
  pm.enableIRPrinting(
      /*shouldPrintBeforePass=*/nullptr,
      /*shouldPrintAfterPass=*/
      [](mlir::Pass *pass, mlir::Operation *) {
        return ::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP");
      },
      /*printModuleScope=*/false,
      /*printAfterOnlyOnChange=*/true,
      /*printAfterOnlyOnFailure*/ false, llvm::dbgs(), printingFlags);

  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(createConvertTritonGPUToSPIRVPass(computeCapability));
  //  pm.addPass(mlir::arith::createConvertArithToSPIRVPass());
  // Canonicalize to eliminate the remaining UnrealizedConversionCastOp
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  // pm.addPass(mlir::createCanonicalizerPass());
  // Simplify the IR
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  // pm.addPass(mlir::createCanonicalizerPass());

  std::string spirvModule;
  if (failed(pm.run(module))) {
    llvm::errs() << "Pass execution failed";
    return spirvModule;
  }

  llvm::raw_string_ostream os(spirvModule);
  if (failed(translateTritonSPIRVToSPIRVIR(module, os))) {
    llvm::errs() << "Translate to SPIRV IR failed";
    return spirvModule;
  }

  return spirvModule;
}

} // namespace triton
} // namespace mlir
