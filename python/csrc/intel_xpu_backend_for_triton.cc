
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <Python.h>
#include <cctype>
#include <fstream>
#include <optional>
#include <pybind11/buffer_info.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <regex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Target/SPIRV/SPIRVTranslation.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Transforms/Passes.h>

namespace py = pybind11;

void init_triton_translation(py::module &m) {

  using ret = py::return_value_policy;

  py::class_<mlir::MLIRContext>(m, "context", py::module_local())
      .def(py::init<>())
      .def("load_triton", [](mlir::MLIRContext &self) {
        //        self.getOrLoadDialect<mlir::triton::TritonDialect>();
        //        self.getOrLoadDialect<mlir::index::IndexDialect>();
        //        self.getOrLoadDialect<mlir::triton::TritonDialect>();
        //        self.getOrLoadDialect<mlir::gpu::GPUDialect>();
        //        // we load LLVM because the frontend uses LLVM.undef for
        //        // some placeholders
        //        self.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
      });

  // Ops
  py::class_<mlir::OpState>(m, "OpState", py::module_local())
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("__str__",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self->print(os);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::module_local(),
                                            py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("str",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("bytecode",
           [](mlir::ModuleOp &self) -> py::bytearray {
             std::string bytecode;
             llvm::raw_string_ostream os(bytecode);
             if (failed(mlir::writeBytecodeToFile(self, os)))
               throw std::runtime_error("Failed to write module bytecode");
             return py::bytearray(bytecode);
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::triton::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](mlir::ModuleOp &self,
              std::string &funcName) -> mlir::triton::FuncOp {
             return self.lookupSymbol<mlir::triton::FuncOp>(funcName);
           })
      .def("get_single_function",
           [](mlir::ModuleOp &self) -> mlir::triton::FuncOp {
             llvm::SmallVector<mlir::triton::FuncOp> funcs;
             self.walk(
                 [&](mlir::triton::FuncOp func) { funcs.push_back(func); });
             if (funcs.size() != 1)
               throw std::runtime_error("Expected a single function");
             return funcs[0];
           });

  m.def(
      "parse_mlir_module",
      [](const std::string &mlir, mlir::MLIRContext &context) {
        // initialize registry
        // note: we initialize llvm for undef
        mlir::DialectRegistry registry;
        registry.insert<mlir::triton::TritonDialect,
                        mlir::triton::gpu::TritonGPUDialect,
                        mlir::math::MathDialect, mlir::arith::ArithDialect,
                        mlir::index::IndexDialect, mlir::scf::SCFDialect,
                        mlir::cf::ControlFlowDialect>();
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        // parse module
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(mlir, &context);
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");

        return module->clone();
      },
      ret::take_ownership);

  m.def(
      "translate_triton_gpu_to_spirv",
      [](const std::string &ttgir, py::dict computeCapability) {
        mlir::MLIRContext context;

        // initialize registry
        // note: we initialize llvm for undef
        mlir::DialectRegistry registry;
        registry.insert<mlir::triton::TritonDialect,
                        mlir::triton::gpu::TritonGPUDialect,
                        mlir::math::MathDialect, mlir::arith::ArithDialect,
                        mlir::index::IndexDialect, mlir::scf::SCFDialect,
                        mlir::cf::ControlFlowDialect>();
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        auto capabilities =
            computeCapability.cast<std::map<std::string, int>>();

        // parse module
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceString<mlir::ModuleOp>(ttgir, &context);
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");
        auto spirvModule = ::mlir::triton::translateTritonGPUToSPIRVIR(
            *module, std::move(capabilities));
        if (spirvModule.empty())
          throw std::runtime_error(
              "Failed to translate TritonGPU to SPIRV IR.");

        auto shared =
            (*module)->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared");
        return py::make_tuple<py::return_value_policy::take_ownership>(
            spirvModule, shared.getInt());
      },
      ret::take_ownership);

  m.def("add_external_libs",
        [](mlir::ModuleOp &op, const std::vector<std::string> &names,
           const std::vector<std::string> &paths) {
          ::mlir::triton::addExternalLibs(op, names, paths);
        });

  m.def("compile_spirv_to_spvbin",
        [](const std::string &spirvCode, int capability) -> py::object {
          std::string spvbin;
          llvm::raw_string_ostream os(spvbin);

          if (failed(::mlir::triton::assembleSPIRV(spirvCode, os)))
            llvm::report_fatal_error("Failed to assemble SPIRV.");

          py::bytes bytes(spvbin);
          return std::move(bytes);
        });

  py::class_<mlir::PassManager>(m, "pass_manager", py::module_local())
      .def(py::init<mlir::MLIRContext *>())
      .def("enable_debug",
           [](mlir::PassManager &self) {
             if (!::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP"))
               return;
             self.getContext()->disableMultithreading();
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.elideLargeElementsAttrs(16);
             printingFlags.enableDebugInfo();
             auto print_always = [](mlir::Pass *, mlir::Operation *) {
               return true;
             };
             self.enableIRPrinting(
                 /*shouldPrintBeforePass=*/print_always,
                 /*shouldPrintAfterPass=*/print_always,
                 /*printModuleScope=*/true,
                 /*printAfterOnlyOnChange=*/false,
                 /*printAfterOnlyOnFailure*/ true, llvm::dbgs(), printingFlags);
           })
      .def("run",
           [](mlir::PassManager &self, mlir::ModuleOp &mod) {
             // TODO: maybe dump module to file and print error for better
             // diagnostics
             if (mlir::failed(self.run(mod.getOperation())))
               throw std::runtime_error("PassManager::run failed");
           })
      .def(
          "add_sccp_pass",
          [](mlir::PassManager &self) { self.addPass(mlir::createSCCPPass()); })
      .def("add_tritongpu_coalesce_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUCoalescePass());
           })
      .def("add_symbol_dce_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createSymbolDCEPass());
           })
      .def("add_inliner_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createInlinerPass());
           })
      .def("add_canonicalizer_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createCanonicalizerPass());
           })
      .def("add_cse_pass",
           [](mlir::PassManager &self) { self.addPass(mlir::createCSEPass()); })
      .def("add_licm_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createLoopInvariantCodeMotionPass());
           })
      .def("add_triton_combine_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::triton::createCombineOpsPass());
           })
      .def("add_reorder_broadcast_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::triton::createReorderBroadcastPass());
           })
      .def("add_rewrite_tensor_pointer_pass",
           [](mlir::PassManager &self, int computeCapability) {
             self.addPass(mlir::triton::createRewriteTensorPointerPass(
                 computeCapability));
           })
      .def(
          "add_convert_triton_to_tritongpu_pass",
          [](mlir::PassManager &self, int numWarps, int threadsPerWarp) {
            self.addPass(mlir::triton::createConvertTritonToTritonGPUPass(
                numWarps, threadsPerWarp));
          },
          py::arg("numWarps") = 4, py::arg("threadsPerWarp") = 32)
      .def("add_tritongpu_pipeline_pass",
           [](mlir::PassManager &self, int numStages) {
             self.addPass(mlir::createTritonGPUPipelinePass(numStages));
           })
      .def("add_tritongpu_prefetch_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUPrefetchPass());
           })
      .def("add_tritongpu_optimize_dot_operands_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUOptimizeDotOperandsPass());
           })
      .def("add_tritongpu_remove_layout_conversions_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPURemoveLayoutConversionsPass());
           })
      .def("add_tritongpu_reorder_instructions_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUReorderInstructionsPass());
           })
      .def("add_tritongpu_decompose_conversions_pass",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createTritonGPUDecomposeConversionsPass());
           })
      .def("add_scf_to_cfg",
           [](mlir::PassManager &self) {
             self.addPass(mlir::createConvertSCFToCFPass());
           })
      .def("add_tritongpu_rewrite_tensor_pointer_pass",
           [](mlir::PassManager &self, py::dict computeCapability) {
             self.addPass(mlir::createTritonGPURewriteTensorPointerPass(80));
           });
  ;
}

void init_intel_xpu_backend_for_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  init_triton_translation(subm);
}
