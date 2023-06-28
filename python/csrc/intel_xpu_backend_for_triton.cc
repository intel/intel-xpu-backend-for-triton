
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"

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


#include "triton/Target/SPIRV/SPIRVTranslation.h"

namespace py = pybind11;

void init_triton_translation(py::module &m) {

  using ret = py::return_value_policy;

  m.def(
          "translate_triton_gpu_to_spirv",
          [](const std::string &ttgir, py::dict computeCapability) {
            mlir::MLIRContext context;

            // initialize registry
            // note: we initialize llvm for undef
            mlir::DialectRegistry registry;
            registry.insert<
                    mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
                    mlir::math::MathDialect, mlir::arith::ArithDialect,
                    mlir::index::IndexDialect, mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
            context.appendDialectRegistry(registry);
            context.loadAllAvailableDialects();

            auto capabilities = computeCapability.cast<std::map<std::string, int>>();

            // parse module
            mlir::OwningOpRef<mlir::ModuleOp> module =
                    mlir::parseSourceString<mlir::ModuleOp>(ttgir, &context);
            if (!module)
              throw std::runtime_error("Parse MLIR file failed.");
            auto spirvModule = ::mlir::triton::translateTritonGPUToSPIRVIR(*module, capabilities);
            if (spirvModule.empty())
              throw std::runtime_error("Failed to translate TritonGPU to SPIRV IR.");

            auto shared = (*module)->getAttrOfType<mlir::IntegerAttr>("triton_gpu.shared");
            return py::make_tuple<py::return_value_policy::take_ownership>(spirvModule, shared.getInt());
          },
          ret::take_ownership);

  m.def("compile_spirv_to_spvbin",
        [](const std::string &spirvCode, int capability) -> py::object {
          std::string spvbin;
          llvm::raw_string_ostream os(spvbin);

          if (failed(::mlir::triton::assembleSPIRV(spirvCode, os)))
            llvm::report_fatal_error("Failed to assemble SPIRV.");

          py::bytes bytes(spvbin);
          return std::move(bytes);
        });
}

void init_intel_xpu_backend_for_triton(py::module &m) {
  py::module subm = m.def_submodule("triton");
  init_triton_translation(subm);
}