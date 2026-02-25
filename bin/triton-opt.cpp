#include "./RegisterTritonDialects.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/FileSystem.h"

int main(int argc, char **argv) {
  if (std::string filename =
      mlir::triton::tools::getStrEnv("TRITON_PASS_PLUGIN_PATH");
      !filename.empty()) {
    auto executablePath = llvm::sys::fs::getMainExecutable(argv[0], (void*)&main);
    std::string fakeTriton =
      llvm::sys::path::parent_path(executablePath).str() + "/faketriton/libtriton.so";
    std::string error;
    llvm::sys::DynamicLibrary::getPermanentLibrary(fakeTriton.c_str(), &error);
  }

  mlir::DialectRegistry registry;
  registerTritonDialects(registry);
  registry.insert<mlir::spirv::SPIRVDialect>();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Triton (GPU) optimizer driver\n", registry));
}
