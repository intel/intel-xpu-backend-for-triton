#ifndef TRITON_SPIRVTRANSLATION_H
#define TRITON_SPIRVTRANSLATION_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace triton {

// add external dependent libs
void addExternalLibs(mlir::ModuleOp &module,
                     const std::vector<std::string> &names,
                     const std::vector<std::string> &paths);

LogicalResult assembleSPIRV(std::string spirvCode, raw_ostream &output);

LogicalResult disassembleSPIRV(uint32_t *binary_ptr, size_t binary_size,
                               raw_ostream &output);

// Translate TritonGPU dialect to SPIRV, return null if failed.
std::string translateTritonGPUToSPIRVIR(mlir::ModuleOp module,
                                        std::map<std::string, int>);

} // namespace triton
} // namespace mlir

#endif
