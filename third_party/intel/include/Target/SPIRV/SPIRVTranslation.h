#ifndef TRITON_TARGET_SPIRVTRANSLATION_H
#define TRITON_TARGET_SPIRVTRANSLATION_H

#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate TritonGPU IR to SPIRV code. \p isLTS indicates whether the target
// driver is a LTS driver, which restricts the set of SPIR-V extensions the
// translator is allowed to use.
std::string translateLLVMIRToSPIRV(llvm::Module &module, bool isLTS);

} // namespace triton

#endif
