#ifndef TRITON_TARGET_SPIRVTRANSLATION_H
#define TRITON_TARGET_SPIRVTRANSLATION_H

#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate TritonGPU IR to SPIRV code. \p isLTS indicates whether the target
// driver is a LTS driver, which restricts the set of SPIR-V extensions the
// translator is allowed to use. \p allowLongVector selects
// `SPV_EXT_long_vector` instead of `SPV_INTEL_vector_compute` for vectors
// beyond the core sizes; it must be false when the module is consumed by a
// compiler that predates the extension. \p allowMaximumRegisters must be
// false when the consumer predates `SPV_INTEL_maximum_registers`, in which
// case the caller is responsible for not having set a `MaximumRegisters`
// kernel attribute in the first place (see `compiler.py`'s `make_llir`).
std::string translateLLVMIRToSPIRV(llvm::Module &module, bool isLTS,
                                   bool allowLongVector,
                                   bool allowMaximumRegisters);

} // namespace triton

#endif
