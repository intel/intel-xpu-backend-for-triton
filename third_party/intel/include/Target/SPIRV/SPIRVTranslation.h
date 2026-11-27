#ifndef TRITON_TARGET_SPIRVTRANSLATION_H
#define TRITON_TARGET_SPIRVTRANSLATION_H

#include <map>
#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate TritonGPU IR to SPIRV code. \p isLTS restricts the set of SPIR-V
// extensions the translator is allowed to use. \p probedExtensions gives the
// ground-truth probe result (see `compiler.py`'s
// `has_spirv_capability`/`ocloc_accepts_*`) for extensions not covered by
// \p isLTS alone, keyed by name (e.g. `SPV_EXT_long_vector`); a missing key
// is treated as `false`.
std::string
translateLLVMIRToSPIRV(llvm::Module &module, bool isLTS,
                       const std::map<std::string, bool> &probedExtensions);

} // namespace triton

#endif
