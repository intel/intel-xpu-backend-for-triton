#ifndef TRITON_TARGET_SPIRVTRANSLATION_H
#define TRITON_TARGET_SPIRVTRANSLATION_H

#include <map>
#include <string>

namespace llvm {
class Module;
} // namespace llvm

namespace triton {

// Translate TritonGPU IR to SPIRV code. \p isLTS indicates whether the target
// driver is a LTS driver, which restricts the set of SPIR-V extensions the
// translator is allowed to use. \p probedExtensions carries the ground-truth
// probe result (see `compiler.py`'s `has_spirv_capability`/`ocloc_accepts_*`)
// for each SPIR-V extension whose availability is not implied by \p isLTS
// alone, keyed by extension name (e.g. `SPV_EXT_long_vector`,
// `SPV_INTEL_maximum_registers`); an extension missing from the map is
// treated the same as a `false` entry. The caller is responsible for not
// having emitted IR that depends on an extension this disallows (e.g. a
// `MaximumRegisters` kernel attribute, see `compiler.py`'s `make_llir`).
std::string
translateLLVMIRToSPIRV(llvm::Module &module, bool isLTS,
                       const std::map<std::string, bool> &probedExtensions);

} // namespace triton

#endif
