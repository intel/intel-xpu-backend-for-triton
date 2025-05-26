#include "intel/include/Target/SPIRV/SPIRVTranslation.h"

#include "LLVMSPIRVLib.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"

#if defined(LLVM_SPIRV_BACKEND_TARGET_PRESENT)
namespace llvm {

using namespace llvm;
using namespace SPIRV;

// The LLVM SPIR-V backend exposes an API call that translates LLVM module to
// SPIR-V and writes results into a string as binary SPIR-V output, providing
// diagnostics on fail and means of configuring translation.
extern "C" bool SPIRVTranslate(Module *M, std::string &SpirvObj,
                               std::string &ErrMsg,
                               const std::vector<std::string> &AllowExtNames,
                               llvm::CodeGenOptLevel OLevel,
                               Triple TargetTriple);

static inline Triple::SubArchType
spirvVersionToSubArch(SPIRV::VersionNumber VN) {
  switch (VN) {
  case SPIRV::VersionNumber::SPIRV_1_0:
    return Triple::SPIRVSubArch_v10;
  case VersionNumber::SPIRV_1_1:
    return Triple::SPIRVSubArch_v11;
  case VersionNumber::SPIRV_1_2:
    return Triple::SPIRVSubArch_v12;
  case VersionNumber::SPIRV_1_3:
    return Triple::SPIRVSubArch_v13;
  case VersionNumber::SPIRV_1_4:
    return Triple::SPIRVSubArch_v14;
  case VersionNumber::SPIRV_1_5:
    return Triple::SPIRVSubArch_v15;
  case VersionNumber::SPIRV_1_6:
    return Triple::SPIRVSubArch_v16;
  }
  return Triple::NoSubArch;
}

bool runSpirvBackend(Module *M, std::string &Result, std::string &ErrMsg,
                     const SPIRV::TranslatorOpts &TranslatorOpts) {
  static const std::string DefaultTriple = "spirv64v1.6-unknown-unknown";
  static const std::vector<std::string> AllowExtNames{"all"};

  // Correct the Triple value if needed
  Triple TargetTriple(M->getTargetTriple());
  if (TargetTriple.isSPIR()) {
    TargetTriple.setArch(TargetTriple.getArch() == Triple::spir64
                             ? Triple::spirv64
                             : Triple::spirv32,
                         TargetTriple.getSubArch());
    M->setTargetTriple(TargetTriple);
    // We need to reset Data Layout to conform with the TargetMachine
    M->setDataLayout("");
  }
  if (TargetTriple.getTriple().empty())
    TargetTriple.setTriple(DefaultTriple);
  if (TranslatorOpts.getMaxVersion() != VersionNumber::MaximumVersion) {
    TargetTriple.setArch(TargetTriple.getArch(),
                         spirvVersionToSubArch(TranslatorOpts.getMaxVersion()));
    M->setTargetTriple(TargetTriple);
  }

  // Translate the Module into SPIR-V
  return SPIRVTranslate(M, Result, ErrMsg, AllowExtNames,
                        CodeGenOptLevel::Aggressive, TargetTriple);
}

bool runSpirvBackend(Module *M, std::ostream &OS, std::string &ErrMsg,
                     const SPIRV::TranslatorOpts &TranslatorOpts) {
  std::string Result;
  bool Status = runSpirvBackend(M, Result, ErrMsg, TranslatorOpts);
  if (Status)
    OS << Result;
  return Status;
}

} // namespace llvm

#endif // LLVM_SPIRV_BACKEND_TARGET_PRESENT

namespace triton {

class SmallVectorBuffer : public std::streambuf {
  // All memory management is delegated to llvm::SmallVectorImpl
  llvm::SmallVectorImpl<char> &OS;

  // Since we don't touch any pointer in streambuf(pbase, pptr, epptr) this is
  // the only method we need to override.
  virtual std::streamsize xsputn(const char *s, std::streamsize n) override {
    OS.append(s, s + n);
    return n;
  }

public:
  SmallVectorBuffer() = delete;
  SmallVectorBuffer(const SmallVectorBuffer &) = delete;
  SmallVectorBuffer &operator=(const SmallVectorBuffer &) = delete;
  SmallVectorBuffer(llvm::SmallVectorImpl<char> &O) : OS(O) {}
};

static SPIRV::TranslatorOpts getSPIRVOpts() {
  SPIRV::TranslatorOpts SPIRVOpts{SPIRV::VersionNumber::SPIRV_1_4};
  static constexpr std::array<SPIRV::ExtensionID, 17> AllowedExtensions{
      SPIRV::ExtensionID::SPV_EXT_shader_atomic_float_add,
      SPIRV::ExtensionID::SPV_INTEL_arbitrary_precision_integers,
      SPIRV::ExtensionID::SPV_INTEL_arithmetic_fence,
      SPIRV::ExtensionID::SPV_INTEL_bfloat16_conversion,
      SPIRV::ExtensionID::SPV_INTEL_cache_controls,
      SPIRV::ExtensionID::SPV_INTEL_fp_fast_math_mode,
      SPIRV::ExtensionID::SPV_INTEL_inline_assembly,
      SPIRV::ExtensionID::SPV_INTEL_kernel_attributes,
      SPIRV::ExtensionID::SPV_INTEL_memory_access_aliasing,
      SPIRV::ExtensionID::SPV_INTEL_split_barrier,
      SPIRV::ExtensionID::SPV_INTEL_subgroup_matrix_multiply_accumulate,
      SPIRV::ExtensionID::SPV_INTEL_subgroups,
      SPIRV::ExtensionID::SPV_INTEL_tensor_float32_conversion,
      SPIRV::ExtensionID::SPV_INTEL_unstructured_loop_controls,
      SPIRV::ExtensionID::SPV_INTEL_vector_compute,
      SPIRV::ExtensionID::SPV_KHR_bit_instructions,
      SPIRV::ExtensionID::SPV_KHR_non_semantic_info};

  SPIRVOpts.setMemToRegEnabled(true);
  SPIRVOpts.setPreserveOCLKernelArgTypeMetadataThroughString(true);
  SPIRVOpts.setPreserveAuxData(false);
  SPIRVOpts.setSPIRVAllowUnknownIntrinsics({"llvm.genx.GenISA."});

  for (auto &Ext : AllowedExtensions)
    SPIRVOpts.setAllowedToUseExtension(Ext, true);
  return SPIRVOpts;
}

std::string translateLLVMIRToSPIRV(llvm::Module &module) {
  llvm::SmallVector<char, 0> buffer;

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);

  if (module.materializeAll()) {
    llvm::errs() << "SPIRVTranslation: failed to read the LLVM module IR!";
    llvm::errs().flush();
    std::string result(buffer.begin(), buffer.end());
    return result;
  }

  // emit
  SmallVectorBuffer StreamBuf(buffer);
  std::ostream OS(&StreamBuf);
  std::string Err;

  SPIRV::TranslatorOpts SPIRVOpts = getSPIRVOpts();

#if defined(LLVM_SPIRV_BACKEND_TARGET_PRESENT)
  int SpvTranslateMode = 0;
  if (const char *EnvIsBackend = std::getenv("TRITON_USE_SPIRV_BACKEND"))
    llvm::StringRef(EnvIsBackend).getAsInteger(10, SpvTranslateMode);
  auto success = SpvTranslateMode
                     ? llvm::runSpirvBackend(&module, OS, Err, SPIRVOpts)
                     : llvm::writeSpirv(&module, SPIRVOpts, OS, Err);
#else
  auto success = llvm::writeSpirv(&module, SPIRVOpts, OS, Err);
#endif // LLVM_SPIRV_BACKEND_TARGET_PRESENT

  if (!success) {
    llvm::errs() << "SPIRVTranslation: SPIRV translation failed with"
                 << Err.c_str();
    llvm::errs().flush();
  }

  std::string result(buffer.begin(), buffer.end());
  return result;
}

} // namespace triton
