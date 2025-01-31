#include "intel/include/Target/SPIRV/SPIRVTranslation.h"
#include <optional>

#include "LLVMSPIRVLib.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

using namespace llvm;
using namespace SPIRV;

// TODO: The LLVM SPIR-V backend API has changed in
// https://github.com/llvm/llvm-project/pull/124745 to improve the way SPIR-V
// Backend API works with user facing options and allow for multithreading
// within the host application. This PR in llvm-project breaks existing API
// contract in how options are being interpreted inside the call, and we need to
// update this file accordingly. After this change is visible in the LLVM
// version from cmake/llvm-hash.txt we will need to update the call to
// SPIRVTranslateModule(M, Result, ErrMsg, AllowExtNames, Opts) in a style of
// SPIRVTranslate(M, Result, ErrMsg, {"all"}, CodeGenOptLevel::Aggressive,
// Triple("spirv64v1.6-unknown-unknown")).

// The LLVM SPIR-V backend exposes an API call that translates LLVM module to
// SPIR-V and writes results into a string as binary SPIR-V output, providing
// diagnostics on fail and means of configuring translation
// (https://github.com/llvm/llvm-project/pull/107216).
extern "C" bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &AllowExtNames,
                     const std::vector<std::string> &Opts);

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
  static const std::vector<std::string> Opts{
      "--avoid-spirv-capabilities", "Shader", "--translator-compatibility-mode",
      "--spirv-ext=all", "-spirv-O3"};
  static const std::vector<std::string> AllowExtNames;

  // Correct the Triple value if needed
  Triple TargetTriple(M->getTargetTriple());
  if (TargetTriple.isSPIR()) {
    TargetTriple.setArch(TargetTriple.getArch() == Triple::spir64
                             ? Triple::spirv64
                             : Triple::spirv32,
                         TargetTriple.getSubArch());
    M->setTargetTriple(TargetTriple.str());
    // We need to reset Data Layout to conform with the TargetMachine
    M->setDataLayout("");
  }
  if (TranslatorOpts.getMaxVersion() != VersionNumber::MaximumVersion) {
    if (TargetTriple.getTriple().empty())
      TargetTriple.setTriple(DefaultTriple);
    TargetTriple.setArch(TargetTriple.getArch(),
                         spirvVersionToSubArch(TranslatorOpts.getMaxVersion()));
    M->setTargetTriple(TargetTriple.str());
  }

  // Translate the Module into SPIR-V
  return SPIRVTranslateModule(M, Result, ErrMsg, AllowExtNames, Opts);
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

std::string translateLLVMIRToSPIRV(llvm::Module &module) {
  // initLLVM();

  llvm::SmallVector<char, 0> buffer;

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createVerifierPass());
  pm.run(module);
  // module->print(llvm::outs(), nullptr);

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

  SPIRV::TranslatorOpts SPIRVOpts;
  SPIRVOpts.enableAllExtensions();
  SPIRVOpts.setAllowedToUseExtension(
      SPIRV::ExtensionID::SPV_KHR_untyped_pointers, false);
  SPIRVOpts.setMemToRegEnabled(true);
  SPIRVOpts.setPreserveOCLKernelArgTypeMetadataThroughString(true);
  SPIRVOpts.setPreserveAuxData(false);
  SPIRVOpts.setSPIRVAllowUnknownIntrinsics({"llvm.genx.GenISA."});

  int SpvTranslateMode = 0;
  if (const char *EnvIsBackend = std::getenv("TRITON_USE_SPIRV_BACKEND"))
    llvm::StringRef(EnvIsBackend).getAsInteger(10, SpvTranslateMode);
  auto success = SpvTranslateMode
                     ? llvm::runSpirvBackend(&module, OS, Err, SPIRVOpts)
                     : llvm::writeSpirv(&module, SPIRVOpts, OS, Err);

  if (!success) {
    llvm::errs() << "SPIRVTranslation: SPIRV translation failed with"
                 << Err.c_str();
    llvm::errs().flush();
  }

  std::string result(buffer.begin(), buffer.end());
  return result;
}

} // namespace triton
