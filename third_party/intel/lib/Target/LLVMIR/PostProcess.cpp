#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"
#include "third_party/intel/include/Target/LLVMIR/SLPVectorizer.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir::triton::intel {

void postProcessLLVMIR(llvm::Module &mod) {
  bool trace = tools::getBoolEnv("LLVM_IR_ENABLE_DUMP");

  auto print = [&](llvm::StringRef title, llvm::Module &mod) {
    if (!trace)
      return;
    for (auto &f : mod) {
      if (f.isDeclaration() ||
          f.getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
        continue;
      llvm::errs() << title << ":\n" << f << "\n";
      break;
    }
  };

  // __devicelib_assert_fail must be a declaration so that
  // IGC can replace it with a runtime assert function.
  // If a 'fallback' implementation is defined in SYCL libarary, the
  // assertion does not work correctly.
  for (auto &f : mod) {
    if (f.getName().str() == "__devicelib_assert_fail") {
      assert(f.isDeclaration() &&
             "__devicelib_assert_fail must be a declaration!");
    }
  }
  print("PostProcessing: Before SLPVectorizer", mod);
  SLPVectorizer(mod, trace);
  print("PostProcessing: After SLPVectorizer", mod);
}

} // namespace mlir::triton::intel
