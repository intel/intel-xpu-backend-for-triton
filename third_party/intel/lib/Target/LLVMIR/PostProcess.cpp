#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"
#include "third_party/intel/include/Target/LLVMIR/DSE.h"
#include "third_party/intel/include/Target/LLVMIR/LICM.h"
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

  print("PostProcessing: Before SLPVectorizer", mod);
  SLPVectorizer(mod, trace);
  print("PostProcessing: After SLPVectorizer", mod);

  print("PostProcessing: Before LICM", mod);
  LICM(mod, trace);
  print("PostProcessing: After LICM", mod);

  print("PostProcessing: Before DSE", mod);
  DSE(mod, trace);
  print("PostProcessing: After DSE", mod);
}

} // namespace mlir::triton::intel
