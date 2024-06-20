#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"
#include "third_party/intel/include/Target/LLVMIR/DSE.h"
#include "third_party/intel/include/Target/LLVMIR/LICM.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir::triton::intel {
void postProcessLLVMIR(llvm::Module &mod) {
  auto print = [](llvm::StringRef title, llvm::Module &mod) {
    if (!tools::getBoolEnv("LLVM_IR_ENABLE_DUMP"))
      return;
    for (auto &f : mod) {
      if (f.isDeclaration() ||
          f.getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
        continue;
      llvm::errs() << title << ":\n" << f << "\n";
      break;
    }
  };

  print("PostProcessing: Before LICM", mod);
  LICM(mod);
  print("PostProcessing: After LICM", mod);
  print("PostProcessing: Before DSE", mod);
  DSE(mod);
  print("PostProcessing: After DSE", mod);
}
} // namespace mlir::triton::intel