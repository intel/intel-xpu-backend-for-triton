#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"
#include "third_party/intel/include/Target/LLVMIR/DSE.h"
#include "third_party/intel/include/Target/LLVMIR/LICM.h"

namespace mlir::triton::intel {
void postProcessLLVMIR(llvm::Module &mod) {
  LICM(mod);
  DSE(mod);
}
} // namespace mlir::triton::intel