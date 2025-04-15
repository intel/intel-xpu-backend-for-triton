#include "third_party/intel/include/Target/LLVMIR/PostProcess.h"

#include "llvm/IR/Module.h"

namespace mlir::triton::intel {

void postProcessLLVMIR(llvm::Module &mod) {
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
}

} // namespace mlir::triton::intel
