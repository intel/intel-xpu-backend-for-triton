#ifndef TRITON_TARGET_LLVMIR_LICM_H
#define TRITON_TARGET_LLVMIR_LICM_H

namespace llvm {
class Module;
} // namespace llvm

namespace mlir::triton::intel {
void LICM(llvm::Module &module);
} // namespace mlir::triton::intel

#endif // TRITON_TARGET_LLVMIR_LICM_H
