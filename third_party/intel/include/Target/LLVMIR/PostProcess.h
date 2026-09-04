#ifndef TRITON_TARGET_LLVMIR_POSTPROCESS_H
#define TRITON_TARGET_LLVMIR_POSTPROCESS_H

namespace llvm {
class Module;
} // namespace llvm

namespace mlir::triton::intel {
void postProcessLLVMIR(llvm::Module &module);
} // namespace mlir::triton::intel

#endif // TRITON_TARGET_LLVMIR_POSTPROCESS_H
