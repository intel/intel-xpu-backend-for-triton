#ifndef TRITON_TARGET_LLVMIR_CODESINK_H
#define TRITON_TARGET_LLVMIR_CODESINK_H

namespace llvm {
class Module;
} // namespace llvm

namespace mlir::triton::intel {
void CodeSink(llvm::Module &module, bool trace);
} // namespace mlir::triton::intel

#endif // TRITON_TARGET_LLVMIR_CODESINK_H
