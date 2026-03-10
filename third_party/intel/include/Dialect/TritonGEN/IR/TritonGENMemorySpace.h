#ifndef TRITON_DIALECT_TRITONGENMEMORYSPACE_H
#define TRITON_DIALECT_TRITONGENMEMORYSPACE_H

namespace mlir::triton::TritonGEN {

/// TritonGEN memory space identifiers following SPIRV storage class convention
/// https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/docs/SPIRVRepresentationInLLVM.rst#address-spaces
enum TritonGENMemorySpace {
  kFunction = 0,        // OpenCL workitem address space
  kCrossWorkgroup = 1,  // OpenCL Global memory
  kUniformConstant = 2, // OpenCL Constant memory
  kWorkgroup = 3,       // OpenCL Local memory
  kGeneric = 4          // OpenCL Generic memory
};

} // namespace mlir::triton::TritonGEN

#endif // TRITON_DIALECT_TRITONGENMEMORYSPACE_H
