#ifndef TRITON_INTEL_GPU_DIALECT
#define TRITON_INTEL_GPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonIntelGPU_Dialect : Dialect {
  let name = "ttig";

  let cppNamespace = "::mlir::triton::gpu::intel";

  let description = [{
    Triton Intel GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
    "mlir::triton::TritonGEN::TritonGENDialect",
  ];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to indicate the minimum subgroup
    /// size supported by the target device.
    static constexpr llvm::StringRef getMinSGSizeAttrName() {
      return "ttig.min_sg_size";
    }

    /// Get the name of the attribute used to indicate whether subgroup 2D block
    /// operations (e.g., 2D block read/write) are available.
    static constexpr llvm::StringRef getSupportSG2DBlockAttrName() {
      return "ttig.support_sg_2d_block";
    }

    /// Get the name of the attribute used to indicate whether the DPAS
    /// instruction is available.
    static constexpr llvm::StringRef getSupportDPASAttrName() {
      return "ttig.support_dpas";
    }

    /// Get the name of the attribute used to indicate whether the BF16 conversion
    /// instruction is available.
    static constexpr llvm::StringRef getSupportBF16ConversionAttrName() {
      return "ttig.support_bf16_conversion";
    }

    /// Get the name of the attribute used to convay information required for lowering
    /// memory operations (e.g. load, prefetches) to 2D block HW instructions.
    static constexpr llvm::StringRef getBlockIOAttrName() {
      return "ttig.block_io";
    }

    /// Get the name of the attribute used to specify the target architecture. This
    /// attribute matches architecture in a target triple used for the resulting LLVM
    /// IR module.
    static constexpr llvm::StringRef getTargetArchAttrName() {
      return "ttig.target_arch";
    }

    /// Get the name of the attribute used to indicate whether the native 16bit
    /// atomic operations are available.
    static constexpr llvm::StringRef getSupport16BitAtomicsAttrName() {
      return "ttig.support_16bit_atomics";
    }

    /// FIXME: Remove once IGC can split large 2D block loads.
    /// Get the name of the attribute used to indicate that a region (e.g. a loop)
    /// contains chained dot operations.
    static constexpr llvm::StringRef getContainsChainedDotAttrName() {
      return "ttig.contains_chained_dot";
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
