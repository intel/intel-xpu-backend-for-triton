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

    /// Get the name of the attribute used to indicate whether the native 16bit
    /// atomic operations are available.
    static constexpr llvm::StringRef getSupport16BitAtomicsAttrName() {
      return "ttig.support_16bit_atomics";
    }

    /// Get the name of the attribute used to indicate whether subgroup 2D block
    /// operations (e.g., 2D block read/write) are available.
    static constexpr llvm::StringRef getSupport2DBlockIOAttrName() {
      return "ttig.support_2d_block_io";
    }

    /// Get the name of the attribute used to indicate whether bfloat16
    /// arithmetic is available.
    static constexpr llvm::StringRef getSupportBFloat16ArithmeticAttrName() {
      return "ttig.support_bfloat16_arithmetic";
    }

    /// Get the name of the attribute used to indicate whether bfloat16
    /// conversion is available.
    static constexpr llvm::StringRef getSupportBFloat16ConversionAttrName() {
      return "ttig.support_bfloat16_conversion";
    }

    /// Get the name of the attribute used to indicate whether predicated io
    /// is available.
    static constexpr llvm::StringRef getSupportPredicatedIOAttrName() {
      return "ttig.support_predicated_io";
    }

    /// Get the name of the attribute used to indicate whether the DPAS
    /// operation is available.
    static constexpr llvm::StringRef getSupportDPASAttrName() {
      return "ttig.support_subgroup_matrix_multiply_accumulate";
    }

    /// Get the name of the attribute used to indicate whether the DPAS
    /// operation supports BF8/HF8.
    static constexpr llvm::StringRef getSupportDPASWithBF8AttrName() {
      return "ttig.support_subgroup_matrix_multiply_accumulate_bf8";
    }

    /// Get the name of the attribute used to indicate whether the Block Scale DPAS
    /// operation is available.
    static constexpr llvm::StringRef getSupportBlockScaleDPASAttrName() {
      return "ttig.support_subgroup_scaled_matrix_multiply_accumulate";
    }

    /// Get the name of the attribute used to indicate whether the F4 conversion
    /// operation is available.
    static constexpr llvm::StringRef getSupportF4ConversionAttrName() {
      return "ttig.support_f4_conversion";
    }

    /// Get the name of the attribute used to indicate whether the
    /// H/BF8 to/from HF conversion instruction is available.
    static constexpr llvm::StringRef getSupportF8ConversionAttrName() {
      return "ttig.support_f8_conversion";
    }

    /// Get the name of the attribute used to convey information required for lowering
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

    /// Get the name of the attribute used to indicate whether prefetching 256
    /// bytes is supported.
    static constexpr llvm::StringRef getSupportPrefetch256BAttrName() {
      return "ttig.support_prefetch_256b";
    }

    /// FIXME: Remove once IGC can split large 2D block loads.
    /// Get the name of the attribute used to indicate that a load operation
    /// should use 'one matrix per load'.
    static constexpr llvm::StringRef getOneMatrixPerLoadAttrName() {
      return "ttig.one_matrix_per_load";
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
