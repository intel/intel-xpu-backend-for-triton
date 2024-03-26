//===- GPUToTritonGENPass.cpp - MLIR GPU to TritonGEN lowering passes ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate TritonGEN IR operations for
// higher-level GPU operations.
//
//===----------------------------------------------------------------------===//

#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"

#include "GPUOpsLowering.h"
#include "IndexIntrinsicsOpLowering.h"
#include "OpToFuncCallLowering.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTGPUTOTRITONGEN
#include "intel/include/GPUToTritonGEN/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

/// Import the GPU Ops to TritonGEN Patterns.
#include "GPUToTritonGEN.cpp.inc"

// A pass that replaces all occurrences of GPU device operations with their
// corresponding TritonGEN equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct GPUToTritonGENPass
    : public triton::impl::ConvertGPUToTritonGENBase<GPUToTritonGENPass> {
  GPUToTritonGENPass() = default;
  GPUToTritonGENPass(unsigned indexBitwidth) {
    if (this->indexBitwidth.getNumOccurrences() == 0)
      this->indexBitwidth = indexBitwidth;
  }

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp m = getOperation();
    MLIRContext *ctx = m.getContext();

    // Request C wrapper emission.
    for (auto func : m.getOps<func::FuncOp>()) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(ctx));
    }

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(
        ctx, DataLayout(cast<DataLayoutOpInterface>(m.getOperation())));
    if (indexBitwidth != kDeriveIndexBitwidthFromDataLayout)
      options.overrideIndexBitwidth(indexBitwidth);

    // Apply in-dialect lowering. In-dialect lowering will replace
    // ops which need to be lowered further, which is not supported by a
    // single conversion pass.
    {
      RewritePatternSet patterns(ctx);
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
        return signalPassFailure();
    }

    LLVMTypeConverter converter(ctx, options);

    populateGpuMemorySpaceAttributeConversions(
        converter, [](mlir::gpu::AddressSpace space) -> unsigned {
          switch (space) {
          case mlir::gpu::AddressSpace::Global:
            return TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
          case mlir::gpu::AddressSpace::Workgroup:
            return TritonGEN::TritonGENMemorySpace::kWorkgroup;
          case mlir::gpu::AddressSpace::Private:
            return TritonGEN::TritonGENMemorySpace::kFunction;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

    RewritePatternSet llvmPatterns(ctx);

    mlir::arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
    populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
    populateGPUToTritonGENConversionPatterns(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGPUToTritonGENConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::triton::configureGPUToTritonGENConversionLegality(
    ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<triton::TritonGEN::TritonGENDialect>();
  target.addIllegalDialect<mlir::gpu::GPUDialect>();
  target.addIllegalOp<LLVM::CosOp, LLVM::ExpOp, LLVM::LogOp, LLVM::SinOp>();

  // TODO: Remove once we support replacing non-root ops.
  target.addLegalOp<mlir::gpu::YieldOp, mlir::gpu::GPUModuleOp,
                    mlir::gpu::ModuleEndOp>();
}

template <typename OpTy>
static void populateOpPatterns(LLVMTypeConverter &converter,
                               RewritePatternSet &patterns, StringRef f32Func,
                               StringRef f64Func) {
  patterns.add<ScalarizeVectorOpLowering<OpTy>>(converter);
  patterns.add<OpToFuncCallLowering<OpTy>>(converter, f32Func, f64Func);
}

void mlir::triton::populateGPUToTritonGENConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
  patterns.add<
      GPUIndexIntrinsicOpLowering<mlir::gpu::ThreadIdOp, TritonGEN::ThreadIdXOp,
                                  TritonGEN::ThreadIdYOp,
                                  TritonGEN::ThreadIdZOp>,
      GPUIndexIntrinsicOpLowering<mlir::gpu::BlockIdOp, TritonGEN::BlockIdXOp,
                                  TritonGEN::BlockIdYOp, TritonGEN::BlockIdZOp>,
      GPUIndexIntrinsicOpLowering<mlir::gpu::BlockDimOp, TritonGEN::BlockDimXOp,
                                  TritonGEN::BlockDimYOp,
                                  TritonGEN::BlockDimZOp>,
      GPUIndexIntrinsicOpLowering<mlir::gpu::GridDimOp, TritonGEN::GridDimXOp,
                                  TritonGEN::GridDimYOp, TritonGEN::GridDimZOp>,
      SingleDimLaunchConfigLowering<mlir::gpu::SubgroupIdOp,
                                    TritonGEN::SubgroupIdOp>>(converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      /*allocaAddrSpace=*/TritonGEN::TritonGENMemorySpace::kFunction,
      /*workgroupAddrSpace=*/TritonGEN::TritonGENMemorySpace::kWorkgroup);

  const llvm::StringRef prefix("_Z15__spirv_ocl_");
  populateOpPatterns<math::ExpOp>(converter, patterns, (prefix + "expf").str(),
                                  (prefix + "expd").str());
  populateOpPatterns<math::LogOp>(converter, patterns, (prefix + "logf").str(),
                                  (prefix + "logd").str());
  populateOpPatterns<math::CosOp>(converter, patterns, (prefix + "cosf").str(),
                                  (prefix + "cosd").str());
  populateOpPatterns<math::SinOp>(converter, patterns, (prefix + "sinf").str(),
                                  (prefix + "sind").str());

  populateOpPatterns<math::AbsFOp>(converter, patterns, "__imf_fabsf",
                                   "__imf_fabs");
  populateOpPatterns<math::AtanOp>(converter, patterns, "__imf_atanf",
                                   "__imf_atan");
  populateOpPatterns<math::Atan2Op>(converter, patterns, "__imf_atan2f",
                                    "__imf_atan2");
  populateOpPatterns<math::CbrtOp>(converter, patterns, "__imf_cbrtf",
                                   "__imf_cbrt");
  populateOpPatterns<math::CeilOp>(converter, patterns, "__imf_ceilf",
                                   "__imf_ceil");
  populateOpPatterns<math::ErfOp>(converter, patterns, "__imf_erff",
                                  "__imf_erf");
  populateOpPatterns<math::ExpM1Op>(converter, patterns, "__imf_expm1f",
                                    "__imf_expm1");
  populateOpPatterns<arith::RemFOp>(converter, patterns, "__imf_fmodf",
                                    "__imf_fmod");
  populateOpPatterns<math::Log1pOp>(converter, patterns, "__imf_log1pf",
                                    "__imf_log1p");
  populateOpPatterns<math::Log10Op>(converter, patterns, "__imf_log10f",
                                    "__imf_log10");
  populateOpPatterns<math::PowFOp>(converter, patterns, "__imf_powf",
                                   "__imf_pow");
  populateOpPatterns<math::RsqrtOp>(converter, patterns, "__imf_rsqrtf",
                                    "__imf_rsqrt");
  populateOpPatterns<math::TanhOp>(converter, patterns, "__imf_tanhf",
                                   "__imf_tanh");
  populateOpPatterns<math::TanOp>(converter, patterns, "__imf_tanf",
                                  "__imf_tan");
  populateOpPatterns<math::ErfOp>(converter, patterns, "__imf_erff",
                                  "__imf_erf");
}

std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
mlir::triton::createLowerGPUToTritonGENPass(unsigned indexBitwidth) {
  return std::make_unique<GPUToTritonGENPass>(indexBitwidth);
}
