//===- GPUToGENPass.cpp - MLIR GPU to GEN lowering passes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to generate GEN IR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "intel/include/GPUToGEN/GPUToGENPass.h"

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
#include "triton/Dialect/GEN/IR/Dialect.h"

#include "GPUOpsLowering.h"
#include "IndexIntrinsicsOpLowering.h"
#include "OpToFuncCallLowering.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTGPUTOGEN
#include "intel/include/GPUToGEN/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

/// Import the GPU Ops to GEN Patterns.
#include "GPUToGEN.cpp.inc"

// A pass that replaces all occurrences of GPU device operations with their
// corresponding GEN equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
struct GPUToGENPass : public triton::impl::ConvertGPUToGENBase<GPUToGENPass> {
  GPUToGENPass() = default;
  GPUToGENPass(unsigned indexBitwidth) {
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
            return GEN::GENMemorySpace::kCrossWorkgroup;
          case mlir::gpu::AddressSpace::Workgroup:
            return GEN::GENMemorySpace::kWorkgroup;
          case mlir::gpu::AddressSpace::Private:
            return GEN::GENMemorySpace::kFunction;
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
    populateGPUToGENConversionPatterns(converter, llvmPatterns);
    LLVMConversionTarget target(getContext());
    configureGPUToGENConversionLegality(target);
    if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::triton::configureGPUToGENConversionLegality(
    ConversionTarget &target) {
  target.addIllegalOp<func::FuncOp>();
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addLegalDialect<triton::GEN::GENDialect>();
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

void mlir::triton::populateGPUToGENConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
  patterns
      .add<GPUIndexIntrinsicOpLowering<mlir::gpu::ThreadIdOp, GEN::ThreadIdXOp,
                                       GEN::ThreadIdYOp, GEN::ThreadIdZOp>,
           GPUIndexIntrinsicOpLowering<mlir::gpu::BlockIdOp, GEN::BlockIdXOp,
                                       GEN::BlockIdYOp, GEN::BlockIdZOp>,
           GPUIndexIntrinsicOpLowering<mlir::gpu::BlockDimOp, GEN::BlockDimXOp,
                                       GEN::BlockDimYOp, GEN::BlockDimZOp>,
           GPUIndexIntrinsicOpLowering<mlir::gpu::GridDimOp, GEN::GridDimXOp,
                                       GEN::GridDimYOp, GEN::GridDimZOp>>(
          converter);
  patterns.add<GPUFuncOpLowering>(
      converter,
      /*allocaAddrSpace=*/GEN::GENMemorySpace::kFunction,
      /*workgroupAddrSpace=*/GEN::GENMemorySpace::kWorkgroup,
      StringAttr::get(&converter.getContext(),
                      GEN::GENDialect::getKernelFuncAttrName()));

  const llvm::StringRef prefix("_Z15__spirv_ocl_");

  populateOpPatterns<math::ExpOp>(converter, patterns, (prefix + "expf").str(),
                                  (prefix + "expd").str());
  populateOpPatterns<math::LogOp>(converter, patterns, (prefix + "logf").str(),
                                  (prefix + "logd").str());
  populateOpPatterns<math::CosOp>(converter, patterns, (prefix + "cosf").str(),
                                  (prefix + "cosd").str());
  populateOpPatterns<math::SinOp>(converter, patterns, (prefix + "sinf").str(),
                                  (prefix + "sind").str());
}

std::unique_ptr<OperationPass<mlir::gpu::GPUModuleOp>>
mlir::triton::createLowerGPUToGENPass(unsigned indexBitwidth) {
  return std::make_unique<GPUToGENPass>(indexBitwidth);
}
