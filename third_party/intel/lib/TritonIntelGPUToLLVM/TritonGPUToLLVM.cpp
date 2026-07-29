#include "PipelineManager.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"
#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"

#include "intel/include/Analysis/Allocation.h"
#include "intel/include/Analysis/Membar.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_CONVERTTRITONINTELGPUTOLLVM
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;

namespace {

/// Returns true if the module allocates shared memory with a partitioned
/// layout, i.e. one base pointer per physical shared memory partition.
bool hasPartitionedSharedMemory(ModuleOp mod) {
  auto isPartitioned = [](Type ty) {
    auto memDescTy = dyn_cast<triton::gpu::MemDescType>(ty);
    return memDescTy && isa<triton::gpu::PartitionedSharedEncodingAttr>(
                            memDescTy.getEncoding());
  };
  WalkResult res = mod.walk([&](Operation *op) {
    if (llvm::any_of(op->getResultTypes(), isPartitioned) ||
        llvm::any_of(op->getOperandTypes(), isPartitioned))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return res.wasInterrupted();
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<triton::TritonGEN::TritonGENDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::gpu::intel::TritonIntelGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addDynamicallyLegalOp<LLVM::CallOp>([](LLVM::CallOp op) {
      return op.getCConv() == triton::gpu::intel::getRequiredCConv(op);
    });
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::gpu::intel::impl::ConvertTritonIntelGPUToLLVMBase<
          ConvertTritonGPUToLLVM> {
  using ConvertTritonIntelGPUToLLVMBase::ConvertTritonIntelGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, TritonGEN::TritonGENDialect,
                    spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::triton::intel::TritonGPUToLLVMPipelineManager pipelineManager(
        mod, context);
    mlir::LowerToLLVMOptions option(context);
    auto targetInfo = mlir::triton::intel::createTargetInfo(mod);
    TritonIntelGPUToLLVMTypeConverter typeConverter(context, option,
                                                    *targetInfo);
    TritonLLVMConversionTarget convTarget(*context);
    int numWarps = triton::gpu::lookupNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(
        mod, ::mlir::triton::intel::allocationAnalysisScratchSizeFn);
    ModuleMembarAnalysis membarPass(&allocation, ::mlir::intel::membarFilter);
    membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonIntelGPUToLLVMTypeConverter typeConverter(context, option,
                                                      *targetInfo);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      pipelineManager.populateFunctionConversionPatterns(
          funcPatterns, typeConverter, numWarps, *targetInfo);

      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    // Convert call and ret ops
    {
      mlir::LowerToLLVMOptions option(context);
      TritonIntelGPUToLLVMTypeConverter typeConverter(context, option,
                                                      *targetInfo);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    mlir::triton::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    mlir::triton::intel::ModuleStrideAnalysis strideAnalysis(mod,
                                                             axisInfoAnalysis);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    pipelineManager.populateConversionPatterns(patterns, axisInfoAnalysis,
                                               strideAnalysis, typeConverter,
                                               *targetInfo, benefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    fixUpLoopAnnotation(mod);

    mod.walk([&](LLVM::LLVMFuncOp funcOp) {
      for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
        funcOp.removeArgAttr(i, "tt.divisibility");
        funcOp.removeArgAttr(i, "tt.constancy");
        funcOp.removeArgAttr(i, "tt.contiguity");
      }
    });
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));

    // Shared memory is allocated statically: `global_smem` is an internal
    // array sized after `ttg.shared` (a compile time constant computed by the
    // `intel-allocate-shared-memory` pass), which the Level Zero runtime
    // allocates directly from the module. No kernel argument is needed.
    //
    // Fall back to a dynamic allocation (size 0, external linkage, base passed
    // in as a kernel argument by `tritonintelgpu-rewrite-stack-ptr`) when:
    //  - requested explicitly via `dynamic-shared-memory`,
    //  - `ttg.shared` is missing, so the size is unknown,
    //  - the module uses partitioned shared memory: its per-partition bases
    //    would become constants, which LLVM folds into an `extractelement` from
    //    a constant vector of pointers - not expressible in SPIR-V without the
    //    SPV_INTEL_masked_gather_scatter extension (rejected by the driver).
    auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
    bool useDynamic =
        dynamicSharedMemory || !sharedAttr || hasPartitionedSharedMemory(mod);
    int64_t sharedMemSize = useDynamic ? 0 : sharedAttr.getInt();

    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, sharedMemSize);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false,
        useDynamic ? LLVM::Linkage::External : LLVM::Linkage::Internal,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(TritonGEN::TritonGENMemorySpace::kWorkgroup));
  }
};

} // anonymous namespace
