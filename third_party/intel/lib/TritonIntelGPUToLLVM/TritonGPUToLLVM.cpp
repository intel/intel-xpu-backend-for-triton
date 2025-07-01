#include "PipelineManager.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "intel/include/Analysis/AxisInfo.h"
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
  ConvertTritonGPUToLLVM() = default;
  ConvertTritonGPUToLLVM(bool advancedPath, bool oneMatrixPerLoadForBT,
                         bool useTileLoadLinearLayout) {
    this->advancedPath = advancedPath;
    this->oneMatrixPerLoadForBT = oneMatrixPerLoadForBT;
    this->useTileLoadLinearLayout = useTileLoadLinearLayout;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, TritonGEN::TritonGENDialect,
                    spirv::SPIRVDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    bool isAdvancedPathEnabled =
        mlir::triton::tools::getBoolEnv("TRITON_INTEL_ADVANCED_PATH") ||
        advancedPath;
    if (isAdvancedPathEnabled)
      assert(mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                              getSupportSG2DBlockAttrName()) &&
             mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                              getSupportDPASAttrName()) &&
             "Target do not support blocked load/mma");
    mlir::triton::intel::TritonGPUToLLVMPipelineManager pipelineManager(
        mod, context, isAdvancedPathEnabled, oneMatrixPerLoadForBT,
        useTileLoadLinearLayout);
    mlir::LowerToLLVMOptions option(context);
    auto targetInfo = mlir::triton::intel::createTargetInfo(mod);
    TritonIntelGPUToLLVMTypeConverter typeConverter(
        context, option, *targetInfo, isAdvancedPathEnabled);
    TritonLLVMConversionTarget convTarget(*context);
    int numWarps = triton::gpu::lookupNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    if (!pipelineManager.skipSharedMemoryAllocation()) {
      ModuleAllocation allocation(
          mod, ::mlir::triton::intel::allocationAnalysisScratchSizeFn);
      ModuleMembarAnalysis membarPass(&allocation, ::mlir::intel::membarFilter);
      membarPass.run();
    }

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonIntelGPUToLLVMTypeConverter typeConverter(
          context, option, *targetInfo, isAdvancedPathEnabled);
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
      TritonIntelGPUToLLVMTypeConverter typeConverter(
          context, option, *targetInfo, isAdvancedPathEnabled);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    mlir::triton::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    pipelineManager.populateConversionPatterns(
        patterns, axisInfoAnalysis, typeConverter, *targetInfo, benefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

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
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(TritonGEN::TritonGENMemorySpace::kWorkgroup));
  }
};

} // anonymous namespace
