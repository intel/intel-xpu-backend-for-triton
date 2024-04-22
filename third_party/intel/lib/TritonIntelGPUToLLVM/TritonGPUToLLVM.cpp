#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "intel/include/GPUToTritonGEN/GPUToTritonGENPass.h"
#include "intel/include/TritonGENToLLVM/TritonGENToLLVMPass.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONINTELGPUTOLLVM
#include "intel/include/TritonIntelGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace mlir {
FailureOr<LLVM::LLVMFuncOp>
convertFuncOpToLLVMFuncOp(FunctionOpInterface funcOp,
                          ConversionPatternRewriter &rewriter,
                          const LLVMTypeConverter &converter);
}

using namespace mlir;
using namespace mlir::triton;

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

/// FuncOp legalization pattern that converts MemRef arguments to pointers to
/// MemRef descriptors (LLVM struct data types) containing all the MemRef type
/// information.

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, int numWarps,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), numWarps(numWarps) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    // 1. Modify the function type to add the new argument.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.push_back(ptrTy);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add the new argument.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    auto amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
    amendedArgAttrs.emplace_back(DictionaryAttr::get(ctx));
    amendedAttrs.push_back(rewriter.getNamedAttr(
        funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(amendedArgAttrs)));
    // 3. Add a new argument to the region
    auto amendedFuncOp = rewriter.create<triton::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    region.addArgument(ptrTy, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Prevent LLVM's inliner to inline this function
    auto amendedFuncOp = funcOp;
    if (!LLVM::isKernel(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp)
      return failure();

    MLIRContext *ctx = funcOp->getContext();
    auto mod = funcOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (LLVM::isKernel(funcOp))
      newFuncOp.setCConv(LLVM::CConv::SPIR_KERNEL);

    auto maxWorkGroupSizeAttr = rewriter.getArrayAttr(
        {rewriter.getStringAttr(
             TritonGEN::TritonGENDialect::getMaxWorkGroupSizeAttrName()),
         rewriter.getStringAttr(std::to_string(threadsPerWarp * numWarps) +
                                ",1,1")});
    auto reqSubGroupSizeAttr = rewriter.getArrayAttr(
        {rewriter.getStringAttr(
             TritonGEN::TritonGENDialect::getReqdSubGroupSizeAttrName()),
         rewriter.getStringAttr(std::to_string(threadsPerWarp))});
    newFuncOp.setPassthroughAttr(
        ArrayAttr::get(ctx, {reqSubGroupSizeAttr, maxWorkGroupSizeAttr}));

    if (!LLVM::isKernel(funcOp)) {
      newFuncOp.setPassthroughAttr(
          ArrayAttr::get(ctx, rewriter.getStringAttr("noinline")));
      rewriter.eraseOp(amendedFuncOp);
    }

    // required by AxisInfoAnalysis
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addIllegalDialect<triton::TritonGEN::TritonGENDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::gpu::intel::TritonIntelGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonIntelGPUToLLVMBase<
          ConvertTritonGPUToLLVM> {
  using ConvertTritonIntelGPUToLLVMBase::ConvertTritonIntelGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, TritonGEN::TritonGENDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonGPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps,
                                         /*benefit=*/1);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    mlir::triton::intel::TargetInfo targetInfo;
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    using namespace mlir::triton::intel;
    populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::intel::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);
    populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      benefit);
    mlir::triton::intel::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo, benefit);
    mlir::triton::intel::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                                      targetInfo, benefit);
    mlir::triton::intel::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                                      benefit);

    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, benefit);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::intel::populateHistogramOpToLLVMPatterns(typeConverter,
                                                           patterns, benefit);
    mlir::triton::intel::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                                      targetInfo, benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::intel::populateMemoryOpToLLVMPattern(typeConverter, patterns,
                                                       benefit);
    mlir::triton::intel::populateControlFlowOpToLLVMPattern(typeConverter,
                                                            patterns, benefit);
    mlir::triton::intel::populateMakeRangeOpToLLVMPattern(typeConverter,
                                                          patterns, benefit);
    mlir::triton::intel::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateTritonGENToLLVMConversionPatterns(typeConverter,
                                                            patterns);
    mlir::triton::populateGPUToTritonGENConversionPatterns(typeConverter,
                                                           patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Fold CTAId when there is only 1 CTA.
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}

} // namespace triton
} // namespace mlir
