#include "intel/include/TritonIntelGPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToGENX/GPUToGENXPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/GENToLLVM/GENToLLVMPass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/GEN/IR/Dialect.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetPlatform.hpp"

#include "PatternTritonGPUOpToLLVM.h"
#include "TypeConverter.h"
#include "Utility.h"
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
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

// pass ws related named attrs.
static void addAttrs(Operation *op, ArrayRef<mlir::NamedAttribute> attrs) {
  for (const NamedAttribute attr : attrs)
    op->setAttr(attr.getName(), attr.getValue());
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
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
    if (!LLVM::utils::isKernel(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        amendedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();
    NamedAttrList attrs;
    auto mod = funcOp->getParentOfType<ModuleOp>();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    if (LLVM::utils::isKernel(funcOp))
      attrs.append(GENX::GENXDialect::getKernelFuncAttrName(),
                   rewriter.getI32IntegerAttr(1));
    attrs.append(GENX::GENXDialect::getMaxWorkGroupSizeAttrName(),
                 rewriter.getI32ArrayAttr({threadsPerWarp * numWarps, 1, 1}));
    attrs.append(GENX::GENXDialect::getReqdSubGroupSizeAttrName(),
                 rewriter.getI32ArrayAttr(threadsPerWarp));
    newFuncOp->setDialectAttrs(attrs);
    if (!LLVM::utils::isKernel(funcOp)) {
      // The noinline attribute will be used by the LLVM codegen to prevent
      // inlining.
      // https://github.com/llvm/llvm-project/blob/main/mlir/lib/Dialect/LLVMIR/IR/LLVMInlining.cpp#L267
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
    addLegalDialect<GENX::GENXDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<triton::GEN::GENDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonGPUToLLVM
    : public triton::impl::ConvertTritonIntelGPUToLLVMBase<
          ConvertTritonGPUToLLVM> {
  using ConvertTritonIntelGPUToLLVMBase::ConvertTritonIntelGPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::nvgpu::NVGPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, GEN::GENDialect, GENX::GENXDialect>();
  }

  ConvertTritonGPUToLLVM(int32_t computeCapability)
      : ConvertTritonIntelGPUToLLVMBase({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonIntelGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Preprocess
    decomposeInsertSliceAsyncOp(mod);

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonIntelGPUToLLVMTypeConverter typeConverter(context, option);
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
    int benefit = 10;
    using namespace mlir::triton::intel;
    populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, computeCapability, benefit);
    populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      benefit);
    populateReduceOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                                   benefit);
    populateScanOpToLLVMPatterns(typeConverter, patterns, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    populateBarrierOpToLLVMPatterns(typeConverter, patterns, benefit);
    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, benefit);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, benefit);
    populateHistogramOpToLLVMPatterns(typeConverter, patterns, benefit);
    populatePrintOpToLLVMPattern(typeConverter, patterns, benefit);
    populateAssertOpToLLVMPattern(typeConverter, patterns, benefit);
    populateMemoryOpToLLVMPattern(typeConverter, patterns, benefit);
    populateControlFlowOpToLLVMPattern(typeConverter, patterns, benefit);
    populateMakeRangeOpToLLVMPattern(typeConverter, patterns, benefit);
    populateSPMDOpToLLVMPattern(typeConverter, patterns, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateGENToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToGENXConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Fold CTAId when there is only 1 CTA.
    if (numCTAs == 1) {
      mod.walk([](triton::nvgpu::ClusterCTAIdOp id) {
        OpBuilder b(id);
        Value zero = LLVM::utils::createConstantI32(id->getLoc(), b, 0);
        id.replaceAllUsesWith(zero);
      });
    }
  }

private:
  // pass ws related named attrs.
  static void addWSNamedAttrs(Operation *op,
                              ArrayRef<mlir::NamedAttribute> attrs) {
    for (const NamedAttribute attr : attrs)
      if (attr.getName() == "async_agent" ||
          attr.getName() == "agent.mutex_role")
        op->setAttr(attr.getName(), attr.getValue());
  }

  void decomposeInsertSliceAsyncOp(ModuleOp mod) const {
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    // TODO(Keren): This is a hacky knob that may cause performance regression
    // when decomposition has been performed. We should remove this knob once we
    // have thorough analysis on async wait. Currently, we decompose
    // `insert_slice_async` into `load` and `insert_slice` without knowing which
    // `async_wait` is responsible for the `insert_slice_async`. To guarantee
    // correctness, we blindly set the `async_wait` to wait for all async ops.
    //
    // There are two options to improve this:
    // 1. We can perform a dataflow analysis to find the `async_wait` that is
    // responsible for the `insert_slice_async` in the backend.
    // 2. We can modify the pipeline to perform the decomposition before the
    // `async_wait` is inserted. However, it is also risky because we don't know
    // the correct vectorized shape yet in the pipeline pass. Making the
    // pipeline pass aware of the vectorization could introduce additional
    // dependencies on the AxisInfoAnalysis and the Coalesce analysis.
    bool decomposed = false;
    // insert_slice_async %src, %dst, %idx, %mask, %other
    // =>
    // %tmp = load %src, %mask, %other
    // %res = insert_slice %tmp into %dst[%idx]
    mod.walk([&](triton::gpu::InsertSliceAsyncOp insertSliceAsyncOp) -> void {
      OpBuilder builder(insertSliceAsyncOp);

      // Get the vectorized load size
      auto src = insertSliceAsyncOp.getSrc();
      auto dst = insertSliceAsyncOp.getDst();
      auto mask = insertSliceAsyncOp.getMask();
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis.getPtrContiguity(src);
      if (mask)
        inVec =
            std::min<unsigned>(axisInfoAnalysis.getMaskAlignment(mask), inVec);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = inVec;
      if (outVec > 1)
        minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              computeCapability)
              .contains(byteWidth)) {
        return;
      }

      // load
      auto tmpTy =
          RankedTensorType::get(srcTy.getShape(), resElemTy, srcBlocked);
      auto loadOp = builder.create<triton::LoadOp>(
          insertSliceAsyncOp.getLoc(), tmpTy, insertSliceAsyncOp.getSrc(),
          insertSliceAsyncOp.getMask(), insertSliceAsyncOp.getOther(),
          // TODO(Chenggang): confirm `boundaryCheck` and `padding`
          /*boundaryCheck=*/nullptr, /*padding=*/nullptr,
          insertSliceAsyncOp.getCache(), insertSliceAsyncOp.getEvict(),
          insertSliceAsyncOp.getIsVolatile());
      addWSNamedAttrs(loadOp, insertSliceAsyncOp->getAttrs());

      // insert_slice
      auto axis = insertSliceAsyncOp.getAxis();
      auto intAttr = [&](int64_t v) { return builder.getI64IntegerAttr(v); };
      auto offsets = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(0));
      auto sizes = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      auto strides = SmallVector<OpFoldResult>(dstTy.getRank(), intAttr(1));
      offsets[axis] = insertSliceAsyncOp.getIndex();
      for (size_t i = 0; i < dstTy.getRank(); i++) {
        if (i != axis)
          sizes[i] = intAttr(dstTy.getShape()[i]);
      }
      auto insertSliceOp = builder.create<tensor::InsertSliceOp>(
          insertSliceAsyncOp.getLoc(), loadOp, insertSliceAsyncOp.getDst(),
          offsets, sizes, strides);
      addWSNamedAttrs(insertSliceOp, insertSliceAsyncOp->getAttrs());

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(computeCapability))
        asyncCommitGroupOp.erase();
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
      if (!triton::gpu::AsyncWaitOp::isSupported(computeCapability)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        auto newWaitOp =
            builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        addWSNamedAttrs(newWaitOp, asyncWaitOp->getAttrs());
        asyncWaitOp.erase();
      }
    });
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass() {
  return std::make_unique<ConvertTritonGPUToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonIntelGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
