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
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx, Target target)
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
                   triton::Target target, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), numWarps(numWarps),
        target(target) {}

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

    switch (target) {
    case Target::NVVM:
    case Target::ROCDL:
      if (LLVM::utils::isKernel(funcOp)) {
        // Set an attribute to indicate this function is a kernel entry.
        newFuncOp->setAttr("nvvm.kernel",
                           rewriter.getIntegerAttr(type::u1Ty(ctx), 1));
      }
      // Set an attribute for maxntidx, it could be used in latter LLVM codegen
      // for `nvvm.annotation` metadata.
      newFuncOp->setAttr("nvvm.maxntid",
                         rewriter.getDenseI32ArrayAttr(32 * numWarps));
      break;
    case Target::GENX:
      NamedAttrList attrs;
      auto mod = funcOp->getParentOfType<ModuleOp>();
      int threadsPerWarp =
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
      if (LLVM::utils::isKernel(funcOp))
        attrs.append(GENX::GENXDialect::getKernelFuncAttrName(),
                     rewriter.getI32IntegerAttr(1));
      attrs.append(GENX::GENXDialect::getMaxWorkGroupSizeAttrName(),
                   rewriter.getI32ArrayAttr({threadsPerWarp * numWarps, 1, 1}));
      attrs.append(GENX::GENXDialect::getReqdSubGroupSizeAttrName(),
                   rewriter.getI32ArrayAttr(threadsPerWarp));
      newFuncOp->setDialectAttrs(attrs);
      break;
    }
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
  triton::Target target;
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx, Target target)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    switch (target) {
    case Target::NVVM:
      addLegalDialect<NVVM::NVVMDialect>();
      addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
      break;
    case Target::GENX:
      addLegalDialect<GENX::GENXDialect>();
      break;
    default:
      break;
    }
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

  ConvertTritonGPUToLLVM(int32_t computeCapability, Target target)
      : ConvertTritonIntelGPUToLLVMBase({computeCapability, target}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonIntelGPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context, target);
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
      TritonLLVMFunctionConversionTarget funcTarget(*context, target);
      RewritePatternSet funcPatterns(context);
      funcPatterns.add<FuncOpConversion>(typeConverter, numWarps, target,
                                         /*benefit=*/1);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter, target);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int benefit = 10;
    using namespace mlir::triton::intel;
    populateConvertLayoutOpToLLVMPatterns(typeConverter, patterns, target,
                                          benefit);
    populateDotOpToLLVMPatterns(typeConverter, patterns, target, benefit);
    populateElementwiseOpToLLVMPatterns(typeConverter, patterns,
                                        axisInfoAnalysis, computeCapability,
                                        target, benefit);
    populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                      target, benefit);
    populateReduceOpToLLVMPatterns(typeConverter, patterns, computeCapability,
                                   target, benefit);
    populateScanOpToLLVMPatterns(typeConverter, patterns, target, benefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    populateBarrierOpToLLVMPatterns(typeConverter, patterns, target, benefit);
    populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns, target,
                                       benefit);
    populateClusterOpsToLLVMPatterns(typeConverter, patterns, target, benefit);
    populateHistogramOpToLLVMPatterns(typeConverter, patterns, target, benefit);
    populatePrintOpToLLVMPattern(typeConverter, patterns, target, benefit);
    populateAssertOpToLLVMPattern(typeConverter, patterns, target, benefit);
    populateMemoryOpToLLVMPattern(typeConverter, patterns, target, benefit);
    populateControlFlowOpToLLVMPattern(typeConverter, patterns, target,
                                       benefit);
    populateMakeRangeOpToLLVMPattern(typeConverter, patterns, target, benefit);
    populateSPMDOpToLLVMPattern(typeConverter, patterns, target, benefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    switch (target) {
    case Target::NVVM:
      mlir::populateGpuToNVVMConversionPatterns(typeConverter, patterns);
      break;
    case Target::GENX:
      mlir::triton::populateGENToLLVMConversionPatterns(typeConverter,
                                                        patterns);
      mlir::populateGpuToGENXConversionPatterns(typeConverter, patterns);
      break;
    default:
      break;
    }
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
  void initSharedMemory(LLVMTypeConverter &typeConverter, Target target) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    switch (target) {
    case Target::NVVM:
    case Target::ROCDL: {
      // Set array size 0 and external linkage indicates that we use dynamic
      // shared allocation to allow a larger shared memory size for each kernel.
      //
      // Ask for 16B alignment on global_smem because that's the largest we
      // should ever need (4xi32).
      auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
      auto global = b.create<LLVM::GlobalOp>(
          loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
          "global_smem", /*value=*/Attribute(), /*alignment=*/16,
          // Add ROCm support.
          static_cast<unsigned>(NVVM::NVVMMemorySpace::kSharedMemorySpace));
    } break;
    case Target::GENX: {
    } break;
    default:
      break;
    }
  }

  // pass ws related named attrs.
  static void addWSNamedAttrs(Operation *op,
                              ArrayRef<mlir::NamedAttribute> attrs) {
    for (const NamedAttribute attr : attrs)
      if (attr.getName() == "async_agent" ||
          attr.getName() == "agent.mutex_role")
        op->setAttr(attr.getName(), attr.getValue());
  }

  void decomposeInsertSliceAsyncOp(ModuleOp mod) const {

    // The function has been deprecated upstream but is required to work on
    // genx. The current rewrite pattern for InsertSliceAsync generates PTX and
    // there is no matching instruciton on genx at the moment.
    // FIXME: remove this function once a suitable replacement is available.
    if (target != triton::Target::GENX)
      return;

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
createConvertTritonIntelGPUToLLVMPass(int32_t computeCapability,
                                      Target target) {
  return std::make_unique<ConvertTritonGPUToLLVM>(computeCapability, target);
}

} // namespace triton
} // namespace mlir
