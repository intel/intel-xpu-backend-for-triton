#include "triton/Conversion/TritonGPUToSPIRV/TritonGPUToSPIRVPass.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "ConvertLayoutOpToSPIRV.h"
#include "DotOpToSPIRV.h"
#include "ElementwiseOpToSPIRV.h"
#include "LoadStoreOpToSPIRV.h"
#include "ReduceOpToSPIRV.h"
#include "TritonGPUToSPIRV.h"
#include "TypeConverter.h"
#include "ViewOpToSPIRV.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonGPUToSPIRV/Passes.h.inc"

namespace {

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getElemsPerThread;
using ::mlir::triton::gpu::getOrder;
using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::getSizePerThread;
using ::mlir::triton::gpu::getThreadsPerCTA;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

class TritonSPIRVFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonSPIRVFunctionConversionTarget(MLIRContext &ctx, SPIRVTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
    addIllegalOp<mlir::func::FuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ReturnOpConversion : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};

struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(SPIRVTypeConverter &converter, MLIRContext *context, int numWarps,
                   ModuleAllocation &allocation,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, context, benefit), allocation(allocation),
      numWarps(numWarps) {}
#if 0
  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = spirv::PointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()), spirv::StorageClass::Workgroup);
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
#endif

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = dyn_cast<ModuleOp>(funcOp->getParentOp());
    if (!mod)
      return failure();

    auto amendedFuncOp = funcOp;
//    if (!allocation.isRoot(funcOp))
//      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    auto newFuncOp = convertFuncOpToSPIRVFuncOp(amendedFuncOp, rewriter);
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    if (allocation.isRoot(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      newFuncOp->setAttr(spirv::getEntryPointABIAttrName(),
                         spirv::EntryPointABIAttr::get(getContext(), nullptr, std::nullopt));
    } else {
      // The noinline function control will be used by the SPIRV codegen to prevent
      // inlining.
      newFuncOp.setFunctionControl(spirv::FunctionControl::DontInline);
//      rewriter.eraseOp(amendedFuncOp);
    }
//    // Set an attribute for maxntidx, it could be used in latter LLVM codegen
//    // for `nvvm.annotation` metadata.
//    newFuncOp->setAttr("nvvm.maxntid", rewriter.getI32ArrayAttr(32 * numWarps));
    // The call graph is updated by mapping the old function to the new one.
    allocation.mapFuncOp(funcOp, newFuncOp);

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
  ModuleAllocation &allocation;
};

class TritonSPIRVConversionTarget : public ConversionTarget {
public:
  explicit TritonSPIRVConversionTarget(MLIRContext &ctx, SPIRVTypeConverter& typeConverter)
          : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
//    addIllegalDialect<triton::TritonDialect>();
//    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<mlir::func::FuncDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addDynamicallyLegalOp<mlir::func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
  }
};

class ConvertTritonGPUToSPIRV
    : public ConvertTritonGPUToSPIRVBase<ConvertTritonGPUToSPIRV> {

public:
  explicit ConvertTritonGPUToSPIRV(std::map<std::string, int> computeCapability) {
    this->computeCapability = std::move(computeCapability);
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    spirv::Capability caps_opencl[] = {
            spirv::Capability::Addresses,
            spirv::Capability::Float16Buffer,
            spirv::Capability::Int64,
            spirv::Capability::Int16,
            spirv::Capability::Int8,
            spirv::Capability::Kernel,
            spirv::Capability::Linkage,
            spirv::Capability::Vector16,
            spirv::Capability::GenericPointer,
            spirv::Capability::Groups,
            spirv::Capability::Float16,
            spirv::Capability::Float64,
            spirv::Capability::AtomicFloat32AddEXT,
            spirv::Capability::ExpectAssumeKHR,
    };
    spirv::Extension exts_opencl[] = {
            spirv::Extension::SPV_EXT_shader_atomic_float_add,
            spirv::Extension::SPV_KHR_expect_assume};
    auto triple = spirv::VerCapExtAttr::get(
            spirv::Version::V_1_4, caps_opencl, exts_opencl, context);
    auto targetAttr = spirv::TargetEnvAttr::get(
            triple, spirv::getDefaultResourceLimits(context),
            spirv::ClientAPI::OpenCL,
            spirv::Vendor::Unknown,
            spirv::DeviceType::Unknown,
            spirv::TargetEnvAttr::kUnknownDeviceID);

    mod->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

    SPIRVConversionOptions options;
    // TODO: need confirm
    options.use64bitIndex = true;
    TritonGPUToSPIRVTypeConverter spirvTypeConverter(targetAttr, options);
    TritonSPIRVFunctionConversionTarget funcTarget(*context, spirvTypeConverter);
    TritonSPIRVConversionTarget spirvTarget(*context, spirvTypeConverter);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Step 1: Decompose unoptimized layout conversions to use shared memory
    // Step 2: Decompose insert_slice_async to use load + insert_slice for
    //   pre-Ampere architectures or unsupported vectorized load sizes
    // Step 3: Allocate shared memories and insert barriers
    // Step 4: Convert SCF to CFG
    // Step 5: Get axis and shared memory info
    // Step 6: Convert FuncOp to spirv::FuncOp via partial conversion
    // Step 7: Convert the rest of ops via partial conversion
    //
    // The reason for putting step 3 before step 4 is that the membar
    // analysis currently only supports SCF but not CFG. The reason for a
    // separation between 5/7 is that, step 6 is out of the scope of Dialect
    // Conversion, thus we need to make sure the smem is not revised during the
    // conversion of step 7.

    // Step 1
    decomposeMmaToDotOperand(mod, numWarps, threadsPerWarp);
    decomposeBlockedToDotOperand(mod);

    // Step 2
    if (failed(decomposeInsertSliceAsyncOp(mod)))
      return signalPassFailure();

    // Step 3
    ModuleAllocation allocation(mod);
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    // Step 4
    RewritePatternSet scf_patterns(context);
    mlir::populateSCFToControlFlowConversionPatterns(scf_patterns);
    mlir::ConversionTarget scf_target(*context);
    scf_target.addIllegalOp<scf::ForOp, scf::IfOp, scf::ParallelOp,
            scf::WhileOp, scf::ExecuteRegionOp>();
    scf_target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(
            applyPartialConversion(mod, scf_target, std::move(scf_patterns))))
      return signalPassFailure();

    // Step 5 - get axis and shared memory info
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(context, 32),
                                        allocation.getSharedMemorySize()));

    // Step 6
    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(spirvTypeConverter, context, numWarps, allocation,
                                        1 /*benefit*/);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();

    initSharedMemory(allocation, spirvTypeConverter);

    // Step 7 - rewrite rest of ops
    // We set a higher benefit here to ensure triton's patterns runs before
    // arith patterns for some encoding not supported by the community
    // patterns.
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo indexCacheInfo{
            &baseIndexCache, &indexCache, &indexInsertPoint};

    RewritePatternSet patterns(context);
    // Normal conversions
    populateTritonGPUToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                     axisInfoAnalysis, allocation,
                                    indexCacheInfo, /*benefit=*/10);
    // ConvertLayoutOp
    populateConvertLayoutOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                          axisInfoAnalysis, allocation,
                                          indexCacheInfo, /*benefit=*/10);
    // DotOp
    populateDotOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                axisInfoAnalysis, allocation,
            /*benefit=*/10);
    // ElementwiseOp
    populateElementwiseOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                        axisInfoAnalysis, &allocation, smem,
            /*benefit=*/10);
    // LoadStoreOp
    populateLoadStoreOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                      axisInfoAnalysis, allocation,
                                      indexCacheInfo, /*benefit=*/10);
    // ReduceOp
    populateReduceOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                   axisInfoAnalysis, allocation,
                                   indexCacheInfo, /*benefit=*/10);
    // ViewOp
    populateViewOpToSPIRVPatterns(spirvTypeConverter, context, patterns, numWarps,
                                 axisInfoAnalysis, &allocation, smem,
            /*benefit=*/10);

    // Add arith/math's patterns to help convert scalar expression to SPIRV.
    mlir::arith::populateArithToSPIRVPatterns(spirvTypeConverter,
                                                            patterns);
    mlir::populateMathToSPIRVPatterns(spirvTypeConverter, patterns);
    mlir::populateFuncToSPIRVPatterns(spirvTypeConverter, patterns);
    mlir::populateGPUToSPIRVPatterns(spirvTypeConverter, patterns);
    mlir::cf::populateControlFlowToSPIRVPatterns(spirvTypeConverter, patterns);

//    ::llvm::DebugFlag = true;
//    ::llvm::setCurrentDebugType("dialect-conversion");
    if (failed(applyPartialConversion(mod, spirvTarget, std::move(patterns))))
      return signalPassFailure();
//    ::llvm::DebugFlag = false;
  }

private:
  Value smem;

  using IndexCacheKeyT = std::pair<Attribute, RankedTensorType>;
  DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
      baseIndexCache;
  DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
           CacheKeyDenseMapInfo>
      indexCache;

  void initSharedMemory(ModuleAllocation &allocation,
                        TritonGPUToSPIRVTypeConverter &typeConverter) {

    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    auto shareMemSize = allocation.getSharedMemorySize();
    if (shareMemSize)
    mod.walk([&](FunctionOpInterface funcOp) {
      auto ptrTy = spirv::PointerType::get(typeConverter.convertType(b.getI8Type()),
                                           spirv::StorageClass::Workgroup);
      funcOp.insertArgument(funcOp.getNumArguments(), ptrTy, {}, funcOp.getLoc());
      Value funcSmem = funcOp.getArgument(funcOp.getNumArguments() - 1);
      allocation.setFunctionSharedMemoryValue(funcOp, funcSmem);
    });
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps, int threadsPerWarp) const {
    // Replace `mma -> dot_op` with `mma -> blocked -> dot_op`
    // unless certain conditions are met
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcMma =
          srcType.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcMma && dstDotOp && !isMmaToDotShortcut(srcType, dstType)) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::BlockedEncodingAttr::get(
                mod.getContext(), srcType.getShape(), getSizePerThread(srcMma),
                getOrder(srcMma), numWarps, threadsPerWarp));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  void decomposeBlockedToDotOperand(ModuleOp mod) const {
    // Replace `blocked -> dot_op` with `blocked -> shared -> dot_op`
    // because the codegen doesn't handle `blocked -> dot_op` directly
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
      auto dstType = cvtOp.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto dstDotOp =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (srcBlocked && dstDotOp) {
        auto tmpType = RankedTensorType::get(
            dstType.getShape(), dstType.getElementType(),
            triton::gpu::SharedEncodingAttr::get(
                mod.getContext(), dstDotOp, srcType.getShape(),
                getOrder(srcBlocked), srcType.getElementType()));
        auto tmp = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), tmpType, cvtOp.getOperand());
        auto newConvert = builder.create<triton::gpu::ConvertLayoutOp>(
            cvtOp.getLoc(), dstType, tmp);
        cvtOp.replaceAllUsesWith(newConvert.getResult());
        cvtOp.erase();
      }
    });
  }

  LogicalResult decomposeInsertSliceAsyncOp(ModuleOp mod) const {
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
      auto srcTy = src.getType().cast<RankedTensorType>();
      auto dstTy = dst.getType().cast<RankedTensorType>();
      auto srcBlocked =
          srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
      auto resSharedLayout =
          dstTy.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      auto resElemTy = dstTy.getElementType();
      unsigned inVec = axisInfoAnalysis.getPtrContiguity(src);
      unsigned outVec = resSharedLayout.getVec();
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(
              80)
              .contains(byteWidth))
        return;

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

      // Replace
      insertSliceAsyncOp.replaceAllUsesWith(insertSliceOp.getResult());
      insertSliceAsyncOp.erase();
      decomposed = true;
    });

    mod.walk([&](triton::gpu::AsyncCommitGroupOp asyncCommitGroupOp) -> void {
      if (!triton::gpu::AsyncCommitGroupOp::isSupported(80))
        asyncCommitGroupOp.erase();
    });

    mod.walk([&](triton::gpu::AsyncWaitOp asyncWaitOp) -> void {
      if (!triton::gpu::AsyncWaitOp::isSupported(80)) {
        // async wait is supported in Ampere and later
        asyncWaitOp.erase();
      } else if (decomposed) {
        // Wait for all previous async ops
        OpBuilder builder(asyncWaitOp);
        builder.create<triton::gpu::AsyncWaitOp>(asyncWaitOp.getLoc(), 0);
        asyncWaitOp.erase();
      }
    });
    return success();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonGPUToSPIRVPass(std::map<std::string, int> computeCapability) {
  return std::make_unique<::ConvertTritonGPUToSPIRV>(computeCapability);
}

} // namespace triton
} // namespace mlir
