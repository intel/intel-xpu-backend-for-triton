#include "triton/Conversion/TritonGPUToSPIRV/TritonGPUToSPIRVPass.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRV.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
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
  explicit TritonSPIRVFunctionConversionTarget(
      MLIRContext &ctx, SPIRVTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<spirv::SPIRVDialect>();
    addIllegalOp<mlir::func::FuncOp>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<spirv::FuncOp>();
    if (funcOp->hasAttr(spirv::getEntryPointABIAttrName())) {
      // A GPU kernel
      if (op.getNumOperands() > 0) {
        return rewriter.notifyMatchFailure(
            op, "Kernel functions do not support return with operands");
      }
      rewriter.replaceOpWithNewOp<spirv::ReturnOp>(
          op, TypeRange(), ValueRange(), op->getAttrs());
    } else {
      // A device function
      Operation *newOp;
      if (adaptor.getOperands().size() < 2) {
        // Single or no return value.
        Type resultType = nullptr;
        if (adaptor.getOperands().size() == 1) {
          resultType = funcOp.getResultTypes().front();
          newOp = rewriter.create<spirv::ReturnValueOp>(
              op.getLoc(), TypeRange(), adaptor.getOperands());
        } else {
          newOp = rewriter.create<spirv::ReturnOp>(op.getLoc(), TypeRange());
        };
      } else {
        // Pack the result types into a struct.
        Type packedResultsTy = nullptr;
        unsigned numResults = funcOp.getNumResults();
        auto resultTypes = llvm::to_vector<4>(funcOp.getResultTypes());

        if (numResults == 1) {
          packedResultsTy =
              getTypeConverter()->convertType(resultTypes.front());
        } else {
          SmallVector<Type> convertedTypes;
          for (auto t : resultTypes) {
            auto converted = getTypeConverter()->convertType(t);
            if (!converted)
              return failure();
            convertedTypes.push_back(converted);
          }

          packedResultsTy = spirv::StructType::get(convertedTypes);
        }
        Value packedResults =
            rewriter.create<spirv::UndefOp>(op.getLoc(), packedResultsTy);
        auto loc = op.getLoc();
        for (auto it : llvm::enumerate(adaptor.getOperands())) {
          packedResults = insert_val(packedResultsTy, it.value(), packedResults,
                                     rewriter.getI32ArrayAttr(it.index()));
        }
        newOp = rewriter.create<spirv::ReturnValueOp>(op.getLoc(), TypeRange(),
                                                      packedResults);
      }
      newOp->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, newOp->getResults());
    }
    return success();
  }
};

struct FuncOpConversion : public FuncOpConversionBase {
  FuncOpConversion(SPIRVTypeConverter &converter, MLIRContext *context,
                   int numWarps, ModuleAllocation &allocation,
                   PatternBenefit benefit)
      : FuncOpConversionBase(converter, context, benefit),
        allocation(allocation), numWarps(numWarps) {}

  triton::FuncOp amendFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Push back a variable that indicates the current stack pointer of shared
    // memory to the function arguments.
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    auto ptrTy = spirv::PointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()),
        spirv::StorageClass::Workgroup);
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
    ModuleOp mod = dyn_cast<ModuleOp>(funcOp->getParentOp());
    if (!mod)
      return failure();

    auto amendedFuncOp = funcOp;
    if (!allocation.isRoot(funcOp))
      amendedFuncOp = amendFuncOp(funcOp, rewriter);

    auto newFuncOp = convertFuncOpToSPIRVFuncOp(amendedFuncOp, rewriter);
    if (!newFuncOp) {
      return failure();
    }

    auto ctx = funcOp->getContext();

    if (allocation.isRoot(funcOp)) {
      // Set an attribute to indicate this function is a kernel entry.
      //    // Set an attribute for maxntidx, it could be used in latter LLVM
      //    codegen
      //    // for `nvvm.annotation` metadata.
      //    newFuncOp->setAttr("nvvm.maxntid", rewriter.getI32ArrayAttr(32 *
      //    numWarps));
      newFuncOp->setAttr(
          spirv::getEntryPointABIAttrName(),
          spirv::EntryPointABIAttr::get(getContext(), nullptr, std::nullopt));
    } else {
      // The noinline function control will be used by the SPIRV codegen to
      // prevent inlining.
      newFuncOp.setFunctionControl(spirv::FunctionControl::DontInline);
      rewriter.eraseOp(amendedFuncOp);
    }
    // The call graph is updated by mapping the old function to the new one.
    allocation.mapFuncOp(funcOp, newFuncOp);

    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  int numWarps{0};
  ModuleAllocation &allocation;
};

// CallOpInterfaceLowering is adapted from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L485
struct CallOpConversion : public OpConversionPattern<triton::CallOp> {
  CallOpConversion(SPIRVTypeConverter &converter, MLIRContext *context,
                   int numWarps, ModuleAllocation &allocation,
                   PatternBenefit benefit)
      : OpConversionPattern<triton::CallOp>(converter, context, benefit),
        numWarps(numWarps), allocation(allocation) {}

  LogicalResult
  matchAndRewrite(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto promotedOperands = promoteOperands(callOp, adaptor, rewriter);
    auto newCallOp =
        convertCallOpToSPIRVCallOp(callOp, promotedOperands, rewriter);
    if (!newCallOp)
      return failure();
    allocation.mapCallOp(callOp, newCallOp);
    auto results = getCallOpResults(callOp, newCallOp, rewriter);
    rewriter.replaceOp(callOp, results);
    return success();
  }

private:
  SmallVector<Value, 4>
  promoteOperands(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    // Get the last argument of the caller, which is the current stack pointer
    // of shared memory and append it to the operands of the callOp.
    auto loc = callOp.getLoc();
    auto caller = callOp->getParentOfType<FunctionOpInterface>();
    auto base = allocation.getFunctionSharedMemoryBase(caller);
    auto *funcAllocation = allocation.getFuncData(caller);
    auto bufferId = funcAllocation->getBufferId(callOp);
    auto offset = funcAllocation->getOffset(bufferId);
    auto ptrTy = spirv::PointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()),
        spirv::StorageClass::Workgroup);
    if (!base) {
      base = rewriter.create<spirv::UndefOp>(callOp.getLoc(), ptrTy);
    }
    auto offsetValue = gep(ptrTy, base, i32_val(offset));
    SmallVector<Value, 4> promotedOperands;
    auto opOperands = callOp->getOpOperands();
    auto spirvOperands = adaptor.getOperands();
    for (auto it : llvm::zip(opOperands, adaptor.getOperands())) {
      auto &operand = std::get<0>(it);
      auto &spirvOperand = std::get<1>(it);
      promotedOperands.push_back(spirvOperand);
    }
    promotedOperands.push_back(offsetValue);
    return promotedOperands;
  }

  spirv::FunctionCallOp
  convertCallOpToSPIRVCallOp(triton::CallOp callOp,
                             ArrayRef<Value> promotedOperands,
                             ConversionPatternRewriter &rewriter) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (numResults == 1) {
        packedResult = getTypeConverter()->convertType(resultTypes.front());
      } else {
        SmallVector<Type> convertedTypes;
        for (auto t : resultTypes) {
          auto converted = getTypeConverter()->convertType(t);
          if (!converted)
            return nullptr;
          convertedTypes.push_back(converted);
        }

        packedResult = spirv::StructType::get(convertedTypes);
      }
    }

    auto newCallOp = rewriter.create<spirv::FunctionCallOp>(
        callOp.getLoc(), packedResult ? TypeRange(packedResult) : TypeRange(),
        promotedOperands, callOp->getAttrs());
    return newCallOp;
  }

  SmallVector<Value>
  getCallOpResults(triton::CallOp callOp, spirv::FunctionCallOp newCallOp,
                   ConversionPatternRewriter &rewriter) const {
    auto numResults = callOp.getNumResults();
    SmallVector<Value> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(rewriter.create<spirv::CompositeExtractOp>(
            callOp.getLoc(), newCallOp->getResult(0), i));
      }
    }
    return results;
  }

  int numWarps{0};
  ModuleAllocation &allocation;
};

class TritonSPIRVConversionTarget : public ConversionTarget {
public:
  explicit TritonSPIRVConversionTarget(MLIRContext &ctx,
                                       SPIRVTypeConverter &typeConverter)
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
  explicit ConvertTritonGPUToSPIRV(
      std::map<std::string, int> computeCapability) {
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
    auto triple = spirv::VerCapExtAttr::get(spirv::Version::V_1_4, caps_opencl,
                                            exts_opencl, context);
    auto targetAttr = spirv::TargetEnvAttr::get(
        triple, spirv::getDefaultResourceLimits(context),
        spirv::ClientAPI::OpenCL, spirv::Vendor::Unknown,
        spirv::DeviceType::Unknown, spirv::TargetEnvAttr::kUnknownDeviceID);

    mod->setAttr(spirv::getTargetEnvAttrName(), targetAttr);

    SPIRVConversionOptions options;
    // TODO: need confirm
    options.use64bitIndex = true;
    TritonGPUToSPIRVTypeConverter spirvTypeConverter(targetAttr, options);
    TritonSPIRVFunctionConversionTarget funcTarget(*context,
                                                   spirvTypeConverter);
    TritonSPIRVConversionTarget spirvTarget(*context, spirvTypeConverter);

    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Preprocess
    decomposeMmaToDotOperand(mod, numWarps, threadsPerWarp);
    decomposeBlockedToDotOperand(mod);
    decomposeInsertSliceAsyncOp(mod);

    // Allocate shared memory and set barrier
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

    // Lower functions
    RewritePatternSet func_patterns(context);
    func_patterns.add<FuncOpConversion>(spirvTypeConverter, context, numWarps,
                                        allocation, 1 /*benefit*/);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(func_patterns))))
      return signalPassFailure();
    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(allocation, spirvTypeConverter);

    // Convert call and ret ops
    RewritePatternSet funcPatterns(context);
    funcPatterns.add<CallOpConversion>(spirvTypeConverter, context, numWarps,
                                       allocation,
                                       /*benefit=*/1);
    funcPatterns.add<ReturnOpConversion>(spirvTypeConverter, context,
                                         /*benefit=*/1);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    // Rewrite ops
    RewritePatternSet patterns(context);
    // TritonGPU lowering patterns
    OpBuilder::InsertPoint indexInsertPoint;
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo indexCacheInfo{
        &baseIndexCache, &indexCache, &indexInsertPoint};
    // TODO: enable index cache if there are multiple functions
    if (axisInfoAnalysis.getNumFunctions() > 1) {
      indexCacheInfo = {nullptr, nullptr, nullptr};
    }
    // Normal conversions
    populateTritonGPUToSPIRVPatterns(spirvTypeConverter, context, patterns,
                                     numWarps, axisInfoAnalysis, allocation,
                                     indexCacheInfo, /*benefit=*/10);
    // ConvertLayoutOp
    populateConvertLayoutOpToSPIRVPatterns(
        spirvTypeConverter, context, patterns, numWarps, axisInfoAnalysis,
        allocation, indexCacheInfo, /*benefit=*/10);
    // DotOp
    populateDotOpToSPIRVPatterns(spirvTypeConverter, context, patterns,
                                 numWarps, axisInfoAnalysis, allocation,
                                 /*benefit=*/10);
    // ElementwiseOp
    populateElementwiseOpToSPIRVPatterns(spirvTypeConverter, context, patterns,
                                         numWarps, axisInfoAnalysis,
                                         &allocation, nullptr,
                                         /*benefit=*/10, computeCapability);
    // LoadStoreOp
    populateLoadStoreOpToSPIRVPatterns(spirvTypeConverter, context, patterns,
                                       numWarps, axisInfoAnalysis, allocation,
                                       indexCacheInfo, /*benefit=*/10);
    // ReduceOp
    populateReduceOpToSPIRVPatterns(spirvTypeConverter, context, patterns,
                                    numWarps, axisInfoAnalysis, allocation,
                                    indexCacheInfo, /*benefit=*/10);
    // ViewOp
    populateViewOpToSPIRVPatterns(spirvTypeConverter, context, patterns,
                                  numWarps, axisInfoAnalysis, &allocation,
                                  nullptr,
                                  /*benefit=*/10);

    // Add arith/math's patterns to help convert scalar expression to SPIRV.
    mlir::arith::populateArithToSPIRVPatterns(spirvTypeConverter, patterns);
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
    mod.walk([&](FunctionOpInterface funcOp) {
      Value funcSmem;
      if (allocation.isRoot(funcOp)) {
        if (shareMemSize) {
          auto ptrTy =
              spirv::PointerType::get(typeConverter.convertType(b.getI8Type()),
                                      spirv::StorageClass::Workgroup);
          funcOp.insertArgument(funcOp.getNumArguments(), ptrTy, {},
                                funcOp.getLoc());
          funcSmem = funcOp.getArgument(funcOp.getNumArguments() - 1);
        }
      } else {
        funcSmem = funcOp.getArgument(funcOp.getNumArguments() - 1);
      }
      allocation.setFunctionSharedMemoryValue(funcOp, funcSmem);
    });
    mod->setAttr("triton_gpu.shared",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                        allocation.getSharedMemorySize()));
  }

  void decomposeMmaToDotOperand(ModuleOp mod, int numWarps,
                                int threadsPerWarp) const {
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
      unsigned minVec = std::min(outVec, inVec);
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto byteWidth = bitWidth / 8;

      // If the load byte width is not eligible or the current compute
      // capability does not support async copy, then we do decompose
      if (triton::gpu::InsertSliceAsyncOp::getEligibleLoadByteWidth(80)
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
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonGPUToSPIRVPass(
    std::map<std::string, int> computeCapability) {
  return std::make_unique<::ConvertTritonGPUToSPIRV>(computeCapability);
}

} // namespace triton
} // namespace mlir
