#include "TritonGPUToSPIRV.h"
#include "Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

struct ReturnOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::ReturnOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::ReturnOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op, TypeRange(), ValueRange(),
                                                 op->getAttrs());
    return success();
  }
};

struct BroadcastOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::BroadcastOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::BroadcastOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Following the order of indices in the legacy code, a broadcast of:
    //   [s(0), s(1) ... s(k-1),    1, s(k+1), s(k+2) ... s(n-1)]
    // =>
    //   [s(0), s(1) ... s(k-1), s(k), s(k+1), s(k+2) ... s(n-1)]
    //
    // logically maps to a broadcast within a thread's scope:
    //   [cta(0)..cta(k-1),     1,cta(k+1)..cta(n-1),spt(0)..spt(k-1),
    //   1,spt(k+1)..spt(n-1)]
    // =>
    //   [cta(0)..cta(k-1),cta(k),cta(k+1)..cta(n-1),spt(0)..spt(k-1),spt(k),spt(k+1)..spt(n-1)]
    //
    // regardless of the order of the layout
    //
    Location loc = op->getLoc();
    Value src = adaptor.getSrc();
    Value result = op.getResult();
    auto srcTy = op.getSrc().getType().cast<RankedTensorType>();
    auto resultTy = result.getType().cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding();
    auto resultLayout = resultTy.getEncoding();
    auto srcShape = srcTy.getShape();
    auto resultShape = resultTy.getShape();
    unsigned rank = srcTy.getRank();

    assert(rank == resultTy.getRank());
    auto order = triton::gpu::getOrder(srcLayout);
    auto srcOffsets = emitOffsetForLayout(srcLayout, srcTy);
    auto resultOffsets = emitOffsetForLayout(resultLayout, resultTy);
    SmallVector<Value> srcVals =
        getTypeConverter()->unpackLLElements(loc, src, rewriter, srcTy);

    DenseMap<SmallVector<unsigned>, Value, SmallVectorKeyInfo> srcValues;
    for (size_t i = 0; i < srcOffsets.size(); i++) {
      srcValues[srcOffsets[i]] = srcVals[i];
    }

    SmallVector<Value> resultVals;
    for (size_t i = 0; i < resultOffsets.size(); i++) {
      auto offset = resultOffsets[i];
      for (size_t j = 0; j < srcShape.size(); j++)
        if (srcShape[j] == 1)
          offset[j] = 0;
      resultVals.push_back(srcValues.lookup(offset));
    }

    Value resultStruct =
        getTypeConverter()->packLLElements(loc, resultVals, rewriter, resultTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct AssertOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AssertOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AssertOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto elems = getTypeConverter()->unpackLLElements(
        loc, adaptor.getCondition(), rewriter, op.getCondition().getType());
    auto elemTy = elems[0].getType();
    Value condition = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition = logic_or(
            condition,
            logic_cmp_eq(elem, rewriter.create<spirv::ConstantOp>(
                                   loc, elemTy, rewriter.getZeroAttr(elemTy))));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    spirvAssert(op, condition, adaptor.getMessage(), adaptor.getFile(),
                adaptor.getFunc(), adaptor.getLine(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void spirvAssert(Operation *op, Value condition, StringRef message,
                          StringRef file, StringRef func, int line,
                          ConversionPatternRewriter &rewriter) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto funcOp = getAssertfailDeclaration(rewriter);
    StringRef funcName("__assert_fail");
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Value messageString =
        spirv::addStringToModule(loc, rewriter, "assertMessage_", message);
    Value fileString =
        spirv::addStringToModule(loc, rewriter, "assertFile_", file);
    Value funcString =
        spirv::addStringToModule(loc, rewriter, "assertFunc_", func);
    Value lineNumber = i32_val(line);
    Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));

    SmallVector<Value> operands = {messageString, fileString, lineNumber,
                                   funcString, charSize};

    auto ret = call(TypeRange(), funcName, operands);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }

  static spirv::FuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("__assert_fail");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<spirv::FuncOp>(*funcOp);

    // void __assert_fail(const char * assertion, const char * file, unsigned
    // int line, const char * function);
    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType{ptr_ty(i8_ty, spirv::StorageClass::Generic),
                               ptr_ty(i8_ty, spirv::StorageClass::Generic),
                               i32_ty,
                               ptr_ty(i8_ty, spirv::StorageClass::Generic),
                               rewriter.getIntegerType(sizeof(size_t) * 8)};

    mlir::FunctionType funcType =
        mlir::FunctionType::get(rewriter.getContext(), argsType, {});

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<spirv::FuncOp>(UnknownLoc::get(ctx), funcName,
                                          funcType);
  }
};

struct MakeRangeOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::MakeRangeOp> {

  MakeRangeOpSPIRVConversion(
      TritonGPUToSPIRVTypeConverter &converter, MLIRContext *context,
      ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::MakeRangeOp>(
            converter, context, indexCacheInfo, benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto rankedTy = op.getResult().getType().cast<RankedTensorType>();
    auto shape = rankedTy.getShape();
    auto layout = rankedTy.getEncoding();

    auto elemTy = rankedTy.getElementType();
    assert(elemTy.isInteger(32));
    Value start = rewriter.create<spirv::ConstantOp>(
        loc, elemTy, rewriter.getIntegerAttr(elemTy, op.getStart()));
    auto idxs = emitIndices(loc, rewriter, layout, rankedTy);
    unsigned elems = idxs.size();
    SmallVector<Value> retVals(elems);
    // TODO: slice layout has more elements than expected.
    // Unexpected behavior for make range, but generally OK when followed by
    // expand dims + broadcast. very weird behavior otherwise potentially.
    for (const auto &multiDim : llvm::enumerate(idxs)) {
      assert(multiDim.value().size() == 1);
      retVals[multiDim.index()] = add(multiDim.value()[0], start);
    }
    Value result =
        getTypeConverter()->packLLElements(loc, retVals, rewriter, rankedTy);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct GetProgramIdOpToSPIRVConversion
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);

    Value blockId = rewriter.create<::mlir::gpu::BlockIdOp>(
        loc, rewriter.getIndexType(), dims[op.getAxisAsInt()]);
    Value blockId_idx =
        rewriter.create<::mlir::arith::TruncIOp>(loc, i32_ty, blockId);
    auto *typeConverter = this->template getTypeConverter<SPIRVTypeConverter>();
    auto indexType = typeConverter->getIndexType();

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TypeRange{i32_ty}, ValueRange{blockId_idx});
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct GetNumProgramsOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    assert(op.getAxis() < 3);

    Value blockId =
        rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxis()]);
    rewriter.replaceOpWithNewOp<arith::TruncIOp>(op, i32_ty, blockId);

    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct AddPtrOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AddPtrOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AddPtrOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto offsetTy = op.getOffset().getType();
    auto ptrTy = op.getPtr().getType();
    auto resultTensorTy = resultTy.dyn_cast<RankedTensorType>();
    if (resultTensorTy) {
      unsigned elems = getTotalElemsPerThread(resultTy);
      Type elemTy =
          getTypeConverter()->convertType(resultTensorTy.getElementType());
      auto ptrs = getTypeConverter()->unpackLLElements(loc, adaptor.getPtr(),
                                                       rewriter, ptrTy);
      auto offsets = getTypeConverter()->unpackLLElements(
          loc, adaptor.getOffset(), rewriter, offsetTy);
      SmallVector<Value> resultVals(elems);
      for (unsigned i = 0; i < elems; ++i) {
        resultVals[i] = gep(elemTy, ptrs[i], offsets[i]);
      }
      Value view = getTypeConverter()->packLLElements(loc, resultVals, rewriter,
                                                      resultTy);
      rewriter.replaceOp(op, view);
    } else {
      assert(resultTy.isa<triton::PointerType>());
      Type llResultTy = getTypeConverter()->convertType(resultTy);
      Value result = gep(llResultTy, adaptor.getPtr(), adaptor.getOffset());
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct AllocTensorOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::AllocTensorOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::AllocTensorOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AllocTensorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value smemBase = getSharedMemoryBase(loc, rewriter, op.getResult());
    auto resultTy = op.getType().dyn_cast<RankedTensorType>();
    auto spirvElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto elemPtrTy = ptr_ty(spirvElemTy, spirv::StorageClass::Workgroup);
    smemBase = bitcast(smemBase, elemPtrTy);
    auto order = resultTy.getEncoding().cast<SharedEncodingAttr>().getOrder();
    // Workaround for 3D tensors
    // TODO: we need to modify the pipeline pass to give a proper shared
    // encoding to 3D tensors
    SmallVector<unsigned> newOrder;
    if (resultTy.getShape().size() == 3)
      newOrder = {1 + order[0], 1 + order[1], 0};
    else
      newOrder = SmallVector<unsigned>(order.begin(), order.end());

    auto smemObj = SharedMemoryObject(smemBase, resultTy.getShape(), newOrder,
                                      loc, rewriter);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct ExtractSliceOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::ExtractSliceOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::ExtractSliceOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = extract_slice %src[%offsets]
    Location loc = op->getLoc();
    auto srcTy = op.getSource().getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    assert(srcLayout && "Unexpected resultLayout in ExtractSliceOpConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by ExtractSliceOpConversion");

    // newBase = base + offset
    // Triton supports either static and dynamic offsets
    auto smemObj =
        getSharedMemoryObjectFromStruct(loc, adaptor.getSource(), rewriter);
    SmallVector<Value, 4> opOffsetVals;
    SmallVector<Value, 4> offsetVals;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i))
        opOffsetVals.emplace_back(adaptor.getOffsets()[i]);
      else
        opOffsetVals.emplace_back(i32_val(op.getStaticOffset(i)));
      offsetVals.emplace_back(add(smemObj.offsets[i], opOffsetVals[i]));
    }
    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, opOffsetVals, smemObj.strides);
    // newShape = rank_reduce(shape)
    // Triton only supports static tensor sizes
    SmallVector<Value, 4> strideVals;
    for (auto i = 0; i < op.static_sizes().size(); ++i) {
      if (op.getStaticSize(i) == 1) {
        offsetVals.erase(offsetVals.begin() + i);
      } else {
        strideVals.emplace_back(smemObj.strides[i]);
      }
    }

    auto spirvElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto elemPtrTy = ptr_ty(spirvElemTy, spirv::StorageClass::Workgroup);
    smemObj = SharedMemoryObject(gep(elemPtrTy, smemObj.base, offset),
                                 strideVals, offsetVals);
    auto retVal = getStructFromSharedMemoryObject(loc, smemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

struct AsyncWaitOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::AsyncWaitOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::AsyncWaitOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: implement the async memory fetch.
#if 0
    PTXBuilder ptxBuilder;
    auto &asyncWaitOp = *ptxBuilder.create<>("cp.async.wait_group");
    auto num = op->getAttrOfType<IntegerAttr>("num").getInt();
    asyncWaitOp(ptxBuilder.newConstantOperand(num));

    auto ctx = op.getContext();
    auto loc = op.getLoc();
    auto voidTy = void_ty(ctx);
    ptxBuilder.launch(rewriter, loc, voidTy);
#endif
    // Safe to remove the op since it doesn't have any return value.
    rewriter.eraseOp(op);
    return success();
  }
};

struct AsyncCommitGroupOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::AsyncCommitGroupOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::AsyncCommitGroupOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: implement the async memory fetch.
#if 0
    PTXBuilder ptxBuilder;
    ptxBuilder.create<>("cp.async.commit_group")->operator()();
    ptxBuilder.launch(rewriter, op.getLoc(), void_ty(op.getContext()));
    // Safe to remove the op since it doesn't have any return value.
#endif
    rewriter.eraseOp(op);
    return success();
  }
};

void populateTritonGPUToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<AddPtrOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<AllocTensorOpSPIRVConversion>(typeConverter, context, allocation,
                                             benefit);
  patterns.add<AsyncCommitGroupOpSPIRVConversion>(typeConverter, context,
                                                  benefit);
  patterns.add<AsyncWaitOpSPIRVConversion>(typeConverter, context, benefit);
  patterns.add<BroadcastOpSPIRVConversion>(typeConverter, context, benefit);

  patterns.add<ExtractSliceOpSPIRVConversion>(typeConverter, context,
                                              allocation, benefit);
  patterns.add<GetProgramIdOpToSPIRVConversion>(typeConverter, context,
                                                benefit);
  patterns.add<GetNumProgramsOpSPIRVConversion>(typeConverter, context,
                                                benefit);
  patterns.add<MakeRangeOpSPIRVConversion>(typeConverter, context,
                                           indexCacheInfo, benefit);
  patterns.add<ReturnOpSPIRVConversion>(typeConverter, context, benefit);
  //  patterns.add<PrintfOpConversion>(typeConverter, benefit);
  patterns.add<AssertOpSPIRVConversion>(typeConverter, context, benefit);
}
