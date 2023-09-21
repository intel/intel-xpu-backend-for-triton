#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"

#include "LoadStoreOpToSPIRV.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreSPIRVConversionBase {
  explicit LoadStoreSPIRVConversionBase(
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : axisAnalysisPass(axisAnalysisPass) {}

  unsigned getContiguity(Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return 1;
    return axisAnalysisPass.getPtrContiguity(ptr);
  }

  unsigned getVectorSize(Value ptr) const {
    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return 1;
    auto contiguity = getContiguity(ptr);
    auto pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
    // The maximum vector size is 128 bits on NVIDIA GPUs.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

struct LoadOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::LoadOp>,
      public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::LoadOp>::ConvertTritonGPUOpToSPIRVPattern;

  LoadOpSPIRVConversion(TritonGPUToSPIRVTypeConverter &converter,
                        MLIRContext *context,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::LoadOp>(converter, context,
                                                         benefit),
        LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    Value spirvPtr = adaptor.getPtr();
    Value spirvMask = adaptor.getMask();
    Value spirvOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getResult().getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());
    if (spirvMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the SPIRV values for pointers
    auto ptrElems = getTypeConverter()->unpackLLElements(
        loc, spirvPtr, rewriter, ptr.getType());
    assert(ptrElems.size() == numElems);

    // Get the SPIRV values for mask
    SmallVector<Value> maskElems;
    if (spirvMask) {
      maskElems = getTypeConverter()->unpackLLElements(loc, spirvMask, rewriter,
                                                       mask.getType());
      assert(maskElems.size() == numElems);
    }

    bool otherIsSplatConstInt = false;
    DenseElementsAttr constAttr;
    int64_t splatVal = 0;
    if (other && valueElemTy.isa<IntegerType>() &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        constAttr.getElementType().isa<IntegerType>()) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    SmallVector<Value> otherElems;
    if (other) {
      otherElems = getTypeConverter()->unpackLLElements(
          loc, spirvOther, rewriter, other.getType());
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == numElems);

      SmallVector<Type> retTys(nWords, IntegerType::get(getContext(), width));
      Type retTy =
          retTys.size() > 1
              ? mlir::VectorType::get({(int64_t)nWords},
                                      IntegerType::get(getContext(), width))
              : retTys[0];

      Value pred = mask ? maskElems[vecStart] : int_val(1, 1);

      // scalar load
      // Create block structure for the masked load.
      auto *preheader = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *tailblock = rewriter.splitBlock(preheader, opPosition);
      tailblock->addArgument(retTy, loc);
      auto *condblock = rewriter.createBlock(tailblock);

      // Test the mask
      rewriter.setInsertionPoint(preheader, preheader->end());

      // Prediction false to use the other value.
      Value other_ = undef(retTy);
      if (other) {
        for (size_t ii = 0; ii < nWords; ++ii) {
          size_t size = width / valueElemNBits;

          Value v;
          if (size > 1) {
            Type vecTy = mlir::VectorType::get({(int64_t)size}, valueElemTy);
            v = undef(vecTy);
            for (size_t s = 0; s < size; ++s) {
              Value falseVal = otherElems[vecStart + ii * size + s];
              v = insert_val(vecTy, falseVal, v, rewriter.getI32ArrayAttr(s));
            }
            v = bitcast(v, IntegerType::get(getContext(), width));
          } else {
            Value falseVal = otherElems[vecStart + ii * size];
            v = bitcast(falseVal, IntegerType::get(getContext(), width));
          }

          if (nWords > 1)
            other_ = insert_val(retTy, v, other_, rewriter.getI32ArrayAttr(ii));
          else
            other_ = v;
        }
      }

      rewriter.create<mlir::cf::CondBranchOp>(loc, pred, condblock, tailblock,
                                              ValueRange{other_});

      // Do the load
      rewriter.setInsertionPoint(condblock, condblock->end());

      Value ptrElem =
          bitcast(ptrElems[vecStart],
                  ptr_ty(retTy, spirv::StorageClass::CrossWorkgroup));

      uint32_t alignment = nWords * width / 8;
      Value ret = rewriter.create<spirv::LoadOp>(
          loc, ptrElem,
          spirv::MemoryAccessAttr::get(rewriter.getContext(),
                                       spirv::MemoryAccess::Aligned),
          rewriter.getI32IntegerAttr(alignment));
      rewriter.create<mlir::cf::BranchOp>(loc, tailblock, ValueRange{ret});

      rewriter.setInsertionPoint(tailblock, tailblock->begin());

      ret = *tailblock->args_begin();
      // Extract and store return values
      SmallVector<Value> rets;
      int elemsPerWord = width / valueElemNBits;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr;
        if (retTy.isa<mlir::VectorType>()) {
          curr = extract_val(IntegerType::get(getContext(), width), ret,
                             rewriter.getI32ArrayAttr(ii));
        } else {
          curr = ret;
        }
        if (elemsPerWord > 1)
          curr = bitcast(curr, mlir::VectorType::get({(int64_t)elemsPerWord},
                                                     valueElemTy));
        else
          curr = bitcast(curr, valueElemTy);
        rets.push_back(curr);
      }

      for (size_t ii = 0; ii < vec; ++ii) {
        Value loaded;
        if (elemsPerWord > 1)
          loaded = extract_val(valueElemTy, rets[ii / elemsPerWord],
                               rewriter.getI32ArrayAttr(ii % elemsPerWord));
        else
          loaded = rets[ii / elemsPerWord];
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type spirvResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = getTypeConverter()->packLLElements(
        loc, loadedVals, rewriter, spirvResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct StoreOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::StoreOp>,
      public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::StoreOp>::ConvertTritonGPUOpToSPIRVPattern;

  StoreOpSPIRVConversion(TritonGPUToSPIRVTypeConverter &converter,
                         MLIRContext *context,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::StoreOp>(converter, context,
                                                          benefit),
        LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();

    Value spirvPtr = adaptor.getPtr();
    Value spirvMask = adaptor.getMask();
    Value spirvValue = adaptor.getValue();

    auto loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    unsigned vec = getVectorSize(ptr);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = getTypeConverter()->unpackLLElements(
        loc, spirvPtr, rewriter, ptr.getType());
    auto valueElems = getTypeConverter()->unpackLLElements(
        loc, spirvValue, rewriter, value.getType());
    assert(ptrElems.size() == valueElems.size());

    // Determine the vectorization size
    SmallVector<Value> maskElems;
    if (spirvMask) {
      Value mask = op.getMask();
      maskElems = getTypeConverter()->unpackLLElements(loc, spirvMask, rewriter,
                                                       mask.getType());
      assert(valueElems.size() == maskElems.size());

      unsigned maskAlign = getMaskAlignment(mask);
      vec = std::min(vec, maskAlign);
    }

    // numElements = 1 for scalar
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    auto numElems = tensorTy ? tensorTy.getNumElements() : 1;
    Value mask = getMask(valueTy, rewriter, loc);
    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      Value maskVal = spirvMask ? and_(mask, maskElems[vecStart]) : mask;

      // scalar store
      // Create block structure for the masked load.
      auto *preheader = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *tailblock = rewriter.splitBlock(preheader, opPosition);
      auto *condblock = rewriter.createBlock(tailblock);

      // Test the mask
      rewriter.setInsertionPoint(preheader, preheader->end());
      rewriter.create<mlir::cf::CondBranchOp>(loc, maskVal, condblock,
                                              tailblock);

      // Do the Store
      rewriter.setInsertionPoint(condblock, condblock->end());

      Type valArgTy = IntegerType::get(ctx, width);
      Type valueVectorTy =
          nWords > 1 ? mlir::VectorType::get({(int64_t)nWords}, valArgTy)
                     : valArgTy;

      auto wordTy =
          wordNElems > 1 ? vec_ty(valueElemTy, wordNElems) : valueElemTy;

      Value valToStore = undef(valueVectorTy);
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        Value spirvWord = undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1)) {
            elem = sext(i8_ty, elem);
          }
          if (wordNElems > 1)
            spirvWord =
                insert_element(wordTy, spirvWord, elem, i32_val(elemIdx));
          else
            spirvWord = elem;
        }
        spirvWord = bitcast(spirvWord, valArgTy);

        if (nWords > 1)
          valToStore = insert_element(valueVectorTy, valToStore, spirvWord,
                                      i32_val(wordIdx));
        else
          valToStore = spirvWord;
      }

      Value ptrElem =
          bitcast(ptrElems[vecStart],
                  ptr_ty(valueVectorTy, spirv::StorageClass::CrossWorkgroup));

      uint32_t alignment = nWords * width / 8;
      rewriter.create<spirv::StoreOp>(
          loc, ptrElem, valToStore,
          spirv::MemoryAccessAttr::get(rewriter.getContext(),
                                       spirv::MemoryAccess::Aligned),
          rewriter.getI32IntegerAttr(alignment));
      rewriter.create<mlir::cf::BranchOp>(loc, tailblock);

      rewriter.setInsertionPoint(tailblock, tailblock->begin());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AtomicCASOp>,
      public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToSPIRVPattern;

  AtomicCASOpSPIRVConversion(TritonGPUToSPIRVTypeConverter &converter,
                             MLIRContext *context, ModuleAllocation &allocation,
                             ModuleAxisInfoAnalysis &axisAnalysisPass,
                             PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::AtomicCASOp>(
            converter, context, allocation, benefit),
        LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    Value spirvPtr = adaptor.getPtr();
    Value spirvCmp = adaptor.getCmp();
    Value spirvVal = adaptor.getVal();

    auto ptrElements = getTypeConverter()->unpackLLElements(
        loc, spirvPtr, rewriter, op.getPtr().getType());
    auto cmpElements = getTypeConverter()->unpackLLElements(
        loc, spirvCmp, rewriter, op.getCmp().getType());
    auto valElements = getTypeConverter()->unpackLLElements(
        loc, spirvVal, rewriter, op.getVal().getType());

    auto TensorTy = op.getResult().getType().dyn_cast<RankedTensorType>();
    Type valueElemTy =
        TensorTy ? getTypeConverter()->convertType(TensorTy.getElementType())
                 : op.getResult().getType();
    auto tid = tid_val();
    Value pred = icmp_eq(tid, i32_val(0));

    Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
    atomPtr =
        bitcast(atomPtr, ptr_ty(valueElemTy, spirv::StorageClass::Workgroup));

    Value casPtr = ptrElements[0];
    Value casCmp = cmpElements[0];
    Value casVal = valElements[0];

    // Create block structure for the prediction of the cas.
    auto *preheader = rewriter.getInsertionBlock();
    auto opPosition = rewriter.getInsertionPoint();
    auto *tailblock = rewriter.splitBlock(preheader, opPosition);
    tailblock->addArgument(valueElemTy, loc);
    auto *condblock = rewriter.createBlock(tailblock);

    // Test the prediction
    rewriter.setInsertionPoint(preheader, preheader->end());
    Value other = undef(valueElemTy);
    rewriter.create<mlir::cf::CondBranchOp>(loc, pred, condblock, tailblock,
                                            ValueRange{other});

    // Do the Atomic
    rewriter.setInsertionPoint(condblock, condblock->end());

    auto old = rewriter.create<spirv::AtomicCompareExchangeOp>(
        loc, valueElemTy, casPtr, mlir::spirv::Scope::Device,
        mlir::spirv::MemorySemantics::AcquireRelease |
            mlir::spirv::MemorySemantics::MakeAvailable |
            mlir::spirv::MemorySemantics::MakeVisible,
        mlir::spirv::MemorySemantics::AcquireRelease |
            mlir::spirv::MemorySemantics::MakeAvailable |
            mlir::spirv::MemorySemantics::MakeVisible,
        casVal, casCmp);

    rewriter.create<mlir::cf::BranchOp>(loc, tailblock, ValueRange{old});
    rewriter.setInsertionPoint(tailblock, tailblock->begin());

    // do the barrier on all thread.
    barrier();
    // get the old value from the bb arg.
    Value old_in_tail = *tailblock->args_begin();

    // Create block structure for the prediction of the store.
    preheader = rewriter.getInsertionBlock();
    opPosition = rewriter.getInsertionPoint();
    tailblock = rewriter.splitBlock(preheader, opPosition);
    condblock = rewriter.createBlock(tailblock);

    // Test the prediction
    rewriter.setInsertionPoint(preheader, preheader->end());
    rewriter.create<mlir::cf::CondBranchOp>(loc, pred, condblock, tailblock);

    // Do the store
    rewriter.setInsertionPoint(condblock, condblock->end());

    store(old_in_tail, atomPtr);

    rewriter.create<mlir::cf::BranchOp>(loc, tailblock);
    rewriter.setInsertionPoint(tailblock, tailblock->begin());
    barrier();
    Value ret = load(atomPtr);
    barrier();
    rewriter.replaceOp(op, {ret});
    return success();
  }
};

struct AtomicRMWOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::AtomicRMWOp>,
      public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::AtomicRMWOp>::ConvertTritonGPUOpToSPIRVPattern;

  AtomicRMWOpSPIRVConversion(TritonGPUToSPIRVTypeConverter &converter,
                             MLIRContext *context, ModuleAllocation &allocation,
                             ModuleAxisInfoAnalysis &axisAnalysisPass,
                             PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::AtomicRMWOp>(
            converter, context, allocation, benefit),
        LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value spirvPtr = adaptor.getPtr();
    Value spirvVal = adaptor.getVal();
    Value spirvMask = adaptor.getMask();

    auto valElements = getTypeConverter()->unpackLLElements(
        loc, spirvVal, rewriter, val.getType());
    auto ptrElements = getTypeConverter()->unpackLLElements(
        loc, spirvPtr, rewriter, ptr.getType());
    SmallVector<Value> maskElements;
    if (spirvMask)
      maskElements = getTypeConverter()->unpackLLElements(
          loc, spirvMask, rewriter, op.getMask().getType());

    auto valueTy = op.getResult().getType();
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // tensor
    if (tensorTy) {
      auto valTy = val.getType().cast<RankedTensorType>();
      if (valTy.getElementType().isF16()) {
        auto vec = getVectorSize(ptr);
        // We only do the fp16 atomic when it is able to be packed to 32 bits.
        if (vec > 1 && ((vec % 2) == 0))
          return rewriteFP16Atomic(op, adaptor, rewriter);
      }
    }
    // mask
    Value mask = getMask(valueTy, rewriter, loc);

    // vec = 1 for scalar
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += 1) {
      Value rmwVal = valElements[i];
      Value rmwPtr = ptrElements[i];
      Value rmwMask = spirvMask ? and_(mask, maskElements[i]) : mask;

      // Create block structure for the masked rmw.
      auto *preheader = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *tailblock = rewriter.splitBlock(preheader, opPosition);
      tailblock->addArgument(valueElemTy, loc);
      auto *condblock = rewriter.createBlock(tailblock);

      // Test the mask
      auto retType = valueElemTy;
      rewriter.setInsertionPoint(preheader, preheader->end());
      Value other = undef(retType);
      rewriter.create<mlir::cf::CondBranchOp>(loc, rmwMask, condblock,
                                              tailblock, ValueRange{other});

      // Do the Atomic
      rewriter.setInsertionPoint(condblock, condblock->end());

      Value ptrElem =
          bitcast(rmwPtr, ptr_ty(retType, spirv::StorageClass::CrossWorkgroup));
      Value ret;
      switch (atomicRmwAttr) {

#define DISPATCH(rwmop__, sprivop__)                                           \
  case (rwmop__):                                                              \
    ret = rewriter.create<sprivop__>(                                          \
        loc, retType, ptrElem, mlir::spirv::Scope::Device,                     \
        mlir::spirv::MemorySemantics::AcquireRelease |                         \
            mlir::spirv::MemorySemantics::MakeAvailable |                      \
            mlir::spirv::MemorySemantics::MakeVisible,                         \
        rmwVal);                                                               \
    break;

        DISPATCH(RMWOp::AND, spirv::AtomicAndOp);
        DISPATCH(RMWOp::OR, spirv::AtomicOrOp);
        DISPATCH(RMWOp::XOR, spirv::AtomicXorOp);
        DISPATCH(RMWOp::ADD, spirv::AtomicIAddOp);
        DISPATCH(RMWOp::FADD, spirv::EXTAtomicFAddOp);
        DISPATCH(RMWOp::MAX, spirv::AtomicSMaxOp);
        DISPATCH(RMWOp::MIN, spirv::AtomicSMinOp);
        DISPATCH(RMWOp::UMAX, spirv::AtomicUMaxOp);
        DISPATCH(RMWOp::UMIN, spirv::AtomicUMinOp);
        DISPATCH(RMWOp::XCHG, spirv::AtomicExchangeOp);

#undef DISPATCH

      default:
        return failure();
      }

      rewriter.create<mlir::cf::BranchOp>(loc, tailblock, ValueRange{ret});
      rewriter.setInsertionPoint(tailblock, tailblock->begin());

      ret = *tailblock->args_begin();
      if (tensorTy) {
        resultVals[i] = ret;
      } else {
        if (op->user_begin() == op->user_end()) {
          rewriter.replaceOp(op, {ret});
          return success();
        }
        Value atomPtr = getSharedMemoryBase(loc, rewriter, op.getOperation());
        atomPtr = bitcast(atomPtr,
                          ptr_ty(valueElemTy, spirv::StorageClass::Workgroup));
        // Only threads with rmwMask = True store the result
        store(ret, atomPtr);
        barrier();
        ret = load(atomPtr);
        barrier();
        rewriter.replaceOp(op, {ret});
      }
    }
    if (tensorTy) {
      Type structTy = getTypeConverter()->convertType(tensorTy);
      Value resultStruct = getTypeConverter()->packLLElements(
          loc, resultVals, rewriter, structTy);
      rewriter.replaceOp(op, {resultStruct});
    }
    return success();
  }

  LogicalResult rewriteFP16Atomic(triton::AtomicRMWOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value spirvPtr = adaptor.getPtr();
    Value spirvVal = adaptor.getVal();
    Value spirvMask = adaptor.getMask();

    auto valElements = getTypeConverter()->unpackLLElements(
        loc, spirvVal, rewriter, val.getType());
    auto ptrElements = getTypeConverter()->unpackLLElements(
        loc, spirvPtr, rewriter, ptr.getType());
    SmallVector<Value> maskElements;
    if (spirvMask)
      maskElements = getTypeConverter()->unpackLLElements(
          loc, spirvMask, rewriter, op.getMask().getType());

    auto valueTy = op.getResult().getType();
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto vec = getVectorSize(ptr);
    int numElems = tensorTy.getNumElements();
    // tensor
    auto valTy = val.getType().cast<RankedTensorType>();
    vec = std::min<unsigned>(vec, 2);
    // mask
    Value mask = getMask(valueTy, rewriter, loc);
    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwPtr = ptrElements[i];
      Value rmwMask = spirvMask ? and_(mask, maskElements[i]) : mask;

      // Create block structure for the masked rmw.
      auto *preheader = rewriter.getInsertionBlock();
      auto opPosition = rewriter.getInsertionPoint();
      auto *tailblock = rewriter.splitBlock(preheader, opPosition);
      tailblock->addArgument(vecTy, loc);
      auto *condblock = rewriter.createBlock(tailblock);

      // Test the mask
      auto retType = vecTy;
      rewriter.setInsertionPoint(preheader, preheader->end());
      Value other = undef(retType);
      rewriter.create<mlir::cf::CondBranchOp>(loc, rmwMask, condblock,
                                              tailblock, ValueRange{other});

      // Do the Atomic
      rewriter.setInsertionPoint(condblock, condblock->end());

      // fetch the value
      Value ptrElem =
          bitcast(rmwPtr, ptr_ty(retType, spirv::StorageClass::CrossWorkgroup));
      Value origin = rewriter.create<spirv::LoadOp>(loc, ptrElem);

      // do-while loop
      auto *loop = rewriter.splitBlock(rewriter.getInsertionBlock(),
                                       rewriter.getInsertionPoint());
      loop->addArgument(vecTy, loc);

      rewriter.create<mlir::cf::BranchOp>(loc, loop, ValueRange{origin});
      rewriter.setInsertionPoint(loop, loop->begin());
      origin = *loop->args_begin();

      Value modify = undef(retType);
      for (int ii = 0; ii < vec; ++ii) {
        Value loadVal = extract_element(origin, i32_val(ii));
        Value rmwVal = valElements[i + ii];
        switch (atomicRmwAttr) {

#define DISPATCH(rwmop__, sprivop__)                                           \
  case (rwmop__):                                                              \
    loadVal = rewriter.create<sprivop__>(loc, loadVal, rmwVal);                \
    break;

          DISPATCH(RMWOp::FADD, spirv::FAddOp);

#undef DISPATCH

        default:
          return failure();
        }
        modify = insert_element(retType, modify, loadVal, i32_val(ii));
      }

      Type writeTy = rewriter.getIntegerType(valueElemNBits * vec);
      modify = bitcast(modify, writeTy);

      Value ptrWrite =
          bitcast(rmwPtr, ptr_ty(writeTy, spirv::StorageClass::CrossWorkgroup));

      Value comparator = bitcast(origin, writeTy);
      Value exchange = rewriter.create<spirv::AtomicCompareExchangeOp>(
          loc, writeTy, ptrWrite, mlir::spirv::Scope::Device,
          mlir::spirv::MemorySemantics::AcquireRelease |
              mlir::spirv::MemorySemantics::MakeAvailable |
              mlir::spirv::MemorySemantics::MakeVisible,
          mlir::spirv::MemorySemantics::None, modify, comparator);
      Value equal = icmp_eq(comparator, exchange);

      rewriter.create<mlir::cf::CondBranchOp>(
          loc, equal, tailblock, ValueRange{bitcast(modify, retType)}, loop,
          ValueRange{bitcast(exchange, retType)});
      rewriter.setInsertionPoint(tailblock, tailblock->begin());

      Value ret = *tailblock->args_begin();

      for (int ii = 0; ii < vec; ++ii) {
        resultVals[i + ii] =
            vec == 1 ? ret : extract_element(valueElemTy, ret, i32_val(ii));
      }
    }

    Type structTy = getTypeConverter()->convertType(tensorTy);
    Value resultStruct =
        getTypeConverter()->packLLElements(loc, resultVals, rewriter, structTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

struct InsertSliceOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<tensor::InsertSliceOp> {
  using ConvertTritonGPUOpToSPIRVPattern<
      tensor::InsertSliceOp>::ConvertTritonGPUOpToSPIRVPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %dst = insert_slice %src into %dst[%offsets]
    Location loc = op->getLoc();
    Value dst = op.getDest();
    Value src = op.getSource();
    Value res = op.getResult();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    auto *funcAllocation = allocation->getFuncData(funcOp);
    assert(funcAllocation->getBufferId(res) == Allocation::InvalidBufferId &&
           "Only support in-place insert_slice for now");

    auto srcTy = src.getType().dyn_cast<RankedTensorType>();
    auto srcLayout = srcTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcLayout && "Unexpected srcLayout in InsertSliceOpSPIRVConversion");

    auto dstTy = dst.getType().dyn_cast<RankedTensorType>();
    auto dstLayout = dstTy.getEncoding().dyn_cast<SharedEncodingAttr>();
    auto spirvDst = adaptor.getDest();
    assert(dstLayout && "Unexpected dstLayout in InsertSliceOpSPIRVConversion");
    assert(op.hasUnitStride() &&
           "Only unit stride supported by InsertSliceOpSPIRVConversion");

    // newBase = base + offset
    // Triton support either static and dynamic offsets
    auto smemObj = getSharedMemoryObjectFromStruct(loc, spirvDst, rewriter);
    SmallVector<Value, 4> offsets;
    SmallVector<Value, 4> srcStrides;
    auto mixedOffsets = op.getMixedOffsets();
    for (auto i = 0; i < mixedOffsets.size(); ++i) {
      if (op.isDynamicOffset(i)) {
        offsets.emplace_back(adaptor.getOffsets()[i]);
      } else {
        offsets.emplace_back(i32_val(op.getStaticOffset(i)));
      }
      // Like insert_slice_async, we only support slice from one dimension,
      // which has a slice size of 1
      if (op.getStaticSize(i) != 1) {
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }

    // Compute the offset based on the original strides of the shared memory
    // object
    auto offset = dot(rewriter, loc, offsets, smemObj.strides);
    auto elemTy = getTypeConverter()->convertType(dstTy.getElementType());
    auto elemPtrTy = ptr_ty(elemTy, spirv::StorageClass::Workgroup);
    auto smemBase = gep(elemPtrTy, smemObj.base, offset);

    auto spirvSrc = adaptor.getSource();
    auto srcIndices = emitIndices(loc, rewriter, srcLayout, srcTy);
    storeDistributedToShared(src, spirvSrc, srcStrides, srcIndices, dst,
                             smemBase, elemTy, loc, rewriter);
    // Barrier is not necessary.
    // The membar pass knows that it writes to shared memory and will handle it
    // properly.
    rewriter.replaceOp(op, spirvDst);
    return success();
  }
};

struct InsertSliceAsyncOpSPIRVConversion
    : public ConvertTritonGPUOpToSPIRVPattern<triton::gpu::InsertSliceAsyncOp>,
      public LoadStoreSPIRVConversionBase {
  using ConvertTritonGPUOpToSPIRVPattern<
      triton::gpu::InsertSliceAsyncOp>::ConvertTritonGPUOpToSPIRVPattern;

  InsertSliceAsyncOpSPIRVConversion(
      TritonGPUToSPIRVTypeConverter &converter, MLIRContext *context,
      ModuleAllocation &allocation,
      ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass, PatternBenefit benefit)
      : ConvertTritonGPUOpToSPIRVPattern<triton::gpu::InsertSliceAsyncOp>(
            converter, context, allocation, indexCacheInfo, benefit),
        LoadStoreSPIRVConversionBase(axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::InsertSliceAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // insert_slice_async %src, %dst, %index, %mask, %other
    auto loc = op.getLoc();
    Value src = op.getSrc();
    Value dst = op.getDst();
    Value res = op.getResult();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    auto *funcAllocation = allocation->getFuncData(funcOp);
    assert(funcAllocation->getBufferId(res) == Allocation::InvalidBufferId &&
           "Only support in-place insert_slice_async for now");

    auto srcTy = src.getType().cast<RankedTensorType>();
    auto resTy = dst.getType().cast<RankedTensorType>();
    auto resElemTy = getTypeConverter()->convertType(resTy.getElementType());
    auto srcBlockedLayout = srcTy.getEncoding().cast<BlockedEncodingAttr>();
    auto resSharedLayout = resTy.getEncoding().cast<SharedEncodingAttr>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 &&
           "insert_slice_async: Unexpected rank of %src");

    Value spirvDst = adaptor.getDst();
    Value spirvSrc = adaptor.getSrc();
    Value spirvMask = adaptor.getMask();
    Value spirvOther = adaptor.getOther();
    Value spirvIndex = adaptor.getIndex();

    // %src
    auto srcElems = getTypeConverter()->unpackLLElements(
        loc, spirvSrc, rewriter, src.getType());

    // %dst
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    auto smemObj = getSharedMemoryObjectFromStruct(loc, spirvDst, rewriter);
    auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
    SmallVector<Value, 4> offsetVals;
    SmallVector<Value, 4> srcStrides;
    for (auto i = 0; i < dstShape.size(); ++i) {
      if (i == axis) {
        offsetVals.emplace_back(spirvIndex);
      } else {
        offsetVals.emplace_back(i32_val(0));
        srcStrides.emplace_back(smemObj.strides[i]);
      }
    }
    // Compute the offset based on the original dimensions of the shared
    // memory object
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    auto dstPtrTy = ptr_ty(resElemTy, spirv::StorageClass::Workgroup);
    Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);

    // %mask
    SmallVector<Value> maskElems;
    if (spirvMask) {
      maskElems = getTypeConverter()->unpackLLElements(loc, spirvMask, rewriter,
                                                       mask.getType());
      assert(srcElems.size() == maskElems.size());
    }

    // %other
    SmallVector<Value> otherElems;
    if (spirvOther) {
      // FIXME(Keren): always assume other is 0 for now
      // It's not necessary for now because the pipeline pass will skip
      // generating insert_slice_async if the load op has any "other" tensor.
      // assert(false && "insert_slice_async: Other value not supported yet");
      otherElems = getTypeConverter()->unpackLLElements(
          loc, spirvOther, rewriter, other.getType());
      assert(srcElems.size() == otherElems.size());
    }

    // We don't use getVec() here because we are copying from memory to memory.
    // If contiguity > vector size, we can have one pointer maintaining the
    // start of the vector and the other pointer moving to the next vector.
    unsigned inVec = getContiguity(src);
    unsigned outVec = resSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned numElems = getTotalElemsPerThread(srcTy);
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    auto sizePerThread = srcBlockedLayout.getSizePerThread();
    auto threadsPerCTA = getThreadsPerCTA(srcBlockedLayout);
    auto inOrder = srcBlockedLayout.getOrder();
    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, resSharedLayout, resElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    // If perPhase * maxPhase > threadsPerCTA, we will have elements
    // that share the same tile indices. The index calculation will
    // be cached.
    auto numSwizzleRows = std::max<unsigned>(
        (perPhase * maxPhase) / threadsPerCTA[inOrder[1]], 1);
    // A sharedLayout encoding has a "vec" parameter.
    // On the column dimension, if inVec > outVec, it means we have to divide
    // single vector read into multiple ones
    auto numVecCols = std::max<unsigned>(inVec / outVec, 1);

    auto srcIndices = emitIndices(loc, rewriter, srcBlockedLayout, srcTy);

    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // 16 * 8 = 128bits
      auto maxBitWidth =
          std::max<unsigned>(128, resElemTy.getIntOrFloatBitWidth());
      auto vecBitWidth = resElemTy.getIntOrFloatBitWidth() * minVec;
      auto bitWidth = std::min<unsigned>(maxBitWidth, vecBitWidth);
      auto numWords = vecBitWidth / bitWidth;
      auto numWordElems = bitWidth / resElemTy.getIntOrFloatBitWidth();
      auto byteWidth = bitWidth / 8;
      assert(byteWidth == 16 || byteWidth == 8 || byteWidth == 4);

      Type spirvElemTy;
      constexpr unsigned opaqueElemBitwidth = 32;
      if (bitWidth > opaqueElemBitwidth) {
        spirvElemTy =
            VectorType::get(ceil<unsigned>(bitWidth, opaqueElemBitwidth),
                            IntegerType::get(getContext(), opaqueElemBitwidth));
      } else {
        spirvElemTy = IntegerType::get(getContext(), bitWidth);
      }
      size_t nWords = ceil<unsigned>(bitWidth, opaqueElemBitwidth);

      Value basePtr = sharedPtrs[elemIdx];
      spirv::PointerType spirvBasePtrType =
          spirv::PointerType::get(spirvElemTy, spirv::StorageClass::Workgroup);
      basePtr = bitcast(basePtr, spirvBasePtrType);
      for (size_t wordIdx = 0; wordIdx < numWords; ++wordIdx) {
        Value spirvDestPtr = gep(spirvBasePtrType, basePtr, i32_val(wordIdx));

        auto wordElemIdx = wordIdx * numWordElems;
        auto srcPtr = srcElems[elemIdx + wordElemIdx];
        spirv::PointerType srcPtrType =
            srcPtr.getType().dyn_cast<spirv::PointerType>();
        spirv::PointerType spirvSrcPtrType =
            spirv::PointerType::get(spirvElemTy, srcPtrType.getStorageClass());
        Value spirvSrcPtr =
            bitcast(srcElems[elemIdx + wordElemIdx], spirvSrcPtrType);

        Value ret;
        if (spirvMask) {
          Value maskVal = maskElems[elemIdx];

          Value other_ = undef(spirvElemTy);
          ;
          for (size_t ii = 0; ii < nWords; ++ii) {
            if (nWords > 1) {
              Value v = int_val(opaqueElemBitwidth, 0);
              other_ = insert_val(spirvElemTy, v, other_,
                                  rewriter.getI32ArrayAttr(ii));
            } else {
              other_ = int_val(bitWidth, 0);
            }
          }

          // Create block structure for the masked memory copy.
          auto *preheader = rewriter.getInsertionBlock();
          auto opPosition = rewriter.getInsertionPoint();
          auto *tailblock = rewriter.splitBlock(preheader, opPosition);
          tailblock->addArgument(spirvElemTy, loc);
          auto *condblock = rewriter.createBlock(tailblock);

          // Test the mask
          rewriter.setInsertionPoint(preheader, preheader->end());
          rewriter.create<mlir::cf::CondBranchOp>(
              loc, maskVal, condblock, tailblock, ValueRange{other_});

          // Do the memory load block
          rewriter.setInsertionPoint(condblock, condblock->end());
          Value val = rewriter.create<spirv::LoadOp>(op.getLoc(), spirvSrcPtr);
          rewriter.create<mlir::cf::BranchOp>(loc, tailblock, ValueRange{val});

          // The memory copy insert position
          rewriter.setInsertionPoint(tailblock, tailblock->begin());

          ret = *tailblock->args_begin();
        } else {
          ret = rewriter.create<spirv::LoadOp>(op.getLoc(), spirvSrcPtr);
        }

        // Extract and store return values
        rewriter.create<spirv::StoreOp>(op.getLoc(), spirvDestPtr, ret);

        // the cp.async is treated as a weak memory operation in the CUDA memory
        // consistency model. So no explicit synchronization required here.
      }
    }

    rewriter.replaceOp(op, spirvDst);
    return success();
  }
};

void populateLoadStoreOpToSPIRVPatterns(
    TritonGPUToSPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
    RewritePatternSet &patterns, int numWarps,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit) {
  patterns.add<LoadOpSPIRVConversion>(typeConverter, context, axisInfoAnalysis,
                                      benefit);
  patterns.add<StoreOpSPIRVConversion>(typeConverter, context, axisInfoAnalysis,
                                       benefit);
  patterns.add<AtomicCASOpSPIRVConversion>(typeConverter, context, allocation,
                                           axisInfoAnalysis, benefit);
  patterns.add<AtomicRMWOpSPIRVConversion>(typeConverter, context, allocation,
                                           axisInfoAnalysis, benefit);
  patterns.add<InsertSliceOpSPIRVConversion>(typeConverter, context, allocation,
                                             indexCacheInfo, benefit);
  patterns.add<InsertSliceAsyncOpSPIRVConversion>(typeConverter, context,
                                                  allocation, indexCacheInfo,
                                                  axisInfoAnalysis, benefit);
}
