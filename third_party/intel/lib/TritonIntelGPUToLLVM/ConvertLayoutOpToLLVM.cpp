#include "PatternTritonGPUOpToLLVM.h"

#include "llvm/ADT/TypeSwitch.h"

#include "intel/include/Analysis/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

namespace mlir::triton::gpu {
namespace {

struct ConvertLayoutOpUsingLinearLayoutsConversion
    : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  const TargetInfoBase &targetInfo;

  // Set benefit to 2 so that this pattern applies before other convert-layout
  // conversions.  TODO(jlebar): Eventually we want this to be the only pattern.
  explicit ConvertLayoutOpUsingLinearLayoutsConversion(
      LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern(typeConverter, benefit), targetInfo(targetInfo) {
  }

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();

    RankedTensorType srcTy = op.getSrc().getType();
    auto dstTy = op.getType();

    if (auto dstTensorTy = cast<RankedTensorType>(dstTy)) {
      if (intel::isBlockIONoOpConversion(srcTy, dstTensorTy)) {
        // TODO: replace this with proper conversion once conversion is removed
        // from LoadStoreOpToLLVM.
        rewriter.replaceOp(op, op.getSrc());
        return success();
      }
    }

    LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
    LinearLayout srcLayout =
        toLinearLayout(srcTy.getShape(), srcTy.getEncoding());
    LinearLayout dstLayout =
        toLinearLayout(dstTy.getShape(), dstTy.getEncoding());

    StringAttr kLane = str_attr("lane");

    assert(to_vector(conversion.getInDimNames()) ==
           to_vector(conversion.getOutDimNames()));
    auto dims = conversion.getInDimNames();
    if (llvm::is_contained(dims, kLane)) {
      // If the operation is a supported sub-group shuffle, perform via shuffle
      // operations.
      if (intel::cvtIsSubGroupShuffle(srcTy, dstTy)) {
        performSubGroupShuffle(op, srcLayout, dstLayout, adaptor, rewriter);
        return success();
      }
      // If the operation is a supported sub-group transposition, perform via
      // SLM.
      if (intel::cvtIsSubGroupTranspose(srcTy, dstTy)) {
        performSubGroupTranspose(op, srcLayout, dstLayout, adaptor, rewriter);
        return success();
      }
    }
    return failure();
  }

  int getNumContiguousRowsForShuffle(const LinearLayout &srcLayout,
                                     const LinearLayout &dstLayout) const {
    MLIRContext *ctx = getContext();

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp =
        *dstLayout.invertAndCompose(srcLayout).quotient({kWarp, kBlock});
    // Basic case: the number of contiguous rows is 1.
    if (comp.getBasis(kRegister, 0)[1] == 1)
      return 1;
    // In other case, we only allow all threads handled by a single element to
    // be contiguous, so we can simply:
    return comp.getOutDimSize(kRegister);
  }

  void performSubGroupShuffle(ConvertLayoutOp op, const LinearLayout &srcLayout,
                              const LinearLayout &dstLayout, OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    assert(intel::cvtIsSubGroupShuffle(op.getSrc().getType(), op.getType()) &&
           "Expecting sub-group shuffle");

    MLIRContext *ctx = op->getContext();
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp = dstLayout.invertAndCompose(srcLayout);
    LinearLayout conversion = *comp.quotient(kBlock)->quotient(kWarp);

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // FIXME: This workaround addresses the incorrect sgsize and SLM offset in
    // ReduceOp and ConvertLayoutOp, which prevents a segmentation fault.
    // However, this is a temporary solution. Once the OutDimSize computation
    // issue in LinearLayout is resolved, this workaround should be removed.
    int32_t numElems = std::min((int32_t)op.getType().getNumElements(),
                                conversion.getOutDimSize(kLane));
    int32_t subGroupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());
    if (!op->hasAttr("allocation.offset")) {
      op->setAttr("allocation.offset",
                  rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
    }

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    // TODO: Drop 'BFloat16Type' and 'IntegerType' cases when supported at MLIR
    // upstream level. We are not enabling support for all types here as that
    // should be done upstream.
    Type origElemTy = inVals.front().getType();
    TypeSwitch<Type>(origElemTy)
        .Case([&](BFloat16Type) {
          auto intTy = i16_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return b.bitcast(val, intTy);
          });
        })
        .Case([&](IntegerType intTy) {
          constexpr unsigned minWidth = 8;
          if (intTy.getWidth() >= minWidth)
            return;
          auto dstTy = i8_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return b.zext(dstTy, val);
          });
        })
        .Case([&](LLVM::LLVMPointerType) {
          Type dstType = i64_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return b.ptrtoint(dstType, val);
          });
        });

    SmallVector<Value> outVals = performSubGroupShuffle(
        loc, inVals, numElems, subGroupSize, rewriter,
        getNumContiguousRowsForShuffle(srcLayout, dstLayout));

    // TODO: Drop 'BFloat16Type' and 'IntegerType' cases when supported at MLIR
    // upstream level. We are not enabling support for all types here as that
    // should be done upstream.
    TypeSwitch<Type>(origElemTy)
        .Case([&](BFloat16Type) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return b.bitcast(val, origElemTy); });
        })
        .Case([&](IntegerType intTy) {
          // Check whether conversion took place.
          if (intTy == outVals.front().getType())
            return;
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return b.trunc(origElemTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType ptrTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return b.inttoptr(ptrTy, val); });
        });

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
  }

  SmallVector<Value> performSubGroupShuffle(Location loc,
                                            ArrayRef<Value> inVals,
                                            int32_t numElems,
                                            int32_t subGroupSize,
                                            ConversionPatternRewriter &rewriter,
                                            int numContiguousRows) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> res;
    Value width = b.i32_val(subGroupSize);
    // A work-item may handle more than one element. There are two cases we
    // support:
    if (numContiguousRows == 1) {
      // 1. Elements held by a work-item are strided rows in the abstract slice
      // matrix: Output element `i` will take the `i / 16`th value from the `i %
      // 16`th thread.
      for (Value val : inVals) {
        for (int32_t i = 0; i < numElems; ++i) {
          res.push_back(
              rewriter
                  .create<mlir::gpu::ShuffleOp>(loc, val, b.i32_val(i), width,
                                                mlir::gpu::ShuffleMode::IDX)
                  .getShuffleResult());
        }
      }
    } else {
      // 2. Elements held by a work-item are contiguous rows in the abstract
      // slice matrix: Output element `i` will take the `i % 16`th value from
      // the `i / 16`th thread.
      for (int32_t i = 0; i < numElems; ++i) {
        for (Value val : inVals) {
          res.push_back(
              rewriter
                  .create<mlir::gpu::ShuffleOp>(loc, val, b.i32_val(i), width,
                                                mlir::gpu::ShuffleMode::IDX)
                  .getShuffleResult());
        }
      }
    }
    return res;
  }

  int getNumContiguousRowsForTranspose(const LinearLayout &srcLayout,
                                       const LinearLayout &dstLayout) const {
    MLIRContext *ctx = getContext();

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");
    LinearLayout comp =
        *dstLayout.invertAndCompose(srcLayout).quotient({kWarp, kBlock});
    // Basic case: the number of contiguous rows is 0.
    if (comp.getBasis(kLane, 0)[0] == 1)
      return 1;
    // In other case, we only allow all threads handled by a single element to
    // be contiguous, so we can simply:
    int32_t sizePerThread = comp.getOutDimSize(kRegister);
    int32_t threadsPerWarp = comp.getOutDimSize(kLane);
    assert(sizePerThread % threadsPerWarp == 0 && "Invalid transpose");
    return sizePerThread / threadsPerWarp;
  }

  void performSubGroupTranspose(ConvertLayoutOp op,
                                const LinearLayout &srcLayout,
                                const LinearLayout &dstLayout,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    assert(intel::cvtIsSubGroupTranspose(op.getSrc().getType(), op.getType()) &&
           "Expecting sub-group transpose");

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> inVals =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);

    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    Type origElemTy = inVals.front().getType();

    TypeSwitch<Type>(origElemTy)
        .Case([&](FloatType floatTy) {
          // TODO: Support FP4.
          Type dstType = int_ty(floatTy.getWidth());
          assert(intel::isValidElementTypeForSubGroupTranspose(dstType) &&
                 "Expecting valid type");
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return b.bitcast(val, dstType);
          });
        })
        .Case([&](IntegerType intTy) {
          if (intel::isValidElementTypeForSubGroupTranspose(intTy))
            return;
          Type dstType = i8_ty;
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return b.zext(dstType, val);
          });
        })
        .Case([&](LLVM::LLVMPointerType) {
          Type dstType = i64_ty;
          assert(intel::isValidElementTypeForSubGroupTranspose(dstType) &&
                 "i64 type should be supported");
          llvm::transform(inVals, std::begin(inVals), [&](Value val) -> Value {
            return b.ptrtoint(dstType, val);
          });
        })
        .Default([](auto) { llvm_unreachable("Unsupported type"); });

    SmallVector<Value> outVals = performSubGroupTranspose(
        loc, inVals, rewriter,
        getNumContiguousRowsForTranspose(srcLayout, dstLayout));

    TypeSwitch<Type>(origElemTy)
        .Case([&](FloatType floatTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return b.bitcast(val, origElemTy); });
        })
        .Case([&](IntegerType intTy) {
          // Check whether conversion took place.
          if (intTy == outVals.front().getType())
            return;
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return b.trunc(origElemTy, val); });
        })
        .Case([&](LLVM::LLVMPointerType ptrTy) {
          llvm::transform(
              outVals, std::begin(outVals),
              [&](Value val) -> Value { return b.inttoptr(ptrTy, val); });
        })
        .Default([](auto) { llvm_unreachable("Unsupported type"); });

    Value result = packLLElements(loc, getTypeConverter(), outVals, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
  }

  SmallVector<Value>
  unwrapFromVectors(Location loc, ArrayRef<Value> vecs,
                    ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Value> res;
    for (Value vec : vecs) {
      for (unsigned i = 0, n = cast<VectorType>(vec.getType()).getShape()[0];
           i < n; ++i)
        res.push_back(b.extract_element(vec, b.i32_val(i)));
    }
    return res;
  }

  static unsigned getVecLoadWidth(unsigned threadsPerWarp) {
    assert(llvm::isPowerOf2_32(threadsPerWarp) &&
           "Expecting power of 2 sub-group size");
    constexpr unsigned maxVecWidth = 16;
    return std::min(maxVecWidth, threadsPerWarp);
  }

  SmallVector<Value>
  performSubGroupTranspose(Location loc, ArrayRef<Value> inVals,
                           ConversionPatternRewriter &rewriter,
                           int numContiguousRows) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type elementType = inVals.front().getType();
    auto mod = rewriter.getInsertionPoint()->getParentOfType<ModuleOp>();

    Value smemBase = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                               &*rewriter.getInsertionPoint());
    Type ptrType = smemBase.getType();

    int numRows = inVals.size();
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    // Add an element that won't be accessed at the end of the row to avoid bank
    // conflicts.
    int rowLength = threadsPerWarp + 1;
    Type offsetType = getTypeConverter()->getIndexType();
    unsigned offsetBitWidth = offsetType.getIntOrFloatBitWidth();
    Value subGroupId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::SubgroupIdOp>(
            loc, /*upper_bound=*/IntegerAttr{}));
    Value subGroupLocalId = getValueOrCreateCastToIndexLike(
        rewriter, loc, offsetType,
        rewriter.create<mlir::gpu::LaneIdOp>(loc,
                                             /*upper_bound=*/IntegerAttr{}));
    Value subGroupOffset =
        b.mul(subGroupId, b.int_val(offsetBitWidth, rowLength * numRows));
    Value subGroupBasePtr =
        b.gep(ptrType, elementType, smemBase, ValueRange{subGroupOffset},
              LLVM::GEPNoWrapFlags::inbounds);
    Value base = subGroupBasePtr;
    // Store in matrix, transposed
    for (Value val : inVals) {
      rewriter.create<TritonGEN::SubGroupBlockWriteOp>(loc, base, val);
      base = b.gep(base.getType(), elementType, base,
                   ArrayRef<LLVM::GEPArg>{rowLength},
                   LLVM::GEPNoWrapFlags::inbounds);
    }

    // Load from matrix, non-trasposed.

    // Each work-item will load a row (but the last garbage element) and go to
    // the next row it needs to handle.

    int32_t workItemStride =
        numContiguousRows == 1 ? rowLength * threadsPerWarp : rowLength;
    Value workItemOffset =
        b.mul(subGroupLocalId,
              b.int_val(offsetBitWidth, numContiguousRows * rowLength));
    Value workItemBasePtr =
        b.gep(ptrType, elementType, subGroupBasePtr, ValueRange{workItemOffset},
              LLVM::GEPNoWrapFlags::inbounds);
    int32_t rowsPerThread = numRows / threadsPerWarp;
    assert((numContiguousRows == 1 || numContiguousRows == rowsPerThread) &&
           "In case of more than one contiguous rows per thread, these must be "
           "consecutive");
    // We may not be able to load rows in a single operation if the sub-group
    // size exceeds a given threshold (16):
    unsigned vecLoadWidth = getVecLoadWidth(threadsPerWarp);
    SmallVector<Value> transposedVecs;
    VectorType vecType = vec_ty(elementType, vecLoadWidth);
    assert(threadsPerWarp % vecLoadWidth == 0 &&
           "Column must be loadable with N loads");
    for (unsigned i = 0; i < rowsPerThread; ++i) {
      for (unsigned j = 0; j < threadsPerWarp; j += vecLoadWidth) {
        transposedVecs.push_back(b.load(vecType, workItemBasePtr));
        workItemBasePtr =
            b.gep(workItemBasePtr.getType(), vecType, workItemBasePtr,
                  ArrayRef<LLVM::GEPArg>{1}, LLVM::GEPNoWrapFlags::inbounds);
      }
      workItemBasePtr =
          b.gep(workItemBasePtr.getType(), elementType, workItemBasePtr,
                // "Go back" to the first column and increment by the stride.
                ArrayRef<LLVM::GEPArg>{workItemStride - threadsPerWarp},
                LLVM::GEPNoWrapFlags::inbounds);
    }
    return unwrapFromVectors(loc, transposedVecs, rewriter);
  }
};

struct ConvertLayoutOpGuard : public ConvertOpToLLVMPattern<ConvertLayoutOp> {
  using ConvertOpToLLVMPattern<ConvertLayoutOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getType();
    assert(!intel::cvtIsSubGroupShuffle(srcTy, dstTy) &&
           "Failed to lower layout conversion through sub-group shuffles");
    assert(!intel::cvtIsSubGroupTranspose(srcTy, dstTy) &&
           "Failed to lower layout conversion through sub-group transpose");
    return failure();
  }
};

} // namespace

} // namespace mlir::triton::gpu

void mlir::triton::intel::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // We prefer using the Intel specific linear layout conversion, so it gets a
  // higher benefit.
  patterns.add<gpu::ConvertLayoutOpUsingLinearLayoutsConversion>(
      typeConverter, targetInfo, benefit.getBenefit() + 2);
  // This guards is to make sure we don't fall back to the generic patterns
  // for some specific cases. We check that we've lowered all those cases
  // for which the default allocation analysis for scratch buffer size was
  // not used. Otherwise, SLM corruption might occur.
  patterns.add<gpu::ConvertLayoutOpGuard>(typeConverter,
                                          benefit.getBenefit() + 1);
  mlir::triton::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                                      patterns, benefit);
}
