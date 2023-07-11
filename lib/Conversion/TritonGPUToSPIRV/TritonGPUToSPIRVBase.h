#ifndef TRITON_TRITONGPUTOSPIRVBASE_H
#define TRITON_TRITONGPUTOSPIRVBASE_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "triton/Analysis/Allocation.h"

#include "TypeConverter.h"
//
#include "DotOpHelpers.h" // This cannot be removed so far. The utility defined marco has conflict with SPIRV header.
#include "Utility.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/AxisInfo.h"
#include <set>
using namespace mlir;
using namespace mlir::triton;

using ::mlir::spirv::delinearize;
using ::mlir::spirv::SharedMemoryObject;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

namespace mlir {
namespace spirv {

// Helper function for using printf in spirv conversion.
void vprintf(StringRef msg, ValueRange args,
             ConversionPatternRewriter &rewriter);

void vprintf_array(Value thread, ArrayRef<Value> arr, std::string info,
                   std::string elem_repr, ConversionPatternRewriter &builder);

} // namespace spirv
} // namespace mlir

struct FuncOpConversionBase : public OpConversionPattern<triton::FuncOp> {
protected:
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

  /// Helper function for wrapping all attributes into a single DictionaryAttr
  static auto wrapAsStructAttrs(OpBuilder &b, ArrayAttr attrs) {
    return DictionaryAttr::get(b.getContext(),
                               b.getNamedAttr("llvm.struct_attrs", attrs));
  }

protected:
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  // Convert input FuncOp to LLVMFuncOp by using the LLVMTypeConverter provided
  // to this legalization pattern.
  spirv::FuncOp
  convertFuncOpToSPIRVFuncOp(triton::FuncOp funcOp,
                             ConversionPatternRewriter &rewriter) const {
    // Convert the original function arguments. They are converted using the
    // TypeConverter provided to this legalization pattern.
    auto varargsAttr = funcOp->getAttrOfType<BoolAttr>("func.varargs");
    if (varargsAttr && varargsAttr.getValue()) {
      funcOp->emitError() << "Conversion to SPIRV FuncOp doesn't support "
                             "variadic param function.";
      return nullptr;
    }

    auto fnType = funcOp.getFunctionType();
    TypeConverter::SignatureConversion result(fnType.getNumInputs());
    for (const auto &argType : enumerate(fnType.getInputs())) {
      auto convertedType = getTypeConverter()->convertType(argType.value());
      if (!convertedType) {
        funcOp->emitError()
            << "Conversion to SPIRV FuncOp meet unsupported type.";
        return nullptr;
      }
      result.addInputs(argType.index(), convertedType);
    }

    // Pack the result types into a struct.
    Type packedResultType;
    unsigned numResults = funcOp.getNumResults();
    if (numResults != 0) {
      auto resultTypes = llvm::to_vector<4>(funcOp.getResultTypes());
      if (numResults == 1) {
        packedResultType = getTypeConverter()->convertType(resultTypes.front());
      } else {
        SmallVector<Type> convertedTypes;
        for (auto t : resultTypes) {
          auto converted = getTypeConverter()->convertType(t);
          if (!converted) {
            funcOp->emitError()
                << "Conversion to SPIRV FuncOp meet unsupported type.";
            return nullptr;
          }
          convertedTypes.push_back(converted);
        }
        packedResultType = spirv::StructType::get(convertedTypes);
      }
    }

    auto spirvType = rewriter.getFunctionType(
        result.getConvertedTypes(),
        packedResultType ? TypeRange(packedResultType) : TypeRange());

    // Propagate argument/result attributes to all converted arguments/result
    // obtained after converting a given original argument/result.
    SmallVector<NamedAttribute, 4> attributes;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, attributes);
    if (ArrayAttr resAttrDicts = funcOp.getAllResultAttrs()) {
      assert(!resAttrDicts.empty() && "expected array to be non-empty");
      auto newResAttrDicts =
          (funcOp.getNumResults() == 1)
              ? resAttrDicts
              : rewriter.getArrayAttr(
                    {wrapAsStructAttrs(rewriter, resAttrDicts)});
      attributes.push_back(
          rewriter.getNamedAttr(funcOp.getResAttrsAttrName(), newResAttrDicts));
    }
    if (ArrayAttr argAttrDicts = funcOp.getAllArgAttrs()) {
      SmallVector<Attribute, 4> newArgAttrs(spirvType.getNumInputs());
      for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
        auto mapping = result.getInputMapping(i);
        assert(mapping && "unexpected deletion of function argument");
        for (size_t j = 0; j < mapping->size; ++j)
          newArgAttrs[mapping->inputNo + j] = argAttrDicts[i];
      }
      attributes.push_back(rewriter.getNamedAttr(
          funcOp.getArgAttrsAttrName(), rewriter.getArrayAttr(newArgAttrs)));
    }
    for (const auto &pair : llvm::enumerate(attributes)) {
      if (pair.value().getName() == "llvm.linkage") {
        attributes.erase(attributes.begin() + pair.index());
        break;
      }
    }

    // Create an SPIRV function, use external linkage by default until MLIR
    // functions have linkage.
    spirv::FunctionControl linkage = spirv::FunctionControl::None;
    if (funcOp->hasAttr("llvm.linkage")) {
      funcOp->emitError() << "Contains llvm.linkage attribute not in SPIRV";
      return nullptr;
#if 0
      auto attr =
              funcOp->getAttr("llvm.linkage").dyn_cast<mlir::LLVM::LinkageAttr>();
      if (!attr) {
        funcOp->emitError()
                << "Contains llvm.linkage attribute not of type LLVM::LinkageAttr";
        return nullptr;
      }
      linkage = attr.getLinkage();
#endif
      // TODO: Need to map the llvm link attribute to SPIRV Function Control.
#if 0
      // LLVM linkage types
      enum class Linkage : uint64_t {
        Private = 0,
        Internal = 1,
        AvailableExternally = 2,
        Linkonce = 3,
        Weak = 4,
        Common = 5,
        Appending = 6,
        ExternWeak = 7,
        LinkonceODR = 8,
        WeakODR = 9,
        External = 10,
      };
      enum class FunctionControl : uint32_t {
        None = 0,
        Inline = 1,
        DontInline = 2,
        Pure = 4,
        Const = 8,
        OptNoneINTEL = 65536,
      };
#endif
    }

    auto newFuncOp = rewriter.create<spirv::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), spirvType, linkage, attributes);

    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &result)))
      return nullptr;

    return newFuncOp;
  }
};

using IndexCacheKeyT = std::pair<Attribute, RankedTensorType>;

struct CacheKeyDenseMapInfo {
  static IndexCacheKeyT getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return std::make_pair(
        mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
        RankedTensorType{});
  }
  static IndexCacheKeyT getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    auto tombstone = llvm::DenseMapInfo<RankedTensorType>::getTombstoneKey();
    return std::make_pair(
        mlir::Attribute(static_cast<mlir::Attribute::ImplType *>(pointer)),
        tombstone);
  }
  static unsigned getHashValue(IndexCacheKeyT key) {
    auto shape = key.second.getShape();
    return llvm::hash_combine(mlir::hash_value(key.first),
                              mlir::hash_value(key.second));
  }
  static bool isEqual(IndexCacheKeyT LHS, IndexCacheKeyT RHS) {
    return LHS == RHS;
  }
};

class ConvertTritonGPUOpToSPIRVPatternBase {
public:
  // Two levels of value cache in emitting indices calculation:
  // Key: pair<layout, shape>
  struct IndexCacheInfo {
    DenseMap<IndexCacheKeyT, SmallVector<Value>, CacheKeyDenseMapInfo>
        *baseIndexCache;
    DenseMap<IndexCacheKeyT, SmallVector<SmallVector<Value>>,
             CacheKeyDenseMapInfo> *indexCache;
    OpBuilder::InsertPoint *indexInsertPoint;
  };

  explicit ConvertTritonGPUOpToSPIRVPatternBase(
      TritonGPUToSPIRVTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  explicit ConvertTritonGPUOpToSPIRVPatternBase(
      TritonGPUToSPIRVTypeConverter &typeConverter,
      IndexCacheInfo indexCacheInfo)
      : converter(&typeConverter), indexCacheInfo(indexCacheInfo) {}

  explicit ConvertTritonGPUOpToSPIRVPatternBase(
      TritonGPUToSPIRVTypeConverter &typeConverter,
      ModuleAllocation &allocation)
      : converter(&typeConverter), allocation(&allocation) {}

  explicit ConvertTritonGPUOpToSPIRVPatternBase(
      TritonGPUToSPIRVTypeConverter &typeConverter,
      ModuleAllocation &allocation, IndexCacheInfo indexCacheInfo)
      : converter(&typeConverter), indexCacheInfo(indexCacheInfo),
        allocation(&allocation) {}

  TritonGPUToSPIRVTypeConverter *getTypeConverter() const { return converter; }

  static Value
  getStructFromSharedMemoryObject(Location loc,
                                  const SharedMemoryObject &smemObj,
                                  ConversionPatternRewriter &rewriter) {
    auto elems = smemObj.getElems();
    auto types = smemObj.getTypes();
    auto structTy = spirv::StructType::get(types);
    // pack into struct
    Value spirvStruct = rewriter.create<spirv::UndefOp>(loc, structTy);
    for (const auto &v : llvm::enumerate(elems)) {
      assert(v.value() && "can not insert null values");
      spirvStruct = insert_val(structTy, v.value(), spirvStruct,
                               rewriter.getI32ArrayAttr(v.index()));
    }
    return spirvStruct;
  }

  Value getThreadId(ConversionPatternRewriter &rewriter, Location loc) const {
    auto spirvIndexTy = this->getTypeConverter()->getIndexType();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{spirvIndexTy},
        ValueRange{rewriter.create<::mlir::gpu::ThreadIdOp>(
            loc, rewriter.getIndexType(), ::mlir::gpu::Dimension::x)});

    Value threadId = rewriter.create<::mlir::arith::TruncIOp>(
        loc, i32_ty, cast.getResult(0));
    return threadId;
  }

  Value getProgramId(ConversionPatternRewriter &rewriter, Location loc,
                     mlir::gpu::Dimension dim = mlir::gpu::Dimension::x) const {
    auto spirvIndexTy = this->getTypeConverter()->getIndexType();
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{spirvIndexTy},
        ValueRange{rewriter.create<::mlir::gpu::BlockIdOp>(
            loc, rewriter.getIndexType(), dim)});

    Value threadId = rewriter.create<::mlir::arith::TruncIOp>(
        loc, i32_ty, cast.getResult(0));
    return threadId;
  }

  // -----------------------------------------------------------------------
  // Shared memory utilities
  // -----------------------------------------------------------------------
  template <typename T>
  Value getSharedMemoryBase(Location loc, ConversionPatternRewriter &rewriter,
                            T value) const {
    auto ptrTy = spirv::PointerType::get(
        this->getTypeConverter()->convertType(rewriter.getI8Type()),
        spirv::StorageClass::Workgroup);
    FunctionOpInterface funcOp;
    if constexpr (std::is_pointer_v<T>)
      funcOp = value->template getParentOfType<FunctionOpInterface>();
    else
      funcOp = value.getParentRegion()
                   ->template getParentOfType<FunctionOpInterface>();
    auto *funcAllocation = allocation->getFuncData(funcOp);
    auto smem = allocation->getFunctionSharedMemoryBase(funcOp);
    auto bufferId = funcAllocation->getBufferId(value);
    assert(bufferId != Allocation::InvalidBufferId && "BufferId not found");
    size_t offset = funcAllocation->getOffset(bufferId);
    Value offVal = i32_val(offset);
    Value base = gep(ptrTy, smem, offVal);
    return base;
  }

  DenseMap<unsigned, Value>
  getSwizzledSharedPtrs(Location loc, unsigned inVec, RankedTensorType srcTy,
                        triton::gpu::SharedEncodingAttr resSharedLayout,
                        Type resElemTy, SharedMemoryObject smemObj,
                        ConversionPatternRewriter &rewriter,
                        SmallVectorImpl<Value> &offsetVals,
                        SmallVectorImpl<Value> &srcStrides) const {
    // This utililty computes the pointers for accessing the provided swizzled
    // shared memory layout `resSharedLayout`. More specifically, it computes,
    // for all indices (row, col) of `srcEncoding` such that idx % inVec = 0,
    // the pointer: ptr[(row, col)] = base + (rowOff * strides[ord[1]] +
    // colOff) where :
    //   compute phase = (row // perPhase) % maxPhase
    //   rowOff = row
    //   colOff = colOffSwizzled + colOffOrdered
    //     colOffSwizzled = ((col // outVec) ^ phase) * outVec
    //     colOffOrdered = (col % outVec) // minVec * minVec
    //
    // Note 1:
    // -------
    // Because swizzling happens at a granularity of outVec, we need to
    // decompose the offset into a swizzled factor and a non-swizzled
    // (ordered) factor
    //
    // Note 2:
    // -------
    // If we have x, y, z of the form:
    // x = 0b00000xxxx
    // y = 0byyyyy0000
    // z = 0b00000zzzz
    // then (x + y) XOR z = 0byyyyxxxx XOR 0b00000zzzz = (x XOR z) + y
    // This means that we can use some immediate offsets for shared memory
    // operations.
    auto dstPtrTy = ptr_ty(resElemTy, spirv::StorageClass::Workgroup);
    auto dstOffset = dot(rewriter, loc, offsetVals, smemObj.strides);
    Value dstPtrBase = gep(dstPtrTy, smemObj.base, dstOffset);

    auto srcEncoding = srcTy.getEncoding();
    auto srcShape = srcTy.getShape();
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    // swizzling params as described in TritonGPUAttrDefs.td
    unsigned outVec = resSharedLayout.getVec();
    unsigned perPhase = resSharedLayout.getPerPhase();
    unsigned maxPhase = resSharedLayout.getMaxPhase();
    // order
    auto inOrder = triton::gpu::getOrder(srcEncoding);
    auto outOrder = triton::gpu::getOrder(resSharedLayout);
    // tensor indices held by the current thread, as LLVM values
    auto srcIndices = emitIndices(loc, rewriter, srcEncoding, srcTy);
    // return values
    DenseMap<unsigned, Value> ret;
    // cache for non-immediate offsets
    DenseMap<unsigned, Value> cacheCol, cacheRow;
    unsigned minVec = std::min(outVec, inVec);
    for (unsigned elemIdx = 0; elemIdx < numElems; elemIdx += minVec) {
      // extract multi dimensional index for current element
      auto idx = srcIndices[elemIdx];
      Value idxCol = idx[outOrder[0]]; // contiguous dimension
      Value idxRow = idx[outOrder[1]]; // discontiguous dimension
      Value strideCol = srcStrides[outOrder[0]];
      Value strideRow = srcStrides[outOrder[1]];
      // extract dynamic/static offset for immediate offsetting
      unsigned immedateOffCol = 0;
      if (auto add = dyn_cast_or_null<spirv::IAddOp>(idxCol.getDefiningOp()))
        if (auto _cst = dyn_cast_or_null<spirv::ConstantOp>(
                add.getOperand2().getDefiningOp())) {
          unsigned cst =
              _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
          unsigned key = cst % (outVec * maxPhase);
          cacheCol.insert({key, idxCol});
          idxCol = cacheCol[key];
          immedateOffCol = cst / (outVec * maxPhase) * (outVec * maxPhase);
        }
      // extract dynamic/static offset for immediate offsetting
      unsigned immedateOffRow = 0;
      if (auto add = dyn_cast_or_null<spirv::IAddOp>(idxRow.getDefiningOp()))
        if (auto _cst = dyn_cast_or_null<spirv::ConstantOp>(
                add.getOperand2().getDefiningOp())) {
          unsigned cst =
              _cst.getValue().cast<IntegerAttr>().getValue().getSExtValue();
          unsigned key = cst % (perPhase * maxPhase);
          cacheRow.insert({key, idxRow});
          idxRow = cacheRow[key];
          immedateOffRow = cst / (perPhase * maxPhase) * (perPhase * maxPhase);
        }
      // compute phase = (row // perPhase) % maxPhase
      Value phase = urem(udiv(idxRow, i32_val(perPhase)), i32_val(maxPhase));
      // row offset is simply row index
      Value rowOff = mul(idxRow, strideRow);
      // because swizzling happens at a granularity of outVec, we need to
      // decompose the offset into a swizzled factor and a non-swizzled
      // (ordered) factor: colOffSwizzled = ((col // outVec) ^ phase) * outVec
      // colOffOrdered = (col % outVec) // minVec * minVec
      Value colOffSwizzled = xor_(udiv(idxCol, i32_val(outVec)), phase);
      colOffSwizzled = mul(colOffSwizzled, i32_val(outVec));
      Value colOffOrdered = urem(idxCol, i32_val(outVec));
      colOffOrdered = udiv(colOffOrdered, i32_val(minVec));
      colOffOrdered = mul(colOffOrdered, i32_val(minVec));
      Value colOff = add(colOffSwizzled, colOffOrdered);
      // compute non-immediate offset
      Value offset = add(rowOff, mul(colOff, strideCol));
      Value currPtr = gep(dstPtrTy, dstPtrBase, offset);
      // compute immediate offset
      Value immedateOff =
          add(mul(i32_val(immedateOffRow), srcStrides[outOrder[1]]),
              i32_val(immedateOffCol));
      ret[elemIdx] = gep(dstPtrTy, currPtr, immedateOff);
    }
    return ret;
  }

  SmallVector<Value>
  loadSharedToDistributed(Value dst, ArrayRef<SmallVector<Value>> dstIndices,
                          Value src, SharedMemoryObject smemObj, Type elemTy,
                          Location loc,
                          ConversionPatternRewriter &rewriter) const {
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto dstShape = dstTy.getShape();
    assert(dstShape.size() == 2 &&
           "Unexpected rank of loadSharedToDistributed");
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto dstDistributedLayout = dstTy.getEncoding();
    if (auto mmaLayout = dstDistributedLayout.dyn_cast<MmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout Shared->MMAv1 is not supported yet");
    }
    auto srcSharedLayout =
        srcTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto srcElemTy = srcTy.getElementType();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcSharedLayout);
    auto outOrd = triton::gpu::getOrder(dstDistributedLayout);
    unsigned outVec =
        inOrd == outOrd
            ? triton::gpu::getContigPerThread(dstDistributedLayout)[outOrd[0]]
            : 1;
    unsigned inVec = srcSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned outElems = triton::gpu::getTotalElemsPerThread(dstTy);
    assert(outElems == dstIndices.size());

    DenseMap<unsigned, Value> sharedPtrs = getSwizzledSharedPtrs(
        loc, outVec, dstTy, srcSharedLayout, srcElemTy, smemObj, rewriter,
        smemObj.offsets, smemObj.strides);
    assert(outElems % minVec == 0 && "Unexpected number of elements");
    unsigned numVecs = outElems / minVec;
    auto wordTy = minVec == 1 ? elemTy : vec_ty(elemTy, minVec);
    SmallVector<Value> outVals(outElems);
    for (unsigned i = 0; i < numVecs; ++i) {
      Value smemAddr = sharedPtrs[i * minVec];
      smemAddr =
          bitcast(smemAddr, ptr_ty(wordTy, spirv::StorageClass::Workgroup));
      Value valVec = load(smemAddr);
      for (unsigned v = 0; v < minVec; ++v) {
        if (minVec != 1) {
          Value currVal = extract_element(dstElemTy, valVec, i32_val(v));
          outVals[i * minVec + v] = currVal;
        } else {
          outVals[i * minVec + v] = valVec;
        }
      }
    }
    return outVals;
  }

  void storeDistributedToShared(Value src, Value llSrc,
                                ArrayRef<Value> dstStrides,
                                ArrayRef<SmallVector<Value>> srcIndices,
                                Value dst, Value smemBase, Type elemTy,
                                Location loc,
                                ConversionPatternRewriter &rewriter) const {
    auto srcTy = src.getType().cast<RankedTensorType>();
    auto srcShape = srcTy.getShape();
    assert(srcShape.size() == 2 &&
           "Unexpected rank of storeDistributedToShared");
    auto dstTy = dst.getType().cast<RankedTensorType>();
    auto srcDistributedLayout = srcTy.getEncoding();
    if (auto mmaLayout = srcDistributedLayout.dyn_cast<MmaEncodingAttr>()) {
      assert((!mmaLayout.isVolta()) &&
             "ConvertLayout MMAv1->Shared is not supported yet");
    }
    auto dstSharedLayout =
        dstTy.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto dstElemTy = dstTy.getElementType();
    auto inOrd = triton::gpu::getOrder(srcDistributedLayout);
    auto outOrd = dstSharedLayout.getOrder();
    unsigned inVec =
        inOrd == outOrd
            ? triton::gpu::getContigPerThread(srcDistributedLayout)[inOrd[0]]
            : 1;
    unsigned outVec = dstSharedLayout.getVec();
    unsigned minVec = std::min(outVec, inVec);
    unsigned perPhase = dstSharedLayout.getPerPhase();
    unsigned maxPhase = dstSharedLayout.getMaxPhase();
    unsigned numElems = triton::gpu::getTotalElemsPerThread(srcTy);
    assert(numElems == srcIndices.size());
    auto inVals =
        getTypeConverter()->unpackLLElements(loc, llSrc, rewriter, srcTy);
    Type wordTy;
    if (minVec == 1)
      wordTy = elemTy;
    else
      wordTy = vec_ty(elemTy, minVec);
    //    auto elemPtrTy = ptr_ty(elemTy);
    Value outVecVal = i32_val(outVec);
    Value minVecVal = i32_val(minVec);
    Value word;

    SmallVector<Value> srcStrides = {dstStrides[0], dstStrides[1]};
    SmallVector<Value> offsetVals = {i32_val(0), i32_val(0)};
    SharedMemoryObject smemObj(smemBase, srcStrides, offsetVals);

    DenseMap<unsigned, Value> sharedPtrs =
        getSwizzledSharedPtrs(loc, inVec, srcTy, dstSharedLayout, dstElemTy,
                              smemObj, rewriter, offsetVals, srcStrides);

    for (unsigned i = 0; i < numElems; ++i) {
      if (minVec > 1) {
        if (i % minVec == 0)
          word = undef(wordTy);
        word = insert_element(wordTy, word, inVals[i], i32_val(i % minVec));
      } else {
        word = inVals[i];
      }
      if (i % minVec == minVec - 1) {
        Value smemAddr = sharedPtrs[i / minVec * minVec];
        if (minVec > 1)
          smemAddr =
              bitcast(smemAddr, ptr_ty(wordTy, spirv::StorageClass::Workgroup));
        store(word, smemAddr);
      }
    }
  }

  // -----------------------------------------------------------------------
  // Utilities
  // -----------------------------------------------------------------------
  Value getMask(Type valueTy, ConversionPatternRewriter &rewriter,
                Location loc) const {
    auto tensorTy = valueTy.dyn_cast<RankedTensorType>();
    Value mask = int_val(1, 1);
    auto tid = tid_val();
    if (tensorTy) {
      auto layout = tensorTy.getEncoding();
      auto shape = tensorTy.getShape();
      unsigned rank = shape.size();
      auto sizePerThread = triton::gpu::getSizePerThread(layout);
      auto threadsPerWarp = triton::gpu::getThreadsPerWarp(layout);
      auto warpsPerCTA = triton::gpu::getWarpsPerCTA(layout);
      auto order = triton::gpu::getOrder(layout);
      auto shapePerCTA = triton::gpu::getShapePerCTA(layout, shape);
      Value warpSize = i32_val(product<unsigned>(threadsPerWarp));
      Value laneId = urem(tid, warpSize);
      Value warpId = udiv(tid, warpSize);
      SmallVector<Value> multiDimWarpId =
          delinearize(rewriter, loc, warpId, warpsPerCTA, order);
      SmallVector<Value> multiDimThreadId =
          delinearize(rewriter, loc, laneId, threadsPerWarp, order);
      for (unsigned dim = 0; dim < rank; ++dim) {
        // if there is no data replication across threads on this dimension
        if (shape[dim] >= shapePerCTA[dim])
          continue;
        // Otherwise, we need to mask threads that will replicate data on this
        // dimension. Calculate the thread index on this dimension for the CTA
        Value threadDim =
            add(mul(multiDimWarpId[dim], i32_val(threadsPerWarp[dim])),
                multiDimThreadId[dim]);
        mask = logic_and(mask,
                         icmp_slt(mul(threadDim, i32_val(sizePerThread[dim])),
                                  i32_val(shape[dim])));
      }
    } else {
      // If the tensor is not ranked, then it is a scalar and only thread 0 can
      // write
      mask = logic_and(mask, icmp_eq(tid, i32_val(0)));
    }
    return mask;
  }

  Value dot(ConversionPatternRewriter &rewriter, Location loc,
            ArrayRef<Value> offsets, ArrayRef<Value> strides) const {
    assert(offsets.size() == strides.size());
    Value ret = i32_val(0);
    for (auto [offset, stride] : llvm::zip(offsets, strides)) {
      ret = add(ret, mul(offset, stride));
    }
    return ret;
  }

  struct SmallVectorKeyInfo {
    static unsigned getHashValue(const SmallVector<unsigned> &key) {
      return llvm::hash_combine_range(key.begin(), key.end());
    }
    static bool isEqual(const SmallVector<unsigned> &lhs,
                        const SmallVector<unsigned> &rhs) {
      return lhs == rhs;
    }
    static SmallVector<unsigned> getEmptyKey() {
      return SmallVector<unsigned>();
    }
    static SmallVector<unsigned> getTombstoneKey() {
      return {std::numeric_limits<unsigned>::max()};
    }
  };

  // -----------------------------------------------------------------------
  // Get offsets / indices for any layout
  // -----------------------------------------------------------------------

  SmallVector<Value> emitBaseIndexForLayout(Location loc,
                                            ConversionPatternRewriter &rewriter,
                                            Attribute layout,
                                            RankedTensorType type) const {
    IndexCacheKeyT key = std::make_pair(layout, type);
    auto cache = indexCacheInfo.baseIndexCache;
    auto insertPt = indexCacheInfo.indexInsertPoint;
    if (cache && cache->count(key) > 0) {
      return cache->lookup(key);
    } else {
      ConversionPatternRewriter::InsertionGuard guard(rewriter);
      if (cache)
        restoreInsertionPointIfSet(insertPt, rewriter);
      SmallVector<Value> result;
      if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
        result =
            emitBaseIndexForBlockedLayout(loc, rewriter, blockedLayout, type);
      } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
        if (mmaLayout.isVolta())
          assert(0 && "add mma layout support");
        if (mmaLayout.isAmpere())
          assert(0 && "add mma layout support");
      } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
        auto parentLayout = sliceLayout.getParent();
        auto parentShape = sliceLayout.paddedShape(type.getShape());
        RankedTensorType parentTy = RankedTensorType::get(
            parentShape, type.getElementType(), parentLayout);
        result = emitBaseIndexForLayout(loc, rewriter, parentLayout, parentTy);
        result.erase(result.begin() + sliceLayout.getDim());
      } else {
        llvm_unreachable("unsupported emitBaseIndexForLayout");
      }
      if (cache) {
        cache->insert(std::make_pair(key, result));
        *insertPt = rewriter.saveInsertionPoint();
      }
      return result;
    }
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForLayout(Attribute layout, RankedTensorType type) const {
    if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
      return emitOffsetForBlockedLayout(blockedLayout, type);
    if (auto mmaLayout = layout.dyn_cast<MmaEncodingAttr>()) {
      if (mmaLayout.isVolta())
        assert(0 && "add mma layout support");
      if (mmaLayout.isAmpere())
        assert(0 && "add mma layout support");
    }
    if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>())
      return emitOffsetForSliceLayout(sliceLayout, type);
    llvm_unreachable("unsupported emitOffsetForLayout");
  }

  // -----------------------------------------------------------------------
  // Emit indices
  // -----------------------------------------------------------------------
  SmallVector<SmallVector<Value>> emitIndices(Location loc,
                                              ConversionPatternRewriter &b,
                                              Attribute layout,
                                              RankedTensorType type) const {
    IndexCacheKeyT key(layout, type);
    auto cache = indexCacheInfo.indexCache;
    auto insertPt = indexCacheInfo.indexInsertPoint;
    if (cache && cache->count(key) > 0) {
      return cache->lookup(key);
    } else {
      ConversionPatternRewriter::InsertionGuard guard(b);
      if (cache)
        restoreInsertionPointIfSet(insertPt, b);
      SmallVector<SmallVector<Value>> result;
      if (auto blocked = layout.dyn_cast<BlockedEncodingAttr>()) {
        result = emitIndicesForDistributedLayout(loc, b, blocked, type);
      } else if (auto mma = layout.dyn_cast<MmaEncodingAttr>()) {
        assert(0 && "add mma layout support");
      } else if (auto slice = layout.dyn_cast<SliceEncodingAttr>()) {
        result = emitIndicesForDistributedLayout(loc, b, slice, type);
      } else {
        llvm_unreachable(
            "emitIndices for layouts other than blocked & slice not "
            "implemented yet");
      }
      if (cache) {
        cache->insert(std::make_pair(key, result));
        *insertPt = b.saveInsertionPoint();
      }
      return result;
    }
  }

private:
  void restoreInsertionPointIfSet(OpBuilder::InsertPoint *insertPt,
                                  ConversionPatternRewriter &rewriter) const {
    if (insertPt->isSet()) {
      rewriter.restoreInsertionPoint(*insertPt);
    } else {
      auto func =
          rewriter.getInsertionPoint()->getParentOfType<spirv::FuncOp>();
      rewriter.setInsertionPointToStart(&func.getBody().front());
    }
  }

  // -----------------------------------------------------------------------
  // Blocked layout indices
  // -----------------------------------------------------------------------

  // Get an index-base for each dimension for a \param blocked_layout.
  SmallVector<Value> emitBaseIndexForBlockedLayout(
      Location loc, ConversionPatternRewriter &rewriter,
      const BlockedEncodingAttr &blocked_layout, RankedTensorType type) const {
    auto shape = type.getShape();
    Value threadId = getThreadId(rewriter, loc);
    auto threadsPerWarp = blocked_layout.getThreadsPerWarp();
    Value warpSize = i32_val(product(threadsPerWarp));
    Value laneId = urem(threadId, warpSize);
    Value warpId = udiv(threadId, warpSize);
    auto sizePerThread = blocked_layout.getSizePerThread();
    auto warpsPerCTA = blocked_layout.getWarpsPerCTA();
    auto order = blocked_layout.getOrder();
    unsigned rank = shape.size();

    // delinearize threadId to get the base index
    SmallVector<Value> multiDimWarpId =
        delinearize(rewriter, loc, warpId, warpsPerCTA, order);
    SmallVector<Value> multiDimThreadId =
        delinearize(rewriter, loc, laneId, threadsPerWarp, order);

    SmallVector<Value> multiDimBase(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // Wrap around multiDimWarpId/multiDimThreadId in case
      // shape[k] > shapePerCTA[k]
      auto maxWarps =
          ceil<unsigned>(shape[k], sizePerThread[k] * threadsPerWarp[k]);
      auto maxThreads = ceil<unsigned>(shape[k], sizePerThread[k]);
      multiDimWarpId[k] = urem(multiDimWarpId[k], i32_val(maxWarps));
      multiDimThreadId[k] = urem(multiDimThreadId[k], i32_val(maxThreads));
      // multiDimBase[k] = (multiDimThreadId[k] +
      //                    multiDimWarpId[k] * threadsPerWarp[k]) *
      //                   sizePerThread[k];
      Value threadsPerWarpK = i32_val(threadsPerWarp[k]);
      Value sizePerThreadK = i32_val(sizePerThread[k]);
      multiDimBase[k] =
          mul(sizePerThreadK, add(multiDimThreadId[k],
                                  mul(multiDimWarpId[k], threadsPerWarpK)));
    }
    return multiDimBase;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForBlockedLayout(const BlockedEncodingAttr &blockedLayout,
                             RankedTensorType type) const {
    auto shape = type.getShape();
    auto sizePerThread = blockedLayout.getSizePerThread();
    auto threadsPerWarp = blockedLayout.getThreadsPerWarp();
    auto warpsPerCTA = blockedLayout.getWarpsPerCTA();
    auto order = blockedLayout.getOrder();

    unsigned rank = shape.size();
    SmallVector<unsigned> shapePerCTA = getShapePerCTA(blockedLayout);
    SmallVector<unsigned> tilesPerDim(rank);
    for (unsigned k = 0; k < rank; ++k)
      tilesPerDim[k] = ceil<unsigned>(shape[k], shapePerCTA[k]);

    SmallVector<SmallVector<unsigned>> offset(rank);
    for (unsigned k = 0; k < rank; ++k) {
      // 1 block in minimum if shape[k] is less than shapePerCTA[k]
      for (unsigned blockOffset = 0; blockOffset < tilesPerDim[k];
           ++blockOffset)
        for (unsigned warpOffset = 0; warpOffset < warpsPerCTA[k]; ++warpOffset)
          for (unsigned threadOffset = 0; threadOffset < threadsPerWarp[k];
               ++threadOffset)
            for (unsigned elemOffset = 0; elemOffset < sizePerThread[k];
                 ++elemOffset)
              offset[k].push_back(blockOffset * sizePerThread[k] *
                                      threadsPerWarp[k] * warpsPerCTA[k] +
                                  warpOffset * sizePerThread[k] *
                                      threadsPerWarp[k] +
                                  threadOffset * sizePerThread[k] + elemOffset);
    }

    unsigned elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
    unsigned totalSizePerThread = product<unsigned>(sizePerThread);
    SmallVector<SmallVector<unsigned>> reorderedOffset(elemsPerThread);
    for (unsigned n = 0; n < elemsPerThread; ++n) {
      unsigned linearNanoTileId = n / totalSizePerThread;
      unsigned linearNanoTileElemId = n % totalSizePerThread;
      SmallVector<unsigned> multiDimNanoTileId =
          getMultiDimIndex<unsigned>(linearNanoTileId, tilesPerDim, order);
      SmallVector<unsigned> multiDimNanoTileElemId = getMultiDimIndex<unsigned>(
          linearNanoTileElemId, sizePerThread, order);
      for (unsigned k = 0; k < rank; ++k) {
        unsigned reorderedMultiDimId =
            multiDimNanoTileId[k] *
                (sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k]) +
            multiDimNanoTileElemId[k];
        reorderedOffset[n].push_back(offset[k][reorderedMultiDimId]);
      }
    }
    return reorderedOffset;
  }

  // Emit indices calculation within each ConversionPattern, and returns a
  // [elemsPerThread X rank] index matrix.
  SmallVector<SmallVector<Value>> emitIndicesForDistributedLayout(
      Location loc, ConversionPatternRewriter &rewriter, Attribute layout,
      RankedTensorType type) const {
    // step 1, delinearize threadId to get the base index
    auto multiDimBase = emitBaseIndexForLayout(loc, rewriter, layout, type);
    // step 2, get offset of each element
    auto offset = emitOffsetForLayout(layout, type);
    // step 3, add offset to base, and reorder the sequence of indices to
    // guarantee that elems in the same sizePerThread are adjacent in order
    auto shape = type.getShape();
    unsigned rank = shape.size();
    unsigned elemsPerThread = offset.size();
    SmallVector<SmallVector<Value>> multiDimIdx(elemsPerThread,
                                                SmallVector<Value>(rank));
    for (unsigned n = 0; n < elemsPerThread; ++n)
      for (unsigned k = 0; k < rank; ++k)
        multiDimIdx[n][k] = add(multiDimBase[k], i32_val(offset[n][k]));
    return multiDimIdx;
  }

  SmallVector<SmallVector<unsigned>>
  emitOffsetForSliceLayout(const SliceEncodingAttr &sliceLayout,
                           RankedTensorType type) const {
    auto parentEncoding = sliceLayout.getParent();
    unsigned dim = sliceLayout.getDim();
    auto parentShape = sliceLayout.paddedShape(type.getShape());
    RankedTensorType parentTy = RankedTensorType::get(
        parentShape, type.getElementType(), parentEncoding);
    auto parentOffsets = emitOffsetForLayout(parentEncoding, parentTy);

    unsigned numOffsets = parentOffsets.size();
    SmallVector<SmallVector<unsigned>> resultOffsets;
    std::set<SmallVector<unsigned>> uniqueOffsets;

    for (unsigned i = 0; i < numOffsets; ++i) {
      SmallVector<unsigned> offsets = parentOffsets[i];
      offsets.erase(offsets.begin() + dim);
      if (uniqueOffsets.find(offsets) == uniqueOffsets.end()) {
        resultOffsets.push_back(offsets);
        uniqueOffsets.insert(offsets);
      }
    }
    return resultOffsets;
  }

protected:
  TritonGPUToSPIRVTypeConverter *converter;
  ModuleAllocation *allocation;
  IndexCacheInfo indexCacheInfo;
};

template <typename SourceOp>
class ConvertTritonGPUOpToSPIRVPattern
    : public OpConversionPattern<SourceOp>,
      public ConvertTritonGPUOpToSPIRVPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUOpToSPIRVPattern(
      TritonGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context,
      PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        ConvertTritonGPUOpToSPIRVPatternBase(typeConverter) {}

  explicit ConvertTritonGPUOpToSPIRVPattern(
      TritonGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context,
      IndexCacheInfo indexCacheInfo, PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        ConvertTritonGPUOpToSPIRVPatternBase(typeConverter, indexCacheInfo) {}

  explicit ConvertTritonGPUOpToSPIRVPattern(
      TritonGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context,
      ModuleAllocation &allocation, PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        ConvertTritonGPUOpToSPIRVPatternBase(typeConverter, allocation) {}

  explicit ConvertTritonGPUOpToSPIRVPattern(
      TritonGPUToSPIRVTypeConverter &typeConverter, MLIRContext *context,
      ModuleAllocation &allocation, IndexCacheInfo indexCacheInfo,
      PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit),
        ConvertTritonGPUOpToSPIRVPatternBase(typeConverter, allocation,
                                             indexCacheInfo) {}

protected:
  TritonGPUToSPIRVTypeConverter *getTypeConverter() const {
    SPIRVTypeConverter *ret =
        ((ConvertTritonGPUOpToSPIRVPatternBase *)this)->getTypeConverter();
    return (TritonGPUToSPIRVTypeConverter *)ret;
  }
};

#endif
