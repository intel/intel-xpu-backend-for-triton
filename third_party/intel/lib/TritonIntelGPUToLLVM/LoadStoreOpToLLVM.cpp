#include "Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "intel/include/Analysis/StrideInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/BlockIOUtils.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include <limits>
#include <optional>
#include <triton/Tools/Sys/GetEnv.h>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

#define S(v) StringAttr::get(ctx, (v))

namespace {

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

/// Unpacked tensor descriptor fields: { shape[rank], stride[rank], base }.
struct DescriptorFields {
  SmallVector<Value, 4> shapes;  // shapes[0..rank-1]
  SmallVector<Value, 4> strides; // strides[0..rank-1]
  Value base;
};

/// Unpack a tensor descriptor struct into its constituent fields.
/// TensorDescType struct layout: { shape[rank], stride[rank], base }
/// This differs from block pointers which have: { offset[rank], shape[rank],
/// stride[rank], base }. For tensor descriptors, offsets (indices) are
/// supplied explicitly via the op operands rather than being embedded in the
/// struct.
static DescriptorFields unpackDescriptor(Value llDesc, unsigned rank,
                                         Location loc,
                                         ConversionPatternRewriter &rewriter) {
  const SmallVector<Value> &elems = unpackLLElements(loc, llDesc, rewriter);
  assert(elems.size() == 2 * rank + 1 && "unexpected descriptor struct size");
  DescriptorFields f;
  f.shapes.assign(elems.begin(), elems.begin() + rank);
  f.strides.assign(elems.begin() + rank, elems.begin() + 2 * rank);
  f.base = elems[2 * rank];
  return f;
}

/// Verify that a rank-reducing tensor descriptor load/store is
/// shape-compatible.
///
/// For rank-reducing ops, the result/source tensor shape must exactly match the
/// inner dimensions of the descriptor block type after stripping the leading
/// (outer) dimensions:
///   desc_shape[i] == 1                        for all i in [0, rankDelta)
///   desc_shape[rankDelta + i] == tensor_shape[i]   for all i in [0, rank)
///
/// This is not checked by the upstream verifier (which only checks total
/// element count), so we assert it here before relying on the mapping in
/// lowering.
static void assertDescriptorInnerShapeCompatible(
    Operation *op, ArrayRef<int64_t> descBlockShape,
    ArrayRef<int64_t> tensorShape, bool permuteInnerDims = false) {
  const size_t descRank = descBlockShape.size();
  const size_t rank = tensorShape.size();
  assert(descRank >= rank && "descriptor rank must be >= tensor rank");
  const size_t rankDelta = descRank - rank;
  for (size_t i = 0; i < rankDelta; ++i) {
    if (descBlockShape[i] != 1) {
      op->emitError()
          << "rank-reducing descriptor op only supports dropping leading "
             "dimensions of size 1, but descriptor dim["
          << i << "] = " << descBlockShape[i];
      llvm_unreachable("rank-reducing descriptor dropped dim mismatch");
    }
  }

  for (size_t i = 0; i < rank; ++i) {
    size_t descDim = rankDelta + i;
    if (permuteInnerDims && rank >= 2) {
      if (i == rank - 2)
        descDim = descRank - 1;
      else if (i == rank - 1)
        descDim = descRank - 2;
    }

    if (descBlockShape[descDim] != tensorShape[i]) {
      op->emitError()
          << "rank-reducing descriptor op requires that the result/source "
             "tensor shape matches the inner dimensions of the descriptor "
             "block type, but descriptor inner dim["
          << i << "] = " << descBlockShape[descDim] << " != tensor dim[" << i
          << "] = " << tensorShape[i];
      llvm_unreachable("rank-reducing descriptor inner shape mismatch");
    }
  }
}

/// Compute the 2D prefetch shape for each warp given an input 2D tensor.
/// Because a cache line is 64 bytes, and we want to prefetch one cache line a
/// time (per thread), the maximum number of bytes per column is 64. We know
/// that the maximum size for each 2D prefetch is 2048 bytes, therefore the
/// maximum number of rows is given by 2048/64=32.
SmallVector<unsigned, 2> get2DPrefetchShapePerWarp(RankedTensorType tensorTy) {
  Type eltTy = tensorTy.getElementType();
  const ArrayRef<int64_t> tensorShape = tensorTy.getShape();
  unsigned rank = tensorShape.size();
  unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
  unsigned elemSizeInBytes = elemSizeInBits / 8;
  unsigned maxBytesPerCol = 64;
  unsigned numRows = std::min<unsigned>(tensorShape[rank - 2], 32);
  unsigned numCols = std::min<unsigned>(tensorShape[rank - 1],
                                        maxBytesPerCol / elemSizeInBytes);
  return {numRows, numCols};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass),
        strideAnalysis(strideAnalysis) {}

  int getStride(Value ptr, unsigned dim) const {
    triton::intel::StrideInfo *strideInfo = strideAnalysis.getStrideInfo(ptr);
    if (strideInfo) {
      const auto &stride = strideInfo->getStride();
      if (dim < stride.size()) {
        return stride[dim];
      }
      // There is only one case that the regular pointer is defined as the
      // function args.
      assert(stride.size() == 1 && stride[0] == -1 &&
             "get the stride of invalid dim from regular pointer");
    }
    return -1;
  }

  unsigned getContiguity(Value ptr) const {
    return const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
        .getContiguity(ptr);
  }

  static bool hasSupport256bLoadStore(Operation *op) {
    auto mod = op->getParentOfType<ModuleOp>();
    return mod && mod->hasAttr(
                      TritonIntelGPUDialect::getSupport256bLoadStoreAttrName());
  }

  /// Maximum number of elements per-thread vector load/store. 128 bits by
  /// default; 256 bits when the target supports wider load/stores.
  static unsigned getMaxVecWidth(bool support256bLoadStore,
                                 unsigned pointeeBitWidth) {
    unsigned maxBits = support256bLoadStore ? 256u : 128u;
    return std::max(1u, maxBits / std::max(8u, pointeeBitWidth));
  }

  unsigned getVectorSize(bool support256bLoadStore, Value ptr) const {
    if (!isa<RankedTensorType>(ptr.getType()))
      return 1;

    unsigned contiguity = getContiguity(ptr);
    unsigned pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
    return std::min<unsigned>(
        getMaxVecWidth(support256bLoadStore, pointeeBitWidth), contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
        .getMaskAlignment(mask);
  }

  /// Compute vectorization factor for descriptor load/store gather fallback.
  /// Queries the descriptor's address-level AxisInfo (analogous to how
  /// getVectorSize queries the pointer operand's AxisInfo for LoadOp).
  unsigned getDescriptorVecSize(bool support256bLoadStore, Value desc,
                                RankedTensorType resultType, Type valueElemTy,
                                StringAttr blockIOAttr) const {
    unsigned rank = resultType.getRank();
    if (rank == 0)
      return 1;

    AxisInfo *descAxisInfo =
        const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
            .getAxisInfo(desc);
    if (!descAxisInfo || static_cast<unsigned>(descAxisInfo->getRank()) != rank)
      return 1;

    auto linAttr = triton::gpu::toGenericLinearEncoding(resultType);
    SmallVector<unsigned> order = linAttr.getOrder();
    SmallVector<unsigned> contigPerThread = linAttr.getContigPerThread();

    // Map result layout's fast dimension to the corresponding descriptor
    // dimension. Column-major descriptor loads transpose the inner 2
    // dimensions: the result layout's fast dim (order[0]) maps to the other of
    // the two innermost descriptor dimensions.
    unsigned descDim = order[0];
    if (blockIOAttr) {
      auto mode = symbolizeBlockIOMode(blockIOAttr.getValue());
      if (mode && *mode == BlockIOMode::ColumnMajor && rank >= 2 &&
          order[0] >= rank - 2) {
        descDim = (order[0] == rank - 2) ? rank - 1 : rank - 2;
      }
    }

    unsigned descContiguity = descAxisInfo->getContiguity(descDim);
    if (descContiguity <= 1)
      return 1; // Stride is not 1 on this dimension

    unsigned pointeeBitWidth = valueElemTy.getIntOrFloatBitWidth();
    unsigned maxVec = getMaxVecWidth(support256bLoadStore, pointeeBitWidth);
    unsigned threadContig = contigPerThread[order[0]];
    // Note: descContiguity and threadContig refer to the same logical dimension
    // but are indexed in different coordinate spaces. descContiguity uses
    // descDim (descriptor space, remapped for column-major), while
    // threadContig uses order[0] (result layout space). The min() below
    // correctly intersects address-level and layout-level constraints.

    // Use descriptor's divisibility (bytes) for pointer alignment.
    unsigned descDivisibility = descAxisInfo->getDivisibility(descDim);
    unsigned elemBytes = std::max(1u, pointeeBitWidth / 8);
    unsigned ptrAlignElems = std::max(1u, descDivisibility / elemBytes);

    unsigned vec =
        std::min({maxVec, threadContig, descContiguity, ptrAlignElems});
    assert(vec > 0 && "vec must be positive for Log2_32");
    return std::max(1u, 1u << llvm::Log2_32(vec));
  }

  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  computeGatherScatterOperands(
      Location loc, Value base, ArrayRef<Value> offsets, ArrayRef<Value> shapes,
      ArrayRef<Value> strides, RankedTensorType tensorType, Type valueElemTy,
      ConversionPatternRewriter &rewriter, ArrayRef<int32_t> boundaryCheck = {},
      std::optional<PaddingOption> padding = std::nullopt) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    assert(offsets.size() == shapes.size() &&
           offsets.size() == strides.size() &&
           "invalid length of offsets, shapes and strides");
    size_t tensorRank = tensorType.getRank();
    size_t pointerRank = offsets.size();
    assert(pointerRank >= tensorRank &&
           "descriptor rank must be >= result/source tensor rank");
    const unsigned numElems = getTotalElemsPerThread(tensorType);

    // Get the LLVM values for indices in block
    auto indices = emitIndices(loc, rewriter, targetInfo,
                               tensorType.getEncoding(), tensorType, true);

    auto linearize =
        [](ArrayRef<Value> A, ArrayRef<Value> B, Value init,
           std::function<Value(const Value &, const Value &, const Value &)>
               linearizeFunc) {
          unsigned rank = A.size();
          Value accumulate = init;
          if (rank > 0) {
            for (auto [a, b] : llvm::zip(A, B))
              accumulate = linearizeFunc(a, b, accumulate);
          }
          return accumulate;
        };

    SetVector<unsigned> boundaryProtect(boundaryCheck.begin(),
                                        boundaryCheck.end());
    SmallVector<Value> ptrElems(numElems);
    SmallVector<Value> maskElems;
    for (unsigned i = 0; i < numElems; ++i) {
      SmallVector<Value> index = indices[i];
      SmallVector<Value> indicesInTensor(pointerRank);
      for (unsigned j = 0; j < pointerRank - tensorRank; ++j) {
        indicesInTensor[j] = offsets[j];
      }
      for (unsigned j = 0; j < tensorRank; ++j)
        indicesInTensor[j + (pointerRank - tensorRank)] =
            b.add(index[j], offsets[j + (pointerRank - tensorRank)]);

      // Get the LLVM values for pointers
      Value offset = linearize(
          indicesInTensor, strides, b.i32_val(0),
          [&](const Value &index, const Value &stride, const Value &off) {
            // off = off + index * stride
            return b.add(b.mul(index, b.trunc(i32_ty, stride)), off);
          });

      ptrElems[i] = b.gep(ptr_ty(rewriter.getContext(), 1 /*global*/),
                          valueElemTy, base, offset);

      if (boundaryProtect.size() > 0) {
        // Get the LLVM values for mask
        unsigned dim = 0;
        maskElems.push_back(linearize(
            indicesInTensor, shapes, b.int_val(1, 1),
            [&](const Value &index, const Value &shape, const Value &mask) {
              if (boundaryProtect.contains(dim++)) {
                // mask = mask && (index < shape) && idx >= 0
                auto is_pos_idx = b.icmp_sge(index, b.i32_val(0));
                return b
                    .and_(
                        b.and_(b.icmp_slt(index, b.trunc(i32_ty, shape)), mask),
                        is_pos_idx)
                    .getResult();
              }

              return mask;
            }));
      }
    }

    // Get the LLVM values for `other`
    SmallVector<Value> otherElems;
    if (padding) {
      Value other;
      switch (*padding) {
      case PaddingOption::PAD_ZERO:
        other = LLVM::ConstantOp::create(rewriter, loc, valueElemTy,
                                         rewriter.getZeroAttr(valueElemTy));

        break;
      case PaddingOption::PAD_NAN: {
        assert(!valueElemTy.isIntOrIndex() &&
               "Expect element type to be non-integer type");
        auto apNaN = llvm::APFloat::getNaN(
            cast<FloatType>(valueElemTy).getFloatSemantics());
        other =
            LLVM::ConstantOp::create(rewriter, loc, valueElemTy,
                                     rewriter.getFloatAttr(valueElemTy, apNaN));
      } break;
      }

      for (unsigned i = 0; i < numElems; ++i)
        otherElems.push_back(other);
    }

    return std::make_tuple(ptrElems, maskElems, otherElems);
  }

  /// Convenience overload that unpacks a tensor descriptor struct and
  /// delegates to computeGatherScatterOperands.
  /// TensorDescType struct layout:
  ///   { shape[rank], stride[rank], base }
  /// Offsets are provided externally via the indices operand of the
  /// DescriptorLoadOp/DescriptorStoreOp (not stored in the struct).
  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  convertTensorDescriptorToTensorOfPtr(
      Location loc, Value descriptorStruct, ValueRange indices,
      RankedTensorType tensorType, unsigned descriptorRank, Type valueElemTy,
      ConversionPatternRewriter &rewriter, ArrayRef<int32_t> boundaryCheck = {},
      std::optional<PaddingOption> padding = std::nullopt) const {

    DescriptorFields desc =
        unpackDescriptor(descriptorStruct, descriptorRank, loc, rewriter);

    SmallVector<Value> offsets;
    offsets.reserve(descriptorRank);
    assert(indices.size() == descriptorRank &&
           "unexpected descriptor indices rank");
    for (size_t i = 0; i < descriptorRank; ++i)
      offsets.push_back(indices[i]);

    SmallVector<Value> mappedShapes(descriptorRank),
        mappedStrides(descriptorRank);
    for (size_t i = 0; i < descriptorRank; ++i) {
      mappedShapes[i] = desc.shapes[i];
      mappedStrides[i] = desc.strides[i];
    }

    return computeGatherScatterOperands(loc, desc.base, offsets, mappedShapes,
                                        mappedStrides, tensorType, valueElemTy,
                                        rewriter, boundaryCheck, padding);
  }

  /// Build per-element NaN masks for out-of-bounds elements.
  ///
  /// Returns a vector of i1 values, one per thread element, where `true`
  /// means the element is in-bounds. The caller should select NaN for
  /// elements where the mask is `false`.
  ///
  /// \p offsets     Base offsets for each dimension (e.g. descriptor indices).
  /// \p shapes      Boundary shapes for each dimension (i64).
  /// \p tensorType  The result tensor type (determines layout and element
  ///                count).
  SmallVector<Value> buildNaNMasks(Location loc, ArrayRef<Value> offsets,
                                   ArrayRef<Value> shapes,
                                   RankedTensorType tensorType,
                                   ConversionPatternRewriter &rewriter) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    size_t rank = tensorType.getRank();
    assert(offsets.size() == rank && shapes.size() == rank);

    const unsigned numElems = getTotalElemsPerThread(tensorType);
    auto indices = emitIndices(loc, rewriter, targetInfo,
                               tensorType.getEncoding(), tensorType, true);

    SmallVector<Value> maskElems;
    for (unsigned i = 0; i < numElems; ++i) {
      SmallVector<Value> index = indices[i];
      Value mask = b.int_val(1, 1);
      for (unsigned j = 0; j < rank; ++j) {
        Value idxInTensor = b.add(index[j], offsets[j]);
        Value inBounds = b.icmp_slt(idxInTensor, b.trunc(i32_ty, shapes[j]));
        Value isPos = b.icmp_sge(idxInTensor, b.i32_val(0));
        mask = b.and_(b.and_(inBounds, mask), isPos).getResult();
      }
      maskElems.push_back(mask);
    }
    return maskElems;
  }

  // Ensure the operation doesn't have attributes that the IGC predicated
  // instruction cannot handle.
  template <
      typename OpType,
      typename = std::enable_if_t<llvm::is_one_of<
          OpType, LoadOp, StoreOp, DescriptorLoadOp, DescriptorStoreOp>::value>>
  bool canUsePredicatedInstructions(OpType op) const {
    if (!mlir::LLVM::intel::hasModuleAttr(
            op, TritonIntelGPUDialect::getSupportPredicatedIOAttrName()))
      return false;

    // Predicated load is enabled by default for LoadOp but disabled by default
    // for DescriptorLoadOp. DescriptorLoadOp always generates boundary-check
    // predicates (even when all elements are in-bounds), and the predicated
    // load intrinsic prevents IGC from optimizing these uniformly-true
    // predicates as effectively as the control-flow-based approach. Both can be
    // overridden by env vars. Predicated store is enabled by default for both
    // op types.
    static const std::optional<bool> usePredicatedLoad =
        tools::isEnvValueBool(tools::getStrEnv("TRITON_INTEL_PREDICATED_LOAD"));
    static const std::optional<bool> usePredicatedStore = tools::isEnvValueBool(
        tools::getStrEnv("TRITON_INTEL_PREDICATED_STORE"));

    // SPIRV predicated load/store does not support volatile qualifier.
    if constexpr (std::is_same_v<OpType, LoadOp>) {
      return (!usePredicatedLoad.has_value() || usePredicatedLoad.value()) &&
             !op.getIsVolatile();
    } else if constexpr (std::is_same_v<OpType, StoreOp>) {
      return !usePredicatedStore.has_value() || usePredicatedStore.value();
    } else if constexpr (std::is_same_v<OpType, DescriptorLoadOp>) {
      return usePredicatedLoad.has_value() && usePredicatedLoad.value();
    } else if constexpr (std::is_same_v<OpType, DescriptorStoreOp>) {
      return !usePredicatedStore.has_value() || usePredicatedStore.value();
    }

    llvm_unreachable("unsupported operation type for predicated instruction");
  }

  // Convert Triton cache modifier to Intel GEN load cache control enum.
  //
  // Explicit cache modifiers (cg/cv/ca) always win. When no cache modifier is
  // set, fall back to the frontend-provided eviction policy hint (e.g.
  // inductor's `eviction_policy='evict_last'`) and route it to the closest
  // LSC cache mode:
  //   EVICT_FIRST -> L1IAR_L3C  (invalidate-after-read: data is used once;
  //                              free the L1 line immediately after delivery)
  //   EVICT_LAST  -> L1C_L3C    (cache at all levels: keep the line warm for
  //                              anticipated reuse)
  //   NORMAL      -> DEFAULT    (let the hardware decide)
  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, LoadOp, DescriptorLoadOp>::value>>
  TritonGEN::LoadCacheControl tritonToIntelCacheModifier(OpType &op) const {
    CacheModifier cacheModifier = op.getCache();

    /******** LoadOp ********
     * ""   -> DEFAULT (No cache modifier provided)
     * "cg" -> L1UC_L3C (Cache at global level, not L1)
     * "cv" -> L1UC_L3UC (Do not cache at all)
     * "ca" -> L1C_L3C (Cache at all levels)
     **/
    switch (cacheModifier) {
    case CacheModifier::NONE:
      switch (op.getEvict()) {
      case EvictionPolicy::EVICT_FIRST:
        return TritonGEN::LoadCacheControl::L1IAR_L3C;
      case EvictionPolicy::EVICT_LAST:
        return TritonGEN::LoadCacheControl::L1C_L3C;
      case EvictionPolicy::NORMAL:
        break;
      }
      return TritonGEN::LoadCacheControl::DEFAULT;
    case CacheModifier::CG:
      return TritonGEN::LoadCacheControl::L1UC_L3C;
    case CacheModifier::CV:
      return TritonGEN::LoadCacheControl::L1UC_L3UC;
    case CacheModifier::CA:
      return TritonGEN::LoadCacheControl::L1C_L3C;
    default:
      llvm_unreachable("invalid cache modifier for LoadOp");
    }
  }

  template <typename OpType,
            typename = std::enable_if_t<std::is_same_v<OpType, StoreOp>>>
  TritonGEN::StoreCacheControl tritonToIntelCacheModifier(OpType &op) const {
    CacheModifier cacheModifier = op.getCache();

    /******** StoreOp ********
     * ""   -> DEFAULT (No cache modifier provided)
     * "wb" -> L1WB_L3WB (Cache write-back at all levels.)
     * "cg" -> L1UC_L3WB (Cache at global level, not L1)
     * "cs" -> L1S_L3S (Cache streaming at all levels)
     * "wt" -> L1WT_L3WT (Cache write-through at all levels)
     **/
    switch (cacheModifier) {
    case CacheModifier::NONE:
      return TritonGEN::StoreCacheControl::DEFAULT;
    case CacheModifier::WB:
      return TritonGEN::StoreCacheControl::L1WB_L3WB;
    case CacheModifier::CG:
      return TritonGEN::StoreCacheControl::L1UC_L3WB;
    case CacheModifier::CS:
      return TritonGEN::StoreCacheControl::L1S_L3S;
    case CacheModifier::WT:
      return TritonGEN::StoreCacheControl::L1WT_L3WT;
    default:
      llvm_unreachable("invalid cache modifier for StoreOp");
    }
  }

  template <typename OpType,
            typename = std::enable_if_t<llvm::is_one_of<
                OpType, LoadOp, StoreOp, DescriptorLoadOp>::value>>
  bool getNonTemporalFlag(OpType op) const {
    switch (op.getCache()) {
    case triton::CacheModifier::CG:
    case triton::CacheModifier::CS:
    case triton::CacheModifier::CV:
      return true;
    case triton::CacheModifier::CA:
      return false;
    case triton::CacheModifier::NONE:
      // No explicit cache modifier: derive from eviction policy hint.
      // EVICT_FIRST implies single-use; map to nontemporal so IGC bypasses L1.
      // EVICT_LAST implies reuse; keep the default (temporal) behavior.
      // Only load ops carry eviction policy; stores always return false here.
      if constexpr (std::is_same_v<OpType, LoadOp> ||
                    std::is_same_v<OpType, DescriptorLoadOp>)
        return op.getEvict() == triton::EvictionPolicy::EVICT_FIRST;
      return false;
    default:
      return false;
    }
  }

protected:
  const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass;
  triton::intel::ModuleStrideAnalysis &strideAnalysis;
  const triton::intel::TargetInfo &targetInfo;
};

struct BlockIOConversionBase : public LoadStoreConversionBase {
  explicit BlockIOConversionBase(
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  // Determine whether the given operation can be lowered to using block IO
  // instructions.
  template <typename OpTy,
            std::enable_if_t<
                llvm::is_one_of<OpTy, triton::LoadOp, triton::StoreOp>::value,
                bool> = true>
  static bool isBlockIOCandidate(OpTy op) {
    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    if (!mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                          getSupport2DBlockIOAttrName()))
      return false;

    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr)
      return false;

    std::optional<bool> enableBlockIOForAllLayout =
        mlir::triton::tools::isEnvValueBool(mlir::triton::tools::getStrEnv(
            "TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"));

    // Only lower operation with dpas layout encoding.
    auto tensorTy =
        cast<RankedTensorType>(getPointeeType(op.getPtr().getType()));
    bool hasDpas = hasDpasEncoding(tensorTy) || hasDotDpasEncoding(tensorTy);
    return !enableBlockIOForAllLayout.has_value() ||
           enableBlockIOForAllLayout.value() || hasDpas;
  }

  /// Check whether an op was annotated by the 1D→2D reshape in
  /// MaterializeBlockPointer.
  template <typename OpTy> static bool hasAnnotated1DReshapeStride(OpTy op) {
    if constexpr (std::is_same_v<OpTy, triton::StoreOp> ||
                  std::is_same_v<OpTy, triton::LoadOp>)
      return op->hasAttr(TritonIntelGPUDialect::getBlockIOStrideAttrName());
    return false;
  }

  /// Return the pitch (in bytes) for an op annotated by the 1D→2D reshape.
  /// The stride attribute is in elements; this converts to bytes.
  /// Returns std::nullopt if the attribute is absent, non-positive, or if
  /// the resulting byte pitch does not fit in a signed 32-bit integer (the
  /// HW pitch operand is i32).
  template <typename OpTy>
  static std::optional<int64_t>
  getAnnotated1DReshapePitch(OpTy op, unsigned elemSizeInBits) {
    auto strideAttr = op->template getAttrOfType<IntegerAttr>(
        TritonIntelGPUDialect::getBlockIOStrideAttrName());
    if (!strideAttr)
      return std::nullopt;
    int64_t strideElems = strideAttr.getInt();
    if (strideElems <= 0)
      return std::nullopt;
    int64_t pitchBytes = strideElems * elemSizeInBits / 8;
    if (pitchBytes > std::numeric_limits<int32_t>::max())
      return std::nullopt;
    return pitchBytes;
  }

  // Determine whether the given descriptor op can be lowered to using
  // block IO instructions.
  template <typename OpTy,
            std::enable_if_t<llvm::is_one_of<OpTy, triton::DescriptorLoadOp,
                                             triton::DescriptorStoreOp>::value,
                             bool> = true>
  static bool isDescriptorBlockIOCandidate(OpTy op) {
    ModuleOp mod = op->template getParentOfType<ModuleOp>();
    if (!mod->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                          getSupport2DBlockIOAttrName()))
      return false;

    RankedTensorType tensorTy;
    if constexpr (std::is_same_v<OpTy, triton::DescriptorLoadOp>)
      tensorTy = cast<RankedTensorType>(op.getType());
    else
      tensorTy = cast<RankedTensorType>(op.getSrc().getType());

    // Rank must be at least 2 for 2D block I/O.
    if (tensorTy.getRank() < 2)
      return false;

    // Require block_io attribute (set by MaterializeBlockPointer).
    if (!op->template getAttrOfType<StringAttr>(
            triton::gpu::intel::TritonIntelGPUDialect::getBlockIOAttrName()))
      return false;

    std::optional<bool> enableBlockIOForAllLayout =
        mlir::triton::tools::isEnvValueBool(mlir::triton::tools::getStrEnv(
            "TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"));
    if (enableBlockIOForAllLayout.has_value() &&
        !enableBlockIOForAllLayout.value() && !hasDpasEncoding(tensorTy) &&
        !hasDotDpasEncoding(tensorTy))
      return false;

    return true;
  }

  template <
      typename OpTy,
      std::enable_if_t<llvm::is_one_of<OpTy, triton::gpu::intel::PrefetchOp,
                                       triton::LoadOp, triton::StoreOp>::value,
                       bool> = true>
  static bool isMemoryRowMajor(OpTy op) {
    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    assert(blockIOAttr && "Expecting block IO attribute");

    auto mode = symbolizeBlockIOMode(cast<StringAttr>(blockIOAttr).getValue());
    assert(mode && "Only row_major or column_major is supported");
    return *mode == BlockIOMode::RowMajor;
  }

  static DpasEncodingAttr::OpIdx getOpIdx(RankedTensorType tensorTy) {
    return triton::gpu::intel::getOpIdx(tensorTy);
  }

  static DpasEncodingAttr getDpasLayout(RankedTensorType tensorTy) {
    return triton::gpu::intel::getDpasLayout(tensorTy);
  }

  // Returns the pitch (stride in bytes) for a regular pointer.
  Value getPitch(ConversionPatternRewriter &rewriter, Value ptr,
                 unsigned elemSizeInBits, unsigned dim) const {
    Location loc = ptr.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    int stride = getStride(ptr, dim);
    // If the stride is 0, we assume a minimum pitch of 64 bytes.
    constexpr int MIN_PITCH = 64;
    if (stride == 0)
      return b.i32_val(MIN_PITCH);

    if (stride > 0) {
      unsigned pitch = (unsigned)stride * elemSizeInBits / 8;
      if (pitch < MIN_PITCH)
        return nullptr; // unsupported pitch
      return b.i32_val(pitch);
    }
    assert(stride == -1 && "invalid stride < 0");

    return nullptr;
  }

  /// Configuration produced by configureDPASLoadTypes().
  struct DPASLoadConfig {
    Type packedDPASOperandType; // null if not DPAS
    Type unpackedType;
    Type load2DGenXType;
    Type packedType;
    bool useVNNIFormat = false;
    unsigned tileHeight;
    unsigned tileWidth;
    unsigned vBlocks;
    int64_t numElemsPerLoad;
    unsigned numValuesPerLoad;
  };

  /// Full configuration for emitting a 2D block load sequence.
  /// Produced by buildBlock2DLoadConfig() from a BlockIOTileSizeInfo.
  struct Block2DLoadConfig {
    // Tile geometry (post-DPAS configuration — may differ from sizeInfo).
    unsigned tileHeight;
    unsigned tileWidth;
    unsigned numPackedVals;
    unsigned vBlocks;
    int rowDim;
    int colDim;
    bool isTransposeRequired;

    // Derived sizes.
    unsigned packedElemSizeInBits;
    unsigned threadsPerWarp;
    unsigned numElems;
    int64_t numElemsPerLoad;
    unsigned numValuesPerLoad;

    // LLVM types for the load sequence.
    Type load2DGenXType;
    Type unpackedType;
    Type packedType;
    Type packedDPASOperandType; // null if not DPAS
    bool useVNNIFormat = false;

    // Mappings for register indexing and shuffle.
    LinearLayout regMapping;
    LinearLayout shuffleMapping;
  };

  /// Configure load types for DPAS encoding.
  ///
  /// For the DPAS layout, there are three types of block loads used.
  /// (For non-DPAS layouts, only two types are involved.)
  ///   1. load2DGenXType
  ///   2. packedDPASOperandType – (This is null for non-DPAS layouts.)
  ///   3. unpackedType
  ///
  // clang-format off
  /// The load operation generates the following block load sequence:
  ///   %0 = load_2d %ptr : <load2DGenXType>
  ///   %1 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
  ///         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ///   %2 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
  ///         <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ///   %3 = bitcast %1 : <packedDPASOperandType> -> <unpackedType>
  ///   %4 = bitcast %2 : <packedDPASOperandType> -> <unpackedType>
  ///   <operations for packLLElements>
  // clang-format on
  ///
  /// The `tt.dot` operation generates the DPAS instruction sequence:
  // clang-format off
  ///   <operations for unpackLLElements>
  ///   %5 = bitcast %3 : <unpackedType> -> <packedDPASOperandType>
  ///   %6 = bitcast %4 : <unpackedType> -> <packedDPASOperandType>
  ///   %7 = dpas %5, %6, %other : <packedDPASOperandType>, ...
  // clang-format on
  ///
  /// The LLVM optimizer eliminates redundant pack/unpack element pairs
  /// and corresponding bitcast operations. The final optimized IR for
  /// the dot product becomes:
  ///
  // clang-format off
  ///   %0 = load_2d %ptr : <load2DGenXType>
  ///   %1 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
  ///         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ///   %2 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
  ///         <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ///   %3 = dpas %1, %2, %other : <packedDPASOperandType>, ...
  // clang-format on
  ///
  /// The `packedDPASOperandType` together with the `shufflevector`
  /// operations defines the computation flow for the dot product.
  static DPASLoadConfig configureDPASLoadTypes(
      RankedTensorType tensorType, Type eltTy, Type initialPackedType,
      Type initialLoad2DGenXType, Type initialUnpackedType,
      unsigned elemSizeInBits, unsigned numPackedVals, unsigned threadsPerWarp,
      unsigned tileHeight, unsigned tileWidth, unsigned vBlocks,
      int64_t numElemsPerLoad, unsigned numValuesPerLoad,
      bool isTransposeRequired, MLIRContext *ctx) {
    DPASLoadConfig cfg;
    cfg.packedDPASOperandType = nullptr;
    cfg.unpackedType = initialUnpackedType;
    cfg.load2DGenXType = initialLoad2DGenXType;
    cfg.packedType = initialPackedType;
    cfg.useVNNIFormat = false;
    cfg.tileHeight = tileHeight;
    cfg.tileWidth = tileWidth;
    cfg.vBlocks = vBlocks;
    cfg.numElemsPerLoad = numElemsPerLoad;
    cfg.numValuesPerLoad = numValuesPerLoad;

    if (!hasDpasEncoding(tensorType) && !hasDotDpasEncoding(tensorType))
      return cfg;

    auto dpasLayout = getDpasLayout(tensorType);
    unsigned opsPerChannel = dpasLayout.getOpsPerChannel();
    DpasEncodingAttr::OpIdx opIdx = getOpIdx(tensorType);

    auto numDPASOperandsPerLoad = [=](const SmallVector<unsigned> &shape) {
      unsigned elemsPerLanePerDPASInst =
          product<unsigned>(shape) / threadsPerWarp;
      unsigned numOps = 0;
      // Make sure the tile shape can fit the DPAS instruction shape.
      if (cfg.tileHeight >= shape[isTransposeRequired ? 1 : 0] &&
          (cfg.tileWidth * numPackedVals * cfg.vBlocks) >=
              shape[isTransposeRequired ? 0 : 1]) {
        numOps =
            mlir::ceil<unsigned>(cfg.numElemsPerLoad, elemsPerLanePerDPASInst);
      }
      return std::make_tuple(numOps, elemsPerLanePerDPASInst);
    };

    switch (opIdx) {
    case DpasEncodingAttr::OpIdx::OperandA: {
      auto [numDPASOperands, elemsPerLanePerDPASInst] =
          numDPASOperandsPerLoad(dpasLayout.getDPASInstShapeA());

      if (((opsPerChannel == 4 && elemSizeInBits == 8) ||
           (opsPerChannel == 2 && elemSizeInBits == 16) ||
           (opsPerChannel == 1 && elemSizeInBits == 32)) &&
          numDPASOperands) {
        // Add the packedDPASOperandType to add the shuffle and bitcast ops.
        cfg.packedDPASOperandType = LLVM::getVectorType(
            cfg.packedType, elemsPerLanePerDPASInst / numPackedVals);
        cfg.unpackedType = LLVM::getVectorType(eltTy, elemsPerLanePerDPASInst);
      }
    } break;
    case DpasEncodingAttr::OpIdx::OperandB: {
      auto [numDPASOperands, elemsPerLanePerDPASInst] =
          numDPASOperandsPerLoad(dpasLayout.getDPASInstShapeB());

      if (((opsPerChannel == 4 && elemSizeInBits == 8) ||
           (opsPerChannel == 2 && elemSizeInBits == 16) ||
           (opsPerChannel == 1 && elemSizeInBits == 32)) &&
          numDPASOperands) {
        if (!isTransposeRequired &&
            ((opsPerChannel == 4 && elemSizeInBits == 8) ||
             (opsPerChannel == 2 && elemSizeInBits == 16))) {
          // Use the VNNI packing format for DotOp B layout.
          cfg.numValuesPerLoad = cfg.numElemsPerLoad / opsPerChannel;
          cfg.packedType = IntegerType::get(ctx, 32);
          cfg.load2DGenXType =
              LLVM::getVectorType(cfg.packedType, cfg.numValuesPerLoad);
          cfg.useVNNIFormat = true;
        }

        // Add the packedDPASOperandType to add the shuffle and bitcast
        // ops.
        cfg.packedDPASOperandType = LLVM::getVectorType(
            cfg.packedType, elemsPerLanePerDPASInst / opsPerChannel);
        cfg.unpackedType = LLVM::getVectorType(eltTy, elemsPerLanePerDPASInst);
      }
    } break;
    case DpasEncodingAttr::OpIdx::OperandC: {
      auto [numDPASOperands, elemsPerLanePerDPASInst] =
          numDPASOperandsPerLoad(dpasLayout.getDPASInstShapeC());
      // Block 2D loads multiple Cs.
      if (cfg.numElemsPerLoad >= elemsPerLanePerDPASInst) {
        static const bool multipleCPerLoad = triton::tools::getBoolEnv(
            "TRITON_INTEL_2DBLOCK_MULTIPLE_C_MATRICES_PER_LOAD");
        if (!isTransposeRequired && !multipleCPerLoad) {
          assert(numPackedVals == 1 &&
                 "invalid numPackedVals for DPAS C operand");
          cfg.tileHeight = dpasLayout.getDPASInstShapeC()[0];
          cfg.tileWidth = dpasLayout.getDPASInstShapeC()[1];
          cfg.vBlocks = 1;
          cfg.numElemsPerLoad = elemsPerLanePerDPASInst;
          cfg.numValuesPerLoad = cfg.numElemsPerLoad;
          cfg.load2DGenXType =
              LLVM::getVectorType(cfg.packedType, cfg.numElemsPerLoad);
          cfg.unpackedType = LLVM::getVectorType(eltTy, cfg.numElemsPerLoad);
        } else {
          // Add the packedDPASOperandType to add the shuffle and bitcast
          // ops.
          cfg.packedDPASOperandType = LLVM::getVectorType(
              cfg.packedType, elemsPerLanePerDPASInst / numPackedVals);
          cfg.unpackedType =
              LLVM::getVectorType(eltTy, elemsPerLanePerDPASInst);
        }
      }
    } break;
    }

    return cfg;
  }

  static FailureOr<LinearLayout> computeTransposeShuffleMapping(
      RankedTensorType tensorType, const LinearLayout &regMapping,
      int64_t numElemsPerLoad, unsigned numPackedVals, unsigned tileHeight,
      unsigned threadsPerWarp, bool hasDPASOperandType, MLIRContext *ctx) {
    return triton::gpu::intel::computeTransposeShuffleMapping(
        tensorType, regMapping, numElemsPerLoad, numPackedVals, tileHeight,
        threadsPerWarp, hasDPASOperandType, ctx);
  }

  /// Build a Block2DLoadConfig from a validated BlockIOTileSizeInfo.
  /// Consolidates register mapping, type computation, DPAS configuration,
  /// and shuffle mapping into a single config object.
  static Block2DLoadConfig
  buildBlock2DLoadConfig(RankedTensorType tensorType, Type eltTy,
                         const BlockIOTileSizeInfo &sizeInfo,
                         const LinearLayout &llEncoding,
                         unsigned threadsPerWarp, MLIRContext *ctx) {
    assert(sizeInfo.isValid() && "expected valid tile size info");
    assert(sizeInfo.regPackedBases.has_value() &&
           "invalid register bases for packing elems.");

    Block2DLoadConfig cfg;
    cfg.numPackedVals = sizeInfo.numElemPerPackedVal;
    cfg.rowDim = sizeInfo.rowDim;
    cfg.colDim = sizeInfo.colDim;
    cfg.isTransposeRequired = sizeInfo.transpose;
    cfg.threadsPerWarp = threadsPerWarp;

    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    cfg.packedElemSizeInBits = elemSizeInBits * cfg.numPackedVals;

    // Build register mapping from packed bases.
    StringAttr kRegister = S("register");
    std::vector<std::vector<int>> bases(sizeInfo.regPackedBases->size());
    llvm::transform(*sizeInfo.regPackedBases, bases.begin(),
                    [](int base) { return std::vector<int>{base}; });
    cfg.regMapping = LinearLayout(
        {{kRegister, bases}}, {{kRegister, llEncoding.getInDimSize(kRegister)}},
        /*requireSurjective=*/true);

    cfg.numElems = getTotalElemsPerThread(tensorType);

    // Compute initial load types.
    cfg.numElemsPerLoad =
        mlir::ceil(sizeInfo.tileHeight * sizeInfo.tileWidth *
                       static_cast<int>(cfg.numPackedVals) * sizeInfo.vBlocks,
                   static_cast<int>(threadsPerWarp));
    cfg.numValuesPerLoad = mlir::ceil(static_cast<int>(cfg.numElemsPerLoad),
                                      static_cast<int>(cfg.numPackedVals));
    cfg.packedType = IntegerType::get(ctx, cfg.packedElemSizeInBits);
    cfg.load2DGenXType =
        LLVM::getVectorType(cfg.packedType, cfg.numValuesPerLoad);
    cfg.unpackedType = LLVM::getVectorType(eltTy, cfg.numElemsPerLoad);

    // Apply DPAS-specific type configuration.
    DPASLoadConfig dpasCfg = configureDPASLoadTypes(
        tensorType, eltTy, cfg.packedType, cfg.load2DGenXType, cfg.unpackedType,
        elemSizeInBits, cfg.numPackedVals, threadsPerWarp, sizeInfo.tileHeight,
        sizeInfo.tileWidth, sizeInfo.vBlocks, cfg.numElemsPerLoad,
        cfg.numValuesPerLoad, cfg.isTransposeRequired, ctx);
    cfg.packedDPASOperandType = dpasCfg.packedDPASOperandType;
    cfg.unpackedType = dpasCfg.unpackedType;
    cfg.load2DGenXType = dpasCfg.load2DGenXType;
    cfg.packedType = dpasCfg.packedType;
    cfg.useVNNIFormat = dpasCfg.useVNNIFormat;
    cfg.tileHeight = dpasCfg.tileHeight;
    cfg.tileWidth = dpasCfg.tileWidth;
    cfg.vBlocks = dpasCfg.vBlocks;
    cfg.numElemsPerLoad = dpasCfg.numElemsPerLoad;
    cfg.numValuesPerLoad = dpasCfg.numValuesPerLoad;

    // Build shuffle mapping (identity unless transpose required).
    cfg.shuffleMapping =
        LinearLayout::identity1D(cfg.numElemsPerLoad, kRegister, kRegister);
    if (cfg.isTransposeRequired) {
      auto maybeShuffleMapping = computeTransposeShuffleMapping(
          tensorType, cfg.regMapping, cfg.numElemsPerLoad, cfg.numPackedVals,
          cfg.tileHeight, threadsPerWarp, !!cfg.packedDPASOperandType, ctx);
      assert(succeeded(maybeShuffleMapping) &&
             "validate2DBlockLoadTile should have rejected this configuration");
      cfg.shuffleMapping = *maybeShuffleMapping;
    }

    return cfg;
  }

  /// Unpack a 2D block load result into individual element values.
  /// Populates unpackedLoadedVals[registerIdx] for each unpacked element.
  /// Optionally applies mask/other/NaN padding when otherElems or
  /// nanMaskElems are non-empty.
  static void unpackBlockLoadResult(
      Value ret, MutableArrayRef<Value> unpackedLoadedVals, size_t elemIdx,
      const LinearLayout &regMapping, const LinearLayout &shuffleMapping,
      Type packedDPASOperandType, Type unpackedType, unsigned numValuesPerLoad,
      unsigned numPackedVals, Value pred, ArrayRef<Value> otherElems,
      ArrayRef<Value> nanMaskElems, Location loc,
      ConversionPatternRewriter &rewriter, MLIRContext *ctx) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    StringAttr kRegister = S("register");

    unsigned numElemsPerUnpackedType =
        LLVM::getVectorNumElements(unpackedType).getKnownMinValue();
    unsigned numValsPerDPASOperand =
        packedDPASOperandType
            ? LLVM::getVectorNumElements(packedDPASOperandType)
                  .getKnownMinValue()
            : numValuesPerLoad;
    unsigned numOperandsPerLoad = numValuesPerLoad / numValsPerDPASOperand;

    for (size_t opsIdx = 0; opsIdx < numOperandsPerLoad; ++opsIdx) {
      Value unpackedVal;
      if (numValsPerDPASOperand != numValuesPerLoad) {
        // Decompose the return value to multiple DPAS operands.
        SmallVector<int32_t> indices(numValsPerDPASOperand);
        for (int i = 0; i < numValsPerDPASOperand; ++i) {
          unsigned packedElemIdx =
              (opsIdx * numValsPerDPASOperand + i) * numPackedVals;
          unsigned shuffleIdx =
              shuffleMapping.apply({{kRegister, packedElemIdx}})[0].second;
          indices[i] = shuffleIdx / numPackedVals;
        }
        DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);
        Value dpasOperand = LLVM::ShuffleVectorOp::create(
            rewriter, loc, packedDPASOperandType, ret, ret, attr);

        unpackedVal = b.bitcast(dpasOperand, unpackedType);
      } else {
        unpackedVal = b.bitcast(ret, unpackedType);
      }

      SmallVector<int32_t> unpackIndices(numElemsPerUnpackedType);
      for (int i = 0; i < numElemsPerUnpackedType; ++i) {
        unsigned elemIdxInPackedValue = opsIdx * numElemsPerUnpackedType + i;
        unsigned shuffledIdx =
            shuffleMapping.apply({{kRegister, elemIdxInPackedValue}})[0].second;
        unsigned registerIdx =
            regMapping.apply({{kRegister, elemIdx + shuffledIdx}})[0].second;
        unpackIndices[i] = registerIdx;
      }

      if (otherElems.size()) {
        assert(pred && "pred must be set when otherElems is non-empty");
        Value other = b.undef(unpackedType);
        for (const auto [i, registerIdx] : llvm::enumerate(unpackIndices)) {
          Value falseVal = otherElems[registerIdx];
          other = b.insert_element(other, falseVal, b.i32_val(i));
        }
        unpackedVal = b.select(pred, unpackedVal, other);
      } else if (nanMaskElems.size() != 0) {
        Type unpackedElemType = getElementTypeOrSelf(unpackedType);
        auto floatType = cast<FloatType>(unpackedElemType);

        SmallVector<Attribute> constOtherElems;
        for (auto i = 0; i < numElemsPerUnpackedType; ++i) {
          constOtherElems.push_back(
              FloatAttr::get(unpackedElemType,
                             APFloat::getNaN(floatType.getFloatSemantics())));
        }

        Value other = b.const_val(
            unpackedType,
            DenseElementsAttr::get(
                VectorType::get(numElemsPerUnpackedType, unpackedElemType),
                constOtherElems));

        Value packedPred =
            b.undef(VectorType::get(numElemsPerUnpackedType, i1_ty));

        for (const auto [i, registerIdx] : llvm::enumerate(unpackIndices)) {
          packedPred = b.insert_element(packedPred, nanMaskElems[registerIdx],
                                        b.i32_val(i));
        }
        unpackedVal = b.select(packedPred, unpackedVal, other);
      }

      for (const auto [i, registerIdx] : llvm::enumerate(unpackIndices)) {
        unpackedLoadedVals[registerIdx] =
            b.extract_element(unpackedVal, b.i32_val(i));
      }
    }
  }

  /// Adjust other dimension offsets and optionally add boundary checking.
  /// Used by both LoadOp and DescriptorLoadOp to handle batch dimensions.
  void adjustOtherDimension(Value &adjustedOffset, Value &addrElem, Value &pred,
                            Type eltTy, ArrayRef<Value> strides,
                            ArrayRef<Value> shapes, unsigned dim,
                            bool hasBoundaryCheck, Location loc,
                            ConversionPatternRewriter &rewriter,
                            TritonLLVMOpBuilder &b) const {
    MLIRContext *ctx = rewriter.getContext();
    Type i64Ty = IntegerType::get(ctx, 64);
    adjustedOffset = b.zext(i64Ty, adjustedOffset);
    Value p = b.mul(adjustedOffset, strides[dim]);
    addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, p);
    if (hasBoundaryCheck) {
      // Add boundary checking for other dims with predication.
      pred = maybeAnd(rewriter, loc, pred,
                      b.icmp_ult(adjustedOffset, shapes[dim]));
    }
  }
};

// Compute the 2D prefetch tile shape and warp tiling for cooperative
// prefetching. The tensor shape must be in row-major order.
// Returns: {tileHeight, tileWidth, warpsM, warpsN}
static std::tuple<unsigned, unsigned, unsigned, unsigned>
get2DPrefetchWarpsPerCTA(ArrayRef<int64_t> tensorShape, Type eltTy,
                         unsigned numWarps, bool isPrefetch256BSupported) {
  unsigned rank = tensorShape.size();
  assert(rank >= 2 && "Only rank >= 2 tensor is supported for now");
  unsigned dimM = rank - 2, dimN = rank - 1;
  unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
  unsigned elemSizeInBytes = elemSizeInBits / 8;
  unsigned numColsPerPrefOps = std::min<unsigned>(
      tensorShape[dimN],
      (isPrefetch256BSupported ? 256 : 64) / elemSizeInBytes);
  if (isPrefetch256BSupported && (numColsPerPrefOps * elemSizeInBytes) != 256) {
    // Fallback to 64 bytes per row.
    numColsPerPrefOps =
        std::min<unsigned>(numColsPerPrefOps, 64 / elemSizeInBytes);
  }

  unsigned repNumN = mlir::ceil((unsigned)tensorShape[dimN], numColsPerPrefOps);
  unsigned warpsNumN = std::min(numWarps, repNumN);
  unsigned warpsNumM = mlir::ceil(numWarps, warpsNumN);

  // Get the number of rows per warp to fit the shape to the tensor shape to
  // avoid duplication in prefetching.
  unsigned rowNumPerWarp = mlir::ceil<unsigned>(tensorShape[dimM], warpsNumM);
  constexpr unsigned maxNumRows = 32u;
  unsigned numRowsPerPrefOps = std::min<unsigned>(rowNumPerWarp, maxNumRows);

  return {numRowsPerPrefOps, numColsPerPrefOps, warpsNumM, warpsNumN};
}

// Get the linear layout for cooperative prefetching.
// tileShape and warpsPerCTA are always rank-2 (inner M×N dims).
// For rank > 2 tensors, they are padded with leading 1s for batch dims.
static LinearLayout getPrefetchLinearLayout(MLIRContext *ctx,
                                            ArrayRef<int64_t> tensorShape,
                                            ArrayRef<unsigned> tileShape,
                                            ArrayRef<unsigned> warpsPerCTA) {
  unsigned tensorRank = tensorShape.size();
  assert(tensorRank >= 2 && "Only rank >= 2 tensor is supported");

  // Pad tile/warp shapes with leading 1s for batch dimensions.
  SmallVector<unsigned> fullTileShape(tensorRank, 1);
  SmallVector<unsigned> fullWarpsPerCTA(tensorRank, 1);
  fullTileShape[tensorRank - 2] = tileShape[0];
  fullTileShape[tensorRank - 1] = tileShape[1];
  fullWarpsPerCTA[tensorRank - 2] = warpsPerCTA[0];
  fullWarpsPerCTA[tensorRank - 1] = warpsPerCTA[1];

  SmallVector<unsigned> order(tensorRank);
  for (unsigned i = 0; i < tensorRank; ++i)
    order[i] = tensorRank - i - 1;

  LinearLayout ctaLayout =
      identityStandardND(S("offset"), fullTileShape, order) *
      identityStandardND(S("warp"), fullWarpsPerCTA, order);

  return combineCtaCgaWithShape(std::move(ctaLayout),
                                CGAEncodingAttr::get1CTALayout(ctx, tensorRank),
                                tensorShape);
}

// Prefetch-specific cache control mapping. Differs from LoadOp in that
// `NONE` defaults to L1C_L3C (traditional prefetch-aggressively policy)
// rather than DEFAULT. Explicit user/pass-set modifiers (e.g., `.cg` from
// the AnnotateCacheControl pass) still propagate through.
static TritonGEN::LoadCacheControl prefetchCacheControl(CacheModifier cm) {
  switch (cm) {
  case CacheModifier::NONE:
  case CacheModifier::CA:
    return TritonGEN::LoadCacheControl::L1C_L3C;
  case CacheModifier::CG:
    return TritonGEN::LoadCacheControl::L1UC_L3C;
  case CacheModifier::CV:
    return TritonGEN::LoadCacheControl::L1UC_L3UC;
  default:
    return TritonGEN::LoadCacheControl::L1C_L3C;
  }
}

/// Emit 2D block prefetch operations for a tiled prefetch.
///
/// Converts base dimensions to bytes, determines vBlocks from element size,
/// computes per-tile offsets via the linear layout, and creates
/// Matrix2DBlockPrefetchOp for each tile. Erases the original op on success.
static LogicalResult emit2DBlockPrefetchOps(
    Operation *op, ConversionPatternRewriter &rewriter, Location loc,
    Value base, Value baseWidth, Value baseHeight, Value rowStride,
    Value offsetBaseX, Value offsetBaseY, Type eltTy, unsigned tileWidthInElem,
    unsigned tileHeightInElem, unsigned numTilesPerWarp,
    unsigned tileSizeInElem, const LinearLayout &llEncoding, unsigned rank = 2,
    ArrayRef<Value> extraDimStridesInBytes = {},
    ArrayRef<Value> extraDimBaseOffsets = {}, Value scalarPred = {},
    TritonGEN::LoadCacheControl cacheOpt =
        TritonGEN::LoadCacheControl::L1C_L3C) {
  assert(extraDimStridesInBytes.size() == rank - 2 &&
         extraDimBaseOffsets.size() == rank - 2 &&
         "extraDim arrays must have rank - 2 elements");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();

  // Convert baseWidth and rowStride to bytes and truncate to i32.
  baseWidth = b.mul(baseWidth, b.i64_val(eltTy.getIntOrFloatBitWidth() / 8));
  baseWidth = b.trunc(i32_ty, baseWidth);

  baseHeight = b.trunc(i32_ty, baseHeight);

  Value rowStrideInBytes =
      b.mul(rowStride, b.i64_val(eltTy.getIntOrFloatBitWidth() / 8));
  rowStrideInBytes = b.trunc(i32_ty, rowStrideInBytes);

  unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
  unsigned vBlocks = 1;
  switch (elemSizeInBits) {
  case 8:
    if (tileWidthInElem == 64) {
      // OCL interface supports 8b_?r32x2c for 64 bytes per row of 8 bits
      // element.
      vBlocks = 2;
      tileWidthInElem = 32;
    }
    break;
  case 16:
    if (tileWidthInElem == 32) {
      // OCL interface supports 16b_?r16x2c for 64 bytes per row of 16 bits
      // element.
      vBlocks = 2;
      tileWidthInElem = 16;
    }
    break;
  }

  StringAttr kOffset = S("offset");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");

  Value warpId = arith::IndexCastOp::create(
      rewriter, loc, i32_ty,
      mlir::gpu::SubgroupIdOp::create(rewriter, loc,
                                      /*upperBound=*/nullptr));

  for (unsigned tile = 0; tile < numTilesPerWarp; ++tile) {
    unsigned off = tile * tileSizeInElem;
    auto offsets = applyLinearLayout(
        loc, rewriter, llEncoding,
        {{kOffset, b.i32_val(off)}, {kWarp, warpId}, {kBlock, b.i32_val(0)}});
    Value offsetX = b.add(offsets[rank - 1].second, offsetBaseX);
    Value offsetY = b.add(offsets[rank - 2].second, offsetBaseY);

    // If a uniform predicate is supplied (e.g. from loop-pipeline epilogue
    // predication), set offsetY to baseHeight when the predicate is false so
    // the HW out-of-bounds check skips the prefetch without generating
    // spurious traffic.
    if (scalarPred)
      offsetY = b.select(scalarPred, offsetY, baseHeight);

    // Fold extra dimensions (beyond the inner 2) into the base pointer.
    Value adjustedBase = base;
    Type i8Ty = IntegerType::get(ctx, 8);
    Type i64Ty = IntegerType::get(ctx, 64);
    for (unsigned d = 0; d < rank - 2; ++d) {
      Value extraOff = b.add(b.zext(i64Ty, extraDimBaseOffsets[d]),
                             b.zext(i64Ty, offsets[d].second));
      Value byteOff = b.mul(extraOff, extraDimStridesInBytes[d]);
      adjustedBase = b.gep(ptr_ty(ctx, 1), i8Ty, adjustedBase, byteOff);
    }

    auto newOp = TritonGEN::Matrix2DBlockPrefetchOp::create(
        rewriter, loc,
        /*ptr*/ adjustedBase,
        /*base_width*/ baseWidth,
        /*base_height*/ baseHeight,
        /*base_pitch*/ rowStrideInBytes,
        /*x*/ offsetX,
        /*y*/ offsetY,
        /*elem_size_in_bits*/ elemSizeInBits,
        /*tile_width*/ tileWidthInElem,
        /*tile_height*/ tileHeightInElem,
        /*v_blocks*/ vBlocks,
        /*cache_opt*/ cacheOpt);
    if (failed(newOp.verify())) {
      // Delete the op so that the verifier will not abort the pass
      // pipeline later, as we can fail this path and try a different
      // approach.
      rewriter.eraseOp(newOp);
      return failure();
    }
  }

  rewriter.eraseOp(op);
  return success();
}

struct PrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::PrefetchOp>::ConvertTritonGPUOpToLLVMPattern;

  PrefetchOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>(
            converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::gpu::intel::PrefetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    LogicalResult res = rewriteRegularPointerPrefetch(op, adaptor, rewriter);
    if (succeeded(res))
      return success();

    res = rewriteCooperativePrefetch(op, adaptor, rewriter);
    if (succeeded(res))
      return success();

    // FIXME: the prefetch lowering code should never fail. Currently it does in
    // some cases. We should address those cases instead of removing the
    // prefetch operation.
    op.emitWarning("Prefetch operation could not be converted to LLVM. "
                   "The operation was erased.");
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult
  rewriteRegularPointerPrefetch(triton::gpu::intel::PrefetchOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr)
      return failure();

    const bool memoryRowMajor = isMemoryRowMajor(op);
    // TODO: To support more layouts on memory.
    if (!memoryRowMajor)
      return failure();

    auto tensorOfPointers = cast<RankedTensorType>(op.getPtr().getType());
    std::optional<DotOperandEncodingAttr> encoding =
        getDotEncoding(tensorOfPointers);
    if (!encoding)
      return failure();

    auto dpasLayout = cast<DpasEncodingAttr>(encoding->getParent());
    SmallVector<unsigned> warpsPerCTA(dpasLayout.getWarpsPerCTA());
    ArrayRef<unsigned> cluster = dpasLayout.getRepCluster();
    SmallVector<unsigned> repCluster{cluster.begin(), cluster.end()};
    ArrayRef<int64_t> tensorShape = tensorOfPointers.getShape();
    unsigned rank = tensorShape.size();
    DpasEncodingAttr::OpIdx opIdx = getOpIdx(tensorOfPointers);
    SmallVector<int64_t> repetitions =
        dpasLayout.getDPASRepetitions(tensorShape, opIdx);
    // getDPASRepetitions prepends a batch dimension; strip it.
    SmallVector<unsigned> numReps;
    for (size_t i = 1; i < repetitions.size(); ++i)
      numReps.push_back(repetitions[i]);

    SmallVector<int64_t> shardTensorShape(tensorShape.begin(),
                                          tensorShape.end());
    switch (opIdx) {
    case DpasEncodingAttr::OpIdx::OperandA: {
      shardTensorShape[rank - 2] =
          std::min<int64_t>(tensorShape[rank - 2], dpasLayout.getShapeA()[0]);
      // K dim (last) not distributed across warps.
      warpsPerCTA[rank - 1] = 1;
      repCluster[rank - 1] = 1;
      numReps[rank - 1] = 1;
    } break;
    case DpasEncodingAttr::OpIdx::OperandB: {
      shardTensorShape[rank - 1] =
          std::min<int64_t>(tensorShape[rank - 1], dpasLayout.getShapeB()[1]);
      // K dim (second-to-last) not distributed across warps.
      warpsPerCTA[rank - 2] = 1;
      repCluster[rank - 2] = 1;
      numReps[rank - 2] = 1;
    } break;
    case DpasEncodingAttr::OpIdx::OperandC: {
      llvm_unreachable("unexpected OpIdx::OperandC");
    } break;
    }

    auto ptrType = cast<PointerType>(tensorOfPointers.getElementType());
    Type elementType = ptrType.getPointeeType();
    auto tensorType = RankedTensorType::get(shardTensorShape, elementType,
                                            tensorOfPointers.getEncoding());

    Value mask = op.getMask();
    unsigned maskConstancyHor = std::numeric_limits<unsigned>::max(),
             maskConstancyVer = std::numeric_limits<unsigned>::max();
    if (mask) {
      // No need to check the constancy of scalar mask.
      if (auto maskTy = dyn_cast_or_null<RankedTensorType>(mask.getType())) {
        maskConstancyHor = maskConstancyVer = 1;
        AxisInfo *axisInfo =
            const_cast<triton::intel::ModuleAxisInfoAnalysis &>(
                axisAnalysisPass)
                .getAxisInfo(mask);
        if (axisInfo) {
          maskConstancyHor = axisInfo->getConstancy(rank - 1);
          maskConstancyVer = axisInfo->getConstancy(rank - 2);
        }
      }
    }

    SmallVector<unsigned, 2> prefetchShape =
        get2DPrefetchShapePerWarp(tensorType);
    prefetchShape = {std::min<unsigned>(prefetchShape[0], maskConstancyVer),
                     std::min<unsigned>(prefetchShape[1], maskConstancyHor)};

    SmallVector<int64_t> numPrefetchsPerRep = {
        mlir::ceil<int64_t>(shardTensorShape[rank - 2], prefetchShape[0]),
        mlir::ceil<int64_t>(shardTensorShape[rank - 1], prefetchShape[1])};

    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    unsigned tileWidthInElem = prefetchShape[1];
    unsigned tileHeightInElem = prefetchShape[0];
    unsigned vBlocks = 1;
    switch (elemSizeInBits) {
    case 8:
      if (tileWidthInElem == 64) {
        // OCL interface supports 8b_?r32x2c for 64 bytes per row of 8 bits
        // element.
        vBlocks = 2;
        tileWidthInElem = 32;
      }
      break;
    case 16:
      if (tileWidthInElem == 32) {
        // OCL interface supports 16b_?r16x2c for 64 bytes per row of 16 bits
        // element.
        vBlocks = 2;
        tileWidthInElem = 16;
      }
      break;
    }

    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    std::map<SmallVector<unsigned>, Value> baseAddrs, masks;
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();

    // Get the LLVM values for pointers
    SmallVector<Value> ptrElems = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    // re-arrange the baseAddrs and masks to for large 2D block IO.
    // Layout is unrelated to the scalar type.
    SmallVector<SmallVector<unsigned>> offsets =
        emitOffsetForLayout(*encoding, tensorOfPointers);
    for (size_t i = 0; i < ptrElems.size(); ++i) {
      SmallVector<unsigned> offset = offsets[i];
      baseAddrs[offset] = ptrElems[i];
      if (llMask && maskElems.size() > 1)
        masks[offset] = maskElems[i];
    }

    Value rowStrideInBytes =
        getPitch(rewriter, op.getPtr(), elemSizeInBits, memoryRowMajor ? 0 : 1);
    if (!rowStrideInBytes)
      return failure();

    // If the stride is 0, we want to load only the first row.
    int stride = getStride(op.getPtr(), rank - 2);
    Value baseHeight = b.i32_val(stride == 0 ? 1 : tileHeightInElem);
    Value baseWidth = b.i32_val(
        std::max(64u, vBlocks * tileWidthInElem * (elemSizeInBits / 8)));
    Value offsetBaseX = b.i32_val(0);
    Value offsetBaseY = b.i32_val(0);

    // Compute total batch repetitions for dims beyond the inner 2.
    int64_t totalBatchReps = 1;
    for (unsigned d = 0; d < rank - 2; ++d)
      totalBatchReps *= numReps[d];

    for (int64_t batchIdx = 0; batchIdx < totalBatchReps; ++batchIdx) {
      // Decompose batchIdx into per-dim batch indices.
      SmallVector<unsigned> batchOffsets(rank - 2);
      int64_t remaining = batchIdx;
      for (int d = static_cast<int>(rank) - 3; d >= 0; --d) {
        batchOffsets[d] = remaining % numReps[d];
        remaining /= numReps[d];
      }
      assert(remaining == 0 &&
             "batchIdx decomposition inconsistent with totalBatchReps");

      for (unsigned row = 0; row < numReps[rank - 2]; ++row) {
        for (unsigned col = 0; col < numReps[rank - 1]; ++col) {
          // Prefetch the data for each repetition.
          for (int64_t i = 0; i < numPrefetchsPerRep[0]; ++i)
            for (int64_t j = 0; j < numPrefetchsPerRep[1]; ++j) {
              unsigned offsetN =
                  col * warpsPerCTA[rank - 1] * shardTensorShape[rank - 1] +
                  j * prefetchShape[1];
              unsigned offsetM =
                  row * warpsPerCTA[rank - 2] * shardTensorShape[rank - 2] +
                  i * prefetchShape[0];

              // Build the full offset key for the baseAddrs/masks map.
              SmallVector<unsigned> key;
              for (unsigned d = 0; d < rank - 2; ++d)
                key.push_back(batchOffsets[d] * warpsPerCTA[d] *
                              shardTensorShape[d]);
              key.push_back(offsetM);
              key.push_back(offsetN);

              Value pred;
              if (llMask)
                pred = (maskElems.size() > 1)
                           ? targetInfo.shuffleIdx(rewriter, loc, masks[key], 0)
                           : maskElems[0];
              else
                pred = b.int_val(1, 1);

              // If the mask exists and evaluates to false, we set offsetY to be
              // equal to baseHeight, which causes the HW to ignore the
              // generated prefetch operation (given that the block to be
              // prefetched would be outside the baseWidth X baseHeight shape).
              Value offsetY = b.select(pred, b.i32_val(0), baseHeight);
              Value addr =
                  targetInfo.shuffleIdx(rewriter, loc, baseAddrs[key], 0);

              auto newOp = TritonGEN::Matrix2DBlockPrefetchOp::create(
                  rewriter, loc,
                  /*ptr*/ addr,
                  /*base_width*/ baseWidth,
                  /*base_height*/ baseHeight,
                  /*base_pitch*/ rowStrideInBytes,
                  /*x*/ offsetBaseX,
                  /*y*/ offsetY,
                  /*elem_size_in_bits*/ elemSizeInBits,
                  /*tile_width*/ tileWidthInElem,
                  /*tile_height*/ tileHeightInElem,
                  /*v_blocks*/ vBlocks,
                  /*cache_opt*/ prefetchCacheControl(op.getCache()));
              if (failed(newOp.verify())) {
                // delete the op so that the verifier will not abort the pass
                // pipeline later, as we can fail this path and try a different
                // approach.
                rewriter.eraseOp(newOp);
                return failure();
              }
            }
        }
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

  /// Handle prefetch for loads with BlockedEncodingAttr (non-dot).
  /// Follows the same cooperative 2D-block-prefetch pattern as
  /// DescriptorPrefetchOpConversion::rewriteTensorDescriptorPrefetch() but
  /// extracts base/stride from tensor-of-pointers instead of a descriptor.
  LogicalResult
  rewriteCooperativePrefetch(triton::gpu::intel::PrefetchOp op,
                             OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr)
      return failure();

    // Masked prefetches can arise from loop-pipeline epilogue predication.
    // `MatmulLoopPipeline::predicateOp` wraps the existing mask with
    // `tt::getPredMask`, which yields either `splat(pred)` (when the prefetch
    // had no prior mask) or `arith.andi(splat(pred), priorMask)` (when it
    // did, e.g. the load had a bounds check). The cooperative path uses a
    // single uniform base pointer plus per-warp offsets and can only honor a
    // uniform (scalar) predicate; the per-element `priorMask` component
    // cannot be applied here. Because a prefetch is a hint (dropping a
    // per-element mask is safe — the HW bounds check on the 2D surface
    // still rejects out-of-surface offsets), we extract any uniform
    // component(s) and AND them into a scalar predicate. If no uniform
    // component can be found (e.g., the mask is purely `arith.cmpi`), bail
    // so the generic path can handle it.
    const bool memoryRowMajor = isMemoryRowMajor(op);
    if (!memoryRowMajor)
      return failure();

    // A tensor value is "uniform" if `AxisInfo` proves its constancy covers
    // the full shape in every dimension (same semantics as the regular path
    // at line 1729–1736). Scalars are trivially uniform.
    auto isUniform = [&](Value v) {
      auto ty = dyn_cast<RankedTensorType>(v.getType());
      if (!ty)
        return true;
      AxisInfo *info =
          const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
              .getAxisInfo(v);
      if (!info)
        return false;
      ArrayRef<int64_t> shape = ty.getShape();
      for (unsigned d = 0, e = ty.getRank(); d < e; ++d)
        if (info->getConstancy(d) < static_cast<unsigned>(shape[d]))
          return false;
      return true;
    };

    // Collect operands that `AxisInfo` proves are uniform, descending
    // through `arith.andi` chains (as produced by `tt::getPredMask`).
    // Non-uniform leaves are dropped — safe for a prefetch hint.
    SmallVector<Value> uniformOps;
    std::function<void(Value)> collect = [&](Value v) {
      if (isUniform(v)) {
        uniformOps.push_back(v);
        return;
      }
      if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
        collect(andOp.getLhs());
        collect(andOp.getRhs());
      }
    };

    Value scalarPred;
    if (Value mask = op.getMask()) {
      collect(mask);
      if (uniformOps.empty())
        return failure();

      // Reduce each uniform operand to a scalar `i1` (element 0 of the
      // packed LLVM representation — all lanes hold the same value by
      // definition of uniformity) and AND them together.
      auto tb = TritonLLVMOpBuilder(op.getLoc(), rewriter);
      for (Value v : uniformOps) {
        Value remapped = rewriter.getRemappedValue(v);
        if (!remapped)
          return failure();
        Value scalar;
        if (isa<RankedTensorType>(v.getType())) {
          SmallVector<Value> elems =
              unpackLLElements(op.getLoc(), remapped, rewriter);
          if (elems.empty())
            return failure();
          scalar = elems[0];
        } else {
          scalar = remapped;
        }
        scalarPred = scalarPred ? tb.and_(scalarPred, scalar) : scalar;
      }
    }

    auto tensorOfPointers = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!tensorOfPointers)
      return failure();
    unsigned rank = tensorOfPointers.getRank();
    if (rank < 2)
      return failure();

    ArrayRef<int64_t> tensorShape = tensorOfPointers.getShape();
    auto ptrType = cast<PointerType>(tensorOfPointers.getElementType());
    Type elementType = ptrType.getPointeeType();
    Type eltTy = getTypeConverter()->convertType(elementType);
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    // 2D block prefetch requires byte-addressable element sizes. Sub-byte
    // types (e.g. i1, i4) would trigger division by zero in
    // get2DPrefetchWarpsPerCTA and downstream byte-based math.
    if (elemSizeInBits < 8 || elemSizeInBits % 8 != 0)
      return failure();
    unsigned elemSizeInBytes = elemSizeInBits / 8;

    int numWarps = triton::gpu::lookupNumWarps(op);
    auto m = op->getParentOfType<ModuleOp>();
    bool isPrefetch256BSupported =
        m->hasAttr(TritonIntelGPUDialect::getSupportPrefetch256BAttrName());

    auto [tileHeightInElem, tileWidthInElem, warpsM, warpsN] =
        get2DPrefetchWarpsPerCTA(tensorShape, eltTy, numWarps,
                                 isPrefetch256BSupported);
    auto llEncoding = getPrefetchLinearLayout(
        getContext(), tensorShape, {tileHeightInElem, tileWidthInElem},
        {warpsM, warpsN});

    unsigned tileSizeInElem = tileHeightInElem * tileWidthInElem;
    int64_t totalElems = 1;
    for (auto s : tensorShape)
      totalElems *= s;
    unsigned numTilesPerWarp = totalElems / (tileSizeInElem * numWarps);

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Recover a uniform scalar base pointer from the tensor-of-pointers.
    // The cooperative path relies on a single base with per-warp offsets added
    // later via applyLinearLayout(kWarp=warpId). Using ptrElems[0] would give
    // each warp its register-0 pointer, which already encodes that warp's
    // position — combining it with the warp offset would double-shift the
    // address. Require the tensor-of-pointers to be produced by a chain of
    // `tt.addptr` ending in `tt.splat(%base)` so we can extract the scalar
    // `%base` as the uniform pointer. Otherwise bail out so the generic
    // fallback can handle it.
    Value cur = op.getPtr();
    while (auto addPtrOp = cur.getDefiningOp<triton::AddPtrOp>())
      cur = addPtrOp.getPtr();
    auto splatOp = cur.getDefiningOp<triton::SplatOp>();
    if (!splatOp)
      return failure();
    Value base = rewriter.getRemappedValue(splatOp.getSrc());
    if (!base)
      return failure();

    // Row stride in elements (i64) -- emit2DBlockPrefetchOps converts to bytes.
    int stride = getStride(op.getPtr(), rank - 2);
    if (stride < 0)
      return failure();
    Value rowStride = b.i64_val(stride);

    // Surface dimensions for the 2D block prefetch hardware.
    // baseWidth must equal the row stride (surface width) because the HW
    // computes addresses as base + y * pitch + x. Using the tile width would
    // be wrong when pitch > tile width.
    // baseHeight is the full logical tensor height (surface height), not the
    // tile height, so the hardware's bounds checking spans the whole tensor.
    Value baseWidth = b.i64_val(stride);
    Value baseHeight = b.i64_val(tensorShape[rank - 2]);

    // Offsets start at 0; per-warp offsets computed by emit2DBlockPrefetchOps.
    Value offsetBaseX = b.i32_val(0);
    Value offsetBaseY = b.i32_val(0);

    // Prepare extra-dim strides (bytes, i64) and base offsets (i32) for
    // rank > 2. Silently substituting 0 for an unknown/negative stride
    // would synthesize aliasing addresses across outer-dim slices (all
    // slices would prefetch the same (x,y) surface), so bail out instead.
    SmallVector<Value> extraDimStrides, extraDimBaseOffsets;
    for (unsigned d = 0; d < rank - 2; ++d) {
      int dimStride = getStride(op.getPtr(), d);
      if (dimStride <= 0)
        return failure();
      extraDimStrides.push_back(b.i64_val(dimStride * elemSizeInBytes));
      extraDimBaseOffsets.push_back(b.i32_val(0));
    }

    return emit2DBlockPrefetchOps(
        op, rewriter, loc, base, baseWidth, baseHeight, rowStride, offsetBaseX,
        offsetBaseY, eltTy, tileWidthInElem, tileHeightInElem, numTilesPerWarp,
        tileSizeInElem, llEncoding, rank, extraDimStrides, extraDimBaseOffsets,
        scalarPred, prefetchCacheControl(op.getCache()));
  }
};

struct DescriptorPrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::intel::DescriptorPrefetchOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::DescriptorPrefetchOp>::
      ConvertTritonGPUOpToLLVMPattern;

  DescriptorPrefetchOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<
            triton::gpu::intel::DescriptorPrefetchOp>(converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::gpu::intel::DescriptorPrefetchOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    LogicalResult res = rewriteTensorDescriptorPrefetch(op, adaptor, rewriter);

    // FIXME: the prefetch lowering code should never fail. Currently it does in
    // some cases. We should address those cases instead of removing the
    // prefetch operation.
    if (failed(res)) {
      op.emitWarning("Descriptor prefetch operation could not be converted to "
                     "LLVM. The operation was erased.");
      rewriter.eraseOp(op);
    }

    return success();
  }

  LogicalResult
  rewriteTensorDescriptorPrefetch(triton::gpu::intel::DescriptorPrefetchOp op,
                                  OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr) {
      rewriter.eraseOp(op);
      return success();
    }

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // Extract the tensor type from the TensorDescType.
    // Unlike PrefetchOp which gets its type from PointerType::getPointeeType(),
    // DescriptorPrefetchOp gets it from TensorDescType::getBlockType().
    auto descType = cast<triton::TensorDescType>(op.getDesc().getType());
    RankedTensorType tensorType = descType.getBlockType();
    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    const ArrayRef<int64_t> shapeRef = tensorType.getShape();
    SmallVector<int64_t> tensorShape{shapeRef.begin(), shapeRef.end()};

    // TODO: Revisit when TensorDescriptor supports column-major layout.
    // Currently, TensorDescriptor is always row major.

    unsigned numWarps = triton::gpu::lookupNumWarps(op);

    auto m = op->getParentOfType<ModuleOp>();
    bool isPrefetch256BSupported =
        m->hasAttr(TritonIntelGPUDialect::getSupportPrefetch256BAttrName());

    auto [tileHeightInElem, tileWidthInElem, warpsM, warpsN] =
        get2DPrefetchWarpsPerCTA(tensorShape, eltTy, numWarps,
                                 isPrefetch256BSupported);
    auto llEncoding = getPrefetchLinearLayout(
        getContext(), tensorShape, {tileHeightInElem, tileWidthInElem},
        {warpsM, warpsN});

    unsigned tileSizeInElem = tileHeightInElem * tileWidthInElem;
    int64_t totalElems = 1;
    for (auto s : tensorShape)
      totalElems *= s;
    unsigned numTilesPerWarp = totalElems / (tileSizeInElem * numWarps);

    // Unpack the tensor descriptor struct.
    unsigned rank = tensorType.getRank();
    unsigned rowIdx = rank - 2, colIdx = rank - 1;
    DescriptorFields desc =
        unpackDescriptor(adaptor.getDesc(), rank, loc, rewriter);

    // Get base width and height from the descriptor shape fields.
    // TODO: baseWidth/baseHeight come from the MakeTensorDescOp shape operands
    // which represent the bounds of the underlying memory, not the tile shape
    // (set by MakeTensorDescOp). Verify this is consistent.
    Value baseWidth = desc.shapes[colIdx];
    Value baseHeight = desc.shapes[rowIdx];
    Value rowStride = desc.strides[rowIdx];

    // Get offset bases from the indices operand.
    // For tensor descriptors, indices are supplied explicitly via the op
    // (unlike block pointers where offsets are embedded in the struct).
    ValueRange indices = adaptor.getIndices();
    assert(indices.size() == rank &&
           "Expected indices count to match tensor rank");
    Value offsetBaseY = indices[rowIdx]; // row offset
    Value offsetBaseX = indices[colIdx]; // col offset

    // Prepare extra-dim strides (in bytes, i64) and base offsets (i32) for
    // dimensions beyond the inner 2.
    unsigned elemSizeInBytes = eltTy.getIntOrFloatBitWidth() / 8;
    SmallVector<Value> extraDimStrides, extraDimBaseOffsets;
    for (unsigned d = 0; d < rank - 2; ++d) {
      extraDimStrides.push_back(
          b.mul(desc.strides[d], b.i64_val(elemSizeInBytes)));
      extraDimBaseOffsets.push_back(indices[d]);
    }

    // Emit the 2D block prefetch operations.
    return emit2DBlockPrefetchOps(
        op, rewriter, loc, desc.base, baseWidth, baseHeight, rowStride,
        offsetBaseX, offsetBaseY, eltTy, tileWidthInElem, tileHeightInElem,
        numTilesPerWarp, tileSizeInElem, llEncoding, rank, extraDimStrides,
        extraDimBaseOffsets, /*scalarPred=*/Value{},
        prefetchCacheControl(op.getCache()));
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LoadOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(op.getType()));
    unsigned numElems = getTotalElemsPerThread(op.getType());
    unsigned vec = getVectorSize(hasSupport256bLoadStore(op), ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    // Get the LLVM values for pointers
    SmallVector<Value> ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems);
    }

    // Get the LLVM values for `other`
    // TODO: (goostavz) handle when other is const but not splat, which
    //       should be rarely seen
    SmallVector<Value> otherElems;
    bool otherIsSplatConstInt = false;
    int64_t splatVal = 0;
    DenseElementsAttr constAttr;
    if (other && isa<IntegerType>(valueElemTy) &&
        matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat() &&
        isa<IntegerType>(constAttr.getElementType())) {
      otherIsSplatConstInt = true;
      splatVal = constAttr.getSplatValue<APInt>().getSExtValue();
    }
    if (other) {
      otherElems = unpackLLElements(loc, llOther, rewriter);
    }

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    // Load redundantly in all dims except reg
    llvm::MapVector<StringAttr, int> freeVarMasks =
        getFreeVariableMasks(ptr.getType());
    uint32_t regMask = freeVarMasks[str_attr("register")];

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (unsigned canonicalVecStart = getCanonicalIndex(vecStart, regMask);
          vecStart != canonicalVecStart) {
        // For redundant registers, refer back to the canonical load
        for (int iVec = 0; iVec < vec; ++iVec)
          loadedVals.push_back(loadedVals[canonicalVecStart + iVec]);

        continue;
      }

      // TODO: optimization when ptr is GEP with constant offset
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = maskElems.size() ? maskElems[vecStart] : Value{};

      SmallVector<Type> retTys(nWords, IntegerType::get(ctx, width));
      Type retTy = retTys.size() > 1
                       ? vec_ty(IntegerType::get(ctx, width), nWords)
                       : retTys[0];

      Value other_ = b.undef(retTy);
      if (otherElems.empty()) {
        other_ = LLVM::ConstantOp::create(rewriter, loc, retTy,
                                          rewriter.getZeroAttr(retTy));
      } else {
        for (size_t ii = 0; ii < nWords; ++ii) {
          size_t size = width / valueElemNBits;
          VectorType vecTy = vec_ty(valueElemTy, size);
          Value v = b.undef(vecTy);
          for (size_t s = 0; s < size; ++s) {
            Value falseVal = otherElems[vecStart + ii * size + s];
            Value sVal = createIndexAttrConstant(
                rewriter, loc, typeConverter->getIndexType(), s);
            v = b.insert_element(vecTy, v, falseVal, sVal);
          }
          v = b.bitcast(v, IntegerType::get(ctx, width));

          if (otherIsSplatConstInt) {
            for (size_t s = 0; s < 32; s += valueElemNBits)
              splatVal |= splatVal << valueElemNBits;
            v = b.int_val(width, splatVal);
          }

          other_ =
              (nWords > 1)
                  ? b.insert_element(
                        retTy, other_, v,
                        createIndexAttrConstant(
                            rewriter, loc, typeConverter->getIndexType(), ii))
                  :

                  v;
        }
      }
      assert(other_ && "Expecting a valid value");

      Value addrElem = b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
      uint32_t alignment = nWords * width / 8;
      auto createLoadWithAttrs = [&]() {
        return SmallVector<Value>{b.load(retTy, addrElem, alignment,
                                         op.getIsVolatile(),
                                         getNonTemporalFlag(op))};
      };

      Value ret;
      if (!pred)
        ret = createLoadWithAttrs()[0];
      else if (canUsePredicatedInstructions(op)) {
        auto cacheModifier = tritonToIntelCacheModifier(op);
        ret = TritonGEN::PredicatedLoadOp::create(
            rewriter, loc, retTy, addrElem, pred, other_, cacheModifier);
      } else {
        Block &endBlock = LLVM::intel::createPredicatedBlock(
            rewriter, loc, pred, SmallVector<Value, 1>{other_},
            createLoadWithAttrs);
        ret = *endBlock.args_begin();
      }
      assert(ret && "Expecting a valid value");

      // Extract and store return values
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr = isa<VectorType>(retTy)
                         ? b.extract_element(IntegerType::get(ctx, width), ret,
                                             b.i32_val(ii))
                         : ret;
        unsigned numElem = width / valueElemNBits;
        if (numElem == 1)
          curr = b.bitcast(curr, valueElemTy);
        else
          curr = b.bitcast(curr, LLVM::getVectorType(valueElemTy, numElem));
        rets.push_back(curr);
      }

      int tmp = width / valueElemNBits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value loaded = rets[ii / tmp];
        if (isa<VectorType>(loaded.getType()))
          loaded = b.extract_element(valueElemTy, loaded, b.i32_val(ii % tmp));
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct DescriptorLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::DescriptorLoadOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<
      triton::DescriptorLoadOp>::ConvertOpToLLVMPattern;

  DescriptorLoadOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DescriptorLoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();

    // Get the descriptor and indices
    Value llDesc = adaptor.getDesc();
    ValueRange indices =
        adaptor.getIndices(); // These are the offsets (i32 values)

    // Get result type information
    auto resultType = cast<RankedTensorType>(op.getType());
    Type valueElemTy = typeConverter->convertType(resultType.getElementType());
    unsigned numElems = getTotalElemsPerThread(resultType);

    // Descriptor rank may differ from result rank for rank-reducing loads.
    auto descType = cast<triton::TensorDescType>(op.getDesc().getType());
    RankedTensorType descTensorType = descType.getBlockType();
    size_t descRank = descTensorType.getRank();
    auto blockIOAttr = op->getAttrOfType<StringAttr>(
        TritonIntelGPUDialect::getBlockIOAttrName());
    bool permuteDescDim =
        blockIOAttr && symbolizeBlockIOMode(blockIOAttr.getValue()) ==
                           BlockIOMode::ColumnMajor;
    assertDescriptorInnerShapeCompatible(op, descTensorType.getShape(),
                                         resultType.getShape(), permuteDescDim);

    // Get padding from the propagated attribute (set by
    // MaterializeBlockPointer).
    PaddingOption padding = PaddingOption::PAD_ZERO;
    if (auto paddingAttr = op->getAttrOfType<triton::PaddingOptionAttr>(
            TritonIntelGPUDialect::getDescPaddingAttrName()))
      padding = paddingAttr.getValue();

    // Boundary check all dimensions — tensor descriptors always encode shape
    // bounds and don't have a user-facing boundaryCheck attribute.
    SmallVector<int32_t> allDims(descRank);
    for (size_t i = 0; i < descRank; ++i)
      allDims[i] = static_cast<int32_t>(i);

    // Reuse the shared gather/scatter operand computation.
    // For column_major descriptor loads the result type has its inner 2
    // dimensions transposed relative to the descriptor's natural order. For
    // example, a rank-2 descriptor [N, K] produces result [K, N], and a rank-3
    // descriptor [Batch, N, K] produces result [Batch, K, N]. emitIndices uses
    // the result type's dimension space, but the descriptor struct encodes
    // shapes/strides in descriptor space. Swap the inner 2 dimensions of
    // shapes, strides, and offsets so that they align with the result type's
    // dimension order. Outer (batch) dimensions are preserved unchanged.
    SmallVector<Value> ptrElems, maskElems, otherElems;
    if (permuteDescDim) {
      DescriptorFields desc = unpackDescriptor(llDesc, descRank, loc, rewriter);
      SmallVector<Value> permShapes(descRank), permStrides(descRank),
          permOffsets(descRank);
      for (unsigned i = 0; i < descRank; ++i) {
        permShapes[i] = desc.shapes[i];
        permStrides[i] = desc.strides[i];
        permOffsets[i] = indices[i];
      }
      // Swap only the inner 2 dimensions (2D block I/O constraint).
      assert(descRank >= 2 &&
             "column_major descriptor load requires rank >= 2");
      std::swap(permShapes[descRank - 2], permShapes[descRank - 1]);
      std::swap(permStrides[descRank - 2], permStrides[descRank - 1]);
      std::swap(permOffsets[descRank - 2], permOffsets[descRank - 1]);
      SmallVector<Value> mappedShapes(descRank), mappedStrides(descRank),
          mappedOffsets(descRank);
      for (size_t i = 0; i < descRank; ++i) {
        mappedShapes[i] = permShapes[i];
        mappedStrides[i] = permStrides[i];
        mappedOffsets[i] = permOffsets[i];
      }
      std::tie(ptrElems, maskElems, otherElems) = computeGatherScatterOperands(
          loc, desc.base, mappedOffsets, mappedShapes, mappedStrides,
          resultType, valueElemTy, rewriter, allDims, padding);
    } else {
      std::tie(ptrElems, maskElems, otherElems) =
          convertTensorDescriptorToTensorOfPtr(loc, llDesc, indices, resultType,
                                               descRank, valueElemTy, rewriter,
                                               allDims, padding);
    }

    // Determine vectorization by querying the descriptor's address-level
    // AxisInfo, analogous to how LoadOp queries getVectorSize(ptr).
    unsigned vec = getDescriptorVecSize(
        hasSupport256bLoadStore(op), op.getDesc(), resultType, valueElemTy,
        op->getAttrOfType<StringAttr>(
            TritonIntelGPUDialect::getBlockIOAttrName()));

    // vectorized iteration through all pointer elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const int numVecs = numElems / vec;

    // Load redundantly in all dims except reg
    llvm::MapVector<StringAttr, int> freeVarMasks =
        getFreeVariableMasks(resultType);
    uint32_t regMask = freeVarMasks[str_attr("register")];

    SmallVector<Value> loadedVals;
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (unsigned canonicalVecStart = getCanonicalIndex(vecStart, regMask);
          vecStart != canonicalVecStart) {
        // For redundant registers, refer back to the canonical load
        for (unsigned iVec = 0; iVec < vec; ++iVec)
          loadedVals.push_back(loadedVals[canonicalVecStart + iVec]);
        continue;
      }

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      assert((width / valueElemNBits) * nWords * numVecs == numElems);

      // Get the predicate mask for this element (always present for
      // DescriptorLoadOp since we always do boundary checking)
      Value pred = maskElems[vecStart];

      SmallVector<Type> retTys(nWords, IntegerType::get(ctx, width));
      Type retTy = retTys.size() > 1
                       ? vec_ty(IntegerType::get(ctx, width), nWords)
                       : retTys[0];

      // Build the "other" value for out-of-bounds (same pattern as LoadOp)
      Value other_ = b.undef(retTy);
      for (size_t ii = 0; ii < nWords; ++ii) {
        size_t size = width / valueElemNBits;
        VectorType vecTy = vec_ty(valueElemTy, size);
        Value v = b.undef(vecTy);
        for (size_t s = 0; s < size; ++s) {
          Value falseVal = otherElems[vecStart + ii * size + s];
          Value sVal = createIndexAttrConstant(
              rewriter, loc, typeConverter->getIndexType(), s);
          v = b.insert_element(vecTy, v, falseVal, sVal);
        }
        v = b.bitcast(v, IntegerType::get(ctx, width));
        other_ = (nWords > 1)
                     ? b.insert_element(retTy, other_, v,
                                        createIndexAttrConstant(
                                            rewriter, loc,
                                            typeConverter->getIndexType(), ii))
                     : v;
      }
      assert(other_ && "Expecting a valid value");

      Value addrElem = b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
      uint32_t alignment = nWords * width / 8;

      auto createLoadWithAttrs = [&]() {
        return SmallVector<Value>{b.load(retTy, addrElem, alignment,
                                         /*isVolatile=*/false,
                                         getNonTemporalFlag(op))};
      };

      Value ret;
      // NOTE: For DescriptorLoadOp, pred is always present since we always
      // perform boundary checking for the gather fallback.
      if (canUsePredicatedInstructions(op)) {
        auto cacheModifier = tritonToIntelCacheModifier(op);
        ret = TritonGEN::PredicatedLoadOp::create(
            rewriter, loc, retTy, addrElem, pred, other_, cacheModifier);
      } else {
        Block &endBlock = LLVM::intel::createPredicatedBlock(
            rewriter, loc, pred, SmallVector<Value, 1>{other_},
            createLoadWithAttrs);
        ret = *endBlock.args_begin();
      }
      assert(ret && "Expecting a valid value");

      // Extract and store return values
      SmallVector<Value> rets;
      for (unsigned int ii = 0; ii < nWords; ++ii) {
        Value curr = isa<VectorType>(retTy)
                         ? b.extract_element(IntegerType::get(ctx, width), ret,
                                             b.i32_val(ii))
                         : ret;
        unsigned numElem = width / valueElemNBits;
        if (numElem == 1)
          curr = b.bitcast(curr, valueElemTy);
        else
          curr = b.bitcast(curr, LLVM::getVectorType(valueElemTy, numElem));
        rets.push_back(curr);
      }

      int tmp = width / valueElemNBits;
      for (size_t ii = 0; ii < vec; ++ii) {
        Value loaded = rets[ii / tmp];
        if (isa<VectorType>(loaded.getType()))
          loaded = b.extract_element(valueElemTy, loaded, b.i32_val(ii % tmp));
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, loadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct DescriptorStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::DescriptorStoreOp>,
      public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<
      triton::DescriptorStoreOp>::ConvertOpToLLVMPattern;

  DescriptorStoreOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DescriptorStoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();

    // Get the descriptor, value, and indices
    Value llDesc = adaptor.getDesc();
    Value llValue = adaptor.getSrc();
    auto indices = adaptor.getIndices(); // These are the offsets (i32 values)

    // Get value type information
    auto valueTy = cast<RankedTensorType>(op.getSrc().getType());
    Type valueElemTy = typeConverter->convertType(valueTy.getElementType());
    unsigned numElems = getTotalElemsPerThread(valueTy);

    // Descriptor rank may differ from stored value rank for rank-reducing
    // stores.
    auto descType = cast<triton::TensorDescType>(op.getDesc().getType());
    RankedTensorType descTensorType = descType.getBlockType();
    size_t descRank = descTensorType.getRank();
    assertDescriptorInnerShapeCompatible(op, descTensorType.getShape(),
                                         valueTy.getShape());

    // Boundary check all dimensions — tensor descriptors always encode shape
    // bounds and don't have a user-facing boundaryCheck attribute.
    SmallVector<int32_t> allDims(descRank);
    for (size_t i = 0; i < descRank; ++i)
      allDims[i] = static_cast<int32_t>(i);

    // Reuse the shared gather/scatter operand computation.
    SmallVector<Value> ptrElems, maskElems, dummyOther;
    std::tie(ptrElems, maskElems, dummyOther) =
        convertTensorDescriptorToTensorOfPtr(loc, llDesc, indices, valueTy,
                                             descRank, valueElemTy, rewriter,
                                             allDims);

    // Unpack the value elements
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    // NOTE: DescriptorStoreOp does not have a mask operand.
    // Unlike StoreOp which has an optional mask.
    // We use the redundant thread predicate for deduplication across
    // warps/blocks.
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("register")];

    // Determine vectorization by querying the descriptor's address-level
    // AxisInfo, analogous to how StoreOp queries getVectorSize(ptr).
    unsigned vec = getDescriptorVecSize(hasSupport256bLoadStore(op),
                                        op.getDesc(), valueTy, valueElemTy,
                                        /*blockIOAttr=*/nullptr);

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    unsigned elemsPerThread = numElems;
    const int numVecs = elemsPerThread / vec;

    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = b.undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = b.sext(i8_ty, elem);
          elem = b.bitcast(elem, valueElemTy);

          llWord = b.insert_element(wordTy, llWord, elem, b.i32_val(elemIdx));
        }
        llWord = b.bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      // Combine the thread redundancy predicate with the per-element boundary
      // mask (always present for DescriptorStoreOp since we check all dims).
      Value maskVal = threadPred;
      if (maskElems.size() > 0) {
        auto mask = maskElems[vecStart];
        maskVal = maybeAnd(rewriter, loc, threadPred, mask);
      }

      auto vecTy = vec_ty(valArgTy, nWords);
      Value vecWord = b.undef(vecTy);
      for (size_t index = 0; index < asmArgs.size(); ++index) {
        auto llWord = asmArgs[index].first;
        if (nWords == 1)
          vecWord = llWord;
        else
          vecWord = b.insert_element(vecTy, vecWord, llWord, b.i32_val(index));
      }

      Value addrElem = b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
      uint32_t alignment = nWords * width / 8;

      // NOTE: DescriptorStoreOp does not have isVolatile or cache attributes.
      // StoreOp has these and uses getNonTemporalFlag.
      // For now, generate a simple store without non-temporal hints.
      // TODO: Consider adding cache hint support if needed.
      auto createStoreWithAttrs = [&]() {
        bool isVolatile = false;
        bool isNonTemporal = false;
        b.store(vecWord, addrElem, alignment, isVolatile, isNonTemporal);
        return ArrayRef<Value>();
      };

      if (!maskVal) {
        (void)createStoreWithAttrs();
      } else if (canUsePredicatedInstructions(op)) {
        // DescriptorStoreOp does not have a cache attribute, so use DEFAULT.
        auto cacheModifier = TritonGEN::StoreCacheControl::DEFAULT;
        TritonGEN::PredicatedStoreOp::create(rewriter, loc, addrElem, vecWord,
                                             maskVal, cacheModifier);
      } else {
        LLVM::intel::createPredicatedBlock(rewriter, loc, maskVal,
                                           createStoreWithAttrs);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct DescriptorStoreOpToBlockIOConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::DescriptorStoreOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::DescriptorStoreOp>::ConvertTritonGPUOpToLLVMPattern;

  DescriptorStoreOpToBlockIOConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::DescriptorStoreOp>(converter,
                                                                   benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // --- Pre-conditions ---
    if (!isDescriptorBlockIOCandidate(op))
      return failure();

    // Read memory layout from block_io attribute (set by
    // MaterializeBlockPointer).
    StringRef blockIOName = TritonIntelGPUDialect::getBlockIOAttrName();
    StringAttr blockIOAttr = op->getAttrOfType<StringAttr>(blockIOName);
    assert(
        blockIOAttr &&
        "block_io attribute required; checked by isDescriptorBlockIOCandidate");
    const bool memoryRowMajor = (blockIOAttr.getValue() == "row_major");
    assert(memoryRowMajor && "column_major descriptor store not yet supported");

    // Get source tensor type and encoding.
    auto tensorType = cast<RankedTensorType>(op.getSrc().getType());
    Attribute encoding = tensorType.getEncoding();

    // --- Linear layout and tile size ---
    std::optional<LinearLayout> llEncoding =
        cast<DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    assert(llEncoding.has_value() &&
           "unexpected failure when getting linear layout");

    unsigned contiguousDim = memoryRowMajor ? 1 : 0;
    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();

    // TODO: DescriptorStoreOp has no mask operand, so maskAxisInfo is always
    // null. If masking support is added in the future, axis info should be
    // propagated here.
    AxisInfo *maskAxisInfo = nullptr;

    // Validate the store tile through the shared helper: it computes the tile
    // geometry, enforces the HW address payload restriction, rejects transpose,
    // and forces vBlocks to 1.
    BlockIOTileSizeInfo sizeInfo = BlockIOTileSizeInfo::unknown();
    if (!validate2DBlockStoreTile(llEncoding.value(), contiguousDim,
                                  elemSizeInBits, tensorType, maskAxisInfo,
                                  sizeInfo))
      return failure();

    auto [tileHeight, tileWidth, numPackedVals, vBlocks, rowDim, colDim,
          isTransposeRequired, regPackedBases] = std::move(sizeInfo);
    unsigned packedElemSizeInBits = elemSizeInBits * numPackedVals;

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value warpId = arith::IndexCastOp::create(
        rewriter, loc, i32_ty,
        mlir::gpu::SubgroupIdOp::create(rewriter, loc,
                                        /*upperBound=*/nullptr));

    // --- Unpack tensor descriptor struct ---
    unsigned rank = tensorType.getRank();
    auto descType = cast<triton::TensorDescType>(op.getDesc().getType());
    unsigned descRank = descType.getBlockType().getRank();
    assert(descRank >= rank &&
           "descriptor rank must be >= source rank for descriptor store");
    unsigned rankDelta = descRank - rank;
    assertDescriptorInnerShapeCompatible(op, descType.getBlockType().getShape(),
                                         tensorType.getShape());
    auto mapSrcDimToDescDim = [rankDelta](unsigned dim) {
      return dim + rankDelta;
    };
    DescriptorFields desc =
        unpackDescriptor(adaptor.getDesc(), descRank, loc, rewriter);

    const unsigned descRowDim = mapSrcDimToDescDim(rowDim);
    const unsigned descColDim = mapSrcDimToDescDim(colDim);

    unsigned numElems = getTotalElemsPerThread(tensorType);

    // The base pointer is uniform across all elements (unlike tensor-of-
    // pointers where each element may have a different pointer).
    SmallVector<Value> ptrElems(numElems, desc.base);

    // --- Shapes (base width / height for 2D block IO payload) ---
    // TensorDesc carries shape as i64 values. The 2D block IO payload
    // expects baseWidth in bytes and baseHeight in elements.
    Value shapeRow = desc.shapes[descRowDim]; // i64
    Value shapeCol = desc.shapes[descColDim]; // i64
    Value baseWidth =
        b.trunc(i32_ty, b.mul(shapeCol, b.i64_val(elemSizeInBits / 8)));
    Value baseHeight = b.trunc(i32_ty, shapeRow);

    // --- Pitch (row stride in bytes) ---
    // TODO: DescriptorStoreOp does not expose a "memory order" attribute, so
    // we always use the row-major stride dimension. Once a memory order
    // attribute is added, this should be adjusted.
    Value strideForPitch = desc.strides[descRowDim]; // i64
    Value pitch =
        b.trunc(i32_ty, b.mul(strideForPitch, b.i64_val(elemSizeInBits / 8)));

    // --- Offsets ---
    // Unlike block pointers which store offsets in the struct, tensor
    // descriptors receive offsets via the indices operand.
    auto indices = adaptor.getIndices();
    assert(indices.size() == descRank &&
           "descriptor index count must match descriptor rank");
    SmallVector<Value> baseOffsets(indices.begin(), indices.end());

    // --- Get the LLVM values for store values ---
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    assert(valElems.size() == numElems &&
           "the number of store values does not match the number of elements");

    // Although the getBlockTileShape makes sure there is no duplication within
    // a warp, we still need to deduplicate across warps and blocks.
    const llvm::MapVector<StringAttr, int> &freeVarMasks =
        getFreeVariableMasks(tensorType);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    unsigned threadsPerWarp =
        TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>());

    Type packedType = IntegerType::get(ctx, packedElemSizeInBits);
    unsigned numPackedElemsPerStore = (tileHeight * tileWidth) / threadsPerWarp;
    Type store2DGenXType =
        LLVM::getVectorType(packedType, numPackedElemsPerStore);
    unsigned numElemsPerStore = numPackedElemsPerStore * numPackedVals;
    Type store2DComposeType = LLVM::getVectorType(eltTy, numElemsPerStore);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    assert(regPackedBases.has_value() &&
           "invalid register bases for packing elems.");
    std::vector<std::vector<int>> bases(regPackedBases->size());
    llvm::transform(*regPackedBases, bases.begin(),
                    [](int base) { return std::vector<int>{base}; });
    LinearLayout regMapping({{kRegister, bases}},
                            {{kRegister, llEncoding->getInDimSize(kRegister)}},
                            /*requireSurjective=*/true);

    // --- Emit 2D block stores ---
    for (size_t valIdx = 0; valIdx < numElems; valIdx += numElemsPerStore) {
      unsigned registerIdx = regMapping.apply({{kRegister, valIdx}})[0].second;

      auto offsets = applyLinearLayout(loc, rewriter, *llEncoding,
                                       {{kRegister, b.i32_val(registerIdx)},
                                        {kLane, b.i32_val(0)},
                                        {kWarp, warpId},
                                        {kBlock, b.i32_val(0)}});
      assert(offsets.size() == 2 && "only support 2D tensor for now.");

      Value addrElem = ptrElems[registerIdx];

      // For tensor descriptors, we always have shape information and always
      // perform boundary protection on all dimensions (unlike block pointers
      // where boundaryCheck is user-specified).
      Value offsetX = b.add(baseOffsets[descColDim], offsets[colDim].second);
      Value offsetY = b.add(baseOffsets[descRowDim], offsets[rowDim].second);

      // Tensor descriptors always encode full shape bounds, so we always
      // use the descriptor's baseWidth/baseHeight for HW boundary
      // protection (no need to expand or adjust like block pointers).
      Value adjustedBaseWidth = baseWidth;
      Value adjustedBaseHeight = baseHeight;

      Value pred = threadPred;
      if (pred) {
        // We leverage the GPU block I/O hardware out-of-bound protection
        // feature by setting the offset to an invalid value when 'pred'
        // is false (the HW will not store out-of-bounds values).
        offsetY = b.select(pred, offsetY, adjustedBaseHeight);
      }

      assert(numPackedVals > 0 && "numPackedVals should be greater than zero.");

      // Compose the matrix by stacking the scalars into a vector.
      Value storeVal = LLVM::UndefOp::create(rewriter, loc, store2DComposeType);
      for (size_t i = 0; i < numElemsPerStore; ++i) {
        unsigned registerIdx =
            regMapping.apply({{kRegister, valIdx + i}})[0].second;
        storeVal =
            b.insert_element(storeVal, valElems[registerIdx], b.i32_val(i));
      }
      if (store2DComposeType != store2DGenXType)
        storeVal = b.bitcast(storeVal, store2DGenXType);

      auto newOp = TritonGEN::Matrix2DBlockStoreOp::create(
          rewriter, loc, addrElem, adjustedBaseWidth, adjustedBaseHeight, pitch,
          // offsetX was in terms of original elements. The 2D block IO requires
          // offsetX to be in terms of packed elements.
          b.udiv(offsetX, b.i32_val(numPackedVals)), offsetY,
          packedElemSizeInBits, tileWidth, tileHeight,
          /*v_blocks, only 1 supported*/ 1, storeVal);

      if (failed(newOp.verify())) {
        // Delete the op so that the verifier will not abort the pass
        // pipeline later, as we can fail this path and try a different
        // approach.
        rewriter.eraseOp(newOp);
        return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct StoreOpToBlockIOConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpToBlockIOConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!isBlockIOCandidate(op))
      return failure();

    // Get the max tile shape supported by the layout.
    auto tensorType = cast<RankedTensorType>(op.getValue().getType());
    Attribute encoding = tensorType.getEncoding();
    std::optional<LinearLayout> llEncoding =
        cast<DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    assert(llEncoding.has_value() &&
           "unexpected failure when getting linear layout");

    // TODO: use the axis info to general the handling for both regular
    // pointer and block pointer.
    const bool memoryRowMajor = isMemoryRowMajor(op);
    const unsigned rank = tensorType.getRank();
    if (rank > 2)
      return failure();
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;

    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    // Get the maximum tile shapes for the given mask constancy.
    AxisInfo *maskAxisInfo = nullptr;
    if (op.getMask()) {
      maskAxisInfo =
          const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
              .getAxisInfo(op.getMask());
    }
    BlockIOTileSizeInfo sizeInfo = BlockIOTileSizeInfo::unknown();
    MLIRContext *ctx = rewriter.getContext();

    if (hasAnnotated1DReshapeStride(op)) {
      // For stores annotated by the 1D→2D reshape, the encoding was inferred
      // by tt.reshape and may not satisfy getBlockIOTileSize constraints.
      // Specifically, the reshape produces a blocked encoding with
      // sizePerThread > maxElemPackedVal (e.g., sizePerThread[1]=8 vs
      // maxElemPackedVal=4 for f16). This creates a gap between the
      // register-packed extent and the first lane base that
      // getBlockIOTileSize cannot bridge.
      // TODO: extend getBlockIOTileSize to handle register bases in the
      // contiguous dimension beyond the packing limit, so this special case
      // can be removed.
      auto blockedEnc = dyn_cast<BlockedEncodingAttr>(encoding);
      assert(blockedEnc && "1D reshape store must have BlockedEncodingAttr");
      assert(rank == 2 && "1D reshape always produces rank-2 tensors");
      unsigned numWarpsRow = blockedEnc.getWarpsPerCTA()[0];
      int height = tensorType.getDimSize(0) / numWarpsRow;
      int width = tensorType.getDimSize(rank - 1);
      // Build register bases as identity mapping — every register holds one
      // element (numPackedVals=1), so all bases are included.
      StringAttr kRegister = str_attr("register");
      unsigned regDimSize = llEncoding->getInDimSize(kRegister);
      SetVector<unsigned> regPackBases;
      for (unsigned i = 1; i < regDimSize; i <<= 1)
        regPackBases.insert(i);
      sizeInfo = BlockIOTileSizeInfo(height, width, /*numElemPerPackedVal=*/1,
                                     /*vBlocks=*/1, /*rowDim=*/0,
                                     /*colDim=*/rank - 1, /*transpose=*/false,
                                     std::move(regPackBases));
      // The reshape path bypasses getBlockIOTileSize (and thus
      // validate2DBlockStoreTile), so apply the HW address payload restriction
      // here; vBlocks and transpose are already fixed (1 / false) above.
      if (!sizeInfo.isValid())
        return failure();
      if (!check2DBlockAddressPayloadRestriction(
              elemSizeInBits * sizeInfo.numElemPerPackedVal,
              sizeInfo.tileWidth))
        return failure();
    } else {
      // Validate through the shared helper (the single source of truth for 2D
      // block store eligibility): tile geometry, HW address payload
      // restriction, no transpose, vBlocks forced to 1.
      if (!validate2DBlockStoreTile(llEncoding.value(), contiguousDim,
                                    elemSizeInBits, tensorType, maskAxisInfo,
                                    sizeInfo))
        return failure();
    }

    auto [tileHeight, tileWidth, numPackedVals, vBlocks, rowDim, colDim,
          isTransposeRequired, regPackedBases] = std::move(sizeInfo);
    unsigned packedElemSizeInBits = elemSizeInBits * numPackedVals;

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value warpId = arith::IndexCastOp::create(
        rewriter, loc, i32_ty,
        mlir::gpu::SubgroupIdOp::create(rewriter, loc,
                                        /*upperBound=*/nullptr));

    Value llPtr = adaptor.getPtr();
    Value ptr = op.getPtr();
    unsigned numElems = getTotalElemsPerThread(tensorType);

    // Get the LLVM values for pointers
    SmallVector<Value> ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems &&
           "the number of pointer values is not matched with the number of "
           "elements");

    SmallVector<Value> maskElems;
    Value llMask = adaptor.getMask();
    // Get the LLVM values for mask
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems &&
             "the number of mask values is not matched with the number of "
             "elements");
    }

    Value baseWidth =
        b.i32_val(vBlocks * tileWidth * (packedElemSizeInBits / 8));
    Value baseHeight = b.i32_val(tileHeight);

    // Use the explicit stride attribute if set by the 1D→2D reshape,
    // otherwise compute pitch from pointer element analysis.
    Value pitch;
    if (auto pitchBytes = getAnnotated1DReshapePitch(op, elemSizeInBits)) {
      pitch = b.i32_val(*pitchBytes);
    } else {
      // Always get the stride of the row dim since block store only supports
      // row major matrix.
      pitch = getPitch(rewriter, ptr, elemSizeInBits, rowDim);
    }
    if (!pitch)
      return failure();
    Value offsetBaseX = b.i32_val(0);
    Value offsetBaseY = b.i32_val(0);

    // Get the LLVM values for store values
    SmallVector<Value> valElems =
        unpackLLElements(loc, adaptor.getValue(), rewriter);
    assert(valElems.size() == numElems &&
           "the number of store values does not match the number of elements");

    // Although the getBlockTileShape make sure there is no duplication within
    // warp, still need to deduplicate the value in across warps and blocks;
    const llvm::MapVector<StringAttr, int> &freeVarMasks =
        getFreeVariableMasks(tensorType);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    unsigned threadsPerWarp =
        TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>());

    Type packedType =
        IntegerType::get(ctx, packedElemSizeInBits); // make it opaque type.
    unsigned numPackedElemsPerStore = (tileHeight * tileWidth) / threadsPerWarp;
    Type store2DGenXType =
        LLVM::getVectorType(packedType, numPackedElemsPerStore);
    unsigned numElemsPerStore = numPackedElemsPerStore * numPackedVals;
    Type store2DComposeType = LLVM::getVectorType(eltTy, numElemsPerStore);

    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    StringAttr kBlock = str_attr("block");

    assert(regPackedBases.has_value() &&
           "invalid register bases for packing elems.");
    std::vector<std::vector<int>> bases(regPackedBases->size());
    llvm::transform(*regPackedBases, bases.begin(),
                    [](int base) { return std::vector<int>{base}; });
    LinearLayout regMapping({{kRegister, bases}},
                            {{kRegister, llEncoding->getInDimSize(kRegister)}},
                            /*requireSurjective=*/true);

    // Right now only support to stack the values into a vector in sequential
    // order.
    for (size_t valIdx = 0; valIdx < numElems; valIdx += numElemsPerStore) {
      unsigned registerIdx = regMapping.apply({{kRegister, valIdx}})[0].second;

      // Need to apply the linear layout to get the offsets to the base of
      // the block pointer.
      // TODO: add annotation uniform to the offsets. Make sure the IGC
      // detect the offsets as uniform.
      auto offsets = applyLinearLayout(loc, rewriter, *llEncoding,
                                       {{kRegister, b.i32_val(registerIdx)},
                                        {kLane, b.i32_val(0)},
                                        {kWarp, warpId},
                                        {kBlock, b.i32_val(0)}});
      // TODO: To support rank > 2 tensor, we need to add the offsets of
      // other dim to the base.
      assert(offsets.size() == 2 && "only support 2D tensor for now.");

      // TODO: the threadPred has to be the uniform value. Maybe just add an
      // attribute to notify IGC about this information.
      Value pred = threadPred;
      Value addrElem =
          targetInfo.shuffleIdx(rewriter, loc, ptrElems[registerIdx], 0);
      Value adjustedBaseWidth = baseWidth, adjustedBaseHeight = baseHeight;

      // Adjust the baseWidth, offsetX and base address use the original base
      // of the BLOCK.
      Value offsetX = offsets[colDim].second;
      Value offsetY = b.i32_val(0);
      Value negOffsetX = b.sub(b.i32_val(0), offsetX);
      addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, negOffsetX);
      // The offset is in number of original elements. So we need to scale it
      // by element bytes size.
      adjustedBaseWidth =
          b.add(baseWidth, b.mul(offsetX, b.i32_val(elemSizeInBits / 8)));
      adjustedBaseWidth = b.umax(adjustedBaseWidth, b.i32_val(64));

      // Use the top-left address and mask of the block to store the data.
      // (The first value refer by the registerIdx.)
      if (maskElems.size()) {
        assert(maskElems.size() == valElems.size() &&
               "Invalid size of the masks.");
        auto mask = maskElems[registerIdx];
        pred = maybeAnd(rewriter, loc, pred, mask);
        pred = targetInfo.shuffleIdx(rewriter, loc, pred, 0);
      }

      if (pred) {
        // We leverage the GPU block I/O hardware out-of-bound protection
        // feature by setting the offset to an invalid value when 'pred'
        // is false (the HW will not read out-of-bounds values).
        offsetY = b.select(pred, offsetY, adjustedBaseHeight);
      }
      assert(numPackedVals > 0 && "numPackedVals should be greater than zero.");

      // Compose the matrix by stacking the scalar into vector.
      Value storeVal = LLVM::UndefOp::create(rewriter, loc, store2DComposeType);
      for (size_t i = 0; i < numElemsPerStore; ++i) {
        unsigned registerIdx =
            regMapping.apply({{kRegister, valIdx + i}})[0].second;
        storeVal =
            b.insert_element(storeVal, valElems[registerIdx], b.i32_val(i));
      }
      if (store2DComposeType != store2DGenXType)
        storeVal = b.bitcast(storeVal, store2DGenXType);

      auto newOp = TritonGEN::Matrix2DBlockStoreOp::create(
          rewriter, loc, addrElem, adjustedBaseWidth, adjustedBaseHeight, pitch,
          // offsetX was in terms of original elements. The 2d block io requires
          // offsetX to be in terms of packed elements.
          b.udiv(offsetX, b.i32_val(numPackedVals)), offsetY,
          packedElemSizeInBits, tileWidth, tileHeight,
          /*v_blocks, only 1 supported*/ 1, storeVal);

      if (failed(newOp.verify())) {
        // delete the op so that the verifier will not abort the pass
        // pipeline later, as we can fail this path and try a different
        // approach.
        rewriter.eraseOp(newOp);
        return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct StoreOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto *typeConverter = getTypeConverter();
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();
    Value llMask = adaptor.getMask();

    // Determine the vectorization size
    Type valueTy = op.getValue().getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(hasSupport256bLoadStore(op), ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(op.getMask()));

    Value llPtr = adaptor.getPtr();
    SmallVector<Value> ptrElems = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElems;
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    Value llValue = adaptor.getValue();
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());
    assert(!maskElems.size() ||
           valueElems.size() == maskElems.size() && "Mask size mismatch");

    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("register")];

    const size_t dtsize =
        std::max<int>(1, valueElemTy.getIntOrFloatBitWidth() / 8);
    const size_t valueElemNBits = dtsize * 8;

    unsigned elemsPerThread = getTotalElemsPerThread(valueTy);
    const int numVecs = elemsPerThread / vec;
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }
      // TODO: optimization when ptr is AddPtr with constant offset
      size_t in_off = 0;

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      // TODO(Superjomn) Add cache policy fields to StoreOp.
      // TODO(Superjomn) Deal with cache policy here.

      Type valArgTy = IntegerType::get(ctx, width);
      auto wordTy = vec_ty(valueElemTy, wordNElems);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      for (size_t wordIdx = 0; wordIdx < nWords; ++wordIdx) {
        // llWord is a width-len composition
        Value llWord = b.undef(wordTy);
        // Insert each value element to the composition
        for (size_t elemIdx = 0; elemIdx < wordNElems; ++elemIdx) {
          const size_t elemOffset = vecStart + wordIdx * wordNElems + elemIdx;
          assert(elemOffset < valueElems.size());
          Value elem = valueElems[elemOffset];
          if (elem.getType().isInteger(1))
            elem = b.sext(i8_ty, elem);
          elem = b.bitcast(elem, valueElemTy);

          llWord = b.insert_element(wordTy, llWord, elem, b.i32_val(elemIdx));
        }
        llWord = b.bitcast(llWord, valArgTy);
        std::string constraint =
            (width == 64) ? "l" : ((width == 32) ? "r" : "c");
        asmArgs.emplace_back(llWord, constraint);
      }

      Value maskVal = threadPred;
      if (maskElems.size() > 0) {
        auto mask = maskElems[vecStart];
        maskVal = maybeAnd(rewriter, loc, threadPred, mask);
      }

      auto vecTy = vec_ty(valArgTy, nWords);
      Value vecWord = b.undef(vecTy);
      for (int index = 0; index < asmArgs.size(); ++index) {
        auto llWord = asmArgs[index].first;
        if (nWords == 1)
          vecWord = llWord;
        else
          vecWord = b.insert_element(vecTy, vecWord, llWord, b.i32_val(index));
      }

      Value addrElem = b.bitcast(ptrElems[vecStart], ptr_ty(ctx, 1 /*global*/));
      uint32_t alignment = nWords * width / 8;
      auto createStoreWithAttrs = [&]() {
        bool isVolatile = false;
        b.store(vecWord, addrElem, alignment, isVolatile,
                getNonTemporalFlag(op));
        return ArrayRef<Value>();
      };

      if (!maskVal)
        auto _ = createStoreWithAttrs();
      else if (canUsePredicatedInstructions(op)) {
        auto cacheModifier = tritonToIntelCacheModifier(op);
        TritonGEN::PredicatedStoreOp::create(rewriter, loc, addrElem, vecWord,
                                             maskVal, cacheModifier);
      } else
        LLVM::intel::createPredicatedBlock(rewriter, loc, maskVal,
                                           createStoreWithAttrs);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

void createBarrier(ConversionPatternRewriter &rewriter, Location loc,
                   int numCTAs) {
  assert(numCTAs == 1 && "Expecting numCTA to be 1");
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  b.barrier(triton::gpu::AddrSpace::Local);
}

struct AtomicCASOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicCASOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicCASOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicCASOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());

    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value mask =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    SmallVector<Value> resultVals(elemsPerThread);

    MemSemantic memSem = op.getSem();
    LLVM::AtomicOrdering successOrdering = getMemoryOrdering(memSem)
                                               ? *getMemoryOrdering(memSem)
                                               : LLVM::AtomicOrdering::acq_rel;
    LLVM::AtomicOrdering failureOrdering = LLVM::AtomicOrdering::monotonic;

    bool support16BitAtomics = moduleOp->hasAttr(
        TritonIntelGPUDialect::getSupport16BitAtomicsAttrName());

    for (size_t i = 0; i < elemsPerThread; ++i) {
      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      Value casVal = valElements[i];

      assert((valueElemNBits == 16 || valueElemNBits == 32 ||
              valueElemNBits == 64) &&
             "Unexpected width");

      Value ret;
      Value zero = b.int_val(valueElemNBits, 0);
      if (valueElemNBits == 16 && !support16BitAtomics) {
        op.emitWarning("'tt.atomic_cas' op fp16/bf16 datatype is not supported "
                       "in the target HW, software emulation is an "
                       "experimental feature (use at own risk)");

        Block *endBlock =
            emulate16BitsCAS(rewriter, loc, casPtr, casCmp, casVal,
                             mask ? mask : b.true_val(), {zero});
        ret = endBlock->getArgument(0);
      } else {
        if (op.getResult().use_empty() && memSem != MemSemantic::RELAXED)
          TritonGEN::BarrierOp::create(rewriter, loc,
                                       TritonGEN::MemFence::GLOBAL);

        auto createAtomicCASInstruction = [&]() -> SmallVector<Value, 1> {
          Value localCasCmp = b.bitcast(casCmp, zero.getType());
          Value localCasVal = b.bitcast(casVal, zero.getType());

          auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
              rewriter, loc, casPtr, localCasCmp, localCasVal, successOrdering,
              failureOrdering);
          Value newLoaded =
              LLVM::ExtractValueOp::create(rewriter, loc, cmpxchg, 0);
          return SmallVector<Value, 1>{newLoaded};
        };

        if (mask) {
          Block &endBlock = LLVM::intel::createPredicatedBlock(
              rewriter, loc, mask, {zero}, createAtomicCASInstruction);
          ret = endBlock.getArgument(0);
        } else {
          ret = createAtomicCASInstruction()[0];
        }
      }

      ret = b.bitcast(ret, valueElemTy);

      if (tensorTy) {
        resultVals[i] = ret;
      } else {
        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        targetInfo.storeShared(rewriter, loc, atomPtr, ret, mask);
        createBarrier(rewriter, loc, numCTAs);
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
      }
    }

    if (tensorTy) {
      finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals,
                                  valueElemTy, b, mask, targetInfo,
                                  getTypeConverter());
    }
    return success();
  }

  Block *emulate16BitsCAS(ConversionPatternRewriter &rewriter, Location loc,
                          Value casPtr, Value casCmp, Value casVal, Value mask,
                          ArrayRef<Value> ops) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Block *insertionBlock = rewriter.getInsertionBlock();
    Block *headerBlock =
        rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
    Block *endBlock = rewriter.splitBlock(headerBlock, headerBlock->begin());

    rewriter.setInsertionPointToEnd(insertionBlock);
    cf::CondBranchOp::create(rewriter, loc, mask, headerBlock, endBlock, ops);
    rewriter.setInsertionPointToStart(headerBlock);

    casCmp = b.bitcast(casCmp, i16_ty);
    casVal = b.bitcast(casVal, i16_ty);

    auto intPtr = b.ptrtoint(i64_ty, casPtr);
    auto lowPtrBits = b.and_(intPtr, b.i64_val(3));
    auto elemIndex = b.trunc(i32_ty, b.lshr(lowPtrBits, b.i64_val(1)));
    auto alignedPtr =
        b.inttoptr(casPtr.getType(), b.sub(intPtr, lowPtrBits).getResult());

    auto firstValInt = b.load(i32_ty, alignedPtr, 4, false, false, false, false,
                              LLVM::AtomicOrdering::acquire);

    Block *bodyBlock =
        rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
    auto origValInt =
        bodyBlock->addArgument(firstValInt.getType(), firstValInt.getLoc());
    rewriter.setInsertionPointToEnd(headerBlock);
    cf::BranchOp::create(rewriter, loc, bodyBlock,
                         SmallVector<Value, 1>{firstValInt});
    rewriter.setInsertionPointToStart(bodyBlock);

    auto origValVec = b.bitcast(origValInt, vec_ty(i16_ty, 2));
    auto origVal = b.extract_element(origValVec, elemIndex);

    Value isEqual = b.icmp_eq(origVal, casCmp);

    Block *casBlock =
        rewriter.splitBlock(bodyBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToEnd(bodyBlock);
    SmallVector<Value, 1> exitOps = {origVal};
    cf::CondBranchOp::create(rewriter, loc, isEqual, casBlock, ValueRange{},
                             endBlock, exitOps);
    rewriter.setInsertionPointToStart(casBlock);

    Value newValVec = b.insert_element(origValVec, casVal, elemIndex);
    Value newValInt = b.bitcast(newValVec, i32_ty);

    auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, alignedPtr, origValInt, newValInt,
        LLVM::AtomicOrdering::acq_rel, LLVM::AtomicOrdering::monotonic);

    auto newLoaded = b.extract_val(cmpxchg, 0);
    auto done = b.extract_val(cmpxchg, 1);

    SmallVector<Value, 1> endOps = {origVal};
    cf::CondBranchOp::create(rewriter, loc, done, endBlock, endOps, bodyBlock,
                             SmallVector<Value, 1>{newLoaded});

    for (Value op : ops)
      endBlock->addArgument(op.getType(), op.getLoc());

    rewriter.setInsertionPointToStart(endBlock);
    return endBlock;
  }
};

struct AtomicRMWOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AtomicRMWOp>::ConvertTritonGPUOpToLLVMPattern;

  AtomicRMWOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for AtomicRMWOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    auto atomicRmwAttr = op.getAtomicRmwOp();
    MemSemantic memSem = op.getSem();
    LLVM::AtomicOrdering llvmMemOrdering = getMemoryOrdering(memSem)
                                               ? *getMemoryOrdering(memSem)
                                               : LLVM::AtomicOrdering::acq_rel;

    Value val = op.getVal();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto valueTy = op.getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    const size_t valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(val.getType());
    // vec = 1, numElements = 1 for scalar
    unsigned vec = getVectorSize(hasSupport256bLoadStore(op), ptr);
    int numElems = 1;
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(val.getType());
      auto maxVecSize =
          valueElemNBits / valTy.getElementType().getIntOrFloatBitWidth();
      vec = std::min<unsigned>(vec,
                               valTy.getElementType().isF16() ? maxVecSize : 1);
      // mask
      numElems = tensorTy.getNumElements();
    }
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    bool support16BitAtomics = moduleOp->hasAttr(
        TritonIntelGPUDialect::getSupport16BitAtomicsAttrName());
    if (valueElemNBits == 16 && !support16BitAtomics &&
        !supports16BitEmulation(atomicRmwAttr, valueElemTy))
      return op.emitError(
          "16-bit atomic RMW is only emulated for fp16/bf16 with "
          "FADD/MAX/MIN/XCHG when the target lacks native 16-bit atomics; "
          "this operation or element type is not supported");

    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value rmwVal = b.undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        rmwVal = b.insert_element(vecTy, rmwVal, valElements[i + ii], iiVal);
      }

      Value rmwPtr = ptrElements[i];
      Value rmwMask = llMask
                          ? maybeAnd(rewriter, loc, maskElements[i], threadPred)
                          : threadPred;

      assert((valueElemNBits == 16 || valueElemNBits == 32 ||
              valueElemNBits == 64) &&
             "Unexpected width");

      Value zero =
          TypeSwitch<mlir::Type, Value>(valueElemTy)
              .Case<mlir::IntegerType>(
                  [&](auto ty) { return b.int_val(valueElemNBits, 0); })
              .Case<mlir::Float16Type>([&](auto) { return b.f16_val(0); })
              .Case<mlir::BFloat16Type>([&](auto) { return b.bf16_val(0); })
              .Case<mlir::Float32Type>([&](auto) { return b.f32_val(0); })
              .Case<mlir::Float64Type>([&](auto) { return b.f64_val(0); });

      // TODO: check device capabilities to avoid unnecessary emulation or
      // emit unsupported feature error.
      Value ret;
      if (valueElemNBits == 16 && !support16BitAtomics) {
        op.emitWarning("'tt.atomic_rmw' op fp16/bf16 datatype is not supported "
                       "in the target HW, software emulation is an "
                       "experimental feature (use at own risk)");
        Block *endBlock = AtomicRMWOpConversion::emulate16BitsAtomicRmw(
            rewriter, loc, atomicRmwAttr, valueElemTy, rmwPtr, rmwVal,
            maybeAnd(rewriter, loc, b.true_val(), rmwMask), {zero});
        ret = endBlock->getArgument(0);
      } else {
        if (op.getResult().use_empty() && memSem != MemSemantic::RELAXED)
          TritonGEN::BarrierOp::create(rewriter, loc,
                                       TritonGEN::MemFence::GLOBAL);

        auto createAtomicBinOpInstruction = [&]() -> SmallVector<Value, 1> {
          std::optional<mlir::LLVM::AtomicBinOp> rmwKind =
              matchAtomicOp(atomicRmwAttr);
          if (!rmwKind)
            llvm_unreachable("Unhandled RMWOp in case statement");

          rmwVal = b.bitcast(rmwVal, valueElemTy);
          auto atomRMW = LLVM::AtomicRMWOp::create(
              rewriter, loc, *rmwKind, rmwPtr, rmwVal, llvmMemOrdering);
          return {atomRMW.getRes()};
        };

        if (rmwMask) {
          Block *endBlock = &LLVM::intel::createPredicatedBlock(
              rewriter, loc, rmwMask, {zero}, createAtomicBinOpInstruction);
          ret = endBlock->getArgument(0);
        } else {
          ret = createAtomicBinOpInstruction()[0];
        }
      }
      assert(ret);
      Type retType = (!tensorTy || vec == 1) ? valueElemTy : vecTy;
      ret = b.bitcast(ret, retType);

      if (tensorTy) {
        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret
                       : b.extract_element(valueElemTy, ret, b.i32_val(ii));
        }
      } else {
        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }
        Value atomPtr = LLVM::getSharedMemoryBase(loc, rewriter, targetInfo,
                                                  op.getOperation());
        atomPtr = b.bitcast(atomPtr, ptr_ty(ctx, 3));
        // Only threads with rmwMask = True store the result
        targetInfo.storeShared(rewriter, loc, atomPtr, ret, rmwMask);
        createBarrier(rewriter, loc, numCTAs);
        Value loadVal = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {loadVal});
      }
    }

    if (tensorTy) {
      finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals,
                                  valueElemTy, b, threadPred, targetInfo,
                                  getTypeConverter());
    }
    return success();
  }

  // emulate16BitsAtomicRmw only implements fp16/bf16 (FADD + FP MAX/MIN/XCHG).
  // Must stay in sync with the switch there: integer 16-bit ops would either
  // hit llvm_unreachable or be silently miscompiled as floating-point MAX/MIN.
  static bool supports16BitEmulation(mlir::triton::RMWOp atomicOp,
                                     Type valueElemTy) {
    if (!isa<mlir::Float16Type, mlir::BFloat16Type>(valueElemTy))
      return false;
    switch (atomicOp) {
    case RMWOp::FADD:
    case RMWOp::MAX:
    case RMWOp::MIN:
    case RMWOp::XCHG:
      return true;
    default:
      return false;
    }
  }

  // Emulate 16-bit atomicrmw through a loop with 32-bit cmpxchg.
  // TODO: optimize for the case when rmwMask is a true constant?
  static Block *emulate16BitsAtomicRmw(ConversionPatternRewriter &rewriter,
                                       Location loc,
                                       mlir::triton::RMWOp atomicOp,
                                       Type valueElemTy, Value rmwPtr,
                                       Value rmwVal, Value rmwMask,
                                       ArrayRef<Value> ops) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Block *insertionBlock = rewriter.getInsertionBlock();
    Block *headerBlock =
        rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
    Block *endBlock = rewriter.splitBlock(headerBlock, headerBlock->begin());
    rewriter.setInsertionPointToEnd(insertionBlock);
    cf::CondBranchOp::create(rewriter, loc, rmwMask, headerBlock, endBlock,
                             ops);
    rewriter.setInsertionPointToStart(headerBlock);

    rmwVal = b.bitcast(rmwVal, valueElemTy);

    // Align pointer by 4 bytes by zeroing lower address bits. Atomically read
    // a vector of two fp16 values as a single i32. The second lowest bit is
    // extracted to later be used as an index to extract the required vector
    // element.
    assert(isa<LLVM::LLVMPointerType>(rmwPtr.getType()));
    auto intPtr = b.ptrtoint(i64_ty, rmwPtr);
    auto lowPtrBits = b.and_(intPtr, b.i64_val(3));
    auto elemIndex = b.trunc(i32_ty, b.lshr(lowPtrBits, b.i64_val(1)));
    auto alignPtr =
        b.inttoptr(rmwPtr.getType(), b.sub(intPtr, lowPtrBits).getResult());
    auto firstValInt = b.load(i32_ty, alignPtr, 4, false, false, false, false,
                              LLVM::AtomicOrdering::acquire);

    // Create a loop body block. It has a single parameter which holds the
    // latest loaded i32 value.
    Block *bodyBlock =
        rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
    auto origValInt =
        bodyBlock->addArgument(firstValInt.getType(), firstValInt.getLoc());
    rewriter.setInsertionPointToEnd(headerBlock);
    cf::BranchOp::create(rewriter, loc, bodyBlock,
                         SmallVector<Value, 1>{firstValInt});
    rewriter.setInsertionPointToEnd(bodyBlock);

    // Extract value for modification.
    auto origValVec = b.bitcast(origValInt, vec_ty(valueElemTy, 2));
    Value origVal = b.extract_element(origValVec, elemIndex);

    // Apply operation.
    Value newVal = nullptr;
    switch (atomicOp) {
    case RMWOp::FADD:
      newVal = LLVM::FAddOp::create(rewriter, loc, origVal, rmwVal);
      break;
    case RMWOp::MAX:
      newVal = LLVM::MaximumOp::create(rewriter, loc, origVal, rmwVal);
      break;
    case RMWOp::MIN:
      newVal = LLVM::MinimumOp::create(rewriter, loc, origVal, rmwVal);
      break;
    case RMWOp::XCHG:
      newVal = rmwVal;
      break;
    default:
      llvm_unreachable("Unsupported FP16 atomic op");
    }

    // Use modified value to form a new i32 value to write to memory.
    assert(newVal);
    Value newValVec = b.insert_element(origValVec, newVal, elemIndex);
    Value newValInt = b.bitcast(newValVec, i32_ty);

    // Execute cmpxchg and loop back if it fails.
    auto successOrdering = LLVM::AtomicOrdering::acq_rel;
    auto failureOrdering = LLVM::AtomicOrdering::monotonic;
    auto cmpxchg = LLVM::AtomicCmpXchgOp::create(
        rewriter, loc, alignPtr, origValInt, newValInt, successOrdering,
        failureOrdering);
    auto newLoaded = b.extract_val(cmpxchg, 0);
    auto done = b.extract_val(cmpxchg, 1);
    assert(ops.size() == (size_t)1);
    SmallVector<Value, 1> endOps = {origVal};
    cf::CondBranchOp::create(rewriter, loc, done, endBlock, endOps, bodyBlock,
                             SmallVector<Value, 1>{newLoaded});

    for (Value op : ops)
      endBlock->addArgument(op.getType(), op.getLoc());

    rewriter.setInsertionPointToStart(endBlock);
    return endBlock;
  }
};

struct ExtractDescOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::intel::ExtractDescOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::intel::ExtractDescOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::intel::ExtractDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Struct layout: { shapes[rank], strides[rank], base_ptr }.
    // The index attribute directly gives the struct field position.
    unsigned idx = op.getIndex();
    Type resultTy = getTypeConverter()->convertType(op.getResult().getType());
    Value result = LLVM::ExtractValueOp::create(rewriter, op.getLoc(), resultTy,
                                                adaptor.getDesc(), idx);
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Per-sub-tile addressing result, produced by the addressing callback
/// passed to lowerBlockLoad2D.
struct SubTileAddress {
  Value addrElem;           // base pointer for this sub-tile
  Value offsetX;            // X coordinate (surface column, in elements)
  Value offsetY;            // Y coordinate (surface row)
  Value adjustedBaseWidth;  // possibly widened base_width (bytes)
  Value adjustedBaseHeight; // base_height (used for HW OOB)
  Value pred;               // mask predicate, or null
};

/// Common lowering body for ttig.2d_block_load and ttig.2d_block_load_from_ptr.
///
/// Computes tile parameters, runs the sub-tile splitting loop, emits
/// triton_gen.2Dblockload for each sub-tile, and replaces the op with the
/// packed result.
///
/// The only thing that differs between descriptor and pointer loads is how
/// each sub-tile's address is computed — provided via `computeAddress`.
/// `staticBaseHeight` enables row-broadcast for stride=0 pointer loads.
static LogicalResult lowerBlockLoad2D(
    Operation *op, const BlockIOConversionBase::Block2DLoadConfig &cfg,
    const LinearLayout &llEncoding, Value pitch,
    function_ref<SubTileAddress(unsigned registerIdx,
                                ArrayRef<std::pair<StringAttr, Value>> offsets)>
        computeAddress,
    ArrayRef<Value> otherElems, ArrayRef<Value> nanMaskElems,
    std::optional<int> staticBaseHeight,
    const triton::intel::TargetInfo &targetInfo,
    const LLVMTypeConverter *typeConverter, Location loc,
    ConversionPatternRewriter &rewriter) {

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();

  StringAttr kRegister = S("register");
  StringAttr kLane = S("lane");
  StringAttr kWarp = S("warp");
  StringAttr kBlock = S("block");

  Value warpId = arith::IndexCastOp::create(
      rewriter, loc, i32_ty,
      mlir::gpu::SubgroupIdOp::create(rewriter, loc,
                                      /*upperBound=*/nullptr));

  SmallVector<Value> unpackedLoadedVals(cfg.numElems);
  for (size_t elemIdx = 0; elemIdx < cfg.numElems;
       elemIdx += cfg.numElemsPerLoad) {
    unsigned registerIdx =
        cfg.regMapping.apply({{kRegister, elemIdx}})[0].second;

    auto offsets = applyLinearLayout(loc, rewriter, llEncoding,
                                     {{kRegister, b.i32_val(registerIdx)},
                                      {kLane, b.i32_val(0)},
                                      {kWarp, warpId},
                                      {kBlock, b.i32_val(0)}});

    auto addr = computeAddress(registerIdx, offsets);

    Value offsetY = addr.offsetY;
    if (addr.pred)
      offsetY = b.select(addr.pred, offsetY, addr.adjustedBaseHeight);

    Value ret = TritonGEN::Matrix2DBlockLoadOp::create(
        rewriter, loc, cfg.load2DGenXType,
        /*ptr*/ addr.addrElem,
        /*base_width*/ addr.adjustedBaseWidth,
        /*base_height*/ addr.adjustedBaseHeight,
        /*base_pitch*/ pitch,
        /*x*/ b.udiv(addr.offsetX, b.i32_val(cfg.numPackedVals)),
        /*y*/ offsetY,
        /*elem_size_in_bits*/ cfg.packedElemSizeInBits,
        /*tile_width*/ cfg.tileWidth,
        /*tile_height*/ cfg.tileHeight,
        /*v_blocks*/ cfg.vBlocks,
        /*transpose*/ cfg.isTransposeRequired,
        /*vnni_transform*/ !cfg.isTransposeRequired && cfg.useVNNIFormat);

    // When staticBaseHeight == 1 but tileHeight > 1 (stride=0 broadcast),
    // only the first row contains valid data. Replicate it across the tile.
    if (staticBaseHeight &&
        *staticBaseHeight < static_cast<int>(cfg.tileHeight) &&
        *staticBaseHeight == 1) {
      unsigned numIndicesPerMatrix = cfg.numValuesPerLoad / cfg.vBlocks;
      SmallVector<int32_t> shuffleIndices(cfg.numValuesPerLoad);

      VectorType vecTy = vec_ty(cfg.packedType, cfg.vBlocks);
      Value firstIndexVec = b.undef(vecTy);

      for (unsigned valueIndex = 0; valueIndex < cfg.numValuesPerLoad;
           ++valueIndex) {
        unsigned firstIndexVecIdx = valueIndex / numIndicesPerMatrix;
        if (valueIndex % numIndicesPerMatrix == 0) {
          Value oldVal = b.extract_element(ret, b.i32_val(valueIndex));
          Value newVal = oldVal;
          if (cfg.tileWidth < cfg.threadsPerWarp) {
            assert(cfg.tileWidth * 2 == cfg.threadsPerWarp &&
                   "Expecting tileWidth to be 2x threadsPerWarp");
            Value threadId = getThreadId(rewriter, loc);
            newVal = targetInfo.shuffleIdx(
                rewriter, loc, oldVal,
                b.urem(threadId, b.i32_val(cfg.tileWidth)));
          }
          firstIndexVec =
              b.insert_element(firstIndexVec.getType(), firstIndexVec, newVal,
                               b.i32_val(firstIndexVecIdx));
        }
        shuffleIndices[valueIndex] = firstIndexVecIdx;
      }
      DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(shuffleIndices);
      ret = LLVM::ShuffleVectorOp::create(rewriter, loc, cfg.load2DGenXType,
                                          firstIndexVec, firstIndexVec, attr);
    }

    BlockIOConversionBase::unpackBlockLoadResult(
        ret, unpackedLoadedVals, elemIdx, cfg.regMapping, cfg.shuffleMapping,
        cfg.packedDPASOperandType, cfg.unpackedType, cfg.numValuesPerLoad,
        cfg.numPackedVals, addr.pred, otherElems, nanMaskElems, loc, rewriter,
        ctx);
  }

  Type llvmResultStructTy =
      typeConverter->convertType(op->getResult(0).getType());
  Value resultStruct = packLLElements(loc, typeConverter, unpackedLoadedVals,
                                      rewriter, llvmResultStructTy);
  rewriter.replaceOp(op, {resultStruct});
  return success();
}

struct Subgroup2DBlockLoadOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::intel::Subgroup2DBlockLoadOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::Subgroup2DBlockLoadOp>::
      ConvertTritonGPUOpToLLVMPattern;

  Subgroup2DBlockLoadOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<
            triton::gpu::intel::Subgroup2DBlockLoadOp>(converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::gpu::intel::Subgroup2DBlockLoadOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(op.getType());
    Attribute encoding = tensorType.getEncoding();
    std::optional<LinearLayout> llEncoding =
        cast<DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    assert(llEncoding.has_value() && "expected valid linear layout");

    BlockIOMode memLayout = op.getMemoryLayout();
    bool memoryRowMajor =
        (memLayout == triton::gpu::intel::BlockIOMode::RowMajor);
    unsigned rank = tensorType.getRank();
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;

    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();

    // Tile size computation.
    // FIXME: Remove once IGC can split large 2D block loads.
    bool oneMatrixPerLoadForBT =
        op->hasAttr(TritonIntelGPUDialect::getOneMatrixPerLoadAttrName());
    BlockIOTileSizeInfo sizeInfo = getBlockIOTileSize<true /*load*/>(
        llEncoding.value(), contiguousDim, elemSizeInBits,
        /*maskAxisInfo=*/nullptr, oneMatrixPerLoadForBT);
    assert(sizeInfo.isValid() && "expected valid tile size");

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    unsigned threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());

    Block2DLoadConfig cfg = buildBlock2DLoadConfig(
        tensorType, eltTy, sizeInfo, *llEncoding, threadsPerWarp, ctx);

    Value basePtr = adaptor.getBasePtr();
    Value baseWidth = adaptor.getBaseWidth();
    Value baseHeight = adaptor.getBaseHeight();
    Value pitch = adaptor.getBasePitch();
    Value baseOffsetX = adaptor.getOffsetX();
    Value baseOffsetY = adaptor.getOffsetY();

    Value elemBytes = b.i32_val(elemSizeInBits / 8);

    // FIXME: Workaround for suboptimal IGC instruction scheduling.
    // Remove once IGC handles the or-expression reassociation correctly
    // (see https://github.com/intel/intel-xpu-backend-for-triton/issues/6540).
    //
    // Pre-apply 64-byte alignment compensation to the base pointer and column
    // offset when the descriptor column index is non-zero. This bakes the
    // alignment adjustment into the column index BEFORE per-tile layout offsets
    // are added, ensuring LLVM builds tile 1's x-coordinate as
    // (tile0_x + delta) rather than (descIndex + delta) + misalign. Without
    // this, LLVM's CSE factors out (descIndex + delta) as a common
    // subexpression shared across different operand loads, producing a
    // suboptimal instruction schedule.
    std::optional<int64_t> colConst =
        triton::intel::getFoldedConstantValue(baseOffsetX);
    if (!colConst || *colConst != 0) {
      constexpr int64_t ALIGNMENT_MASK = 0x3f;
      Value baseAddr = b.ptrtoint(int_ty(64), basePtr);
      Value alignedBaseAddr = b.and_(baseAddr, b.i64_val(~ALIGNMENT_MASK));
      basePtr = b.inttoptr(ptr_ty(ctx, 1), alignedBaseAddr);
      Value offsetInBytes =
          b.trunc(i32_ty, b.and_(baseAddr, b.i64_val(ALIGNMENT_MASK)));
      Value misalignElems = b.udiv(offsetInBytes, elemBytes);
      baseWidth = b.add(baseWidth, offsetInBytes);
      baseOffsetX = b.add(baseOffsetX, misalignElems);
    }

    // Build NaN masks if pad_nan is set.
    SmallVector<Value> nanMaskElems;
    if (op.getPadNan()) {
      SmallVector<Value> resultOffsets(rank, b.i32_val(0));
      SmallVector<Value> resultShapes(rank);
      for (unsigned i = 0; i < rank; ++i) {
        if (static_cast<int>(i) == cfg.rowDim)
          resultShapes[i] = baseHeight;
        else if (static_cast<int>(i) == cfg.colDim)
          resultShapes[i] = b.udiv(baseWidth, elemBytes);
        else
          resultShapes[i] = b.i32_val(tensorType.getDimSize(i));
      }
      unsigned surfaceColDim = contiguousDim;
      unsigned surfaceRowDim =
          (contiguousDim == rank - 1) ? rank - 2 : rank - 1;
      resultOffsets[surfaceColDim] = baseOffsetX;
      resultOffsets[surfaceRowDim] = baseOffsetY;
      nanMaskElems =
          buildNaNMasks(loc, resultOffsets, resultShapes, tensorType, rewriter);
    }

    unsigned blockRowIdx = cfg.isTransposeRequired ? cfg.colDim : cfg.rowDim;
    unsigned blockColIdx = cfg.isTransposeRequired ? cfg.rowDim : cfg.colDim;

    // Per-sub-tile: combine base offsets with linear layout offsets.
    auto computeAddress =
        [&](unsigned /*registerIdx*/,
            ArrayRef<std::pair<StringAttr, Value>> offsets) -> SubTileAddress {
      Value addrElem = basePtr;
      Value offsetX, offsetY;
      unsigned surfaceColDim = contiguousDim;
      unsigned surfaceRowDim =
          (contiguousDim == rank - 1) ? rank - 2 : rank - 1;
      for (auto [dim, offsetPair] : llvm::enumerate(offsets)) {
        Value baseOff = b.i32_val(0);
        if (dim == surfaceColDim)
          baseOff = baseOffsetX;
        else if (dim == surfaceRowDim)
          baseOff = baseOffsetY;
        Value adjustedOffset = b.add(baseOff, offsetPair.second);
        if (dim == blockRowIdx)
          offsetY = adjustedOffset;
        else if (dim == blockColIdx)
          offsetX = adjustedOffset;
        else {
          // Batch dimensions: fold into base pointer via GEP.
          Value strideInElems =
              b.zext(int_ty(64), b.mul(baseHeight, b.udiv(pitch, elemBytes)));
          Value offset64 = b.zext(int_ty(64), adjustedOffset);
          Value batchOffset = b.mul(offset64, strideInElems);
          addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, batchOffset);
        }
      }
      return {addrElem,        offsetX, offsetY, baseWidth, baseHeight,
              /*pred=*/Value()};
    };

    return lowerBlockLoad2D(op, cfg, *llEncoding, pitch, computeAddress,
                            /*otherElems=*/{}, nanMaskElems,
                            /*staticBaseHeight=*/std::nullopt, targetInfo,
                            getTypeConverter(), loc, rewriter);
  }
};

struct Subgroup2DBlockLoadFromPtrOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::intel::Subgroup2DBlockLoadFromPtrOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::Subgroup2DBlockLoadFromPtrOp>::
      ConvertTritonGPUOpToLLVMPattern;

  Subgroup2DBlockLoadFromPtrOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      triton::intel::ModuleStrideAnalysis &strideAnalysis,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<
            triton::gpu::intel::Subgroup2DBlockLoadFromPtrOp>(converter,
                                                              benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass, strideAnalysis) {}

  LogicalResult
  matchAndRewrite(triton::gpu::intel::Subgroup2DBlockLoadFromPtrOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = cast<RankedTensorType>(op.getType());
    Attribute encoding = tensorType.getEncoding();
    unsigned rank = tensorType.getRank();
    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    // Memory layout and tile computation.
    BlockIOMode memLayout = op.getMemoryLayout();
    bool memoryRowMajor = (memLayout == BlockIOMode::RowMajor);
    unsigned contiguousDim = memoryRowMajor ? rank - 1 : rank - 2;

    std::optional<LinearLayout> llEncoding =
        cast<DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    assert(llEncoding.has_value() && "expected valid linear layout");

    // Tile size computation.
    // FIXME: Remove once IGC can split large 2D block loads.
    bool oneMatrixPerLoadForBT =
        op->hasAttr(TritonIntelGPUDialect::getOneMatrixPerLoadAttrName());

    AxisInfo *maskAxisInfo = nullptr;
    if (op.getMask())
      maskAxisInfo =
          const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
              .getAxisInfo(op.getMask());

    BlockIOTileSizeInfo sizeInfo = BlockIOTileSizeInfo::unknown();
    bool has1DReshapeStride =
        op->hasAttr(TritonIntelGPUDialect::getBlockIOStrideAttrName());
    auto blockedEnc = dyn_cast<BlockedEncodingAttr>(encoding);
    if (has1DReshapeStride && blockedEnc && rank == 2) {
      unsigned numWarpsRow = blockedEnc.getWarpsPerCTA()[0];
      int height = tensorType.getDimSize(0) / numWarpsRow;
      int width = tensorType.getDimSize(rank - 1);
      StringAttr kReg = S("register");
      unsigned regDimSize = llEncoding->getInDimSize(kReg);
      SetVector<unsigned> regPackBases;
      for (unsigned i = 1; i < regDimSize; i <<= 1)
        regPackBases.insert(i);
      sizeInfo = BlockIOTileSizeInfo(height, width, /*numElemPerPackedVal=*/1,
                                     /*vBlocks=*/1, /*rowDim=*/0,
                                     /*colDim=*/rank - 1, /*transpose=*/false,
                                     std::move(regPackBases));
    } else {
      sizeInfo =
          getBlockIOTileSize<true>(*llEncoding, contiguousDim, elemSizeInBits,
                                   maskAxisInfo, oneMatrixPerLoadForBT);
    }
    if (!sizeInfo.isValid())
      return failure();

    unsigned threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(
        op->getParentOfType<ModuleOp>());

    Block2DLoadConfig cfg = buildBlock2DLoadConfig(
        tensorType, eltTy, sizeInfo, *llEncoding, threadsPerWarp, ctx);

    // Unpack pointer elements.
    SmallVector<Value> ptrElems =
        unpackLLElements(loc, adaptor.getPtr(), rewriter);

    // Unpack mask/other elements.
    SmallVector<Value> maskElems;
    Value llMask = adaptor.getMask();
    if (llMask)
      maskElems = unpackLLElements(loc, llMask, rewriter);

    SmallVector<Value> otherElems;
    Value llOther = adaptor.getOther();
    if (llOther && llMask) {
      Value other = op.getOther();
      DenseElementsAttr constAttr;
      if (matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat()) {
        Type elemTy = constAttr.getElementType();
        auto handleSplatValue = [&](auto splatVal) {
          if (!splatVal.isZero()) {
            otherElems = SmallVector<Value>(
                cfg.numElems,
                LLVM::ConstantOp::create(rewriter, loc, elemTy, splatVal));
          }
        };
        TypeSwitch<mlir::Type>(elemTy)
            .Case<FloatType>([&](FloatType) {
              handleSplatValue(constAttr.getSplatValue<APFloat>());
            })
            .Case<IntegerType>([&](IntegerType) {
              handleSplatValue(constAttr.getSplatValue<APInt>());
            });
      } else {
        otherElems = unpackLLElements(loc, llOther, rewriter);
      }
    }

    // Surface parameters from op attributes.
    Value baseWidth = b.i32_val(op.getBaseWidth() * cfg.vBlocks);
    Value baseHeight = b.i32_val(op.getBaseHeight());
    Value pitch = b.i32_val(op.getBasePitch());

    unsigned blockColIdx = cfg.isTransposeRequired ? cfg.rowDim : cfg.colDim;

    // Per-sub-tile: use ptrElems[registerIdx] with subtraction-based
    // addressing.
    auto computeAddress =
        [&](unsigned registerIdx,
            ArrayRef<std::pair<StringAttr, Value>> offsets) -> SubTileAddress {
      Value addrElem =
          targetInfo.shuffleIdx(rewriter, loc, ptrElems[registerIdx], 0);
      Value offsetX = offsets[blockColIdx].second;
      Value negativeOffsetX = b.sub(b.i32_val(0), offsetX);
      addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, negativeOffsetX);
      Value adjustedBaseWidth =
          b.add(baseWidth, b.mul(offsetX, b.i32_val(elemSizeInBits / 8)));

      Value pred;
      if (maskElems.size())
        pred = targetInfo.shuffleIdx(rewriter, loc, maskElems[registerIdx], 0);

      return {addrElem,          offsetX,    /*offsetY=*/b.i32_val(0),
              adjustedBaseWidth, baseHeight, pred};
    };

    return lowerBlockLoad2D(op, cfg, *llEncoding, pitch, computeAddress,
                            otherElems, /*nanMaskElems=*/{},
                            /*staticBaseHeight=*/op.getBaseHeight(), targetInfo,
                            getTypeConverter(), loc, rewriter);
  }
};

// -----------------------------------------------------------------------
// LocalAtomicScatterRMWOp lowering
// -----------------------------------------------------------------------

struct LocalAtomicScatterRMWInfo {
  RankedTensorType valuesTy;
  Type llvmElemTy;
  LinearLayout regLayout;
  ColumnAction removeBroadcast;
  Value threadPred;
  SmallVector<Value> values;
  SmallVector<Value> maskValues;
  SmallVector<Value> ptrs;
};

// Ported from NVIDIA backend
static FailureOr<LocalAtomicScatterRMWInfo>
prepareLocalAtomicScatterRMW(triton::gpu::LocalAtomicScatterRMWOp op, Value dst,
                             Value indices, Value inputValues, Value mask,
                             ConversionPatternRewriter &rewriter,
                             const triton::intel::TargetInfo &targetInfo,
                             const LLVMTypeConverter *typeConverter) {
  auto loc = op.getLoc();
  auto valuesTy = cast<RankedTensorType>(op.getValues().getType());
  auto memDescTy = cast<MemDescType>(op.getDst().getType());
  if (isa<triton::gpu::PartitionedSharedEncodingAttr>(
          memDescTy.getEncoding())) {
    return failure();
  }

  auto llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, dst, llvmElemTy, rewriter);
  SmallVector<Value> idxValues = unpackLLElements(loc, indices, rewriter);
  SmallVector<Value> values = unpackLLElements(loc, inputValues, rewriter);
  SmallVector<Value> maskValues;
  if (mask)
    maskValues = unpackLLElements(loc, mask, rewriter);

  LinearLayout regLayout = toLinearLayout(valuesTy);
  auto freeVarMasks = regLayout.getFreeVariableMasks();
  auto removeBroadcast = actionRemoveBroadcastedRegs(regLayout);
  Value threadPred =
      emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
  LinearLayout activeRegLayout = regLayout;
  if (!removeBroadcast.isIdentity()) {
    activeRegLayout = removeBroadcast.apply(regLayout);
    values = removeBroadcast.apply(values);
    idxValues = removeBroadcast.apply(idxValues);
    if (!maskValues.empty())
      maskValues = removeBroadcast.apply(maskValues);
  }
  SmallVector<SmallVector<Value>> srcIndices =
      emitIndices(loc, rewriter, targetInfo, activeRegLayout, valuesTy,
                  /*withCTAOffset=*/true);

  SmallVector<Value> ptrs = llvm::map_to_vector(
      computeLocalAddrs(loc, memDescTy, smemObj, llvmElemTy, idxValues,
                        srcIndices, op.getAxis(), rewriter),
      [](const LocalSharedMemoryAddress &addr) { return addr.ptr; });

  return LocalAtomicScatterRMWInfo{valuesTy,        llvmElemTy, regLayout,
                                   removeBroadcast, threadPred, values,
                                   maskValues,      ptrs};
}

struct LocalAtomicScatterRMWOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::gpu::LocalAtomicScatterRMWOp> {

  LocalAtomicScatterRMWOpConversion(const LLVMTypeConverter &converter,
                                    const triton::intel::TargetInfo &targetInfo,
                                    PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::LocalAtomicScatterRMWOp>(
            converter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalAtomicScatterRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RMWOp rmwOp = op.getAtomicRmwOp();
    std::optional<LLVM::AtomicBinOp> atomicBinOp = matchAtomicOp(rmwOp);
    if (!atomicBinOp)
      return op.emitError("unsupported atomic RMW operation for XPU backend");

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto lowering = prepareLocalAtomicScatterRMW(
        op, adaptor.getDst(), adaptor.getIndices(), adaptor.getValues(),
        op.getMask() ? adaptor.getMask() : Value(), rewriter, targetInfo,
        getTypeConverter());

    if (failed(lowering))
      return failure();

    LocalAtomicScatterRMWInfo &info = *lowering;
    bool returnOld = !op.getResult().use_empty();

    bool needs16BitEmulation = requires16BitEmulation(op, info.llvmElemTy);
    if (needs16BitEmulation &&
        !AtomicRMWOpConversion::supports16BitEmulation(rmwOp, info.llvmElemTy))
      return op.emitError(
          "16-bit atomic RMW is only emulated for fp16/bf16 with "
          "FADD/MAX/MIN/XCHG when the target lacks native 16-bit atomics; "
          "this operation or element type is not supported");

    Value zero = emitZeroConstant(b, info.llvmElemTy);

    SmallVector<Value> results;
    if (returnOld)
      results.reserve(info.ptrs.size());

    for (auto [i, ptrAndValue] :
         llvm::enumerate(llvm::zip(info.ptrs, info.values))) {
      auto [elemPtr, packedVal] = ptrAndValue;
      Value elemVal = b.bitcast(packedVal, info.llvmElemTy);
      Value pred =
          maybeAnd(rewriter, loc, info.threadPred,
                   info.maskValues.empty() ? Value() : info.maskValues[i]);

      Value ret = emitOneAtomicRMW(b, rewriter, loc, info.llvmElemTy, rmwOp,
                                   *atomicBinOp, elemPtr, elemVal, pred, zero,
                                   needs16BitEmulation);
      if (returnOld)
        results.push_back(ret);
    }

    if (!returnOld) {
      rewriter.eraseOp(op);
      return success();
    }

    if (!info.removeBroadcast.isIdentity())
      results = broadcastAs(results, info.regLayout);

    finalizeTensorAtomicResults(op, info.valuesTy, rewriter, results,
                                info.llvmElemTy, b, info.threadPred, targetInfo,
                                getTypeConverter());
    return success();
  }

private:
  bool requires16BitEmulation(Operation *op, Type llvmElemTy) const {
    bool support16BitAtomics = op->getParentOfType<ModuleOp>()->hasAttr(
        TritonIntelGPUDialect::getSupport16BitAtomicsAttrName());
    return llvmElemTy.getIntOrFloatBitWidth() == 16 && !support16BitAtomics;
  }

  Value emitZeroConstant(TritonLLVMOpBuilder &b, Type llvmElemTy) const {
    return TypeSwitch<mlir::Type, Value>(llvmElemTy)
        .Case<mlir::IntegerType>(
            [&](auto ty) { return b.int_val(ty.getWidth(), 0); })
        .Case<mlir::Float16Type>([&](auto) { return b.f16_val(0); })
        .Case<mlir::BFloat16Type>([&](auto) { return b.bf16_val(0); })
        .Case<mlir::Float32Type>([&](auto) { return b.f32_val(0); })
        .Case<mlir::Float64Type>([&](auto) { return b.f64_val(0); });
  }

  // Emit one atomic RMW for a single element pointer, returning the old value.
  // 16-bit types without native HW support use CAS-loop emulation.
  Value emitOneAtomicRMW(TritonLLVMOpBuilder &b,
                         ConversionPatternRewriter &rewriter, Location loc,
                         Type llvmElemTy, RMWOp rmwOp,
                         LLVM::AtomicBinOp atomicBinOp, Value elemPtr,
                         Value elemVal, Value pred, Value zero,
                         bool needs16BitEmulation) const {
    if (needs16BitEmulation) {
      Value casGuard = pred ? pred : b.true_val();
      Block *endBlock = AtomicRMWOpConversion::emulate16BitsAtomicRmw(
          rewriter, loc, rmwOp, llvmElemTy, elemPtr, elemVal, casGuard, {zero});
      return endBlock->getArgument(0);
    }

    auto createAtomicOp = [&]() -> SmallVector<Value, 1> {
      auto atomRMW =
          LLVM::AtomicRMWOp::create(rewriter, loc, atomicBinOp, elemPtr,
                                    elemVal, LLVM::AtomicOrdering::monotonic);
      return {atomRMW.getRes()};
    };

    if (pred) {
      Block *endBlock = &LLVM::intel::createPredicatedBlock(
          rewriter, loc, pred, {zero}, createAtomicOp);
      return endBlock->getArgument(0);
    }
    return createAtomicOp()[0];
  }

  const triton::intel::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::intel::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns,
    const intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    intel::ModuleStrideAnalysis &strideAnalysis, PatternBenefit benefit) {

  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               DescriptorLoadOpConversion, StoreOpConversion,
               DescriptorStoreOpConversion, PrefetchOpConversion,
               DescriptorPrefetchOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, strideAnalysis, benefit);
  // Local atomic scatter RMW (shared memory atomics via indices).
  patterns.add<LocalAtomicScatterRMWOpConversion>(typeConverter, targetInfo,
                                                  benefit);
  // Block IO store patterns (loads are handled via ttig.2d_block_load path).
  patterns
      .add<StoreOpToBlockIOConversion, DescriptorStoreOpToBlockIOConversion>(
          typeConverter, targetInfo, axisInfoAnalysis, strideAnalysis,
          benefit.getBenefit() + 2);
  // TTIG ops from LowerTo2DBlockLoad TTGIR pass.
  patterns.add<ExtractDescOpConversion>(typeConverter, benefit);
  patterns.add<Subgroup2DBlockLoadOpConversion,
               Subgroup2DBlockLoadFromPtrOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, strideAnalysis, benefit);
}
