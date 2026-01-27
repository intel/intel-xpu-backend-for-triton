#include "Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "intel/include/Utils/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include <TritonIntelGPUToLLVM/XeAsmFormat.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/Support/FormatVariadic.h>
#include <optional>
#include <triton/Tools/Sys/GetEnv.hpp>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

#define S(v) StringAttr::get(ctx, (v))

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_ctz(unsigned x) {
  unsigned long r;
  _BitScanForward(&r, x);
  return static_cast<int>(r);
}
#endif

namespace {

Value maybeAnd(RewriterBase &rewriter, Location loc, Value a, Value b) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  if (a && b) {
    return tb.and_(a, b);
  }
  return a ? a : b;
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask. The predicate may be null to indicate no
// predication is required.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const triton::intel::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred;
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = maybeAnd(rewriter, loc, pred, dimPred);
    }
  }
  return pred;
}

unsigned getCanonicalIndex(unsigned index, unsigned freeVarMask) {
  return index & ~freeVarMask;
}

/// Holds the values related to a block pointer.
/// It includes the base pointer, base width and height, row and column
/// stride, and offset base for X and Y.
struct BlockPointerValues {
  Value base;
  Value baseWidth;
  Value baseHeight;
  Value rowStride;
  Value colStride;
  Value offsetBaseX;
  Value offsetBaseY;
};

// Unpack values as the params to 2DBlockLoad Payload: offsetBaseY,
// offsetBaseX, baseHeight, baseWidth, rowStride, colStride, base.
// FIXME: Only supports 2D matrices for now.
BlockPointerValues
getValuesFromBlockPointerStruct(Value blockPointerStruct,
                                ConversionPatternRewriter &rewriter) {
  const SmallVector<Value> &elems = unpackLLElements(
      blockPointerStruct.getLoc(), blockPointerStruct, rewriter);
  assert(elems.size() == sizeof(BlockPointerValues) / sizeof(Value) &&
         "unexpected number of values unpacked from a block pointer");
  return {/*base=*/elems[6],       /*baseWidth=*/elems[3],
          /*baseHeight=*/elems[2], /*rowStride=*/elems[4],
          /*colStride=*/elems[5],  /*offsetBaseX=*/elems[1],
          /*offsetBaseY=*/elems[0]};
}

/// Compute the 2D prefetch shape for each warp given an input 2D tensor.
/// Because a cache line is 64 bytes, and we want to prefetch one cache line a
/// time (per thread), the maximum number of bytes per column is 64. We know
/// that the maximum size for each 2D prefetch is 2048 bytes, therefore the
/// maximum number of rows is given by 2048/64=32.
SmallVector<unsigned, 2> get2DPrefetchShapePerWarp(RankedTensorType tensorTy) {
  Type eltTy = tensorTy.getElementType();
  const ArrayRef<int64_t> tensorShape = tensorTy.getShape();
  unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
  unsigned elemSizeInBytes = elemSizeInBits / 8;
  unsigned maxBytesPerCol = 64;
  unsigned numRows = std::min<unsigned>(tensorShape[0], 32);
  unsigned numCols =
      std::min<unsigned>(tensorShape[1], maxBytesPerCol / elemSizeInBytes);
  return {numRows, numCols};
}

/// Get the 2D warps per CTA given the tensor shape and the prefetch
/// shape per warp.
SmallVector<unsigned, 2>
getWarpsPerCTA(const ArrayRef<int64_t> tensorShape,
               const SmallVector<unsigned, 2> &shapePerWarp,
               unsigned numWarps) {
  assert(tensorShape.size() == 2 && shapePerWarp.size() == 2 &&
         "only 2D tensors are supported");

  unsigned repNumPerRow = mlir::ceil((unsigned)tensorShape[1], shapePerWarp[1]);
  unsigned warpNumPerRow = std::min(numWarps, repNumPerRow);
  unsigned warpNumRow = mlir::ceil(numWarps, warpNumPerRow);
  return {warpNumRow, warpNumPerRow};
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  int getStride(Value ptr, unsigned dim) const {
    AxisInfo *axisInfo =
        const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
            .getAxisInfo(ptr);
    if (axisInfo) {
      const SmallVector<int64_t> &stride = axisInfo->getStride();
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

  unsigned getVectorSize(Value ptr) const {
    if (!isTensorOrTensorPointerType(ptr.getType()))
      return 1;

    unsigned contiguity = getContiguity(ptr);
    unsigned pointeeBitWidth = triton::getPointeeBitWidth(ptr.getType());
    // The maximum vector size is 128 bits.
    return std::min<unsigned>(128 / pointeeBitWidth, contiguity);
  }

  unsigned getMaskAlignment(Value mask) const {
    return const_cast<triton::intel::ModuleAxisInfoAnalysis &>(axisAnalysisPass)
        .getMaskAlignment(mask);
  }

  std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
  convertBlockPtrToTensorOfPtr(
      Location loc, Value blockPointerStruct, RankedTensorType tensorType,
      Type valueElemTy, ConversionPatternRewriter &rewriter,
      ArrayRef<int32_t> boundaryCheck = {},
      std::optional<PaddingOption> padding = std::nullopt) const {

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    size_t rank = tensorType.getRank();
    // The block pointer struct is expected to have the following layout:
    //    Struct {
    //      Value offset[rank];
    //      Value shape[rank];
    //      Value stride[rank];
    //      Value base;
    //    }
    // All the values are decomposed by `unpackLLElements` into a vector.
    // Defines the indices for the block pointer struct.
    const unsigned blockOffset = 0, blockShape = 1 * rank,
                   blockStride = 2 * rank, blockBase = 3 * rank;
    const SmallVector<Value> &blockPtr =
        unpackLLElements(loc, blockPointerStruct, rewriter);
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
      SmallVector<Value> indicesInTensor(rank);
      for (unsigned j = 0; j < rank; ++j)
        indicesInTensor[j] = b.add(index[j], blockPtr[blockOffset + j]);

      // Get the LLVM values for pointers
      Value offset = linearize(
          indicesInTensor,
          {blockPtr.begin() + blockStride, blockPtr.begin() + blockBase},
          b.i32_val(0),
          [&](const Value &index, const Value &stride, const Value &off) {
            // off = off + index * stride
            return b.add(b.mul(index, b.trunc(i32_ty, stride)), off);
          });

      ptrElems[i] = b.gep(ptr_ty(rewriter.getContext(), 1 /*global*/),
                          valueElemTy, blockPtr[blockBase], offset);

      if (boundaryProtect.size() > 0) {
        // Get the LLVM values for mask
        unsigned dim = 0;
        maskElems.push_back(linearize(
            indicesInTensor,
            {blockPtr.begin() + blockShape, blockPtr.begin() + blockStride},
            b.int_val(1, 1),
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

  // Ensure the operation doesn't have attributes that the IGC predicated
  // instruction cannot handle.
  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, LoadOp, StoreOp>::value>>
  bool canUsePredicatedInstructions(OpType op) const {
    if (!mlir::LLVM::intel::hasModuleAttr(
            op, TritonIntelGPUDialect::getSupportPredicatedIOAttrName()))
      return false;

    if constexpr (std::is_same_v<OpType, LoadOp>) {
      if (!triton::tools::getBoolEnv("TRITON_INTEL_PREDICATED_LOAD"))
        return false;
      return !op.getIsVolatile() && op.getCache() == CacheModifier::NONE;
    }

    if constexpr (std::is_same_v<OpType, StoreOp>) {
      std::optional<bool> usePredicatedStore =
          mlir::triton::tools::isEnvValueBool(
              mlir::triton::tools::getStrEnv("TRITON_INTEL_PREDICATED_STORE"));
      if (usePredicatedStore.has_value() && !usePredicatedStore.value())
        return false;
    }

    return op.getCache() == CacheModifier::NONE;
  }

  template <typename OpType, typename = std::enable_if_t<llvm::is_one_of<
                                 OpType, LoadOp, StoreOp>::value>>
  bool getNonTemporalFlag(OpType op) const {
    switch (op.getCache()) {
    case triton::CacheModifier::CG:
    case triton::CacheModifier::CS:
    case triton::CacheModifier::CV:
      return true;
    case triton::CacheModifier::CA:
    default:
      return false;
    }
  }

protected:
  const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass;
  const triton::intel::TargetInfo &targetInfo;
};

struct BlockIOConversionBase : public LoadStoreConversionBase {
  explicit BlockIOConversionBase(
      const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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

    static const bool enableBlockIOForAllLayout =
        triton::tools::getBoolEnv("TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS");

    // Only lower operation with dpas layout encoding.
    auto tensorTy =
        cast<RankedTensorType>(getPointeeType(op.getPtr().getType()));
    return enableBlockIOForAllLayout || hasDpasEncoding(tensorTy) ||
           hasDotDpasEncoding(tensorTy);
  }

  static bool
  check2DBlockAddressPayloadRestriction(unsigned packedElemSizeInBits,
                                        unsigned tileWidth) {
    // Return false if tile width is not supported by HW.
    // Note: Tile width is not changeable.
    switch (packedElemSizeInBits) {
    case 8:
      if (tileWidth < 4 || tileWidth > 64)
        return false;
      break;
    case 16:
      if (tileWidth < 2 || tileWidth > 32)
        return false;
      break;
    case 32:
      if (tileWidth > 16)
        return false;
      break;
    case 64:
      if (tileWidth > 8)
        return false;
      break;
    default:
      // invalid element type for 2D block io.
      return false;
    }
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

    // TODO: To support more layouts on memory:
    // https://github.com/intel/intel-xpu-backend-for-triton/issues/4057.
    // Only support rank 2 dot layout, either row major or column major.
    StringRef memoryLayoutInfo = cast<StringAttr>(blockIOAttr).getValue();
    assert((memoryLayoutInfo == "row_major" ||
            memoryLayoutInfo == "column_major") &&
           "Only row_major or column_major is supported");
    return memoryLayoutInfo == "row_major";
  }

  static DpasEncodingAttr::OpIdx getOpIdx(RankedTensorType tensorTy) {
    if (hasDpasEncoding(tensorTy))
      return DpasEncodingAttr::OpIdx::OperandC;

    assert(hasDotDpasEncoding(tensorTy) && "Expecting dot layout");
    DotOperandEncodingAttr dotLayout = getDotEncoding(tensorTy).value();
    return static_cast<DpasEncodingAttr::OpIdx>(dotLayout.getOpIdx());
  }

  static DpasEncodingAttr getDpasLayout(RankedTensorType tensorTy) {
    Attribute encoding = tensorTy.getEncoding();
    return cast<DpasEncodingAttr>(
        hasDpasEncoding(tensorTy)
            ? encoding
            : getDotEncoding(tensorTy).value().getParent());
  }

  // Unpack the base pointers from regular pointer or block pointer.
  SmallVector<Value> getBases(ConversionPatternRewriter &rewriter, Value ptr,
                              const SmallVector<Value> &unpackedPtrs,
                              unsigned numElems) const {
    SmallVector<Value> ptrElems;
    if (isTensorPointerType(ptr.getType())) {
      // The block pointer struct is expected to have the following layout:
      //    Struct {
      //      Value offset[rank];
      //      Value shape[rank];
      //      Value stride[rank];
      //      Value base;
      //    }
      assert((unpackedPtrs.size() - 1) % 3 == 0 &&
             "unexpected number of values unpacked from a block pointer");
      unsigned rank = (unpackedPtrs.size() - 1) / 3;
      unsigned blockBase = 3 * rank;
      ptrElems.assign(numElems, unpackedPtrs[blockBase]);
    } else {
      ptrElems = unpackedPtrs;
    }

    return ptrElems;
  }

  // Unpack the shapes from regular pointer or block pointer.
  SmallVector<Value> getShapes(ConversionPatternRewriter &rewriter, Value ptr,
                               const SmallVector<Value> &unpackedPtrs) const {
    Location loc = ptr.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (isTensorPointerType(ptr.getType())) {
      // The block pointer struct is expected to have the following layout:
      //    Struct {
      //      Value offset[rank];
      //      Value shape[rank];
      //      Value stride[rank];
      //      Value base;
      //    }
      assert((unpackedPtrs.size() - 1) % 3 == 0 &&
             "unexpected number of values unpacked from a block pointer");
      unsigned rank = (unpackedPtrs.size() - 1) / 3;
      unsigned blockShape = 1 * rank, blockStride = 2 * rank;

      return SmallVector<Value>(unpackedPtrs.begin() + blockShape,
                                unpackedPtrs.begin() + blockStride);
    } else {
      // For the regular pointers, there is no shape boundary. Return empty
      // vector.
      return {};
    }
  }

  // Returns the pitch (stride in bytes) from regular pointer or block pointer.
  Value getPitch(ConversionPatternRewriter &rewriter, Value ptr,
                 const SmallVector<Value> &unpackedPtrs,
                 unsigned elemSizeInBits, unsigned dim) const {
    Location loc = ptr.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (isTensorPointerType(ptr.getType())) {
      // The block pointer struct is expected to have the following layout:
      //    Struct {
      //      Value offset[rank];
      //      Value shape[rank];
      //      Value stride[rank];
      //      Value base;
      //    }
      assert((unpackedPtrs.size() - 1) % 3 == 0 &&
             "unexpected number of values unpacked from a block pointer");
      unsigned rank = (unpackedPtrs.size() - 1) / 3;
      unsigned blockStride = 2 * rank;
      Value stride = unpackedPtrs[blockStride + dim];
      return b.mul(b.trunc(i32_ty, stride), b.i32_val(elemSizeInBits / 8));
    } else {
      // Regular pointer.
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
    }

    return nullptr;
  }

  // Returns the strides in elements from regular pointer or block pointer.
  SmallVector<Value> getStrides(ConversionPatternRewriter &rewriter, Value ptr,
                                const SmallVector<Value> &unpackedPtrs) const {
    Location loc = ptr.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    if (isTensorPointerType(ptr.getType())) {
      // The block pointer struct is expected to have the following layout:
      //    Struct {
      //      Value offset[rank];
      //      Value shape[rank];
      //      Value stride[rank];
      //      Value base;
      //    }
      assert((unpackedPtrs.size() - 1) % 3 == 0 &&
             "unexpected number of values unpacked from a block pointer");
      unsigned rank = (unpackedPtrs.size() - 1) / 3;
      unsigned blockStride = 2 * rank, blockBase = 3 * rank;
      ;
      return SmallVector<Value>(unpackedPtrs.begin() + blockStride,
                                unpackedPtrs.begin() + blockBase);
    } else {
      // Regular pointer.
      Type resultType = ptr.getType();
      RankedTensorType tensorType = cast<RankedTensorType>(resultType);
      unsigned rank = tensorType.getRank();
      SmallVector<Value> strides(rank);
      for (unsigned dim = 0; dim < rank; ++dim) {
        int stride = getStride(ptr, dim);
        if (stride < 0)
          return {};
        strides[dim] = b.i32_val(stride);
      }
      return strides;
    }
  }

  // Returns the offsets of the block from regular pointer or block pointer.
  SmallVector<Value> getOffsets(ConversionPatternRewriter &rewriter, Value ptr,
                                const SmallVector<Value> &unpackedPtrs) const {
    Location loc = ptr.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    if (isTensorPointerType(ptr.getType())) {
      // The block pointer struct is expected to have the following layout:
      //    Struct {
      //      Value offset[rank];
      //      Value shape[rank];
      //      Value stride[rank];
      //      Value base;
      //    }
      assert((unpackedPtrs.size() - 1) % 3 == 0 &&
             "unexpected number of values unpacked from a block pointer");
      unsigned rank = (unpackedPtrs.size() - 1) / 3;
      unsigned blockOffset = 0, blockShape = 1 * rank;
      return SmallVector<Value>(unpackedPtrs.begin() + blockOffset,
                                unpackedPtrs.begin() + blockShape);
    } else {
      // For the regular pointers, the offsets have already been added into
      // bases. Return empty vector.
      return {};
    }
  }

  struct BlockIOTileSizeInfo {
    BlockIOTileSizeInfo() = delete;
    BlockIOTileSizeInfo(int tileHeight, int tileWidth, int numElemPerPackedVal,
                        int vBlocks, int rowDim, int colDim, bool transpose,
                        bool vnni,
                        std::optional<SetVector<unsigned>> regPackedBases)
        : tileHeight(tileHeight), tileWidth(tileWidth),
          numElemPerPackedVal(numElemPerPackedVal), vBlocks(vBlocks),
          rowDim(rowDim), colDim(colDim), transpose(transpose), vnni(vnni),
          regPackedBases(regPackedBases) {}
    static BlockIOTileSizeInfo unknown() {
      return {-1, -1, -1, -1, -1, -1, false, false, std::nullopt};
    }

    int tileHeight;
    int tileWidth;
    int numElemPerPackedVal;
    int vBlocks;
    int rowDim;
    int colDim;
    bool transpose;
    bool vnni;
    std::optional<SetVector<unsigned>> regPackedBases;

    bool isValid() const {
      return tileHeight >= 0 && tileWidth >= 0 && numElemPerPackedVal >= 0 &&
             vBlocks >= 0 && rowDim >= 0 && colDim >= 0;
    }
  };

  // Return the tileHeight, tileWidth, numElemPerPackedVal, vBlocks, row Dim and
  // column Dim.
  template <bool IS_LOAD>
  static BlockIOTileSizeInfo
  getBlockIOTileSize(const LinearLayout &ll, unsigned memContiguousDim,
                     unsigned elemSizeInBits, AxisInfo *maskAxisInfo = nullptr,
                     bool oneMatrixPerLoadForBT = false) {

    if (elemSizeInBits > 64)
      return BlockIOTileSizeInfo::unknown();

    const size_t rank = ll.getOutDims().size();
    std::vector<unsigned> tileShape(rank, 1);

    const LinearLayout::BasesT &bases = ll.getBases();
    auto getBase = [&](const std::string &inDim) {
      for (const auto &base : bases) {
        StringAttr attr = base.first;
        if (attr.getValue().compare(inDim) == 0)
          return base.second;
      }
      llvm_unreachable(("Could not find the input dim:" + inDim +
                        ", on the ll:" + ll.toString())
                           .c_str());
    };

    auto validateBase = [](const std::vector<int> &vec) {
      // Check there is only one element that is greater than 0
      return llvm::count_if(vec, [](int x) { return x > 0; }) == 1;
    };

    auto getFirstNonZeroDim = [](const std::vector<int> &vec) {
      auto it = llvm::find_if(vec, [](int x) { return x > 0; });
      return (it != vec.end()) ? std::distance(vec.begin(), it) : -1;
    };

    using BaseType = LinearLayout::BasesT::value_type::second_type;
    const BaseType &basesOfLane = getBase("lane");

    if (!validateBase(basesOfLane[0]))
      return BlockIOTileSizeInfo::unknown();

    // The IGC scalar backend always vectorize the non-uniform value in row
    // major. So the first non-zero dimension of the lane base is used as column
    // dim for block io.
    int fastChangeDim = getFirstNonZeroDim(basesOfLane[0]);

    // The mask constancy has to be power of 2 for block IO.
    if (maskAxisInfo &&
        !llvm::isPowerOf2_64(maskAxisInfo->getConstancy(fastChangeDim)))
      return BlockIOTileSizeInfo::unknown();

    unsigned maskConstancyFastChangeDimLimit =
        maskAxisInfo ? maskAxisInfo->getConstancy(fastChangeDim)
                     : std::numeric_limits<unsigned>::max();
    bool transpose = fastChangeDim != memContiguousDim;

    // Walk thru the register bases in incremental order to get the register
    // index for the packed value for block io.
    // TODO: improve the register packing order to support swizzled linear
    // layout.
    const BaseType &basesOfRegister = getBase("register");
    int numElemPerPackedVal = 1;
    constexpr unsigned MAX_BITS_NORMAL = 64;
    // Hardware supports the d64 for transposing. But for packing
    // transpose, we'd prefer smaller d32 type cause hardware could
    // transpose more to reduce the number of mov operation in register.
    constexpr unsigned MAX_BITS_TRANSPOSE = 32;
    constexpr unsigned MAX_BITS_VNNI = 32;
    constexpr unsigned MAX_BITS_WIDTH_NORMAL = 64 * 8; // 64 bytes.
    constexpr unsigned MAX_BITS_WIDTH_TRANSPOSE =
        8 * 4 * 8; // 8xd32. (and 4xd64)
    constexpr unsigned TRANSPOSE_LOAD_D64_HEIGHT = 8;
    constexpr unsigned MAX_TILE_HEIGHT_STORE = 8;
    constexpr unsigned MAX_TILE_HEIGHT_LOAD = 32;
    unsigned MAX_TILE_HEIGHT;
    if constexpr (IS_LOAD) {
      MAX_TILE_HEIGHT = (transpose && elemSizeInBits == 64)
                            ? TRANSPOSE_LOAD_D64_HEIGHT
                            : MAX_TILE_HEIGHT_LOAD;
    } else {
      MAX_TILE_HEIGHT = MAX_TILE_HEIGHT_STORE;
    }
    unsigned MAX_BITS_WIDTH =
        transpose ? MAX_BITS_WIDTH_TRANSPOSE : MAX_BITS_WIDTH_NORMAL;

    SetVector<unsigned> regPackBases;
    auto packRegister = [&](unsigned dim, unsigned maxPackNum) {
      for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
           ++regBaseIter) {
        if (numElemPerPackedVal >= maxPackNum) {
          // Reached the maximum number of elements per packed value.
          break;
        }
        const std::vector<int> &base = basesOfRegister[regBaseIter];
        if (!validateBase(base))
          continue; // Skip as the register can not be trivial packed.
        int baseDim = getFirstNonZeroDim(base);
        if (dim == baseDim) {
          if (tileShape[dim] != base[dim])
            continue; // Skip the register not in dense tile.
          // The value can be loaded as packed value.
          tileShape[dim] <<= 1;
          numElemPerPackedVal <<= 1;
          regPackBases.insert(1 << regBaseIter);
        }
      }
    };

    packRegister(
        memContiguousDim,
        mlir::ceil<unsigned>(transpose ? MAX_BITS_TRANSPOSE : MAX_BITS_NORMAL,
                             elemSizeInBits));

    // For the transpose case, we have to pack the elements to d32.
    if (transpose &&
        (numElemPerPackedVal * elemSizeInBits) != MAX_BITS_TRANSPOSE)
      return BlockIOTileSizeInfo::unknown();

    // We already get the basic tile shape in packing values.
    // To increase the tile shape along each lane dimension.
    bool vnni = false;
    for (const std::vector<int> &base : basesOfLane) {
      if (!validateBase(base))
        break; // break if the lane bases are not trivial.
      int dim = getFirstNonZeroDim(base);
      if (tileShape[dim] != base[dim]) {
        if (numElemPerPackedVal == 1) {
          // There are no register packing. Try to pack here.
          if (dim != fastChangeDim) {
            // VNNI pack:
            packRegister(dim,
                         mlir::ceil<unsigned>(MAX_BITS_VNNI, elemSizeInBits));
            if ((numElemPerPackedVal * elemSizeInBits) == MAX_BITS_VNNI)
              vnni = true;
          }
        }
        if (tileShape[dim] != base[dim]) {
          // break if we can not increase the tile shape along this dim after
          // packing.
          break;
        }
      }
      tileShape[dim] <<= 1;
    }

    const unsigned numLanes = 1 << basesOfLane.size();
    // The slice of a name is not distributed densely across the lane. It is not
    // supported by block io.
    if ((product<unsigned>(tileShape) / numElemPerPackedVal) != numLanes)
      return BlockIOTileSizeInfo::unknown();

    unsigned sliceRank = 0;
    int rowDim = -1;
    for (size_t i = 0; i < rank; ++i) {
      if (tileShape[i] > 1) {
        sliceRank++;
        // if the slice has more than one non-zero size. Chose the
        // non-fast change dim as the row dim.
        if (i != fastChangeDim)
          rowDim = i;
      }
    }

    // The block IO only supports 2D shape.
    if (sliceRank > 2)
      return BlockIOTileSizeInfo::unknown();

    // When transposed, width and height constraints swap between fastChangeDim
    // and rowDim: fastChangeDim is constrained by block io tile width in the
    // non-transposed case and by block io tile height in the transposed case,
    // while rowDim uses the opposite limits.

    // The tile shape sizes should not exceed the hardware limit.
    unsigned fastChangeDimLimit =
        !transpose ? MAX_BITS_WIDTH / elemSizeInBits : MAX_TILE_HEIGHT;
    unsigned rowDimLimit =
        !transpose ? MAX_TILE_HEIGHT : MAX_BITS_WIDTH / elemSizeInBits;

    // The tile shape sizes should not exceed the mask constancy limit.
    fastChangeDimLimit =
        std::min(fastChangeDimLimit, maskConstancyFastChangeDimLimit);

    unsigned maskConstancyRowDimLimit = std::numeric_limits<unsigned>::max();
    if (rowDim >= 0) {
      // The mask constancy has to be power of 2 for block IO.
      if (maskAxisInfo &&
          !llvm::isPowerOf2_64(maskAxisInfo->getConstancy(rowDim)))
        return BlockIOTileSizeInfo::unknown();
      if (maskAxisInfo)
        maskConstancyRowDimLimit = maskAxisInfo->getConstancy(rowDim);
    }

    rowDimLimit = std::min(rowDimLimit, maskConstancyRowDimLimit);

    if (tileShape[fastChangeDim] > fastChangeDimLimit)
      return BlockIOTileSizeInfo::unknown();

    if (rowDim >= 0 && tileShape[rowDim] > rowDimLimit)
      return BlockIOTileSizeInfo::unknown();

    if (!oneMatrixPerLoadForBT && transpose &&
        tileShape[memContiguousDim] == numElemPerPackedVal) {
      // Increase the tile shape along the col dimension for transpose case.
      for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
           ++regBaseIter) {
        if (regPackBases.contains(1 << regBaseIter))
          continue; // Skip the register already packed.
        const std::vector<int> &base = basesOfRegister[regBaseIter];
        if (!validateBase(base))
          continue; // Skip as the bases are not trivial.
        int dim = getFirstNonZeroDim(base);
        if (dim != fastChangeDim ||
            tileShape[fastChangeDim] != base[fastChangeDim])
          continue; // Skip the register not mapped to the row dim.
        if ((tileShape[fastChangeDim] << 1) > MAX_TILE_HEIGHT)
          break; // The col dim is the height.
        if ((tileShape[fastChangeDim] << 1) > maskConstancyFastChangeDimLimit)
          break; // Should not exceed the mask constancy limit.
        tileShape[fastChangeDim] <<= 1;
        regPackBases.insert(1 << regBaseIter);
      }
    }

    // Note: we only walk thru register packing order by increasing the
    // tileHeight and vBlocks for simplicity. This may cause low efficiency in
    // block store for some cases because the block store doesn't support
    // vBlocks > 1. Illustration of the tile shape and register packing:
    // clang-format off
    //                 vBlocks=2
    //                     ^
    //          ┌───────────────────┐
    //     tileWidth=16        tileWidth=16
    //           ^                   ^
    // ┌───────────────────┬───────────────────┐
    // lane 0 1 2 .....  15 lane 0 1 2 .....  15
    // ┌────┬────┬────┬────┬────┬────┬────┬────┐
    // │R0  │    │    │    │R1  │    │    │    │
    // │    │    │    │    │    │    │    │    │
    // ├────┼────┼────┼────┼────┼────┼────┼────┤
    // │R2  │    │    │    │R3  │    │    │    │
    // │    │    │    │    │    │    │    │    │
    // └────┴────┴────┴────┴────┴────┴────┴────┘
    // We will pack the R0 and R2 as the first matrix. R1 and R3 as the second matrix with vBlocks=2 for 2 matrixes.
    // But the tile shape following maybe more efficient for block store because block store only supports vBlocks=1.
    //               tileWidth=32
    //                     ^
    // ┌───────────────────┬───────────────────┐
    // lane 0 1 2 .....  15 lane 0 1 2 .....  15
    // ┌────┬────┬────┬────┬────┬────┬────┬────┐
    // │R0  │    │    │    │R1  │    │    │    │
    // │    │    │    │    │    │    │    │    │
    // ├────┼────┼────┼────┼────┼────┼────┼────┤
    // │R2  │    │    │    │R3  │    │    │    │
    // │    │    │    │    │    │    │    │    │
    // └────┴────┴────┴────┴────┴────┴────┴────┘
    // clang-format on

    // Increase the tile shape along the row dimension. (Increase the
    // tileHeight.)
    for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
         ++regBaseIter) {
      if (regPackBases.contains(1 << regBaseIter))
        continue; // Skip the register already packed.
      const std::vector<int> &base = basesOfRegister[regBaseIter];
      if (!validateBase(base))
        continue; // Skip as the bases are not trivial.
      int dim = getFirstNonZeroDim(base);
      if (rowDim < 0 && dim != fastChangeDim) {
        rowDim = dim;
        // The mask constancy has to be power of 2 for block IO.
        if (maskAxisInfo &&
            !llvm::isPowerOf2_64(maskAxisInfo->getConstancy(rowDim)))
          return BlockIOTileSizeInfo::unknown();
        if (maskAxisInfo)
          maskConstancyRowDimLimit = maskAxisInfo->getConstancy(rowDim);
      }
      if (dim != rowDim || tileShape[rowDim] != base[rowDim])
        continue; // Skip the register not mapped to the row dim.
      if (!transpose) {
        if ((tileShape[rowDim] << 1) > MAX_TILE_HEIGHT)
          break; // If the tile height is limited, we stop here.
      } else {
        if (((tileShape[rowDim] << 1) * elemSizeInBits) > MAX_BITS_WIDTH)
          break; // The row is the width.
      }
      // The size should not exceed the mask constancy limit.
      if ((tileShape[rowDim] << 1) > maskConstancyRowDimLimit)
        break;
      tileShape[rowDim] <<= 1;
      regPackBases.insert(1 << regBaseIter);
    }

    if (rowDim < 0)
      rowDim = (fastChangeDim != 0) ? 0 : 1;

    if (transpose && elemSizeInBits == 64) {
      // D64 transpose only supports 8 rows.
      if (tileShape[fastChangeDim] != TRANSPOSE_LOAD_D64_HEIGHT)
        return BlockIOTileSizeInfo::unknown();
    }

    unsigned vBlocks = 1;
    if (!transpose) {
      // Increase the tile shape along the column dimension. (Increase the
      // vBlocks.)
      for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
           ++regBaseIter) {
        if (regPackBases.contains(1 << regBaseIter))
          continue; // Skip the register already packed.
        const std::vector<int> &base = basesOfRegister[regBaseIter];
        if (!validateBase(base))
          continue; // Skip as the bases are not trivial.
        int dim = getFirstNonZeroDim(base);
        if (dim != fastChangeDim || (tileShape[dim] * vBlocks) != base[dim])
          continue;
        if ((tileShape[fastChangeDim] * (vBlocks << 1)) >
            maskConstancyFastChangeDimLimit)
          break; // Should not exceed the mask constancy limit.
        vBlocks <<= 1;
        regPackBases.insert(1 << regBaseIter);
      }
    }
    for (unsigned regBaseIter = 0; regBaseIter < basesOfRegister.size();
         ++regBaseIter) {
      if (regPackBases.contains(1 << regBaseIter))
        continue; // Skip the register already packed.
      // insert the remaining register base.
      regPackBases.insert(1 << regBaseIter);
    }

    unsigned packedValueNumber =
    vnni ? 1 : numElemPerPackedVal; // VNNI doesn't really pack elements in

    return BlockIOTileSizeInfo(
        tileShape[transpose ? fastChangeDim : rowDim],
        tileShape[transpose ? rowDim : fastChangeDim] / packedValueNumber,
        packedValueNumber, vBlocks, rowDim, fastChangeDim, transpose, vnni,
        std::move(regPackBases));
  }
};

struct PrefetchOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::gpu::intel::PrefetchOp>::ConvertTritonGPUOpToLLVMPattern;

  PrefetchOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::gpu::intel::PrefetchOp>(
            converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::intel::PrefetchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    LogicalResult res =
        isTensorPointerType(op.getPtr().getType())
            ? rewriteTensorPointerPrefetch(op, adaptor, rewriter)
            : rewriteRegularPointerPrefetch(op, adaptor, rewriter);

    // FIXME: the prefetch lowering code should never fail. Currently it does in
    // some cases. We should address those cases instead of removing the
    // prefetch operation.
    if (failed(res)) {
      op.emitWarning("Prefetch operation could not be converted to LLVM. "
                     "The operation was erased.");
      rewriter.eraseOp(op);
    }

    return success();
  }

  LogicalResult
  rewriteTensorPointerPrefetch(triton::gpu::intel::PrefetchOp op,
                               OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    Attribute blockIOAttr =
        op->getAttr(TritonIntelGPUDialect::getBlockIOAttrName());
    if (!blockIOAttr) {
      rewriter.eraseOp(op);
      return success();
    }

    auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Value ptr = op.getPtr();
    auto ptrType = cast<PointerType>(ptr.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    Type eltTy = getTypeConverter()->convertType(tensorType.getElementType());
    const ArrayRef<int64_t> shapeRef = tensorType.getShape();
    SmallVector<int64_t> tensorShape{shapeRef.begin(), shapeRef.end()};

    const bool memoryRowMajor = isMemoryRowMajor(op);
    if (!memoryRowMajor) {
      // Swap the shape to make it row major and then get the tiling
      // size base on row major shape.
      std::swap(tensorShape[0], tensorShape[1]);
    }
    unsigned numWarps = triton::gpu::lookupNumWarps(op);

    auto m = op->getParentOfType<ModuleOp>();
    bool isPrefetch256BSupported =
        m->hasAttr(TritonIntelGPUDialect::getSupportPrefetch256BAttrName());
    auto [tileHeightInElem, tileWidthInElem, warpsM, warpsN] =
        get2DPrefetchWarpsPerCTA(tensorShape, eltTy, numWarps,
                                 isPrefetch256BSupported);
    auto llEncoding = getLinearLayout(
        tensorShape, {tileHeightInElem, tileWidthInElem}, {warpsM, warpsN});

    unsigned tileSizeInElem = tileHeightInElem * tileWidthInElem;
    unsigned numTilesPerWarp =
        (tensorShape[0] * tensorShape[1]) / (tileSizeInElem * numWarps);

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

    auto [base, baseWidth, baseHeight, rowStride, colStride, offsetBaseX,
          offsetBaseY] =
        getValuesFromBlockPointerStruct(adaptor.getPtr(), rewriter);

    if (!memoryRowMajor) {
      // Swap the width/height and strides to the row major.
      std::swap(baseWidth, baseHeight);
      std::swap(colStride, rowStride);
      std::swap(offsetBaseX, offsetBaseY);
    }

    baseWidth = b.mul(baseWidth, b.i64_val(eltTy.getIntOrFloatBitWidth() / 8));
    baseWidth = b.trunc(i32_ty, baseWidth);

    baseHeight = b.trunc(i32_ty, baseHeight);

    Value rowStrideInBytes =
        b.mul(rowStride, b.i64_val(eltTy.getIntOrFloatBitWidth() / 8));
    rowStrideInBytes = b.trunc(i32_ty, rowStrideInBytes);

    MLIRContext *ctx = getContext();
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
      Value offsetX = b.add(offsets[1].second, offsetBaseX);
      Value offsetY = b.add(offsets[0].second, offsetBaseY);

      auto newOp = TritonGEN::Matrix2DBlockPrefetchOp::create(
          rewriter, loc,
          /*ptr*/ base,
          /*base_width*/ baseWidth,
          /*base_height*/ baseHeight,
          /*base_pitch*/ rowStrideInBytes,
          /*x*/ offsetX,
          /*y*/ offsetY,
          /*elem_size_in_bits*/ elemSizeInBits,
          /*tile_width*/ tileWidthInElem,
          /*tile_height*/ tileHeightInElem,
          /*v_blocks*/ vBlocks,
          /*cache_opt*/ TritonGEN::LoadCacheControl::L1C_L3C);
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
    DpasEncodingAttr::OpIdx opIdx = getOpIdx(tensorOfPointers);
    SmallVector<int64_t> repetitions =
        dpasLayout.getDPASRepetitions(tensorShape, opIdx);
    assert(repetitions.size() == 3 &&
           "getDPASRepetitions always return rank 3 size");
    SmallVector<unsigned> numReps{repetitions.begin() + 1, repetitions.end()};

    SmallVector<int64_t, 2> shardTensorShape;
    switch (opIdx) {
    case DpasEncodingAttr::OpIdx::OperandA: {
      shardTensorShape = {
          std::min<unsigned>(tensorShape[0], dpasLayout.getShapeA()[0]),
          tensorShape[1]};
      warpsPerCTA[1] = 1;
      repCluster[1] = 1;
      numReps[1] = 1;
    } break;
    case DpasEncodingAttr::OpIdx::OperandB: {
      shardTensorShape = {
          tensorShape[0],
          std::min<unsigned>(tensorShape[1], dpasLayout.getShapeB()[1])};
      warpsPerCTA[0] = 1;
      repCluster[0] = 1;
      numReps[0] = 1;
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
          maskConstancyHor = axisInfo->getConstancy(1);
          maskConstancyVer = axisInfo->getConstancy(0);
        }
      }
    }

    SmallVector<unsigned, 2> prefetchShape =
        get2DPrefetchShapePerWarp(tensorType);
    prefetchShape = {std::min<unsigned>(prefetchShape[0], maskConstancyVer),
                     std::min<unsigned>(prefetchShape[1], maskConstancyHor)};

    SmallVector<int64_t> numPrefetchsPerRep = {
        mlir::ceil<int64_t>(shardTensorShape[0], prefetchShape[0]),
        mlir::ceil<int64_t>(shardTensorShape[1], prefetchShape[1])};

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

    Value rowStrideInBytes = getPitch(rewriter, op.getPtr(), ptrElems,
                                      elemSizeInBits, memoryRowMajor ? 0 : 1);
    if (!rowStrideInBytes)
      return failure();

    // If the stride is 0, we want to load only the first row.
    int stride = getStride(op.getPtr(), 0);
    Value baseHeight = b.i32_val(stride == 0 ? 1 : tileHeightInElem);
    Value baseWidth = b.i32_val(
        std::max(64u, vBlocks * tileWidthInElem * (elemSizeInBits / 8)));
    Value offsetBaseX = b.i32_val(0);
    Value offsetBaseY = b.i32_val(0);

    for (int row = 0; row < numReps[0]; ++row) {
      for (int col = 0; col < numReps[1]; ++col) {
        // Prefetch the data for each repetitions.
        for (int i = 0; i < numPrefetchsPerRep[0]; ++i)
          for (int j = 0; j < numPrefetchsPerRep[1]; ++j) {
            unsigned offsetN = col * warpsPerCTA[1] * shardTensorShape[1] +
                               j * prefetchShape[1];
            unsigned offsetM = row * warpsPerCTA[0] * shardTensorShape[0] +
                               i * prefetchShape[0];

            Value pred;
            if (llMask)
              pred = (maskElems.size() > 1)
                         ? targetInfo.shuffleIdx(rewriter, loc,
                                                 masks[{offsetM, offsetN}], 0)
                         : maskElems[0];

            else
              pred = b.int_val(1, 1);

            // If the mask exists and evaluates to false, we set offsetY to be
            // equal to baseHeight, which causes the HW to ignore the generated
            // prefetch operation (given that the block to be prefetched would
            // be outside the baseWidth X baseHeight shape).
            Value offsetY = b.select(pred, b.i32_val(0), baseHeight);
            Value addr = targetInfo.shuffleIdx(
                rewriter, loc, baseAddrs[{offsetM, offsetN}], 0);

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
                /*cache_opt*/ TritonGEN::LoadCacheControl::L1C_L3C);
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

    rewriter.eraseOp(op);
    return success();
  }

private:
  // tensor shape has to be in row major.
  // Returns:
  // Prefetch Op Shape in {M, N}
  // Warps per CTA in {M, N}
  std::tuple<unsigned, unsigned, unsigned, unsigned>
  get2DPrefetchWarpsPerCTA(const ArrayRef<int64_t> tensorShape, Type eltTy,
                           unsigned numWarps,
                           bool isPrefetch256BSupported) const {
    unsigned rank = tensorShape.size();
    assert(rank >= 2 && "Only rank >= 2 tensor is supported for now");
    unsigned dimM = rank - 2, dimN = rank - 1;
    unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();
    unsigned elemSizeInBytes = elemSizeInBits / 8;
    unsigned numColsPerPrefOps = std::min<unsigned>(
        tensorShape[dimN],
        (isPrefetch256BSupported ? 256 : 64) / elemSizeInBytes);
    if (isPrefetch256BSupported &&
        (numColsPerPrefOps * elemSizeInBytes) != 256) {
      // Fallback to 64 bytes per row.
      numColsPerPrefOps =
          std::min<unsigned>(numColsPerPrefOps, 64 / elemSizeInBytes);
    }

    unsigned repNumN =
        mlir::ceil((unsigned)tensorShape[dimN], numColsPerPrefOps);
    unsigned warpsNumN = std::min(numWarps, repNumN);
    unsigned warpsNumM = mlir::ceil(numWarps, warpsNumN);

    // Get the number of rows per warp to fit the shape to the tensor shape to
    // avoid duplication in prefetching.
    unsigned rowNumPerWarp = mlir::ceil<unsigned>(tensorShape[dimM], warpsNumM);
    constexpr unsigned maxNumRows = 32u;
    unsigned numRowsPerPrefOps = std::min<unsigned>(rowNumPerWarp, maxNumRows);
    SmallVector<unsigned, 2> tilePerPrefOps{numRowsPerPrefOps,
                                            numColsPerPrefOps};

    return {numRowsPerPrefOps, numColsPerPrefOps, warpsNumM, warpsNumN};
  }

  // Get the linear layout for the cooperative prefetching.
  LinearLayout getLinearLayout(const ArrayRef<int64_t> tensorShape,
                               const ArrayRef<unsigned> tileShape,
                               const ArrayRef<unsigned> warpsPerCTA) const {
    MLIRContext *ctx = getContext();
    unsigned rank = warpsPerCTA.size();
    assert(rank >= 2 && "Only rank >= 2 tensor is supported for now");
    SmallVector<unsigned> order(rank);
    for (size_t i = 0; i < warpsPerCTA.size(); ++i) {
      // The fastest change dim is the first.
      order[i] = rank - i - 1;
    }
    LinearLayout ctaLayout = identityStandardND(S("offset"), tileShape, order) *
                             identityStandardND(S("warp"), warpsPerCTA, order);

    return combineCtaCgaWithShape(std::move(ctaLayout),
                                  CGAEncodingAttr::get1CTALayout(ctx, rank),
                                  tensorShape);
  }
};

struct LoadOpToBlockIOConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::LoadOp>::ConvertTritonGPUOpToLLVMPattern;

  using ValueTable = std::map<std::pair<int, int>, Value>;

  LoadOpToBlockIOConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // FIXME: Handle the case where padding is set to PAD_NAN (#5145).
    if (op.getPadding() && op.getPadding() == PaddingOption::PAD_NAN)
      return failure();

    if (!isBlockIOCandidate(op))
      return failure();

    // FIXME: Remove once IGC can split large 2D block loads.
    std::optional<bool> oneMatrixPerLoadForBT =
        mlir::triton::tools::isEnvValueBool(mlir::triton::tools::getStrEnv(
            "TRITON_INTEL_ONE_MATRIX_PER_LOAD_BT"));
    if (!oneMatrixPerLoadForBT.has_value())
      oneMatrixPerLoadForBT =
          op->hasAttr(triton::gpu::intel::TritonIntelGPUDialect::
                          getOneMatrixPerLoadAttrName());

    // Get the max tile shape supported by the layout.
    Type resultType = op.getType();
    auto tensorType = cast<RankedTensorType>(resultType);
    Attribute encoding = tensorType.getEncoding();
    std::optional<LinearLayout> llEncoding =
        cast<DistributedEncodingTrait>(encoding).toLinearLayout(
            tensorType.getShape());
    assert(llEncoding.has_value() &&
           "unexpected failure when getting linear layout");

    // TODO: use the axis info to general the handling for both regular pointer
    // and block pointer.
    const bool memoryRowMajor = isMemoryRowMajor(op);
    const unsigned rank = tensorType.getRank();
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
    BlockIOTileSizeInfo sizeInfo = getBlockIOTileSize<true /*load*/>(
        llEncoding.value(), contiguousDim, elemSizeInBits, maskAxisInfo,
        oneMatrixPerLoadForBT.has_value() ? *oneMatrixPerLoadForBT : false);
    if (!sizeInfo.isValid())
      return failure();
    // Extract members to regular variables for C++17 compatibility
    // (capturing structured bindings in lambdas requires C++20)
    int tileHeight = sizeInfo.tileHeight;
    int tileWidth = sizeInfo.tileWidth;
    int numPackedVals = sizeInfo.numElemPerPackedVal;
    int vBlocks = sizeInfo.vBlocks;
    int rowDim = sizeInfo.rowDim;
    int colDim = sizeInfo.colDim;
    bool isTransposeRequired = sizeInfo.transpose;
    bool useVNNIFormat = sizeInfo.vnni;
    std::optional<SetVector<unsigned>> regPackedBases =
        std::move(sizeInfo.regPackedBases);

    unsigned packedElemSizeInBits = elemSizeInBits * numPackedVals;
    if (!check2DBlockAddressPayloadRestriction(packedElemSizeInBits, tileWidth))
      return failure();

    // 2D block load supports 64 bytes per row at most.
    constexpr int MAX_WIDTH = 64;
    unsigned totalBytesPerRowPerMatrix = tileWidth * packedElemSizeInBits / 8;
    if (totalBytesPerRowPerMatrix > MAX_WIDTH)
      return failure();

    // Load multiple dot operands by enlarging the vBlocks.
    vBlocks = std::min(vBlocks,
                       static_cast<int>(MAX_WIDTH / totalBytesPerRowPerMatrix));
    // vBlocks has HW limitation of 4.
    vBlocks = std::min(vBlocks, 4);
    // Limit vBlocks to 1 if block size is smaller than GRF size.
    const unsigned GRF_SIZE = 64;
    if (tileHeight * tileWidth * packedElemSizeInBits / 8 < GRF_SIZE)
      vBlocks = 1;

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();

    unsigned threadsPerWarp =
        TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>());

    StringAttr kRegister = S("register");
    StringAttr kLane = S("lane");
    StringAttr kWarp = S("warp");
    StringAttr kBlock = S("block");
    assert(regPackedBases.has_value() &&
           "invalid register bases for packing elems.");
    std::vector<std::vector<int>> bases(regPackedBases->size());
    llvm::transform(*regPackedBases, bases.begin(),
                    [&](int base) { return std::vector<int>{base}; });
    LinearLayout regMapping({{kRegister, bases}},
                            {{kRegister, llEncoding->getInDimSize(kRegister)}},
                            /*requireSurjective=*/true);

    // Get the LLVM values for pointers
    Value ptr = op.getPtr();
    Value llPtr = adaptor.getPtr();
    unsigned numElems = getTotalElemsPerThread(resultType);
    SmallVector<Value> unpackedPtr =
        unpackLLElements(ptr.getLoc(), llPtr, rewriter);
    SmallVector<Value> ptrElems =
        getBases(rewriter, ptr, unpackedPtr, numElems);
    assert(ptrElems.size() == numElems &&
           "the number of pointer values is not matched with the number of "
           "elements");

    SmallVector<Value> maskElems;
    Value llMask = adaptor.getMask();
    // Get the LLVM values for mask
    if (llMask) {
      Value mask = op.getMask();
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(maskElems.size() == numElems &&
             "the number of mask values is not matched with the number of "
             "elements");
    }

    // Get the LLVM values for other
    Value other = op.getOther();
    SmallVector<Value> otherElems;
    Value llOther = adaptor.getOther();
    DenseElementsAttr constAttr;
    if (other) {
      if (matchPattern(other, m_Constant(&constAttr)) && constAttr.isSplat()) {
        Type elemTy = constAttr.getElementType();
        auto handleSplatValue = [&](auto splatVal) {
          if (!splatVal.isZero()) {
            otherElems = SmallVector<Value>(
                numElems,
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

    int64_t numElemsPerLoad = mlir::ceil(
        tileHeight * tileWidth * numPackedVals * vBlocks, (int)threadsPerWarp);
    unsigned numValuesPerLoad =
        mlir::ceil((unsigned)numElemsPerLoad,
                   useVNNIFormat ? (32 / elemSizeInBits) : numPackedVals);
    Type packedType =
        useVNNIFormat ? i32_ty : IntegerType::get(ctx, packedElemSizeInBits);
    Type load2DGenXType = LLVM::getVectorType(packedType, numValuesPerLoad);
    Type unpackedType = LLVM::getVectorType(eltTy, numElemsPerLoad);

    Value pitch = getPitch(rewriter, ptr, unpackedPtr, elemSizeInBits,
                           memoryRowMajor ? rowDim : colDim);
    if (!pitch)
      return failure();

    SmallVector<Value> baseOffsets = getOffsets(rewriter, ptr, unpackedPtr);

    SmallVector<Value> strides = getStrides(rewriter, ptr, unpackedPtr);

    Type packedDPASOperandType;

    if (hasDpasEncoding(tensorType) || hasDotDpasEncoding(tensorType)) {

      // For the DPAS layout, there are three types of block loads used.
      // (For non-DPAS layouts, only two types are involved.)
      //   1. load2DGenXType –
      //   2. packedDPASOperandType – (This is null for non-DPAS layouts.)
      //   3. unpackedType –
      //
      // clang-format off
      // The `tt.load` operation generates the following block load sequence:
      //   %0 = load_2d %ptr : <load2DGenXType>
      //   %1 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
      //         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
      //   %2 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
      //         <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
      //   %3 = bitcast %1 : <packedDPASOperandType> -> <unpackedType>
      //   %4 = bitcast %2 : <packedDPASOperandType> -> <unpackedType>
      //   <operations for packLLElements>
      // clang-format on
      //
      // The `tt.dot` operation generates the DPAS instruction sequence:
      // clang-format off
      //   <operations for unpackLLElements>
      //   %5 = bitcast %3 : <unpackedType> -> <packedDPASOperandType>
      //   %6 = bitcast %4 : <unpackedType> -> <packedDPASOperandType>
      //   %7 = dpas %5, %6, %other : <packedDPASOperandType>, <packedDPASOperandType>, <packedDPASOperandType>
      // clang-format on
      //
      // The LLVM optimizer eliminates redundant pack/unpack element pairs
      // and corresponding bitcast operations. The final optimized IR for
      // the dot product becomes:
      //
      // clang-format off
      //   %0 = load_2d %ptr : <load2DGenXType>
      //   %1 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
      //         <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
      //   %2 = shufflevector <load2DGenXType> %0, <load2DGenXType> %0,
      //         <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
      //   %3 = dpas %1, %2, %other : <packedDPASOperandType>, <packedDPASOperandType>, <packedDPASOperandType>
      // clang-format on
      //
      // The `packedDPASOperandType` together with the `shufflevector`
      // operations defines the computation flow for the dot product.

      auto dpasLayout = getDpasLayout(tensorType);
      unsigned opsPerChannel = dpasLayout.getOpsPerChannel();
      DpasEncodingAttr::OpIdx opIdx = getOpIdx(tensorType);

      auto numDPASOprandsPerLoad = [=](const SmallVector<unsigned> &shape) {
        unsigned elemsPerLanePerDPASInst =
            product<unsigned>(shape) / threadsPerWarp;
        unsigned numOps = 0;
        // Make sure the tile shape can fit the DPAS instruction shape.
        if (tileHeight >= shape[isTransposeRequired ? 1 : 0] &&
            (tileWidth * numPackedVals * vBlocks) >=
                shape[isTransposeRequired ? 0 : 1]) {
          numOps =
              mlir::ceil<unsigned>(numElemsPerLoad, elemsPerLanePerDPASInst);
        }
        return std::make_tuple(numOps, elemsPerLanePerDPASInst);
      };

      switch (opIdx) {
      case DpasEncodingAttr::OpIdx::OperandA: {
        auto [numDPASOperands, elemsPerLanePerDPASInst] =
            numDPASOprandsPerLoad(dpasLayout.getDPASInstShapeA());

        if (((opsPerChannel == 4 && elemSizeInBits == 8) ||
             (opsPerChannel == 2 && elemSizeInBits == 16) ||
             (opsPerChannel == 1 && elemSizeInBits == 32)) &&
            numDPASOperands) {
          // Add the packedDPASOperandType to add the shuffle and bitcast ops.
          packedDPASOperandType = LLVM::getVectorType(
              packedType, elemsPerLanePerDPASInst / numPackedVals);
          unpackedType = LLVM::getVectorType(eltTy, elemsPerLanePerDPASInst);
        }
      } break;
      case DpasEncodingAttr::OpIdx::OperandB: {
        auto [numDPASOperands, elemsPerLanePerDPASInst] =
            numDPASOprandsPerLoad(dpasLayout.getDPASInstShapeB());

        if (((opsPerChannel == 4 && elemSizeInBits == 8) ||
             (opsPerChannel == 2 && elemSizeInBits == 16) ||
             (opsPerChannel == 1 && elemSizeInBits == 32)) &&
            numDPASOperands) {
          // Block 2D loads multiple Bs.
          unsigned sysDepth = dpasLayout.getSystolicDepth();
          if (!isTransposeRequired &&
              ((opsPerChannel == 4 && elemSizeInBits == 8) ||
               (opsPerChannel == 2 && elemSizeInBits == 16))) {
            // Use the VNNI packing format for DotOp B layout.
            numValuesPerLoad = numElemsPerLoad / opsPerChannel;
            packedType = i32_ty;
            load2DGenXType = LLVM::getVectorType(packedType, numValuesPerLoad);
            useVNNIFormat = true;
          }

          // Add the packedDPASOperandType to add the shuffle and bitcast
          // ops.
          packedDPASOperandType = LLVM::getVectorType(
              packedType, elemsPerLanePerDPASInst / opsPerChannel);
          unpackedType = LLVM::getVectorType(eltTy, elemsPerLanePerDPASInst);
        }
      } break;
      case DpasEncodingAttr::OpIdx::OperandC: {
        auto [numDPASOperands, elemsPerLanePerDPASInst] =
            numDPASOprandsPerLoad(dpasLayout.getDPASInstShapeC());
        // Block 2D loads multiple Cs.
        if (numElemsPerLoad >= elemsPerLanePerDPASInst) {
          static const bool multipleCPerLoad = triton::tools::getBoolEnv(
              "TRITON_INTEL_2DBLOCK_MULTIPLE_C_MATRICES_PER_LOAD");
          if (!isTransposeRequired && !multipleCPerLoad) {
            assert(numPackedVals == 1 &&
                   "invalid numPackedVals for DPAS C operand");
            tileHeight = dpasLayout.getDPASInstShapeC()[0];
            tileWidth = dpasLayout.getDPASInstShapeC()[1];
            vBlocks = 1;
            numElemsPerLoad = elemsPerLanePerDPASInst;
            numValuesPerLoad = numElemsPerLoad;
            load2DGenXType = LLVM::getVectorType(packedType, numElemsPerLoad);
            unpackedType = LLVM::getVectorType(eltTy, numElemsPerLoad);
          } else {
            // Add the packedDPASOperandType to add the shuffle and bitcast
            // ops.
            packedDPASOperandType = LLVM::getVectorType(
                packedType, elemsPerLanePerDPASInst / numPackedVals);
            unpackedType = LLVM::getVectorType(eltTy, elemsPerLanePerDPASInst);
          }
        }
      } break;
      }
    }

    SmallVector<Value> shapes = getShapes(rewriter, ptr, unpackedPtr);
    Value baseWidth, baseHeight;
    if (isTensorPointerType(ptr.getType())) {
      baseWidth = b.trunc(i32_ty, shapes[memoryRowMajor ? colDim : rowDim]);
      baseHeight = b.trunc(i32_ty, shapes[memoryRowMajor ? rowDim : colDim]);
      baseWidth = b.mul(baseWidth, b.i32_val(elemSizeInBits / 8));
    } else {
      // If the stride is 0, we want to load only the first row.
      int stride = getStride(ptr, memoryRowMajor ? rowDim : colDim);
      baseHeight = b.i32_val((stride == 0 ? 1 : tileHeight));
      baseWidth = b.i32_val(vBlocks * tileWidth * (packedElemSizeInBits / 8));
    }

    LinearLayout shuffleMapping =
        LinearLayout::identity1D(numElemsPerLoad, kRegister, kRegister);
    if (isTransposeRequired) {
      // Improve this. The current 2D block load only transposes the matrix at
      // i32 granularity. We still need to perform an additional in-register
      // transpose from i32 -> (N × ElemSizeInBits) tiles, using the tile width.
      // At the moment, we can only achieve this using a bitcast operation,
      // which implicitly uses the sub-group size as the transpose width. To
      // optimize further, we should implement this with inline VISA
      // instructions.

      // tileHeight becomes width after transposing.
      unsigned widthToTranspose = tileHeight;
      if (packedDPASOperandType) {
        // For the DPAS related layout, we will do the shuffle at first in the
        // unpacking of the elements at the DPAS operands granularity.
        // And then we will do the transposing. So the transposing width is DPAS
        // op shapes.
        DpasEncodingAttr::OpIdx opIdx = getOpIdx(tensorType);
        DpasEncodingAttr dpasLayout = getDpasLayout(tensorType);
        switch (opIdx) {
        case DpasEncodingAttr::OpIdx::OperandA: {
          widthToTranspose = dpasLayout.getDPASInstShapeA()[1];
          break;
        }
        case DpasEncodingAttr::OpIdx::OperandB: {
          widthToTranspose = dpasLayout.getDPASInstShapeB()[1];
          break;
        }
        case DpasEncodingAttr::OpIdx::OperandC: {
          widthToTranspose = dpasLayout.getDPASInstShapeC()[1];
          break;
        }
        }
        // For shuffle the transposed Dot operands matrix, we can shuffle the
        // loaded matrix in an reverse order.
        auto invertMapping = regMapping.invert();
        for (unsigned numElemsPerSurjectiveTile = numElemsPerLoad;;
             numElemsPerSurjectiveTile >>= 1) {
          assert(numElemsPerSurjectiveTile > 0 &&
                 "cannot find surjective layout for transpose.");
          auto layout =
              invertMapping.resizeInDim(kRegister, numElemsPerSurjectiveTile)
                  .resizeOutDim(kRegister, numElemsPerSurjectiveTile);
          if (layout.isSurjective()) {
            shuffleMapping =
                layout * LinearLayout::identity1D(numElemsPerLoad /
                                                      numElemsPerSurjectiveTile,
                                                  kRegister, kRegister);
            break;
          }
        }
      }

      // if (numPackedVals > 1 && (widthToTranspose) != threadsPerWarp)
      //   return failure();
    }

    Value warpId = arith::IndexCastOp::create(
        rewriter, loc, i32_ty,
        mlir::gpu::SubgroupIdOp::create(rewriter, loc,
                                        /*upperBound=*/nullptr));

    SmallVector<Value> unpackedLoadedVals(numElems);
    for (size_t elemIdx = 0; elemIdx < numElems; elemIdx += numElemsPerLoad) {
      unsigned registerIdx = regMapping.apply({{kRegister, elemIdx}})[0].second;

      // Need to apply the linear layout to get the offsets to the base of the
      // block pointer.
      // TODO: add annotation uniform to the offsets. Make sure the IGC detect
      // the offsets as uniform.
      auto offsets = applyLinearLayout(loc, rewriter, *llEncoding,
                                       {{kRegister, b.i32_val(registerIdx)},
                                        {kLane, b.i32_val(0)},
                                        {kWarp, warpId},
                                        {kBlock, b.i32_val(0)}});

      // Use the top-left address of the block to load the data.
      Value addrElem = ptrElems[registerIdx];
      Value offsetX, offsetY;
      Value adjustedBaseWidth = baseWidth, adjustedBaseHeight = baseHeight;
      Value pred;
      if (isTensorPointerType(ptr.getType())) {
        unsigned c = isTransposeRequired ? rowDim : colDim;
        unsigned r = isTransposeRequired ? colDim : rowDim;

        // To prevent triggering hardware boundary protection, expand the base
        // shape sufficiently when boundary check is absent.
        SetVector<unsigned> boundaryCheck(op.getBoundaryCheck().begin(),
                                          op.getBoundaryCheck().end());

        for (auto [dim, offset] : llvm::enumerate(offsets)) {
          Value off = b.add(baseOffsets[dim], offset.second);
          if (dim == r) {
            if (boundaryCheck.contains(r)) {
              offsetY = off;
            } else {
              adjustedBaseHeight = b.i32_val(tileHeight);
              // Use i8_ty as pitch is in number of bytes.
              off = b.mul(off, pitch);
              addrElem = b.gep(ptr_ty(ctx, 1), i8_ty, addrElem, off);
              offsetY = b.i32_val(0);
            }
          } else if (dim == c) {
            if (boundaryCheck.contains(dim)) {
              offsetX = off;
            } else {
              adjustedBaseWidth = b.i32_val(std::max(
                  64u, vBlocks * tileWidth * (packedElemSizeInBits / 8)));
              // The offsetX is number of elements instead of packed elements.
              addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, off);
              offsetX = b.i32_val(0);
            }
          } else {
            // Add the offsets of other dim to the base.
            off = b.zext(i64_ty, off);
            Value p = b.mul(off, strides[dim]);
            addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, p);
            if (boundaryCheck.contains(dim)) {
              // Add boundary checking for other dims with predication.
              pred =
                  maybeAnd(rewriter, loc, pred, b.icmp_ult(off, shapes[dim]));
            }
          }
        }
      } else {
        addrElem = targetInfo.shuffleIdx(rewriter, loc, addrElem, 0);

        // Adjust the baseWidth, offsetX and base address use the original base
        // of the BLOCK.
        offsetX = offsets[isTransposeRequired ? rowDim : colDim].second;
        offsetY = b.i32_val(0);
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
          pred =
              targetInfo.shuffleIdx(rewriter, loc, maskElems[registerIdx], 0);
        }
      }

      if (pred) {
        // We leverage the GPU block I/O hardware out-of-bound protection
        // feature by setting the offset to an invalid value when 'pred'
        // is false (the HW will not read out-of-bounds values). Later on,
        // after issuing the 2d block read operation, we will select the
        // result of the load only if the mask evaluate to true, otherwise
        // we will use 'other'.
        offsetY = b.select(pred, offsetY, adjustedBaseHeight);
      }

      assert(numPackedVals > 0 && "numPackedVals should be greater than zero.");
      Value ret = TritonGEN::Matrix2DBlockLoadOp::create(
          rewriter, loc, load2DGenXType,
          /*ptr*/ addrElem,
          /*base_width*/ adjustedBaseWidth,
          /*base_height*/ adjustedBaseHeight,
          /*base_pitch*/ pitch,
          // offsetX was in terms of original elements. The 2d block io requires
          // offsetX to be in terms of packed elements.
          /*x*/ b.udiv(offsetX, b.i32_val(numPackedVals)),
          /*y*/ offsetY,
          /*elem_size_in_bits*/ packedElemSizeInBits,
          /*tile_width*/ tileWidth,
          /*tile_height*/ tileHeight,
          /*v_blocks*/ vBlocks,
          /*transpose*/ isTransposeRequired,
          /*vnni_transform*/ !isTransposeRequired && useVNNIFormat);

      if (!isTensorPointerType(ptr.getType())) {
        // When strides[0] is 0, we only want to load the first row, so we
        // set the base height to be 1. If tile height is bigger than 1,
        // then only the first row contain valid data. To ensure the entire
        // tile is filled with valid data, we must replicate the first row
        // throughout the tile.
        if (auto baseHeightInt =
                mlir::triton::intel::getFoldedConstantValue(baseHeight)) {
          if (baseHeightInt < tileHeight && baseHeightInt == 1) {
            unsigned numIndicesPerMatrix = numValuesPerLoad / vBlocks;
            SmallVector<int32_t> shuffleIndices(numValuesPerLoad);

            // Create a vector to store the data of the first index of each
            // matrix.
            VectorType vecTy = vec_ty(packedType, vBlocks);
            Value firstIndexVec = b.undef(vecTy);

            for (unsigned valueIndex = 0; valueIndex < numValuesPerLoad;
                 ++valueIndex) {
              unsigned firstIndexVecIdx = valueIndex / numIndicesPerMatrix;
              // Handle case where an index spans two rows.
              if (valueIndex % numIndicesPerMatrix == 0) {
                Value oldVal = b.extract_element(ret, b.i32_val(valueIndex));
                Value newVal = oldVal;
                if (tileWidth < threadsPerWarp) {
                  assert(tileWidth * 2 == threadsPerWarp &&
                         "Expecting tileWidth to be 2x threadsPerWarp");
                  Value threadId = getThreadId(rewriter, loc);
                  newVal = targetInfo.shuffleIdx(
                      rewriter, loc, oldVal,
                      b.urem(threadId, b.i32_val(tileWidth)));
                }
                firstIndexVec =
                    b.insert_element(firstIndexVec.getType(), firstIndexVec,
                                     newVal, b.i32_val(firstIndexVecIdx));
              }

              shuffleIndices[valueIndex] = firstIndexVecIdx;
            }
            DenseI32ArrayAttr attr =
                rewriter.getDenseI32ArrayAttr(shuffleIndices);
            ret = LLVM::ShuffleVectorOp::create(rewriter, loc, load2DGenXType,
                                                firstIndexVec, firstIndexVec,
                                                attr);
          }
        }
      }

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
            unsigned elemIdx =
                (opsIdx * numValsPerDPASOperand + i) * numPackedVals;
            unsigned suffleIdx =
                shuffleMapping.apply({{kRegister, elemIdx}})[0].second;
            indices[i] = suffleIdx / numPackedVals;
          }
          DenseI32ArrayAttr attr = rewriter.getDenseI32ArrayAttr(indices);
          Value dpasOperand = LLVM::ShuffleVectorOp::create(
              rewriter, loc, packedDPASOperandType, ret, ret, attr);

          unpackedVal = b.bitcast(dpasOperand, unpackedType);
        } else {
          // if (isTransposeRequired) {
          //   if (numPackedVals > 1 && tileHeight != threadsPerWarp) {
          //     std::string simdAsm = TransposeAsm(
          //         threadsPerWarp, tileHeight, numPackedVals,
          //         threadsPerWarp * numValuesPerLoad * numPackedVals, eltTy,
          //         XeArch::Xe2);
          //
          //     XeBuilder xeBuilder;
          //     XeInstr &transpose = *xeBuilder.create<XeInstr>(simdAsm);
          //     XeBuilder::Operand *res = xeBuilder.newOperand("=rw");
          //     XeBuilder::Operand *unpackIn = xeBuilder.newOperand(ret, "rw");
          //     transpose({res, unpackIn}, /*onlyAttachMLIRArgs=*/true);
          //     unpackedVal =
          //         xeBuilder.launch(rewriter, loc, unpackedType, false);
          //   } else {
          //     // we can use the bitcast to do the transpose
          //     unpackedVal = b.bitcast(ret, unpackedType);
          //   }
          // } else
          unpackedVal = b.bitcast(ret, unpackedType);
        }

        SmallVector<int32_t> unpackIndices(numElemsPerUnpackedType);
        for (int i = 0; i < numElemsPerUnpackedType; ++i) {
          unsigned elemIdxInPackedValue = opsIdx * numElemsPerUnpackedType + i;
          unsigned shuffledIdx =
              shuffleMapping.apply({{kRegister, elemIdxInPackedValue}})[0]
                  .second;
          unsigned registerIdx =
              regMapping.apply({{kRegister, elemIdx + shuffledIdx}})[0].second;
          unpackIndices[i] = registerIdx;
        }
        if (otherElems.size()) {
          assert(maskElems.size() == otherElems.size() &&
                 "Invalid size of the masks");
          Value other = b.undef(unpackedType);
          for (const auto [i, registerIdx] : llvm::enumerate(unpackIndices)) {
            Value falseVal = otherElems[registerIdx];
            other = b.insert_element(other, falseVal, b.i32_val(i));
          }
          unpackedVal = b.select(pred, unpackedVal, other);
        }

        for (const auto [i, registerIdx] : llvm::enumerate(unpackIndices)) {
          unpackedLoadedVals[registerIdx] =
              b.extract_element(unpackedVal, b.i32_val(i));
        }
      }
    }

    auto typeConverter = getTypeConverter();
    Type llvmResultStructTy = typeConverter->convertType(op.getType());
    Value resultStruct = packLLElements(loc, typeConverter, unpackedLoadedVals,
                                        rewriter, llvmResultStructTy);
    rewriter.replaceOp(op, {resultStruct});

    return success();
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  using ConvertOpToLLVMPattern<triton::LoadOp>::ConvertOpToLLVMPattern;

  LoadOpConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::LoadOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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
    unsigned vec = getVectorSize(ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(mask));

    SmallVector<Value> ptrElems, maskElems, otherElems;
    bool otherIsSplatConstInt = false;
    int64_t splatVal = 0;

    if (isTensorPointerType(ptr.getType())) {
      // fallback to gather load.
      auto tensorType = cast<RankedTensorType>(op.getType());
      std::tie(ptrElems, maskElems, otherElems) = convertBlockPtrToTensorOfPtr(
          loc, llPtr, tensorType, valueElemTy, rewriter, op.getBoundaryCheck(),
          op.getPadding());
    } else {
      // Get the LLVM values for pointers
      ptrElems = unpackLLElements(loc, llPtr, rewriter);
      assert(ptrElems.size() == numElems);

      // Get the LLVM values for mask
      if (llMask) {
        maskElems = unpackLLElements(loc, llMask, rewriter);
        assert(maskElems.size() == numElems);
      }

      // Get the LLVM values for `other`
      // TODO: (goostavz) handle when other is const but not splat, which
      //       should be rarely seen
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
      else if (canUsePredicatedInstructions(op))
        ret = TritonGEN::PredicatedLoadOp::create(
            rewriter, loc, retTy, addrElem, b.i64_val(alignment), pred, other_);
      else {
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

struct StoreOpToBlockIOConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>,
      public BlockIOConversionBase {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::StoreOp>::ConvertTritonGPUOpToLLVMPattern;

  StoreOpToBlockIOConversion(
      LLVMTypeConverter &converter, const triton::intel::TargetInfo &targetInfo,
      const triton::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        BlockIOConversionBase(targetInfo, axisAnalysisPass) {}

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
    BlockIOTileSizeInfo sizeInfo = getBlockIOTileSize<false /*store*/>(
        llEncoding.value(), contiguousDim, elemSizeInBits, maskAxisInfo);
    if (!sizeInfo.isValid())
      return failure();
    auto [tileHeight, tileWidth, numPackedVals, vBlocks, rowDim, colDim,
          isTransposeRequired, useVNNIFormat, regPackedBases] =
        std::move(sizeInfo);

    unsigned packedElemSizeInBits = elemSizeInBits * numPackedVals;
    if (!check2DBlockAddressPayloadRestriction(packedElemSizeInBits, tileWidth))
      return failure();

    // Limit vBlock to 1
    vBlocks = 1;

    if (isTransposeRequired || useVNNIFormat) {
      // 2D Block store doesn't support transpose or vnni.
      return failure();
    }

    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value warpId = arith::IndexCastOp::create(
        rewriter, loc, i32_ty,
        mlir::gpu::SubgroupIdOp::create(rewriter, loc,
                                        /*upperBound=*/nullptr));

    Value llPtr = adaptor.getPtr();

    SmallVector<Value> ptrElems, maskElems;
    Value baseWidth, baseHeight, pitch, offsetBaseX, offsetBaseY;

    Value ptr = op.getPtr();
    unsigned numElems = getTotalElemsPerThread(tensorType);
    bool isBlockPointer = isTensorPointerType(ptr.getType());
    if (isBlockPointer) {
      auto [base, width, height, rowStride, colStride, offsetX, offsetY] =
          getValuesFromBlockPointerStruct(llPtr, rewriter);

      ptrElems = SmallVector<Value>(numElems, base);

      Value elemSizeInBytes = b.i32_val(elemSizeInBits / 8);
      width = b.trunc(i32_ty, width);
      rowStride = b.trunc(i32_ty, rowStride);
      // encoded as bytes.
      baseWidth = b.mul(width, elemSizeInBytes);
      baseHeight = b.trunc(i32_ty, height);
      // encoded as bytes.
      pitch = b.mul(rowStride, elemSizeInBytes);
      offsetBaseX = offsetX;
      offsetBaseY = offsetY;
    } else {
      // Get the LLVM values for pointers
      ptrElems = unpackLLElements(loc, llPtr, rewriter);
      assert(ptrElems.size() == numElems &&
             "the number of pointer values is not matched with the number of "
             "elements");

      Value llMask = adaptor.getMask();
      // Get the LLVM values for mask
      if (llMask) {
        Value mask = op.getMask();
        maskElems = unpackLLElements(loc, llMask, rewriter);
        assert(maskElems.size() == numElems &&
               "the number of mask values is not matched with the number of "
               "elements");
      }

      baseWidth = b.i32_val(vBlocks * tileWidth * (packedElemSizeInBits / 8));
      baseHeight = b.i32_val(tileHeight);
      pitch = getPitch(rewriter, ptr, ptrElems, elemSizeInBits,
                       memoryRowMajor ? rowDim : colDim);
      if (!pitch)
        return failure();
      offsetBaseX = b.i32_val(0);
      offsetBaseY = b.i32_val(0);
    }

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
      Value addrElem = ptrElems[registerIdx];
      Value offsetX, offsetY;
      Value adjustedBaseWidth = baseWidth, adjustedBaseHeight = baseHeight;
      if (isBlockPointer) {
        offsetX = b.add(offsetBaseX, offsets[colDim].second);
        offsetY = b.add(offsetBaseY, offsets[rowDim].second);

        // To prevent triggering hardware boundary protection, expand the base
        // shape sufficiently when boundary check is absent.
        SetVector<unsigned> boundaryCheck(op.getBoundaryCheck().begin(),
                                          op.getBoundaryCheck().end());

        if (!boundaryCheck.contains(colDim)) {
          adjustedBaseWidth = b.i32_val(
              std::max(64u, vBlocks * tileWidth * (packedElemSizeInBits / 8)));
          // The offsetX is number of elements instead of packed elements.
          addrElem = b.gep(ptr_ty(ctx, 1), eltTy, addrElem, offsetX);
          offsetX = b.i32_val(0);
        }
        if (!boundaryCheck.contains(rowDim)) {
          adjustedBaseHeight = b.i32_val(tileHeight);
          // Use i8_ty as pitch is in number of bytes.
          Value off = b.mul(offsetY, pitch);
          addrElem = b.gep(ptr_ty(ctx, 1), i8_ty, addrElem, off);
          offsetY = b.i32_val(0);
        }
      } else {
        addrElem = targetInfo.shuffleIdx(rewriter, loc, addrElem, 0);

        // Adjust the baseWidth, offsetX and base address use the original base
        // of the BLOCK.
        offsetX = offsets[colDim].second;
        offsetY = b.i32_val(0);
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
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::StoreOp>(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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
    SmallVector<Value> ptrElems, maskElems;
    unsigned vec = getVectorSize(ptr);
    if (llMask)
      vec = std::min<size_t>(vec, getMaskAlignment(op.getMask()));

    if (isTensorPointerType(ptr.getType())) {
      // fallback to scatter store.
      auto tensorType = cast<RankedTensorType>(valueTy);
      SmallVector<Value> dummyOther;
      std::tie(ptrElems, maskElems, dummyOther) = convertBlockPtrToTensorOfPtr(
          loc, adaptor.getPtr(), tensorType, valueElemTy, rewriter,
          op.getBoundaryCheck());
    } else {
      Value llPtr = adaptor.getPtr();
      ptrElems = unpackLLElements(loc, llPtr, rewriter);
      if (llMask)
        maskElems = unpackLLElements(loc, llMask, rewriter);
    }

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
      else if (canUsePredicatedInstructions(op))
        TritonGEN::PredicatedStoreOp::create(rewriter, loc, addrElem, vecWord,
                                             b.i64_val(alignment), maskVal);
      else
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
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicCASOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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
        if (op.getResult().use_empty())
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
      PatternBenefit benefit)
      : ConvertTritonGPUOpToLLVMPattern<triton::AtomicRMWOp>(converter,
                                                             benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

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
    auto vec = getVectorSize(ptr);
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
      bool support16BitAtomics = moduleOp->hasAttr(
          TritonIntelGPUDialect::getSupport16BitAtomicsAttrName());
      if (valueElemNBits == 16 && !support16BitAtomics) {
        op.emitWarning("'tt.atomic_rmw' op fp16/bf16 datatype is not supported "
                       "in the target HW, software emulation is an "
                       "experimental feature (use at own risk)");
        Block *endBlock = emulate16BitsAtomicRmw(
            rewriter, loc, atomicRmwAttr, valueElemTy, rmwPtr, rmwVal,
            maybeAnd(rewriter, loc, b.true_val(), rmwMask), {zero});
        ret = endBlock->getArgument(0);
      } else {
        if (op.getResult().use_empty())
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

  // Emulate 16-bit atomicrmw through a loop with 32-bit cmpxchg.
  // TODO: optimize for the case when rmwMask is a true constant?
  Block *emulate16BitsAtomicRmw(ConversionPatternRewriter &rewriter,
                                Location loc, mlir::triton::RMWOp atomicOp,
                                Type valueElemTy, Value rmwPtr, Value rmwVal,
                                Value rmwMask, ArrayRef<Value> ops) const {
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

} // namespace

void mlir::triton::intel::populateLoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns,
    const intel::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               StoreOpConversion, PrefetchOpConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit);
  // BlockIO is more efficient than gather load or scatter store.
  patterns.add<LoadOpToBlockIOConversion, StoreOpToBlockIOConversion>(
      typeConverter, targetInfo, axisInfoAnalysis, benefit.getBenefit() + 2);
}
