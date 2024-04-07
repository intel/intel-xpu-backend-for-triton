#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

// Utility
namespace mlir {
namespace triton {

static Type getI1SameShapeFromTensorOrTensorPtr(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(tensorType.getShape(), i1Type,
                                 tensorType.getEncoding());
  } else if (auto ptrType = type.dyn_cast<triton::PointerType>()) {
    Type pointeeType = ptrType.getPointeeType();
    if (auto tensorType = pointeeType.dyn_cast<RankedTensorType>()) {
      return RankedTensorType::get(tensorType.getShape(), i1Type,
                                   tensorType.getEncoding());
    }
  }
  return Type();
}

namespace gpu {

// TODO: Inheritance of layout attributes
// so that all distributed layouts implement
// these utilities

unsigned getTotalElemsPerThread(Attribute layout, ArrayRef<int64_t> shape,
                                Type eltTy) {
  if (auto tritonGPUAttr = layout.dyn_cast<TritonGPU_AttrTrait>()) {
    return tritonGPUAttr.getTotalElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getTotalElemsPerThread not implemented");
    return 0;
  }
}

SmallVector<unsigned> getElemsPerThread(Attribute layout,
                                        ArrayRef<int64_t> shape, Type eltTy) {
  if (auto tritonGPUAttr = layout.dyn_cast<TritonGPU_AttrTrait>()) {
    return tritonGPUAttr.getElemsPerThread(shape, eltTy);
  } else {
    llvm::report_fatal_error("getElemsPerThread not implemented");
    return SmallVector<unsigned>();
  }
}

SmallVector<unsigned> getElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || type.isa<triton::PointerType>())
    return SmallVector<unsigned>(1, 1);
  auto tensorType = type.cast<RankedTensorType>();
  return getElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
                           tensorType.getElementType());
}

unsigned getTotalElemsPerThread(Type type) {
  if (type.isIntOrIndexOrFloat() || type.isa<triton::PointerType>())
    return 1;
  auto tensorType = type.cast<RankedTensorType>();
  return getTotalElemsPerThread(tensorType.getEncoding(), tensorType.getShape(),
                                tensorType.getElementType());
}

SmallVector<unsigned> getThreadsPerWarp(Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return distributedLayout.getThreadsPerWarp();
  } else {
    llvm::report_fatal_error("getThreadsPerWarp not implemented");
    return SmallVector<unsigned>();
  }
}

unsigned getWarpSize(Attribute layout) {
  unsigned size = 1;
  auto threadsPerWarp = getThreadsPerWarp(layout);
  for (auto e : threadsPerWarp) {
    size *= e;
  }
  return size;
}

SmallVector<unsigned>
getThreadsPerWarpWithUniqueData(Attribute layout,
                                ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentThreadsPerWarp =
        getThreadsPerWarpWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> threadsPerWarp = parentThreadsPerWarp;
    threadsPerWarp.erase(threadsPerWarp.begin() + sliceLayout.getDim());
    return threadsPerWarp;
  }
  auto threadsPerWarp = getThreadsPerWarp(layout);
  assert(threadsPerWarp.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < threadsPerWarp.size(); i++) {
    threadsPerWarp[i] = std::min<unsigned>(threadsPerWarp[i], tensorShape[i]);
  }

  return threadsPerWarp;
}

SmallVector<unsigned> getWarpsPerCTA(Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return distributedLayout.getWarpsPerCTA();
  }

  llvm::report_fatal_error("getWarpsPerCTA not implemented");
  return SmallVector<unsigned>();
}

SmallVector<unsigned>
getWarpsPerCTAWithUniqueData(Attribute layout, ArrayRef<int64_t> tensorShape) {
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(tensorShape);
    auto parentWarpsPerCTA =
        getWarpsPerCTAWithUniqueData(parentLayout, parentShape);
    SmallVector<unsigned> warpsPerCTA = parentWarpsPerCTA;
    warpsPerCTA.erase(warpsPerCTA.begin() + sliceLayout.getDim());
    return warpsPerCTA;
  }
  auto warpsPerCTA = getWarpsPerCTA(layout);
  assert(warpsPerCTA.size() == tensorShape.size() &&
         "layout and tensor shape must have the same rank");
  for (unsigned i = 0; i < warpsPerCTA.size(); i++) {
    auto sizePerWarp =
        getSizePerThread(layout)[i] * getThreadsPerWarp(layout)[i];
    auto maxWarpsPerDim = ceil<unsigned>(tensorShape[i], sizePerWarp);
    warpsPerCTA[i] = std::min<unsigned>(warpsPerCTA[i], maxWarpsPerDim);
  }

  return warpsPerCTA;
}

SmallVector<unsigned> getSizePerThread(Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return distributedLayout.getSizePerThread();
  } else {
    llvm::report_fatal_error("getSizePerThread not implemented");
    return {};
  }
}

SmallVector<unsigned> getContigPerThread(Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return distributedLayout.getContigPerThread();
  } else {
    llvm::report_fatal_error("getContigPerThread not implemented");
    return {};
  }
}

SmallVector<unsigned> getUniqueContigPerThread(Attribute layout,
                                               ArrayRef<int64_t> shape) {
  // If slice layout, call recursively on parent layout, and drop
  // sliced dim
  if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto parentShape = sliceLayout.paddedShape(shape);
    auto parentUniqueContigPerThread =
        getUniqueContigPerThread(parentLayout, parentShape);
    parentUniqueContigPerThread.erase(parentUniqueContigPerThread.begin() +
                                      sliceLayout.getDim());
    return parentUniqueContigPerThread;
  }
  // Base case
  auto rank = shape.size();
  SmallVector<unsigned> ret(rank);
  auto contigPerThread = getContigPerThread(layout);
  assert(contigPerThread.size() == rank && "Unexpected contigPerThread size");
  for (int d = 0; d < rank; ++d) {
    ret[d] = std::min<unsigned>(shape[d], contigPerThread[d]);
  }
  return ret;
}

SmallVector<unsigned> getShapePerCTATile(Attribute layout,
                                         ArrayRef<int64_t> tensorShape) {
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return distributedLayout.getShapePerCTATile(tensorShape);
  } else {
    llvm::report_fatal_error("getShapePerCTATile not implemented");
    return SmallVector<unsigned>();
  }
}

bool isExpensiveView(Type srcType, Type dstType) {
  return getTotalElemsPerThread(srcType) != getTotalElemsPerThread(dstType);
}

/* Utility function used by getOrder and getCTAOrder of SliceEncodingAttr.
 * Erase dim and decrease all values larger than dim by 1.
 * Example:    order = [0, 2, 4, 3, 1], dim = 2
 *          resOrder = [0,    3, 2, 1]
 */
static SmallVector<unsigned> eraseOrder(ArrayRef<unsigned> order,
                                        unsigned dim) {
  unsigned rank = order.size();
  assert(dim < rank && "Invalid dim to erase");
  SmallVector<unsigned> resOrder;
  for (unsigned i : order)
    if (i < dim)
      resOrder.push_back(i);
    else if (i > dim)
      resOrder.push_back(i - 1);
  return resOrder;
}

SmallVector<unsigned> getOrder(Attribute layout) {
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>()) {
    return SmallVector<unsigned>(blockedLayout.getOrder().begin(),
                                 blockedLayout.getOrder().end());
  } else if (auto mmaLayout = layout.dyn_cast<MmaEncodingTrait>()) {
    auto distributedLayout = layout.cast<DistributedEncodingTrait>();
    auto rank = distributedLayout.getWarpsPerCTA().size();
    SmallVector<unsigned> order(rank);
    for (auto i = 0; i < rank; ++i)
      order[i] = rank - 1 - i;
    if (auto mfmaLayout = layout.dyn_cast<AMDMfmaEncodingAttr>()) {
      if (mfmaLayout.getIsTransposed()) {
        std::reverse(order.begin(), order.end());
      }
    }
    return order;
  } else if (auto dotLayout = layout.dyn_cast<DotOperandEncodingAttr>()) {
    auto rank = getWarpsPerCTA(dotLayout.getParent()).size();
    SmallVector<unsigned> order(rank);
    for (auto i = 0; i < rank; ++i)
      order[i] = rank - 1 - i;
    return order;
  } else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>()) {
    SmallVector<unsigned> parentOrder = getOrder(sliceLayout.getParent());
    unsigned dim = sliceLayout.getDim();
    SmallVector<unsigned> order;
    for (unsigned d : parentOrder) {
      if (d == dim)
        continue;
      else if (d > dim)
        order.push_back(d - 1);
      else
        order.push_back(d);
    }
    return order;
  } else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>()) {
    return SmallVector<unsigned>(sharedLayout.getOrder().begin(),
                                 sharedLayout.getOrder().end());
  } else {
    llvm::report_fatal_error("Unimplemented usage of getOrder");
  }
  return {};
};

CTALayoutAttr getCTALayout(Attribute layout) {
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return CTALayoutAttr::get(
        layout.getContext(), getCTAsPerCGA(distributedLayout),
        getCTASplitNum(distributedLayout), getCTAOrder(distributedLayout));
  } else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>())
    return sharedLayout.getCTALayout();
  else
    llvm::report_fatal_error("Unimplemented usage of getCTALayout");
  return {};
}

SmallVector<unsigned> getCTAsPerCGA(Attribute layout) {
  ArrayRef<unsigned> ref;
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>())
    return distributedLayout.getCTAsPerCGA();
  else if (layout.isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>())
    return {1, 1};
  else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>())
    ref = sharedLayout.getCTALayout().getCTAsPerCGA();
  else
    llvm::report_fatal_error("Unimplemented usage of getCTAsPerCGA");
  return SmallVector<unsigned>(ref.begin(), ref.end());
}

SmallVector<unsigned> getCTASplitNum(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    return distributedLayout.getCTASplitNum();
  } else if (layout.isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>()) {
    res.resize(2);
    res[0] = res[1] = 1;
  } else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>()) {
    res.assign(sharedLayout.getCTALayout().getCTASplitNum().begin(),
               sharedLayout.getCTALayout().getCTASplitNum().end());
  } else {
    assert(false && "Unimplemented usage of getCTASplitNum");
  }
  return res;
}

SmallVector<unsigned> getCTAOrder(Attribute layout) {
  SmallVector<unsigned> res;
  if (auto distributedLayout = layout.dyn_cast<DistributedEncodingTrait>()) {
    res = distributedLayout.getCTAOrder();
  } else if (layout.isa<AMDMfmaEncodingAttr, AMDWmmaEncodingAttr>()) {
    return {0, 1};
  } else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>()) {
    res = SmallVector<unsigned>(sharedLayout.getCTALayout().getCTAOrder());
  } else {
    llvm::report_fatal_error("Unimplemented usage of getCTAOrder");
  }
  return res;
}

SmallVector<int64_t> getShapePerCTA(ArrayRef<unsigned> CTASplitNum,
                                    ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<int64_t> shapePerCTA(rank);
  for (unsigned i = 0; i < rank; ++i) {
    // This wrapping rule must be consistent with emitCTAOffsetForLayout
    unsigned splitNum = std::min<unsigned>(shape[i], CTASplitNum[i]);
    shapePerCTA[i] = shape[i] / splitNum;
  }
  return shapePerCTA;
}

SmallVector<int64_t> getShapePerCTA(Attribute layout, ArrayRef<int64_t> shape) {
  if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>()) {
    // Special logic for pipeline pass, where shape is 3D and CTALayout is 2D.
    // The first dim of shape is numStages. This is a work around, otherwise too
    // many places would have to be modified in pipeline pass. Maybe we need to
    // refactor this logic in the future.
    auto CTASplitNum = sharedLayout.getCTALayout().getCTASplitNum();
    if (shape.size() == CTASplitNum.size() + 1) {
      auto res = getShapePerCTA(CTASplitNum, shape.drop_front());
      res.insert(res.begin(), shape.front());
      return res;
    }
  }
  return getShapePerCTA(getCTASplitNum(layout), shape);
}

SmallVector<int64_t> getShapePerCTA(Type type) {
  auto tensorType = type.cast<TensorOrMemDesc>();
  return getShapePerCTA(tensorType.getEncoding(), tensorType.getShape());
}

unsigned getNumWarpsPerCTA(Attribute layout) {
  SmallVector<unsigned> warpsPerCTA;
  if (auto blockedLayout = layout.dyn_cast<BlockedEncodingAttr>())
    warpsPerCTA = blockedLayout.getWarpsPerCTA();
  else if (auto sliceLayout = layout.dyn_cast<SliceEncodingAttr>())
    return getNumWarpsPerCTA(sliceLayout.getParent());
  else if (auto mmaLayout = layout.dyn_cast<MmaEncodingTrait>()) {
    // Use the distributed layout interface to get the number of warps per CTA.
    auto distributedLayout = layout.cast<DistributedEncodingTrait>();
    warpsPerCTA = distributedLayout.getWarpsPerCTA();
  } else if (auto mfmaLayout = layout.dyn_cast<AMDMfmaEncodingAttr>())
    warpsPerCTA = mfmaLayout.getWarpsPerCTA();
  else if (auto wmmaLayout = layout.dyn_cast<AMDWmmaEncodingAttr>())
    warpsPerCTA = wmmaLayout.getWarpsPerCTA();
  else if (auto dotLayout = layout.dyn_cast<DotOperandEncodingAttr>())
    return getNumWarpsPerCTA(dotLayout.getParent());
  else if (auto sharedLayout = layout.dyn_cast<SharedEncodingAttr>())
    llvm::report_fatal_error("Cannot get numWarps from SharedEncodingAttr");
  else
    llvm::report_fatal_error("Unimplemented usage of getNumWarpsPerCTA");
  return product<unsigned>(warpsPerCTA);
}

unsigned getNumCTAs(Attribute layout) {
  return product<unsigned>(getCTAsPerCGA(layout));
}

bool isaDistributedLayout(Attribute layout) {
  return layout.isa<BlockedEncodingAttr>() || layout.isa<MmaEncodingTrait>() ||
         layout.isa<SliceEncodingAttr>();
}

template <typename T> bool hasEncoding(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<TensorOrMemDesc>()) {
    auto encoding = tensorType.getEncoding();
    return encoding && encoding.isa<T>();
  }
  return false;
}

bool hasDotOperandEncoding(Value value) {
  return hasEncoding<triton::gpu::DotOperandEncodingAttr>(value);
}

bool isExpensiveCat(CatOp cat, Attribute targetEncoding) {
  // If the new elements per thread is less than the old one, we will need to do
  // convert encoding that goes through shared memory anyway. So we consider it
  // as expensive.
  RankedTensorType tensorTy = cat.getType();
  auto totalElemsPerThread = gpu::getTotalElemsPerThread(tensorTy);
  auto shape = tensorTy.getShape();
  auto elemTy = tensorTy.getElementType();
  auto newTotalElemsPerThread =
      gpu::getTotalElemsPerThread(targetEncoding, shape, elemTy);
  return newTotalElemsPerThread < totalElemsPerThread;
}

LogicalResult CTALayoutAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, ArrayRef<unsigned> CTAsPerCGA,
    ArrayRef<unsigned> CTASplitNum, ArrayRef<unsigned> CTAOrder) {
  if (CTAsPerCGA.size() != CTASplitNum.size() ||
      CTASplitNum.size() != CTAOrder.size()) {
    return emitError() << "CTAsPerCGA, CTASplitNum, and CTAOrder must all have "
                          "the same rank.";
  }

  if (!isPermutationOfIota(CTAOrder)) {
    return emitError()
           << "CTAOrder must be a permutation of 0..(rank-1), but was ["
           << CTAOrder << "]";
  }

  return success();
}

LogicalResult
BlockedEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                            ArrayRef<unsigned> sizePerThread,
                            ArrayRef<unsigned> threadsPerWarp,
                            ArrayRef<unsigned> warpsPerCTA,
                            ArrayRef<unsigned> order, CTALayoutAttr CTALayout) {
  if (sizePerThread.size() != threadsPerWarp.size() ||
      threadsPerWarp.size() != warpsPerCTA.size() ||
      warpsPerCTA.size() != order.size()) {
    return emitError() << "sizePerThread, threadsPerWarp, warpsPerCTA, and "
                          "order must all have the same rank.";
  }

  // Empty CTALayout is allowed, but if it's present its rank must match the
  // BlockedEncodingAttr's rank.
  if (CTALayout.getCTASplitNum().size() != 0 &&
      sizePerThread.size() != CTALayout.getCTASplitNum().size()) {
    return emitError() << "BlockedEncodingAttr and CTALayout's fields must "
                          "have the same rank.";
  }
  if (!isPermutationOfIota(order)) {
    return emitError()
           << "order must be a permutation of 0..(rank-1), but was [" << order
           << "]";
  }
  return success();
}

// 1 element per thread
// order = reverse(arange(rank))
triton::gpu::BlockedEncodingAttr
getDefaultBlockedEncoding(MLIRContext *context, ArrayRef<int64_t> shape,
                          int numWarps, int threadsPerWarp, int numCTAs) {
  int rank = shape.size();
  llvm::SmallVector<unsigned> order(rank);
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  llvm::SmallVector<unsigned> sizePerThread(rank, 1);
  triton::gpu::BlockedEncodingAttr encoding =
      triton::gpu::BlockedEncodingAttr::get(context, shape, sizePerThread,
                                            order, numWarps, threadsPerWarp,
                                            numCTAs);
  return encoding;
}

} // namespace gpu
} // namespace triton
} // namespace mlir

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = attr.dyn_cast<IntegerAttr>();
  if (!intAttr) {
    parser.emitError(parser.getNameLoc(), "expected an integer type in ")
        << desc;
    return failure();
  }
  if (intAttr.getType().isSignedInteger()) {
    int64_t attrVal = intAttr.getSInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else if (intAttr.getType().isSignlessInteger()) {
    int64_t attrVal = intAttr.getInt();
    if (attrVal < 0) {
      parser.emitError(parser.getNameLoc(),
                       "expected an unsigned integer value in ")
          << desc;
      return failure();
    }
    value = attrVal;
  } else {
    value = intAttr.getUInt();
  }
  return success();
}

static LogicalResult parseBoolAttrValue(AsmParser &parser, Attribute attr,
                                        bool &value, StringRef desc) {
  auto boolAttr = attr.dyn_cast<BoolAttr>();
  if (!boolAttr) {
    parser.emitError(parser.getNameLoc(), "expected an bool type in ") << desc;
    return failure();
  }
  value = boolAttr.getValue();
  return success();
}

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
  if (!arrayAttr) {
    parser.emitError(parser.getNameLoc(), "expected an array for ") << desc;
    return failure();
  }
  for (Attribute i : arrayAttr) {
    unsigned value;
    if (parseIntAttrValue(parser, i, value, desc).failed())
      return failure();
    res.push_back(value);
  }
  return success();
};

static LogicalResult parseUInt(AsmParser &parser, const NamedAttribute &attr,
                               unsigned &value, StringRef desc) {
  return parseIntAttrValue(parser, attr.getValue(), value, desc);
};

static LogicalResult parseBool(AsmParser &parser, const NamedAttribute &attr,
                               bool &value, StringRef desc) {
  return parseBoolAttrValue(parser, attr.getValue(), value, desc);
};

// Print the CTALayout if it's not equal to the default.
static void maybePrintCTALayout(mlir::MLIRContext *context,
                                mlir::AsmPrinter &printer, CTALayoutAttr layout,
                                unsigned rank) {
  if (layout != CTALayoutAttr::getDefault(context, rank)) {
    printer << ", CTAsPerCGA = [" << ArrayRef(layout.getCTAsPerCGA()) << "]"
            << ", CTASplitNum = [" << ArrayRef(layout.getCTASplitNum()) << "]"
            << ", CTAOrder = [" << ArrayRef(layout.getCTAOrder()) << "]";
  }
}

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrInterfaces.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"

SliceEncodingAttr BlockedEncodingAttr::squeeze(int axis) {
  return SliceEncodingAttr::get(getContext(), axis, *this);
}
SmallVector<unsigned>
BlockedEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                       Type eltTy) const {
  size_t rank = shape.size();
  auto sizePerThread = getSizePerThread();
  auto warpsPerCTA = getWarpsPerCTA();
  auto threadsPerWarp = getThreadsPerWarp();
  auto shapePerCTA = getShapePerCTA(*this, shape);
  assert(rank == sizePerThread.size() &&
         "unexpected rank in BlockedEncodingAttr::getElemsPerThread");
  SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = sizePerThread[i] * threadsPerWarp[i] * warpsPerCTA[i];
    elemsPerThread[i] = ceil<unsigned>(shapePerCTA[i], t) * sizePerThread[i];
  }
  return elemsPerThread;
}
unsigned BlockedEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                     Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

// If we only had BlockedEncodingAttr, we could simply return ArrayRefs here.
// But we need to have a consistent interface with e.g. SliceEncodingAttr, which
// computes some of these fields.
SmallVector<unsigned> BlockedEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> BlockedEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> BlockedEncodingAttr::getWarpOrder() const {
  return SmallVector<unsigned>(getOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getThreadsPerWarp() const {
  return SmallVector<unsigned>(getThreadsPerWarp__());
}
SmallVector<unsigned> BlockedEncodingAttr::getThreadOrder() const {
  return SmallVector<unsigned>(getOrder());
}
SmallVector<unsigned> BlockedEncodingAttr::getSizePerThread() const {
  return SmallVector<unsigned>(getSizePerThread__());
}
SmallVector<unsigned>
BlockedEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  SmallVector<unsigned> shape;
  for (unsigned d = 0, n = getOrder().size(); d < n; ++d)
    shape.push_back(getSizePerThread()[d] * getThreadsPerWarp()[d] *
                    getWarpsPerCTA()[d]);
  return shape;
}

template <class T>
SmallVector<T> SliceEncodingAttr::paddedShape(ArrayRef<T> shape) const {
  size_t rank = shape.size();
  unsigned dim = getDim();
  SmallVector<T> retShape(rank + 1);
  for (unsigned d = 0; d < rank + 1; ++d) {
    if (d < dim)
      retShape[d] = shape[d];
    else if (d == dim)
      retShape[d] = 1;
    else
      retShape[d] = shape[d - 1];
  }
  return retShape;
}
template SmallVector<unsigned>
SliceEncodingAttr::paddedShape<unsigned>(ArrayRef<unsigned> shape) const;
template SmallVector<int64_t>
SliceEncodingAttr::paddedShape<int64_t>(ArrayRef<int64_t> shape) const;

SmallVector<unsigned>
SliceEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                     Type eltTy) const {
  auto parent = getParent();
  auto parentElemsPerThread =
      ::getElemsPerThread(parent, paddedShape(shape), eltTy);
  parentElemsPerThread.erase(parentElemsPerThread.begin() + getDim());
  return parentElemsPerThread;
}
unsigned SliceEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                   Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}
SmallVector<unsigned> SliceEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res = ::getCTASplitNum(getParent());
  res.erase(res.begin() + getDim());
  return res;
}
SmallVector<unsigned> SliceEncodingAttr::getCTAOrder() const {
  auto parentCTAOrder = ::getCTAOrder(getParent());
  return eraseOrder(parentCTAOrder, getDim());
}
SmallVector<unsigned> SliceEncodingAttr::getCTAsPerCGA() const {
  auto parentCTAsPerCGA = ::getCTAsPerCGA(getParent());
  if (parentCTAsPerCGA[getDim()] == 1) {
    parentCTAsPerCGA.erase(parentCTAsPerCGA.begin() + getDim());
    return parentCTAsPerCGA;
  }
  /* For getCTAsPerCGA of a slice layout, we have two choices:
   * (1) Return CTAsPerCGA of its parent. This is not a perfect solution
   * because the rank of the returned CTAsPerCGA does not match the rank of
   * tensorShape.
   * (2) Get CTAsPerCGA of its parent and erase the sliced dim. This is not a
   * perfect solution because the product of the returned CTAsPerCGA might not
   * match numCTAs.
   * To avoid introducing inconsistencies to the shape and
   * layout system, the usage of directly getting CTAsPerCGA of a slice layout
   * in which the sliced dim is not 1 is banned. You should always consider
   * slice layout as a special case and use getCTAsPerCGA(layout.getParent())
   * in the branch where layout is an instance of SliceEncodingAttr. This is
   * inconvenient but safe.
   */
  llvm::report_fatal_error(
      "getCTAsPerCGA for SliceEncodingAttr is not well-defined");
}
SmallVector<unsigned> SliceEncodingAttr::getWarpsPerCTA() const {
  auto parent = getParent();
  auto parentWarpsPerCTA = ::getWarpsPerCTA(parent);
  assert(parentWarpsPerCTA.size() == 2 ||
         parentWarpsPerCTA[getDim()] == 1 &&
             "getWarpsPerCTA only implemented for 2D slice layout or the "
             "slice dim must have 1 warp in the parent layout");
  SmallVector<unsigned> warpsPerCTA = parentWarpsPerCTA;
  warpsPerCTA.erase(warpsPerCTA.begin() + getDim());
  for (unsigned i = 0; i < warpsPerCTA.size(); i++)
    warpsPerCTA[i] *= parentWarpsPerCTA[getDim()];
  return warpsPerCTA;
}
SmallVector<unsigned> SliceEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> SliceEncodingAttr::getThreadsPerWarp() const {
  auto parent = getParent();
  auto parentThreadsPerWarp = ::getThreadsPerWarp(parent);
  assert(parentThreadsPerWarp.size() == 2 &&
         "getThreadsPerWarp only implemented for 2D slice layout");
  SmallVector<unsigned> threadsPerWarp = parentThreadsPerWarp;
  threadsPerWarp.erase(threadsPerWarp.begin() + getDim());
  for (unsigned i = 0; i < threadsPerWarp.size(); i++)
    threadsPerWarp[i] *= parentThreadsPerWarp[getDim()];
  return threadsPerWarp;
}
SmallVector<unsigned> SliceEncodingAttr::getThreadOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> SliceEncodingAttr::getSizePerThread() const {
  auto sizePerThread = ::getSizePerThread(getParent());
  sizePerThread.erase(sizePerThread.begin() + getDim());
  return sizePerThread;
}
SmallVector<unsigned>
SliceEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  SmallVector<unsigned> shape = ::getShapePerCTATile(getParent(), tensorShape);
  shape.erase(shape.begin() + getDim());
  return shape;
}

//

SmallVector<unsigned>
AMDMfmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                       Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of mfma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto nonKDim = getMDim();
  auto elemsPerThreadPerTile = (nonKDim == 16 ? 4 : 16);
  if (getIsTransposed()) {
    unsigned elemsCol =
        ceil<unsigned>(shape[1], nonKDim * getWarpsPerCTA()[1]) *
        elemsPerThreadPerTile;
    unsigned elemsRow = ceil<unsigned>(shape[0], nonKDim * getWarpsPerCTA()[0]);
    elemsPerThread[0] = elemsRow;
    elemsPerThread[1] = elemsCol;
  } else {
    unsigned elemsCol = ceil<unsigned>(shape[1], nonKDim * getWarpsPerCTA()[1]);
    unsigned elemsRow =
        ceil<unsigned>(shape[0], nonKDim * getWarpsPerCTA()[0]) *
        elemsPerThreadPerTile;
    elemsPerThread[0] = elemsRow;
    elemsPerThread[1] = elemsCol;
  }
  return elemsPerThread;
}

unsigned AMDMfmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                     Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

//

SmallVector<unsigned>
AMDWmmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                       Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of wmma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto mnkDim = getMNKDimPerWMMAInstr();
  auto elemsPerThreadPerTile = getSizePerThread();
  auto warpsPerCTA = getWarpsPerCTA();
  return {ceil<unsigned>(shape[0], mnkDim[0] * warpsPerCTA[0]) *
              elemsPerThreadPerTile[0],
          ceil<unsigned>(shape[1], mnkDim[1] * warpsPerCTA[1]) *
              elemsPerThreadPerTile[1]};
}

unsigned AMDWmmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                     Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

//

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                         Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 ||
         (rank == 3 && isAmpere()) && "Unexpected rank of mma layout");
  assert((isVolta() || isAmpere() || isHopper()) &&
         "For NvidiaMmaEncodingAttr only version 1~3 is supported");

  auto shapePerCTA = getShapePerCTA(getCTALayout().getCTASplitNum(), shape);

  SmallVector<unsigned> elemsPerThread(rank);
  if (isVolta()) {
    auto [isARow, isBRow, isAVec4, isBVec4, id] = decodeVoltaLayoutStates();
    static constexpr std::array<unsigned, 2> fpw{{2, 2}};
    unsigned packSize0 = (isARow || isAVec4) ? 1 : 2;
    unsigned packSize1 = (isBRow && !isBVec4) ? 2 : 1;
    unsigned repM = 2 * packSize0;
    unsigned repN = 2 * packSize1;
    unsigned spwM = fpw[0] * 4 * repM;
    unsigned spwN = fpw[1] * 4 * repN;
    unsigned wptM = getWarpsPerCTA()[0];
    unsigned wptN = getWarpsPerCTA()[1];
    unsigned resM = repM * std::max<int>(1, shapePerCTA[0] / (spwM * wptM));
    unsigned resN = 2 * repN * std::max<int>(1, shapePerCTA[1] / (spwN * wptN));
    elemsPerThread[0] = resM;
    elemsPerThread[1] = resN;
  } else if (isAmpere()) {
    unsigned elemsRow =
        ceil<unsigned>(shapePerCTA[rank - 2], 16 * getWarpsPerCTA()[rank - 2]) *
        2;
    unsigned elemsCol =
        ceil<unsigned>(shapePerCTA[rank - 1], 8 * getWarpsPerCTA()[rank - 1]) *
        2;
    if (rank == 3)
      elemsPerThread[0] = ceil<unsigned>(shapePerCTA[0], getWarpsPerCTA()[0]);
    elemsPerThread[rank - 2] = elemsRow;
    elemsPerThread[rank - 1] = elemsCol;
  } else if (isHopper()) {
    auto wpt = getWarpsPerCTA();
    auto instrMNK = getInstrShape();
    int repM = ceil<unsigned>(shapePerCTA[0], instrMNK[0] * wpt[0]);
    int repN = ceil<unsigned>(shapePerCTA[1], instrMNK[1] * wpt[1]);
    elemsPerThread[0] = 2 * repM;
    elemsPerThread[1] = (instrMNK[1] / 4) * repN;
  } else {
    llvm_unreachable("Unexpected mma version");
  }

  return elemsPerThread;
}

unsigned NvidiaMmaEncodingAttr::getElemsPerThreadOfOperand(
    int opIdx, ArrayRef<int64_t> shape) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of mma layout");
  auto shapePerCTA = getShapePerCTA(*this, shape);
  int res = 0;
  if (isVolta()) {
    llvm_unreachable(
        "getElemsPerThreadOfOperand() not supported for version 1");
  } else if (isAmpere()) {
    llvm_unreachable(
        "getElemsPerThreadOfOperand() not supported for version 2");
  } else if (isHopper()) {
    auto wpt = getWarpsPerCTA();
    auto instrMNK = getInstrShape();
    if (opIdx == 0) {
      int repM = ceil<unsigned>(shapePerCTA[0], instrMNK[0] * wpt[0]);
      int repK = ceil<unsigned>(shapePerCTA[1], instrMNK[2]);
      return 8 * repM * repK;

    } else if (opIdx == 1) {
      int repK = ceil<unsigned>(shapePerCTA[0], instrMNK[2]);
      int repN = ceil<unsigned>(shapePerCTA[1], instrMNK[1] * wpt[1]);
      // benzh@ here need more check
      return 4 * std::max<int>(instrMNK[1] / 32, 1) * repK * repN;
    }
  }
  return res;
}

unsigned NvidiaMmaEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                       Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

//

SmallVector<unsigned>
SharedEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                      Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for shared layout");
  return SmallVector<unsigned>();
}
unsigned SharedEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                    Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for shared layout");
  return 0;
}

SmallVector<unsigned>
DotOperandEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape,
                                          Type eltTy) const {
  llvm_unreachable("getElemsPerThread is not supported for dot operand");
  return SmallVector<unsigned>();
}

unsigned DotOperandEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                        Type eltTy) const {
  if (auto mmaParent = getParent().dyn_cast<MmaEncodingTrait>()) {
    return mmaParent.getTotalElemsPerThreadForOperands(shape, eltTy,
                                                       getKWidth(), getOpIdx());
  }
  if (auto blockedLayout = getParent().dyn_cast<BlockedEncodingAttr>()) {
    auto shapePerCTA = getShapePerCTA(*this, shape);
    auto shapePerCTATile = ::getShapePerCTATile(blockedLayout);
    auto order = blockedLayout.getOrder();
    auto sizePerThread = ::getSizePerThread(blockedLayout);

    int K = getOpIdx() == 0 ? shapePerCTA[1] : shapePerCTA[0];
    int otherDim = getOpIdx() == 1 ? shapePerCTA[1] : shapePerCTA[0];

    bool isM = getOpIdx() == 0;

    int mSizePerThread =
        order[0] == 1 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int nSizePerThread =
        order[0] == 0 ? sizePerThread[order[1]] : sizePerThread[order[0]];
    int sizePerThreadMN = isM ? mSizePerThread : nSizePerThread;

    int mShapePerCTATile =
        order[0] == 1 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
    int nShapePerCTATile =
        order[0] == 0 ? shapePerCTATile[order[1]] : shapePerCTATile[order[0]];
    int shapePerCTAMNTile = isM ? mShapePerCTATile : nShapePerCTATile;

    return K * std::max<int>(otherDim / shapePerCTAMNTile, 1) * sizePerThreadMN;
  }
  llvm_unreachable("unknown dot operand parent layout");
  return 0;
}
SmallVector<unsigned> DotOperandEncodingAttr::getCTAsPerCGA() const {
  return ::getCTAsPerCGA(getParent());
}
SmallVector<unsigned> DotOperandEncodingAttr::getCTAOrder() const {
  return ::getCTAOrder(getParent());
}
SmallVector<unsigned> DotOperandEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res = ::getCTASplitNum(getParent());
  auto rank = res.size();
  assert(rank == 2 || rank == 3 && "Invalid dotLayout");

  // Do not split CTA in K dimension
  getOpIdx() == 0 ? res[rank - 1] = 1 : res[rank - 2] = 1;
  return res;
}
SmallVector<unsigned> DotOperandEncodingAttr::getWarpsPerCTA() const {
  return getParent().cast<DistributedEncodingTrait>().getWarpsPerCTA();
}
SmallVector<unsigned> DotOperandEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> DotOperandEncodingAttr::getThreadOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> DotOperandEncodingAttr::getShapePerCTATile(
    ArrayRef<int64_t> tensorShape) const {
  auto parentLayout = getParent();
  assert(parentLayout && "DotOperandEncodingAttr must have a parent");
  if (auto parentMmaLayout = parentLayout.dyn_cast<MmaEncodingTrait>()) {
    return parentMmaLayout.getShapePerCTATileForDotOperands(tensorShape,
                                                            getOpIdx());
  } else {
    llvm::report_fatal_error(
        "DotOperandEncodingAttr non-NvidiaMmaEncodingAttr parent not "
        "supported yet");
  }
}

//===----------------------------------------------------------------------===//
// Blocked Encoding
//===----------------------------------------------------------------------===//

static std::optional<CTALayoutAttr> getCTALayoutOrError(
    AsmParser &parser, std::optional<SmallVector<unsigned>> CTAsPerCGA,
    std::optional<SmallVector<unsigned>> CTASplitNum,
    std::optional<SmallVector<unsigned>> CTAOrder, unsigned rank) {
  if (CTAsPerCGA && CTASplitNum && CTAOrder) {
    return CTALayoutAttr::get(parser.getContext(), *CTAsPerCGA, *CTASplitNum,
                              *CTAOrder);
  }
  if (!CTAsPerCGA && !CTASplitNum && !CTAOrder) {
    return CTALayoutAttr::getDefault(parser.getContext(), rank);
  }
  parser.emitError(parser.getNameLoc(), "CTAsPerCGA, CTASplitNum, and CTAOrder "
                                        "must all be present or all be absent");
  return std::nullopt;
}

Attribute BlockedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> sizePerThread;
  SmallVector<unsigned> threadsPerWarp;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> order;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "sizePerThread") {
      if (parseIntArrayAttr(parser, attr, sizePerThread,
                            "number of elements per thread")
              .failed())
        return {};
    } else if (attr.getName() == "threadsPerWarp") {
      if (parseIntArrayAttr(parser, attr, threadsPerWarp,
                            "number of threads per warp")
              .failed())
        return {};
    } else if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA,
                            "number of warps per CTA")
              .failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/sizePerThread.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<BlockedEncodingAttr>(parser.getContext(),
                                                sizePerThread, threadsPerWarp,
                                                warpsPerCTA, order, *CTALayout);
}

void BlockedEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "sizePerThread = [" << ArrayRef(getSizePerThread()) << "]"
          << ", threadsPerWarp = [" << ArrayRef(getThreadsPerWarp()) << "]"
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]"
          << ", order = [" << getOrder() << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getSizePerThread().size());

  printer << "}>";
}

//===----------------------------------------------------------------------===//
// MMA encoding
//===----------------------------------------------------------------------===//

Attribute NvidiaMmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  SmallVector<unsigned> instrShape;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "versionMajor") {
      if (parseUInt(parser, attr, versionMajor, "versionMajor").failed())
        return {};
    }
    if (attr.getName() == "versionMinor") {
      if (parseUInt(parser, attr, versionMinor, "versionMinor").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed()) {
        return {};
      }
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<NvidiaMmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA, *CTALayout,
      instrShape);
}

void NvidiaMmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()
          << ", versionMinor = " << getVersionMinor() //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]";

  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());

  printer << ", instrShape = [" << getInstrShape() << "]}>";
}

//===----------------------------------------------------------------------===//
// MFMA encoding
//===----------------------------------------------------------------------===//

Attribute AMDMfmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned versionMajor = 0;
  unsigned versionMinor = 0;
  SmallVector<unsigned> warpsPerCTA;
  SmallVector<unsigned> instrShape;
  bool isTransposed;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "versionMajor") {
      if (parseUInt(parser, attr, versionMajor, "versionMajor").failed())
        return {};
    }
    if (attr.getName() == "versionMinor") {
      if (parseUInt(parser, attr, versionMinor, "versionMinor").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "instrShape") {
      if (parseIntArrayAttr(parser, attr, instrShape, "instrShape").failed())
        return {};
    }
    if (attr.getName() == "isTransposed") {
      if (parseBool(parser, attr, isTransposed, "isTransposed").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDMfmaEncodingAttr>(
      parser.getContext(), versionMajor, versionMinor, warpsPerCTA,
      instrShape[0], instrShape[1], isTransposed, *CTALayout);
}

void AMDMfmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "versionMajor = " << getVersionMajor()                      //
          << ", versionMinor = " << getVersionMinor()                    //
          << ", warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]"    //
          << ", instrShape = [" << ArrayRef{getMDim(), getNDim()} << "]" //
          << ", isTransposed = " << getIsTransposed();
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// WMMA encoding
//===----------------------------------------------------------------------===//

Attribute AMDWmmaEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> warpsPerCTA;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    }
    if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    }
    if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/warpsPerCTA.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<AMDWmmaEncodingAttr>(parser.getContext(),
                                                warpsPerCTA, *CTALayout);
}

void AMDWmmaEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "warpsPerCTA = [" << ArrayRef(getWarpsPerCTA()) << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getWarpsPerCTA().size());
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// Sliced Encoding
//===----------------------------------------------------------------------===//

Attribute SliceEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned dim = attrs.get("dim").cast<IntegerAttr>().getInt();
  Attribute parent = attrs.get("parent");
  return parser.getChecked<SliceEncodingAttr>(parser.getContext(), dim, parent);
}

void SliceEncodingAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{"
          << "dim = " << getDim() << ", "
          << "parent = " << getParent() << "}>";
}

//===----------------------------------------------------------------------===//
// Shared encoding
//===----------------------------------------------------------------------===//

Attribute SharedEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  // Parse the data as a dictionary
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  unsigned vec = 0;
  unsigned perPhase = 0;
  unsigned maxPhase = 0;
  SmallVector<unsigned> order;
  std::optional<SmallVector<unsigned>> CTAsPerCGA;
  std::optional<SmallVector<unsigned>> CTASplitNum;
  std::optional<SmallVector<unsigned>> CTAOrder;
  bool hasLeadingOffset = false;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "vec") {
      if (parseUInt(parser, attr, vec, "vec").failed())
        return {};
    } else if (attr.getName() == "perPhase") {
      if (parseUInt(parser, attr, perPhase, "perPhase").failed())
        return {};
    } else if (attr.getName() == "maxPhase") {
      if (parseUInt(parser, attr, maxPhase, "maxPhase").failed())
        return {};
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else if (attr.getName() == "CTAsPerCGA") {
      if (parseIntArrayAttr(parser, attr, CTAsPerCGA.emplace(), "CTAsPerCGA")
              .failed())
        return {};
    } else if (attr.getName() == "CTASplitNum") {
      if (parseIntArrayAttr(parser, attr, CTASplitNum.emplace(), "CTASplitNum")
              .failed())
        return {};
    } else if (attr.getName() == "CTAOrder") {
      if (parseIntArrayAttr(parser, attr, CTAOrder.emplace(), "CTAOrder")
              .failed())
        return {};
    } else if (attr.getName() == "hasLeadingOffset") {
      if (parseBool(parser, attr, hasLeadingOffset, "hasLeadingOffset")
              .failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }

  std::optional<CTALayoutAttr> CTALayout = getCTALayoutOrError(
      parser, CTAsPerCGA, CTASplitNum, CTAOrder, /*rank=*/order.size());
  if (!CTALayout.has_value())
    return {};

  return parser.getChecked<SharedEncodingAttr>(parser.getContext(), vec,
                                               perPhase, maxPhase, order,
                                               *CTALayout, hasLeadingOffset);
}

void SharedEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{"
          << "vec = " << getVec() //
          << ", perPhase = " << getPerPhase()
          << ", maxPhase = " << getMaxPhase() //
          << ", order = [" << getOrder() << "]";
  maybePrintCTALayout(getContext(), printer, getCTALayout(),
                      /*rank=*/getOrder().size());
  printer << ", hasLeadingOffset = " << getHasLeadingOffset() << "}>";
}

//===----------------------------------------------------------------------===//
// Mfma encoding
//===----------------------------------------------------------------------===//
// TODO: there is a lot of common code with MmaEncoding here

SmallVector<unsigned>
AMDMfmaEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  auto nonKDim = getMDim();
  return {nonKDim * getWarpsPerCTA()[0], nonKDim * getWarpsPerCTA()[1]};
}

SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getThreadOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDMfmaEncodingAttr::getThreadsPerWarp() const {
  unsigned rows, cols;
  if (getMDim() == 32) {
    cols = 2;
    rows = 32;
  } else {
    cols = 4;
    rows = 16;
  }
  if (getIsTransposed()) {
    return {rows, cols};
  } else {
    return {cols, rows};
  }
}

SmallVector<unsigned> AMDMfmaEncodingAttr::getSizePerThread() const {
  unsigned rows, cols;
  if (getMDim() == 32) {
    rows = 16;
    cols = 1;
  } else if (getMDim() == 16) {
    rows = 4;
    cols = 1;
  } else
    llvm_unreachable("Unexpected mfma non-k dim");

  if (getIsTransposed()) {
    return {cols, rows};
  } else {
    return {rows, cols};
  }
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getMFMAInstrShapeForOperands(int kWidth, int opIdx) const {
  unsigned mDim = getMDim();
  unsigned nDim = getNDim();
  assert((mDim == nDim) && (mDim == 32 || mDim == 16 || mDim == 4) ||
         (mDim == 64 && nDim == 4) || (mDim == 4 && nDim == 64));
  constexpr int waveSize = 64; // MFMA is used on wave64 architectures only
  int kGroups = -1;
  if (mDim == nDim)
    kGroups = waveSize / mDim;
  if (mDim == 64 && nDim == 4 || mDim == 4 && nDim == 64)
    kGroups = 1;
  int64_t kDim = kWidth * kGroups;
  if (opIdx == 0)
    return {mDim, kDim};
  else
    assert(opIdx == 1);
  return {kDim, nDim};
}

SmallVector<int64_t>
AMDMfmaEncodingAttr::getMFMARepForOperands(ArrayRef<int64_t> operandShape,
                                           int kWidth, int opIdx) const {
  auto operandTileShape = getMFMAInstrShapeForOperands(kWidth, opIdx);
  auto warpsPerCTA = getWarpsPerCTA();
  if (opIdx == 0)
    return {std::max<int64_t>(1, operandShape[0] /
                                     (operandTileShape[0] * warpsPerCTA[0])),
            std::max<int64_t>(1, operandShape[1] / operandTileShape[1])};
  else {
    assert(opIdx == 1);
    return {std::max<int64_t>(1, operandShape[0] / operandTileShape[0]),
            std::max<int64_t>(1, operandShape[1] /
                                     (operandTileShape[1] * warpsPerCTA[1]))};
  }
}

unsigned AMDMfmaEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const {
  int warpsPerCTAM = getWarpsPerCTA()[0];
  int warpsPerCTAN = getWarpsPerCTA()[1];
  constexpr int waveSize = 64;
  auto tileSize = getMFMAInstrShapeForOperands(kWidth, opIdx);
  auto rep = getMFMARepForOperands(shape, kWidth, opIdx);
  return rep[0] * rep[1] * kWidth;
}

SmallVector<unsigned>
AMDMfmaEncodingAttr::getSizePerThreadForOperands(unsigned opIdx) const {
  if (opIdx == 0) {
    return {4, 1};
  } else if (opIdx == 1) {
    return {1, 4};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
    return {};
  }
}

SmallVector<unsigned>
AMDMfmaEncodingAttr::getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape,
                                                      int opIdx) const {
  auto parentShapePerCTA = getShapePerCTATile(shape);
  if (opIdx == 0) {
    return {parentShapePerCTA[0], 32};
  } else if (opIdx == 1) {
    return {32, parentShapePerCTA[1]};
  }
  llvm_unreachable("DotOperandEncodingAttr opIdx must be 0 or 1");
}

SmallVector<unsigned>
AMDWmmaEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  auto mnkDim = getMNKDimPerWMMAInstr();
  return {mnkDim[0] * getWarpsPerCTA()[0], mnkDim[1] * getWarpsPerCTA()[1]};
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getThreadOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> AMDWmmaEncodingAttr::getThreadsPerWarp() const {
  return {getMNKDimPerWMMAInstr()[0] / getSizePerThread()[0],
          getMNKDimPerWMMAInstr()[1] / getSizePerThread()[1]};
}

SmallVector<unsigned> AMDWmmaEncodingAttr::getSizePerThread() const {
  return {8, 1};
}
SmallVector<unsigned>
AMDWmmaEncodingAttr::getSizePerThreadForOperands(unsigned opIdx) const {
  if (opIdx == 0) {
    return {1, 16};
  } else if (opIdx == 1) {
    return {16, 1};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
}

SmallVector<unsigned>
AMDWmmaEncodingAttr::getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape,
                                                      int opIdx) const {
  auto parentShapePerCTA = getShapePerCTATile(shape);
  if (opIdx == 0) {
    return {parentShapePerCTA[0], static_cast<unsigned>(shape[1])};
  } else if (opIdx == 1) {
    return {static_cast<unsigned>(shape[0]), parentShapePerCTA[1]};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
}

unsigned AMDWmmaEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const {
  auto rep = getWMMARepForOperands(shape, eltTy, kWidth, opIdx);
  return rep[0] * rep[1] * kWidth;
}

SmallVector<int64_t>
AMDWmmaEncodingAttr::getWMMAElemsPerInstrForOperands() const {
  return {16, 16};
}

SmallVector<int64_t>
AMDWmmaEncodingAttr::getWMMARepForOperands(ArrayRef<int64_t> operandShape,
                                           Type elemType, int kWidth,
                                           int opIdx) const {
  auto operandTileShape = getWMMAElemsPerInstrForOperands();
  auto warpsPerCTA = getWarpsPerCTA();
  if (opIdx == 0)
    return {std::max<int64_t>(1, operandShape[0] /
                                     (operandTileShape[0] * warpsPerCTA[0])),
            std::max<int64_t>(1, operandShape[1] / operandTileShape[1])};
  else {
    assert(opIdx == 1);
    return {std::max<int64_t>(1, operandShape[0] / operandTileShape[0]),
            std::max<int64_t>(1, operandShape[1] /
                                     (operandTileShape[1] * warpsPerCTA[1]))};
  }
}

SmallVector<unsigned> AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr() {
  // TODO: move magic numbers out of the code
  return {16, 16, 16};
}

//===----------------------------------------------------------------------===//
// Mma encoding
//===----------------------------------------------------------------------===//

bool NvidiaMmaEncodingAttr::isVolta() const { return getVersionMajor() == 1; }

bool NvidiaMmaEncodingAttr::isTuring() const {
  return getVersionMajor() == 2 && getVersionMinor() == 1;
}

bool NvidiaMmaEncodingAttr::isAmpere() const { return getVersionMajor() == 2; }

bool NvidiaMmaEncodingAttr::isHopper() const { return getVersionMajor() == 3; }

SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAsPerCGA() const {
  return SmallVector<unsigned>(getCTALayout().getCTAsPerCGA());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTAOrder() const {
  return SmallVector<unsigned>(getCTALayout().getCTAOrder());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getCTASplitNum() const {
  return SmallVector<unsigned>(getCTALayout().getCTASplitNum());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__());
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getWarpOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getThreadsPerWarp() const {
  auto rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  if (isVolta()) {
    res[rank - 2] = 4;
    res[rank - 1] = 8;
    return res;
  }
  if (isAmpere()) {
    res[rank - 2] = 8;
    res[rank - 1] = 4;
    return res;
  }
  if (isHopper()) {
    res[rank - 2] = 8;
    res[rank - 1] = 4;
    return res;
  }
  llvm::report_fatal_error(
      "getThreadsPerWarp not implemented for unknown Mma version ");
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getThreadOrder() const {
  return ::getOrder(*this);
}
SmallVector<unsigned> NvidiaMmaEncodingAttr::getSizePerThread() const {
  auto rank = ::getOrder(*this).size();
  SmallVector<unsigned> res(rank, 1);
  if (isAmpere()) {
    res[rank - 2] = 2;
    res[rank - 1] = 2;
    return res;
  }
  if (isVolta()) {
    res[rank - 2] = 1;
    res[rank - 1] = 2;
    return res;
  }
  if (isHopper()) {
    auto instrShape = getInstrShape();
    // TODO(thomas): what are those magic numbers?
    return SmallVector<unsigned>{instrShape[0] * 4 / 32, instrShape[1] / 4};
  }
  llvm_unreachable("Unexpected mma version");
}

SmallVector<unsigned>
NvidiaMmaEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  if (isAmpere()) {
    auto warpsPerCTA = getWarpsPerCTA();
    auto rank = warpsPerCTA.size();
    SmallVector<unsigned> shapePerCTATile(warpsPerCTA.begin(),
                                          warpsPerCTA.end());
    shapePerCTATile[rank - 1] *= 8;
    shapePerCTATile[rank - 2] *= 16;
    return shapePerCTATile;
  }
  if (isVolta()) {
    assert(!tensorShape.empty() && "Volta needs the tensorShape");
    if (tensorShape.size() == 1) // must be SliceEncoding
      return {static_cast<unsigned>(tensorShape[0]),
              static_cast<unsigned>(tensorShape[0])};
    return {static_cast<unsigned>(tensorShape[0]),
            static_cast<unsigned>(tensorShape[1])};
  }
  if (isHopper()) {
    auto instrShape = getInstrShape();
    return {16 * getWarpsPerCTA()[0], instrShape[1] * getWarpsPerCTA()[1]};
  }
  llvm::report_fatal_error("Unexpected MMA layout version found");
}

// Get [isARow, isBRow, isAVec4, isBVec4, id] from versionMinor
std::tuple<bool, bool, bool, bool, int>
NvidiaMmaEncodingAttr::decodeVoltaLayoutStates() const {
  unsigned versionMinor = getVersionMinor();
  bool isARow = versionMinor & (1 << 0);
  bool isBRow = versionMinor & (1 << 1);
  bool isAVec4 = versionMinor & (1 << 2);
  bool isBVec4 = versionMinor & (1 << 3);

  int id = 0;
  for (int i = numBitsToHoldMmaV1ID - 1; i >= 0; --i)
    id = (id << 1) + static_cast<bool>(versionMinor & (1 << (4 + i)));

  return std::make_tuple(isARow, isBRow, isAVec4, isBVec4, id);
}

bool NvidiaMmaEncodingAttr::getMMAv1IsRow(int opIdx) const {
  auto [isARow, isBRow, _0, _1, _2] = decodeVoltaLayoutStates();
  return opIdx == 0 ? isARow : isBRow;
}
bool NvidiaMmaEncodingAttr::getMMAv1IsVec4(int opIdx) const {
  auto [_0, _1, isAVec4, isBVec4, _2] = decodeVoltaLayoutStates();
  return opIdx == 0 ? isAVec4 : isBVec4;
}
int NvidiaMmaEncodingAttr::getMMAv1NumOuter(ArrayRef<int64_t> shape,
                                            int opIdx) const {
  auto spw = getMMAv1ShapePerWarp(opIdx);
  auto rep = getMMAv1Rep(opIdx);
  auto warpsPerCTA = getWarpsPerCTA();
  if (opIdx == 0) {
    return rep[0] * shape[0] / (spw[0] * warpsPerCTA[0]);
  } else {
    return rep[1] * shape[1] / (spw[1] * warpsPerCTA[1]);
  }
}
SmallVector<int> NvidiaMmaEncodingAttr::getMMAv1Rep(int opIdx) const {
  auto [isARow, isBRow, isAVec4, isBVec4, _] = decodeVoltaLayoutStates();
  // A
  if (opIdx == 0) {
    int packSize = (isARow || isAVec4) ? 1 : 2;
    return {2 * packSize, 0, 1};
  }
  // B
  else {
    int packSize = (isBRow && !isBVec4) ? 2 : 1;
    return {0, 2 * packSize, 1};
  }
}
SmallVector<int> NvidiaMmaEncodingAttr::getMMAv1ShapePerWarp(int opIdx) const {
  auto rep = getMMAv1Rep(opIdx);
  if (opIdx == 0) {
    return {8 * rep[0], 0, 1};
  } else {
    return {0, 8 * rep[1], 1};
  }
}
int NvidiaMmaEncodingAttr::getMMAv1Vec(int opIdx) const {
  return 2 * getMMAv1Rep(opIdx)[opIdx];
}
SmallVector<int64_t> NvidiaMmaEncodingAttr::getMMAv2Rep(ArrayRef<int64_t> shape,
                                                        int bitwidth,
                                                        int opIdx) const {
  auto rank = shape.size();
  auto warpsPerCTA = getWarpsPerCTA();
  SmallVector<int> shapePerWarp = {1, 16, 8, 4 * 64 / bitwidth};
  int numRepBatch =
      rank == 3
          ? std::max<int64_t>(1, shape[0] / (shapePerWarp[0] * warpsPerCTA[0]))
          : 1;
  assert(isAmpere());

  if (opIdx == 0)
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] /
                                     (shapePerWarp[1] * warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / shapePerWarp[3])};
  else {
    assert(opIdx == 1);
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / shapePerWarp[3]),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[2] *
                                                    warpsPerCTA[rank - 1]))};
  }
}
unsigned NvidiaMmaEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const {
  auto shapePerCTA = getShapePerCTA(*this, shape);
  int warpsPerCTAM = getWarpsPerCTA()[0];
  int warpsPerCTAN = getWarpsPerCTA()[1];
  // H100
  if (isHopper()) {
    return getTotalElemsPerThread(shape, eltTy);
  }
  // A100
  if (isAmpere()) {
    auto rep = getMMAv2Rep(shapePerCTA, eltTy.getIntOrFloatBitWidth(), opIdx);
    if (opIdx == 0)
      return 4 * rep[0] * rep[1] * rep[2];
    if (opIdx == 1)
      return 4 * rep[0] * rep[1] * std::max<int>(rep[2] / 2, 1);
  }
  // V100
  if (isVolta()) {
    bool isRow = getMMAv1IsRow(opIdx);
    bool isVec4 = getMMAv1IsVec4(opIdx);
    if (opIdx == 0) {
      int packSizeM = (isRow || isVec4) ? 1 : 2;
      int repM = 2 * packSizeM;
      int spwM = 2 * 4 * repM;
      int numM = getMMAv1NumOuter(shape, opIdx);
      int NK = shape[1];
      int vec = 2 * repM;
      // Here we mimic the logic in loadA, the result cannot be calculated
      // directly.
      llvm::DenseSet<std::pair<int, int>> visited;
      auto ld = [&](int m, int k) {
        visited.insert({m, k});
        if (vec > 4) {
          if (isRow)
            visited.insert({m, k + 4});
          else
            visited.insert({m + 1, k});
        }
      };
      for (unsigned k = 0; k < NK; k += 4)
        for (unsigned m = 0; m < numM / 2; ++m)
          if (!visited.count({m, k}))
            ld(m, k);
      return visited.size() * 2;
    }
    if (opIdx == 1) {
      int packSizeN = (isRow && !isVec4) ? 2 : 1;
      int repN = 2 * packSizeN;
      int spwN = 2 * 4 * repN;
      int numN = getMMAv1NumOuter(shape, opIdx);
      int vec = 2 * repN;

      int NK = shape[0];
      // Here we mimic the logic in loadA, the result cannot be calculated
      // directly.
      llvm::DenseSet<std::pair<int, int>> visited;
      int elemsPerLd = vec > 4 ? 4 : 2;
      auto ld = [&](int n, int k) {
        visited.insert({n, k});
        if (vec > 4) {
          if (isRow)
            visited.insert({n + 1, k});
          else
            visited.insert({n, k + 4});
        }
      };

      for (unsigned k = 0; k < NK; k += 4)
        for (unsigned n = 0; n < numN / 2; ++n) {
          if (!visited.count({n, k}))
            ld(n, k);
        }

      return visited.size() * 2;
    }
  }
  llvm_unreachable("unknown mma layout");
}
SmallVector<unsigned>
NvidiaMmaEncodingAttr::getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape,
                                                        int opIdx) const {
  assert(isAmpere() && "mmaLayout version = 1 is not implemented yet");
  auto parentShapePerCTATile = getShapePerCTATile(shape);
  auto rank = parentShapePerCTATile.size();
  if (opIdx == 0) {
    if (rank == 2)
      return {parentShapePerCTATile[rank - 2], 16};
    else
      return {parentShapePerCTATile[0], parentShapePerCTATile[rank - 2], 16};
  } else if (opIdx == 1) {
    if (rank == 2)
      return {16, parentShapePerCTATile[rank - 1]};
    else
      return {parentShapePerCTATile[0], 16, parentShapePerCTATile[rank - 1]};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
}
SmallVector<unsigned>
NvidiaMmaEncodingAttr::getSizePerThreadForOperands(unsigned opIdx) const {
  assert(isAmpere() && "mmaLayout version = 1 is not implemented yet");
  if (opIdx == 0) {
    return {2, 4};
  } else if (opIdx == 1) {
    return {4, 1};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
    return {};
  }
}

//===----------------------------------------------------------------------===//
// DotOperand Encoding
//===----------------------------------------------------------------------===//
SmallVector<unsigned> DotOperandEncodingAttr::getThreadsPerWarp() const {
  llvm::report_fatal_error(
      "getThreadsPerWarp not implemented for DotOperandEncodingAttr");
}
SmallVector<unsigned> DotOperandEncodingAttr::getSizePerThread() const {
  auto parentLayout = getParent();
  assert(parentLayout && "DotOperandEncodingAttr must have a parent");
  if (auto parentMmaLayout = parentLayout.dyn_cast<MmaEncodingTrait>()) {
    return parentMmaLayout.getSizePerThreadForOperands(getOpIdx());
  } else {
    llvm::report_fatal_error(
        "DotOperandEncodingAttr non-NvidiaMmaEncodingAttr parent not "
        "supported yet");
    return {};
  }
}

Attribute DotOperandEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDict(attrs).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};
  unsigned opIdx = attrs.get("opIdx").cast<IntegerAttr>().getInt();
  Attribute parent = attrs.get("parent");
  auto mmaParent = parent.dyn_cast<NvidiaMmaEncodingAttr>();
  unsigned kWidth = 0;
  Attribute _kWidth = attrs.get("kWidth");
  if (_kWidth) {
    if (!mmaParent || mmaParent.isVolta()) {
      auto loc = parser.getNameLoc();
      parser.emitError(loc, "kWidth only supported for MMAv2+ parent");
      return Attribute();
    }
    kWidth = _kWidth.cast<IntegerAttr>().getInt();
  }
  if (parent.isa<AMDWmmaEncodingAttr>()) {
    kWidth = AMDWmmaEncodingAttr::getMNKDimPerWMMAInstr()[2];
  }
  return parser.getChecked<DotOperandEncodingAttr>(parser.getContext(), opIdx,
                                                   parent, kWidth);
}

void DotOperandEncodingAttr::print(mlir::AsmPrinter &printer) const {
  auto mmaParent = getParent().dyn_cast<NvidiaMmaEncodingAttr>();
  printer << "<{"
          << "opIdx = " << getOpIdx() << ", parent = " << getParent();
  if (mmaParent && mmaParent.isAmpere())
    printer << ", kWidth = " << getKWidth();
  printer << "}>";
}

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//

class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto mmaAttr = attr.dyn_cast<MmaEncodingTrait>()) {
      os << "mma";
      return AliasResult::FinalAlias;
    } else if (auto sharedAttr = attr.dyn_cast<SharedEncodingAttr>()) {
      os << "shared";
      return AliasResult::FinalAlias;
    } else if (auto blockedAttr = attr.dyn_cast<BlockedEncodingAttr>()) {
      os << "blocked";
      return AliasResult::FinalAlias;
    } else if (auto warpAttr = attr.dyn_cast<WarpEncodingAttr>()) {
      os << "warp";
      return AliasResult::FinalAlias;
    } /* else if (auto sliceAttr = attr.dyn_cast<SliceEncodingAttr>()) {
      os << "slice";
      return AliasResult::FinalAlias;
    } */
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};

struct TritonGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding = SliceEncodingAttr::get(getDialect()->getContext(), axis,
                                            operandEncoding);
    return success();
  }

  // Infer the encoding of a tt.trans(x) given the encoding of x.
  //
  // Our goal is to choose an encoding so that the trans is a "nop".  For
  // example, in a blocked encoding, the same GPU threads hold the same
  // elements, they're just "renamed" -- what was element [i,j] of the tensor is
  // now element [j,i], but that element is held by the same GPU thread.
  //
  // For most properties of the encoding, we let
  //   outputEnc.prop = inputEnc.prop * trans.order,
  // where `x * y` means we apply permutation y to x.
  //
  // This works because prop[i] tells you something about the i'th dimension of
  // the tensor. (For example, sizePerThread[2] == 4 means that one GPU thread
  // contains 4 elements along dim 2 of the tensor.) The transpose reorders the
  // dimensions according to the perm trans.order, so we achieve our goal of
  // having a "nop" transpose by reordering the values in the prop the same way.
  //
  // The big exception to this is the encoding's `order`.
  //
  // An encoding's order is a list of dimensions, from fastest moving (most
  // minor) to slowest moving.  Thus enc.order[i] does not tell you something
  // about the i'th dimension of the tensor, and it would be disasterously
  // incorrect to do enc.order * trans.order.
  //
  // But!  If we invert enc.order, it *does* meet this criterion.  For example,
  // if enc.order = [2,0,1], inverse(enc.order) = [1,2,0].  If you stare at it,
  // you'll see that inverse(enc.order)[i] == j means that dimension i is the
  // j'th most minor.  Therefore we can safely permute *this* by trans.order.
  //
  // Thus we have
  //
  //   outputEnc.order = inverse(inverse(inputEnc.order) * trans.order)
  //                   = inverse(trans.order) * inputEnc.order.
  //
  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     ArrayRef<int32_t> order, // trans order
                                     Attribute &resultEncoding) const override {
    // Note: inferFooOpEncoding should not crash if given invalid inputs, which
    // happens when someone creates invalid IR.  If we return failure() on
    // error, then MLIR will generate a helpful error message.

    auto invOrder = inversePermutation(order);
    SmallVector<unsigned> invOrderUnsigned(invOrder.begin(), invOrder.end());

    auto permuteCTALayout =
        [&](const CTALayoutAttr &layout) -> FailureOr<CTALayoutAttr> {
      auto n = order.size();
      if (layout.getCTAsPerCGA().size() != n ||
          layout.getCTASplitNum().size() != n ||
          layout.getCTAOrder().size() != n) {
        return failure();
      }

      return CTALayoutAttr::get(
          getDialect()->getContext(),
          applyPermutation(layout.getCTAsPerCGA(), order),
          applyPermutation(layout.getCTASplitNum(), order),
          applyPermutation(invOrderUnsigned, layout.getCTAOrder()));
    };

    if (auto enc = operandEncoding.dyn_cast<SharedEncodingAttr>()) {
      if (enc.getOrder().size() != order.size()) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = SharedEncodingAttr::get(
          getDialect()->getContext(), enc.getVec(), enc.getPerPhase(),
          enc.getMaxPhase(), applyPermutation(invOrderUnsigned, enc.getOrder()),
          *ctaLayout, enc.getHasLeadingOffset());
      return success();
    }

    if (auto enc = operandEncoding.dyn_cast<BlockedEncodingAttr>()) {
      auto n = order.size();
      if (enc.getSizePerThread().size() != n ||
          enc.getThreadsPerWarp().size() != n ||
          enc.getWarpsPerCTA().size() != n || enc.getOrder().size() != n) {
        return failure();
      }
      FailureOr<CTALayoutAttr> ctaLayout = permuteCTALayout(enc.getCTALayout());
      if (failed(ctaLayout)) {
        return failure();
      }
      resultEncoding = BlockedEncodingAttr::get(
          getDialect()->getContext(),
          applyPermutation(enc.getSizePerThread(), order),
          applyPermutation(enc.getThreadsPerWarp(), order),
          applyPermutation(enc.getWarpsPerCTA(), order),
          applyPermutation(invOrderUnsigned, enc.getOrder()), *ctaLayout);
      return success();
    }

    return failure(); // unhandled encoding
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    auto sliceEncoding = operandEncoding.dyn_cast<SliceEncodingAttr>();
    if (!sliceEncoding)
      return emitOptionalError(
          location, "ExpandDimsOp operand encoding must be SliceEncodingAttr");
    if (sliceEncoding.getDim() != axis)
      return emitOptionalError(
          location, "Incompatible slice dimension for ExpandDimsOp operand");
    resultEncoding = sliceEncoding.getParent();
    return success();
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
    auto mmaRetEncoding = retEncoding.dyn_cast<NvidiaMmaEncodingAttr>();
    if (mmaRetEncoding && mmaRetEncoding.isHopper()) {
      auto dotOpEnc = operandEncoding.dyn_cast<DotOperandEncodingAttr>();
      if (!operandEncoding.isa<SharedEncodingAttr>() &&
          !(opIdx == 0 && dotOpEnc && dotOpEnc.getOpIdx() == 0 &&
            dotOpEnc.getParent().isa<NvidiaMmaEncodingAttr>())) {
        return emitOptionalError(
            location, "unexpected operand layout for NvidiaMmaEncodingAttr v3");
      }
    } else if (auto dotOpEnc =
                   operandEncoding.dyn_cast<DotOperandEncodingAttr>()) {
      if (opIdx != dotOpEnc.getOpIdx())
        return emitOptionalError(location, "Wrong opIdx");
      if (retEncoding != dotOpEnc.getParent())
        return emitOptionalError(location, "Incompatible parent encoding");
    } else
      return emitOptionalError(
          location, "Dot's a/b's encoding should be of DotOperandEncodingAttr");
    return success();
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    auto aEncoding =
        operandEncodingA.dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    auto bEncoding =
        operandEncodingB.dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    if (!aEncoding && !bEncoding)
      return mlir::success();
    auto mmaAEncoding =
        aEncoding.getParent().dyn_cast_or_null<NvidiaMmaEncodingAttr>();
    if (mmaAEncoding && mmaAEncoding.isHopper())
      return success();
    // Verify that the encodings are valid.
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");
    if (aEncoding.getKWidth() != bEncoding.getKWidth())
      return op->emitError("mismatching kWidth between A and B operands");
    return success();
  }

  // Given a src shape + encoding and a dst shape, our goal is to compute a dst
  // encoding that makes the reshape a "nop".  That is, if GPU thread [x,y,z]
  // contains elements [a,b,c,d] before the reshape, it contains those same
  // elements after the reshape, they're just "renamed".
  //
  // A dst encoding that satisfies this property does not exist for all inputs.
  // Here are some positive and negative examples.
  //
  //   - NOT OK: 4x4 order=[0,1] -> 16.  Reshape merges elements so
  //     dim 1 is the fastest-changing in the dst, but the src has the opposite
  //     order.
  //   - OK: 2x2x32 order=[1,0,2] -> 4x32.  We choose dst order [0,1].
  //     What's important is that the 2x2 dimensions appear in major-to-minor
  //     order.
  //   - NOT OK: 32x32 sizePerThread=[2,2] -> 1024.  Thread 0 in the src
  //     contains elements [(0,0), (0,1), (1,0), and (1,1)].  We cannot express
  //     this with an encoding based on the dst shape.
  //   - OK: 32x4 sizePerThread=[4,4] -> 128.  dst with sizePerThread=[16] will
  //     contain the same elements as before.
  //
  // Users of this function require that it is symmetrical: if
  // (srcShape,srcEnc,dstShape) => dstEnc, then (dstShape,dstEnc,srcShape) =>
  // srcEnc.
  LogicalResult
  inferReshapeOpNoReorderEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                                  ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                                  std::optional<Location> loc) const override {
    auto src = srcEnc.dyn_cast<BlockedEncodingAttr>();
    if (!src) {
      return emitOptionalError(
          loc, "Non-reordering reshape only supports BlockedEncoding");
    }

    // Nop reshape; we can always infer an encoding.
    if (srcShape == dstShape) {
      dstEnc = srcEnc;
      return success();
    }

    // default -> default encoding is always a nop.
    auto context = srcEnc.getContext();
    int32_t numWarps = product(src.getWarpsPerCTA());
    int32_t threadsPerWarp = product(src.getThreadsPerWarp());
    int32_t numCTAs = product(src.getCTALayout().getCTAsPerCGA());
    if (srcEnc == getDefaultBlockedEncoding(context, srcShape, numWarps,
                                            threadsPerWarp, numCTAs)) {
      dstEnc = getDefaultBlockedEncoding(context, dstShape, numWarps,
                                         threadsPerWarp, numCTAs);
      return success();
    }

    // Feature flag to disable this routine while it's relatively new.
    // TODO(jlebar): Remove this once we're confident in the code.
    if (triton::tools::getBoolEnv(
            "TRITON_DISABLE_RESHAPE_ENCODING_INFERENCE")) {
      return failure();
    }

    // Cowardly refuse to handle encodings with multiple CTAs.  CTAsPerCGA
    // should be like the other fields in blocked encoding, but I'm not sure how
    // to handle CTASplitNum.
    if (!all_of(src.getCTAsPerCGA(), [](int32_t x) { return x == 1; }) ||
        !all_of(src.getCTASplitNum(), [](int32_t x) { return x == 1; })) {
      return emitOptionalError(
          loc, "Non-reordering reshape does not currently support multi-CTA "
               "layouts other than the default layout.");
    }

    // Cowardly refuse to handle encodings where shape[dim] is not divisible by
    // sizePerThread[dim], threadsPerWarp[dim], and warpsPerCTA[dim].  (We make
    // an exception if the block is larger than the shape.)
    auto checkDivisibility = [&](StringRef name, ArrayRef<unsigned> subblock) {
      for (int dim = 0; dim < srcShape.size(); dim++) {
        if (srcShape[dim] >= subblock[dim] &&
            srcShape[dim] % subblock[dim] != 0) {
          return emitOptionalError(loc,
                                   "Can't do a non-reordering reshape because "
                                   "the size of dimension ",
                                   dim, " (", srcShape[dim], ")",
                                   " is not divisible by ", name, "[", dim, "]",
                                   " = ", subblock[dim]);
        }
      }
      return success();
    };
    if (!succeeded(
            checkDivisibility("sizePerThread", src.getSizePerThread())) ||
        !succeeded(
            checkDivisibility("threadsPerWarp", src.getThreadsPerWarp())) ||
        !succeeded(checkDivisibility("warpsPerCTA", src.getWarpsPerCTA()))) {
      return failure();
    }

    SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>> decomp =
        getReshapeDecomposition(srcShape, dstShape);

    // enc.order[i] == j means that dimension j is the enc.order[i]'th most
    // minor. But what we usually want is the inverse: inverse(enc.order)[i] = j
    // means that dimension i is the j'th most minor (larger means more major).
    auto srcInvOrder = inversePermutation(src.getOrder());

    // If src dims [a,b,c] are to be merged, then they must be consecutive in
    // physical order, with `a` being the most major.
    for (const auto &[srcDims, dstDims] : decomp) {
      if (!isConsecutive(to_vector(reverse(gather(srcInvOrder, srcDims))))) {
        return emitOptionalError(loc,
                                 "Cannot do a non-reordering reshape given "
                                 "this src encoding order.  Dimensions [",
                                 join(srcDims, ","),
                                 "] must be physically consecutive.");
      }
    }

    // If src dims [a,b,c] are to be merged, then `c` must fill up sizePerThread
    // / threadsPerWarp / blocksPerCTA before `b` can have any non-1 values.
    // Examples:
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[1,2,2].
    //    The total sizePerThread for dim 2 is 2, which is less than dim 2's
    //    size of 4.  Therefore dim 1 cannot have non-1 sizePerThread.
    //
    //  - OK: shape=[4,4,4], sizePerThread=[1,2,4].
    //    Dim 2's sizePerThread covers its whole size, so dim 1 is allowed to
    //    have non-1 sizePerThread.
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[2,1,4].
    //    Dim 1's sizePerThread does not cover its whole size, so dim 0 is not
    //    allowed to have non-1 sizePerThread.
    //
    //  - NOT OK: shape=[4,4,4], sizePerThread=[1,1,2],
    //            threadsPerWarp=[1,2,1].
    //    Dim 2 has 2 elems per thread and 1 thread per warp.  2*1 is less than
    //    dim 2's size.  Therefore dim 1 must have threadsPerWarp=1.
    //
    // In addition, the encoding's block can be larger than the shape, but only
    // in the most-major dimension of each decomposed chunk, and only after
    // we've "used up" the more minor dims.  Examples:
    //
    //  - OK: shape=[4,4,4], sizePerThread=[1,2,4], threadsPerWarp=[16,2,1],
    //        warpsPerCTA=[4,1,1].
    //    The whole size of dims 0 and 1 are covered by sizePerThread *
    //    threadsPerWarp.  Therefore dim 2 is allowed to have threadsPerWarp and
    //    warpsPerCTA larger than its size.
    for (const auto &[srcDims, dstDims] : decomp) {
      auto shapeRemaining = gather(srcShape, srcDims);
      auto checkSubblock = [&, srcDims = srcDims](ArrayRef<unsigned> subblock) {
        // Iterate minor-to-major (i==0 is most major).
        for (int i = srcDims.size() - 1; i >= 0; i--) {
          int dim = srcDims[i];
          if (subblock[dim] == 1) {
            continue;
          }

          // Check that more-minor dims all have 1 in shapeRemaining.
          for (int j = i + 1; j < srcDims.size(); j++) {
            if (shapeRemaining[j] != 1) {
              return emitOptionalError(
                  loc,
                  "Invalid src encoding for non-reordering reshape.  Must use "
                  "up sizePerThread / threadsPerWarp / warpsPerCTA for "
                  "more-minor dimensions before more major-dims can use them.");
            }
          }

          if (shapeRemaining[i] >= subblock[dim]) {
            assert(shapeRemaining[i] % subblock[dim] == 0); // checked earlier
            shapeRemaining[i] /= subblock[dim];
          } else {
            shapeRemaining[i] = 0;
          }

          // Is the block larger than the shape in this dimension?  This is OK
          // only if we're the most-major dimension of the chunk and in all
          // future chunks, only this most-major dim has a non-1 size.
          if (shapeRemaining[i] == 0 && i != 0) {
            return emitOptionalError(
                loc,
                "Invalid src encoding for non-reordering reshape.  Block "
                "size in dimension ",
                dim,
                " is larger than the shape that dimension, but this is only "
                "allowed for the most-major dimension of a reshape chunk");
          }
        }
        return success();
      };
      if (!succeeded(checkSubblock(src.getSizePerThread())) ||
          !succeeded(checkSubblock(src.getThreadsPerWarp())) ||
          !succeeded(checkSubblock(src.getWarpsPerCTA()))) {
        return failure();
      }
    }

    // Given e.g. src.getSizePerThread(), computeSubblockSize computes e.g.
    // dst.getSizePerThread().  This should be called for each of sizePerThread,
    // threadsPerWarp, and warpsPerCTA, in that order.
    SmallVector<int64_t> dstShapeRemaining(dstShape);
    auto computeSubblockSize = [&](ArrayRef<unsigned> srcSubblock,
                                   SmallVector<unsigned> &dstSubblock,
                                   StringRef fieldName) -> LogicalResult {
      // The dst subblock is "filled up" greedily starting with the most minor
      // dim.  When we're done, we are left with a smaller shape, of size
      // dstShape / dstSubblock, which we store in dstShapeRemaining and use for
      // the next call to computeSubblockSize.
      dstSubblock.resize(dstShape.size());
      for (const auto &[srcDims, dstDims] : decomp) {
        int64_t subblockRemaining = product(gather(srcSubblock, srcDims));
        for (int i = dstDims.size() - 1; i >= 0; i--) {
          auto &val = dstSubblock[dstDims[i]];
          auto &shapeRemaining = dstShapeRemaining[dstDims[i]];
          val = std::min(subblockRemaining, shapeRemaining);

          assert(shapeRemaining % val == 0); // Checked earlier.
          subblockRemaining /= val;
          shapeRemaining /= val;
        }

        // If there are any elems remaining in the subblock, it must be because
        // the block is larger than the shape.  This excess goes into the
        // most-major dim of the subblock.
        dstSubblock[dstDims[0]] *= subblockRemaining;
      }
      return success();
    };

    SmallVector<unsigned> dstSizePerThread;
    SmallVector<unsigned> dstThreadsPerWarp;
    SmallVector<unsigned> dstWarpsPerCTA;
    if (!succeeded(computeSubblockSize(src.getSizePerThread(), dstSizePerThread,
                                       "sizePerThread")) ||
        !succeeded(computeSubblockSize(src.getThreadsPerWarp(),
                                       dstThreadsPerWarp, "threadsPerWarp")) ||
        !succeeded(computeSubblockSize(src.getWarpsPerCTA(), dstWarpsPerCTA,
                                       "warpsPerCTA"))) {
      return failure();
    }

    // Since we know that each set of srcDims is consecutive, we can
    // meaningfully sort decomp by the physical order of the src dimensions,
    // major-to-minor.  This will also be the order of the dst dimensions.
    llvm::sort(decomp, [&](const auto &a, const auto &b) {
      const auto &[srcDimsA, dstDimsA] = a;
      const auto &[srcDimsB, dstDimsB] = b;
      return srcInvOrder[srcDimsA.front()] < srcInvOrder[srcDimsB.front()];
    });

    // Compute the dst order.  Make the dimensions appear in the same order as
    // their corresponding src dimensions.
    SmallVector<unsigned> dstInvOrder(dstShape.size());
    int i = 0;
    for (const auto &[srcDims, dstDims] : decomp) {
      for (auto dim : reverse(dstDims)) {
        dstInvOrder[dim] = i++;
      }
    }
    auto dstOrder = inversePermutation(dstInvOrder);

    // CTALayout can be all 1's because we bailed on multi-CTA layouts above.
    auto CTALayout = CTALayoutAttr::get(
        src.getContext(),
        /*CTAsPerCGA=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTASplitNum=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTAOrder=*/llvm::to_vector(llvm::seq<unsigned>(dstShape.size())));

    dstEnc = BlockedEncodingAttr::get(src.getContext(), dstSizePerThread,
                                      dstThreadsPerWarp, dstWarpsPerCTA,
                                      dstOrder, CTALayout);

    return success();
  }

  LogicalResult
  inferJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                      std::optional<Location> loc) const override {
    auto enc = srcEnc.dyn_cast<BlockedEncodingAttr>();
    if (!enc) {
      return emitOptionalError(loc,
                               "JoinOp can only operate on BlockedEncoding");
    }

    // JoinOp takes two tensors of shape AxBxC and generates a tensor of shape
    // AxBxCx2.  The encoding is the same as the input, but with 2 elems per
    // thread in the new dimension.  The new dimension is most-minor.
    auto append = [](ArrayRef<unsigned> vals, int val) {
      SmallVector<unsigned> ret(vals);
      ret.push_back(val);
      return ret;
    };
    auto appendMinorDim = [](ArrayRef<unsigned> order) {
      SmallVector<unsigned> ret(order);
      ret.insert(ret.begin(), ret.size());
      return ret;
    };
    dstEnc = BlockedEncodingAttr::get(
        enc.getContext(),                    //
        append(enc.getSizePerThread(), 2),   //
        append(enc.getThreadsPerWarp(), 1),  //
        append(enc.getWarpsPerCTA(), 1),     //
        appendMinorDim(enc.getOrder()),      //
        CTALayoutAttr::get(enc.getContext(), //
                           append(enc.getCTAsPerCGA(), 1),
                           append(enc.getCTASplitNum(), 1),
                           appendMinorDim(enc.getCTAOrder())));
    return success();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       std::optional<Location> loc) const override {
    auto enc = srcEnc.dyn_cast<BlockedEncodingAttr>();
    if (!enc) {
      return emitOptionalError(loc,
                               "SplitOp can only operate on BlockedEncoding");
    }

    // SplitOp takes a tensor of shape AxBxCx2 and generates two tensors of
    // shape AxBxC.  The input must have 2 elements per thread in the last
    // dimension, which must be most-minor.  The result encoding is the same as
    // the input, but with the last dimension removed.
    if (enc.getSizePerThread().back() != 2) {
      return emitOptionalError(loc,
                               "SplitOp requires 2 elements per thread in the "
                               "last dimension of the input");
    }
    if (enc.getThreadsPerWarp().back() != 1 ||
        enc.getWarpsPerCTA().back() != 1 || enc.getCTAsPerCGA().back() != 1) {
      return emitOptionalError(
          loc, "SplitOp requires threadsPerWarp, warpsPerCTA, "
               "and CTAsPerCGA = 1 for the last dimension of the input");
    }
    if (enc.getOrder().front() != enc.getOrder().size() - 1) {
      return emitOptionalError(
          loc, "SplitOp requires the last dimension to be most-minor in order");
    }
    if (enc.getCTALayout().getCTAsPerCGA().back() != 1) {
      return emitOptionalError(
          loc,
          "SplitOp requires the last dimension to be most-minor in CTAOrder");
    }

    dstEnc = BlockedEncodingAttr::get(
        enc.getContext(), //
        ArrayRef(enc.getSizePerThread()).drop_back(1),
        ArrayRef(enc.getThreadsPerWarp()).drop_back(1),
        ArrayRef(enc.getWarpsPerCTA()).drop_back(1),
        ArrayRef(enc.getOrder()).drop_front(1),
        CTALayoutAttr::get(enc.getContext(), //
                           ArrayRef(enc.getCTAsPerCGA()).drop_back(1),
                           ArrayRef(enc.getCTASplitNum()).drop_back(1),
                           ArrayRef(enc.getCTAOrder()).drop_front(1)));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Canonicalizer
//===----------------------------------------------------------------------===//

// reshape(cvt) -> reshape
struct CanonicalizeConvertFromReshape
    : public mlir::OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    if (isExpensiveView(convert.getSrc().getType(), op.getType()))
      return failure();
    if (!op.getAllowReorder() || op.getEfficientLayout().has_value())
      return failure();

    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(
        op, op.getType(), convert.getSrc(), op.getAllowReorder());
    return mlir::success();
  }
};

// histogram(cvt) -> histogram
struct CanonicalizeConvertFromHistogram
    : public mlir::OpRewritePattern<triton::HistogramOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::HistogramOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::HistogramOp>(
        op, op->getResult(0).getType(), convert.getSrc());
    return mlir::success();
  }
};

// alloc(cvt) -> alloc
struct CanonicalizeConvertFromAlloc
    : public mlir::OpRewritePattern<triton::gpu::LocalAllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op,
                  PatternRewriter &rewriter) const override {
    if (!op.getInit())
      return failure();
    auto convert = op.getInit().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::gpu::LocalAllocOp>(
        op, op->getResult(0).getType(), convert.getSrc());
    return mlir::success();
  }
};

struct CanonicalizeConvertFromConvert
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    // Convert to the same layout is redundant.
    if (op->getResultTypes() == op->getOperandTypes()) {
      rewriter.replaceOp(op, op->getOperands());
      return success();
    }

    // We don't handle conversions to DotOperandEncodingAttr.  This is a
    // heuristic to accommodate fused attention.
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (dstType.getEncoding().isa<DotOperandEncodingAttr>() &&
        (srcType.getEncoding().isa<NvidiaMmaEncodingAttr>() ||
         srcType.getEncoding().isa<DpasEncodingAttr>()))
      return failure();

    // for hopper MMAv3
    if (dstType.getEncoding().isa<SharedEncodingAttr>() &&
        srcType.getEncoding().isa<NvidiaMmaEncodingAttr>() &&
        llvm::any_of(op.getResult().getUsers(),
                     [](Operation *dot) { return isa<DotOp>(dot); })) {
      return failure();
    }

    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    // cvt(reshape) -> reshape
    if (auto reshape = dyn_cast<ReshapeOp>(arg)) {
      if (!reshape.getAllowReorder() ||
          reshape.getEfficientLayout().has_value() ||
          isExpensiveView(reshape.getSrc().getType(), op.getType()))
        return failure();

      // In TritonGPUToLLVM phase, ViewOp is converted to unpacking and packing
      // operations, which requires the element type to match between unpacking
      // and packing. However, part of values with dot operand encoding will be
      // packed/unpacked as i32 elements instead of the underlying element type.
      // To avoid errors, skip this folding when either the operand or result
      // of view has a dot operand encoding.
      if (hasDotOperandEncoding(op->getOperand(0)) ||
          hasDotOperandEncoding(op->getResult(0)))
        return failure();

      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op->getResult(0).getType(),
                                             reshape.getResult(),
                                             reshape.getAllowReorder());
      return success();
    }

    // cvt(histogram) -> histogram
    if (auto histogram = dyn_cast<HistogramOp>(arg)) {
      // For histogram ops the input and output layouts are independent, so we
      // can always fold convert into the histogram op.
      rewriter.replaceOpWithNewOp<HistogramOp>(op, op->getResult(0).getType(),
                                               histogram.getSrc());
      return success();
    }

    // cvt(local_load) -> local_load.
    if (auto sharedLoad = dyn_cast<LocalLoadOp>(arg)) {
      // Shared_load can load to any layout so we can always fold convert into
      // it.
      rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op->getResult(0).getType(),
                                               sharedLoad.getSrc());
      return success();
    }

    // cvt(cat) -> cat
    if (auto cat = dyn_cast<CatOp>(arg)) {
      if (isExpensiveCat(cat, op.getType().getEncoding()))
        return failure();

      rewriter.replaceOpWithNewOp<CatOp>(op, op->getResult(0).getType(),
                                         cat.getOperands());
      return success();
    }

    // cvt(cvt(x, type1), type2) -> cvt(x, type2)
    if (auto cvt = dyn_cast<ConvertLayoutOp>(arg)) {
      auto srcType = op.getSrc().getType();
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
          op, op->getResultTypes().front(), cvt.getSrc());
      return success();
    }

    // cvt(type1, splat(type2, x)) -> splat(type1, x)
    if (auto splat = dyn_cast<triton::SplatOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op->getResultTypes(),
                                                   splat.getSrc());
      return success();
    }

    // cvt(type1, make_range(type2, x)) -> make_range(type1, x)
    if (auto range = dyn_cast<MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<MakeRangeOp>(
          op, op->getResultTypes(), range.getStart(), range.getEnd());
      return success();
    }

    // cvt(type, constant) -> constant
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(arg))
      if (auto ret = cst.getValue().dyn_cast<SplatElementsAttr>()) {
        auto ty = op->getResultTypes().front().cast<ShapedType>();
        auto newRet =
            SplatElementsAttr::get(ty, ret.getSplatValue<Attribute>());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newRet);
        return success();
      }
    return failure();
  }
};

void ConvertLayoutOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<CanonicalizeConvertFromConvert>(context);
  patterns.add<CanonicalizeConvertFromReshape>(context);
  patterns.add<CanonicalizeConvertFromHistogram>(context);
  patterns.add<CanonicalizeConvertFromAlloc>(context);
}

// LocalAllocOp
void LocalAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // For alloc that cannot be mutated we mark them as no-side effect. This is
  // not fully correct
  if (!getType().getMutableMemory())
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       mlir::triton::gpu::SharedMemory::get());
}

LogicalResult MemDescSubviewOp::verify() {
  auto srcTy = getDesc().getType();
  auto dstTy = getType();

  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitError("result element type must match desc element type");
  }
  if (getOffsets().size() != srcTy.getRank()) {
    return emitError("offsets must have the same rank as input");
  }
  if (srcTy.getRank() < dstTy.getRank()) {
    return emitError("result rank must be less than or equal to input rank");
  }
  auto rankDiff = srcTy.getRank() - dstTy.getRank();
  for (int i = 0; i < dstTy.getRank(); i++) {
    if (dstTy.getDimSize(i) > srcTy.getDimSize(i + rankDiff)) {
      return emitError(
                 "result shape cannot be larger than input shape at dimension ")
             << i;
    }
  }

  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (!!srcEnc != !!dstEnc) {
    return emitError("src and result must both have or not have an encoding");
  }

  if (!isa<SharedEncodingAttr>(srcEnc)) {
    return emitError("src encoding must be SharedEncodingAttr");
  }
  if (!isa<SharedEncodingAttr>(dstEnc)) {
    return emitError("result encoding must be SharedEncodingAttr");
  }

  // TODO(jlebar): Currently we generate illegal encodings, so we can't add a
  // verifier for them.  In particular, we use the same encoding for the src and
  // dst of a subview op, when the subview removes a dimension.  That generates
  // an illegal shared encoding (because the size of `order` doesn't match the
  // rank of the tensor), but it's not checked anywhere, and we believe the
  // resulting code ultimately works.

  return success();
}

void TritonGPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"
#include "triton/Dialect/TritonGPU/IR/OpsEnums.cpp.inc"
      >();
  addInterfaces<TritonGPUOpAsmInterface>();
  addInterfaces<TritonGPUInferLayoutInterface>();
}

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

// verify TritonGPU ops
LogicalResult TritonGPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
