#include "triton/Dialect/Triton/IR/Dialect.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.cpp.inc"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

static LogicalResult parseIntAttrValue(AsmParser &parser, Attribute attr,
                                       unsigned &value, StringRef desc) {
  auto intAttr = dyn_cast<IntegerAttr>(attr);
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

// parse an array of integers
static LogicalResult parseIntArrayAttr(AsmParser &parser,
                                       const NamedAttribute &attr,
                                       SmallVector<unsigned> &res,
                                       StringRef desc) {
  auto arrayAttr = dyn_cast<ArrayAttr>(attr.getValue());
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

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// DpasEncodingAttr
//===----------------------------------------------------------------------===//

SmallVector<unsigned> DpasEncodingAttr::getDPASInstShapeA() const {
  return {getRepeatCount(), getSystolicDepth() * getOpsPerChannel()};
};

SmallVector<unsigned> DpasEncodingAttr::getDPASInstShapeB() const {
  return {getSystolicDepth() * getOpsPerChannel(), getExecutionSize()};
};

SmallVector<unsigned> DpasEncodingAttr::getDPASInstShapeC() const {
  return {getRepeatCount(), getExecutionSize()};
};

SmallVector<unsigned> DpasEncodingAttr::getShapeA() const {
  SmallVector<unsigned> instShapeA = getDPASInstShapeA();
  ArrayRef<unsigned> repCluster = getRepCluster();
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeA[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeA[1];
  return resShape;
}

SmallVector<unsigned> DpasEncodingAttr::getShapeB() const {
  SmallVector<unsigned> instShapeB = getDPASInstShapeB();
  ArrayRef<unsigned> repCluster = getRepCluster();
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeB[0];
  resShape[rank - 1] = instShapeB[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> DpasEncodingAttr::getShapeC() const {
  SmallVector<unsigned> instShapeC = getDPASInstShapeC();
  ArrayRef<unsigned> repCluster = getRepCluster();
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeC[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeC[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> DpasEncodingAttr::getSizePerThread() const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  unsigned threadsPerWarp = getThreadsPerWarp__();
  SmallVector<unsigned> shapeC = getDPASInstShapeC();
  unsigned elemsNum = product<unsigned>(shapeC);
  unsigned elemsPerThread = elemsNum / threadsPerWarp;
  auto repCluster = getRepCluster();
  // The Value is shard to lanes to threads per DPAS instruction.
  if (rank == 3)
    res[0] = repCluster[0];
  res[rank - 2] = elemsPerThread * repCluster[rank - 2];
  res[rank - 1] = repCluster[rank - 1];
  return res;
}

SmallVector<unsigned> DpasEncodingAttr::getDefaultOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned> DpasEncodingAttr::getDefaultThreadOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned> DpasEncodingAttr::getDefaultWarpOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned> DpasEncodingAttr::getRepOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned>
DpasEncodingAttr::getRepOrderForOperand(OpIdx opIdx) const {
  size_t rank = getWarpsPerCTA().size();
  return getOrderForDotOperand(unsigned(opIdx), rank, /*kMajor*/ true);
}

SmallVector<unsigned>
DpasEncodingAttr::getThreadsPerWarpForOperand(int opIdx) const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  assert((opIdx == 0 || opIdx == 1) && "Invalid OpIdx!");
  unsigned execSize = getExecutionSize();
  unsigned subgroupSize = getThreadsPerWarp__();
  unsigned systolicDepth = getSystolicDepth();
  unsigned opsPerChannel = getOpsPerChannel();
  if (subgroupSize < execSize) {
    llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                             "be smaller than the execution size");
  }
  if (opIdx == 0) {
    res[rank - 1] =
        systolicDepth * opsPerChannel / ceil<unsigned>(opsPerChannel, 2);
    res[rank - 2] = ceil<unsigned>(subgroupSize, res[rank - 1]);
  } else {
    res[rank - 1] = execSize;
    res[rank - 2] = subgroupSize / execSize;
  }
  return res;
}

SmallVector<unsigned> DpasEncodingAttr::getCTASplitNum() const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  return res;
}

SmallVector<unsigned> DpasEncodingAttr::getCTAOrder() const {
  size_t rank = getWarpsPerCTA().size();
  auto res = llvm::to_vector(llvm::reverse(llvm::seq<unsigned>(rank)));
  return res;
}

SmallVector<unsigned> DpasEncodingAttr::getCTAsPerCGA() const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  return res;
}

SmallVector<int64_t>
DpasEncodingAttr::getDPASRepetitions(ArrayRef<int64_t> shape,
                                     OpIdx opIdx) const {
  // Always return a 3D shape repetitions for the ease of value handling, same
  // to mma.
  SmallVector<unsigned> warpsPerCTA = getWarpsPerCTA();
  size_t rank = shape.size();
  SmallVector<int64_t> rep(3, 1);
  switch (opIdx) {
  case OpIdx::OperandA: {
    SmallVector<unsigned> shapePerWarp = getShapeA();
    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / shapePerWarp[rank - 1])};
  } break;
  case OpIdx::OperandB: {
    SmallVector<unsigned> shapePerWarp = getShapeB();
    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / shapePerWarp[rank - 2]),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  } break;
  case OpIdx::OperandC: {
    auto shapePerWarp = getShapeC();
    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  } break;
  }

  llvm_unreachable("unexpected opIdx");
}

unsigned DpasEncodingAttr::getTotalElemsPerThreadForOperand(
    ArrayRef<int64_t> shape, mlir::Type eltTy, int kWidth, OpIdx opIdx) const {
  SmallVector<int64_t> shapePerCTA = getShapePerCTA(*this, shape);
  SmallVector<int64_t> rep = getDPASRepetitions(shapePerCTA, opIdx);
  unsigned threadsPerWar = getThreadsPerWarp__();
  size_t rank = shape.size();

  switch (opIdx) {
  case OpIdx::OperandA: {
    SmallVector<unsigned> shapeA = getShapeA();
    auto totalElem = product<unsigned>(shapeA);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * product<int64_t>(rep);
  } break;
  case OpIdx::OperandB: {
    SmallVector<unsigned> shapeB = getShapeB();
    auto totalElem = product<unsigned>(shapeB);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * product<int64_t>(rep);
  } break;
  case OpIdx::OperandC: {
    llvm_unreachable("unexpected OpIdx::OperandC");
  } break;
  }
  llvm_unreachable("unexpected opIdx");
}

SmallVector<unsigned> DpasEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__().begin(),
                               getWarpsPerCTA__().end());
}

SmallVector<unsigned> DpasEncodingAttr::getThreadsPerWarp() const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  unsigned executionSize = getExecutionSize();
  unsigned subGroupSize = getThreadsPerWarp__();
  if (subGroupSize < executionSize) {
    llvm::report_fatal_error("DpasEncodingAttr sub-group size could not be "
                             "smaller than the execution size");
  }
  res[rank - 2] = subGroupSize / executionSize;
  res[rank - 1] = executionSize;
  return res;
}

SmallVector<unsigned>
DpasEncodingAttr::getSizePerThreadForOperand(int kWidth, OpIdx opIdx) const {
  ArrayRef<unsigned> repCluster = getRepCluster();
  size_t rank = repCluster.size();
  assert((rank == 2 || rank == 3) && "unexpected rank number for Dpas layout");

  switch (opIdx) {
  case OpIdx::OperandA: {
    SmallVector<unsigned> shapeA = getDPASInstShapeA();
    unsigned subGroupSize = getThreadsPerWarp__();
    unsigned opsPerChannel = getOpsPerChannel();

    // pack the value to i16 for scalar bit width <=16.
    assert((opsPerChannel == 4 || opsPerChannel == 2 || opsPerChannel == 1) &&
           "invalid opsPerChannel number.");
    unsigned packedOpsPerLane = opsPerChannel == 4 ? 2 : 1;
    unsigned packedColNum = shapeA[1] / packedOpsPerLane;

    if (subGroupSize < packedColNum) {
      llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                               "be smaller than the threads required per row.");
    }
    unsigned rowsPerWarp = mlir::ceil<unsigned>(subGroupSize, packedColNum);
    return {shapeA[0] / rowsPerWarp * repCluster[rank - 2], packedOpsPerLane};
  } break;
  case OpIdx::OperandB: {
    SmallVector<unsigned> shapeB = getShapeB();
    unsigned subGroupSize = getThreadsPerWarp__();
    unsigned executionSize = getExecutionSize();
    if (subGroupSize < executionSize) {
      llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                               "be smaller than the execution size");
    }
    SmallVector<unsigned, 2> threadsPerWarp = {subGroupSize / executionSize,
                                               executionSize};
    return {shapeB[rank - 2] / threadsPerWarp[0],
            shapeB[rank - 1] / threadsPerWarp[1] * repCluster[rank - 1]};
  } break;
  case OpIdx::OperandC: {
    llvm_unreachable("unexpected OpIdx::OperandC");
  } break;
  }
  llvm_unreachable("unexpected opIdx");
}

SmallVector<unsigned> DpasEncodingAttr::getContigPerThread() const {
  size_t rank = getWarpsPerCTA().size();
  assert(rank == 2 || rank == 3);
  SmallVector<unsigned> contigPerThread(rank, 1);

  unsigned threadsPerWarp = getThreadsPerWarp__();
  SmallVector<unsigned> instShapeC = getDPASInstShapeC();
  // The software vectorization vectorized the value as C array: int a[N] ->
  // int a[N][threadsPerWarp]
  if (threadsPerWarp > instShapeC[1]) {
    return contigPerThread;
  }

  if (threadsPerWarp == instShapeC[1]) {
    ArrayRef<unsigned> repCluster = getRepCluster();
    contigPerThread[rank - 2] = instShapeC[0] * repCluster[rank - 2];
    return contigPerThread;
  }

  // threadsPerWarp < shapeC[1]
  llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                           "be smaller than the threads required per row.");
}

DpasEncodingAttr::DPASCapability
DpasEncodingAttr::getDPASCapability(ModuleOp mod) {
  assert(mod && "expected a valid module");

  if (auto minSGSizeAttr = mod->getAttrOfType<IntegerAttr>(
          triton::gpu::intel::TritonIntelGPUDialect::getMinSGSizeAttrName())) {
    unsigned minSGSize = minSGSizeAttr.getInt();
    assert(minSGSize == 8 || minSGSize == 16 && "unsupported minSGSize");
    return DPASCapability(minSGSize);
  }

  return DPASCapability();
}

unsigned DpasEncodingAttr::getOpsPerChannel(Type elemType) {
  assert(elemType.isIntOrFloat() && "unsupported type for DpasEncodingAttr");

  unsigned dpasElemBitWidths = elemType.getIntOrFloatBitWidth();
  if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(elemType))
    dpasElemBitWidths *= 2; // We are upcasting FP8 to FP16.

  return DPASCapability::opsChanBitWidths / dpasElemBitWidths;
}

LogicalResult DpasEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned repeatCount, unsigned systolicDepth, unsigned executionSize,
    unsigned opsPerChan, ::llvm::ArrayRef<unsigned> warpsPerCTA__,
    ::llvm::ArrayRef<unsigned> repCluster, unsigned sugGroupSize) {
  if (repeatCount > 8 || repeatCount < 1) {
    return emitError() << "repeatCount must be in the range [1, 8], but was:"
                       << repeatCount;
  }

  if (!(opsPerChan == 1 || opsPerChan == 2 || opsPerChan == 4)) {
    return emitError() << "opsPerChannel must be 1, 2 or 4, but was:"
                       << opsPerChan;
  }

  if (systolicDepth != 8) {
    return emitError() << "systolicDepth must be 8, but was:" << opsPerChan;
  }

  if (!(repCluster.size() == 2 || repCluster.size() == 3)) {
    return emitError() << "expected rank 2 or 3 of repCluster, but the rank is:"
                       << repCluster.size();
  }

  return success();
}

Attribute DpasEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};
  DictionaryAttr dict;
  if (parser.parseAttribute(dict).failed())
    return {};
  if (parser.parseGreater().failed())
    return {};

  SmallVector<unsigned> warpsPerCTA, repCluster;
  unsigned repeatCount = 0;
  unsigned systolicDepth = 0;
  unsigned executionSize = 0;
  unsigned opsPerChan = 0;
  unsigned threadsPerWarp = 0;

  for (const NamedAttribute &attr : dict) {
    if (attr.getName() == "repeatCount") {
      if (parseUInt(parser, attr, repeatCount, "repeatCount").failed())
        return {};
    }
    if (attr.getName() == "systolicDepth") {
      if (parseUInt(parser, attr, systolicDepth, "systolicDepth").failed())
        return {};
    }
    if (attr.getName() == "executionSize") {
      if (parseUInt(parser, attr, executionSize, "executionSize").failed())
        return {};
    }
    if (attr.getName() == "opsPerChan") {
      if (parseUInt(parser, attr, opsPerChan, "opsPerChan").failed())
        return {};
    }
    if (attr.getName() == "warpsPerCTA") {
      if (parseIntArrayAttr(parser, attr, warpsPerCTA, "warpsPerCTA").failed())
        return {};
    }
    if (attr.getName() == "repCluster") {
      if (parseIntArrayAttr(parser, attr, repCluster, "repCluster").failed())
        return {};
    }
    if (attr.getName() == "threadsPerWarp") {
      if (parseUInt(parser, attr, threadsPerWarp, "threadsPerWarp").failed())
        return {};
    }
  }

  return parser.getChecked<DpasEncodingAttr>(
      parser.getContext(), repeatCount, systolicDepth, executionSize,
      opsPerChan, warpsPerCTA, repCluster, threadsPerWarp);
}

void DpasEncodingAttr::print(AsmPrinter &printer) const {
  SmallVector<unsigned> shapeA = getShapeA();
  ArrayRef<unsigned> rA = shapeA;
  SmallVector<unsigned> shapeB = getShapeB();
  ArrayRef<unsigned> rB = shapeB;
  SmallVector<unsigned> shapeC = getShapeC();
  ArrayRef<unsigned> rC = shapeC;
  SmallVector<unsigned> warpsPerCTA = getWarpsPerCTA();
  ArrayRef<unsigned> repCluster = getRepCluster();
  printer << "<{"
          << "repeatCount = " << getRepeatCount() << ", "
          << "systolicDepth = " << getSystolicDepth() << ", "
          << "executionSize = " << getExecutionSize() << ", "
          << "opsPerChan = " << getOpsPerChannel() << ", "
          << "threadsPerWarp = " << getThreadsPerWarp__() << ", "
          << "warpsPerCTA = [" << llvm::ArrayRef<unsigned>(warpsPerCTA) << "], "
          << "repCluster = [" << repCluster << "], "
          << "A = [" << rA << "], "
          << "B = [" << rB << "], "
          << "C = [" << rC << "]"
          << "}>";
}

LinearLayout DpasEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  return DPAStoLinearLayout(shape, *this);
}

//===----------------------------------------------------------------------===//
// WarpEncodingAttr
//===----------------------------------------------------------------------===//

SmallVector<unsigned>
WarpEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
  size_t rank = shape.size();
  ArrayRef<unsigned> sizePerThread = getSizePerThread_();
  ArrayRef<unsigned> threadsPerWarp = getThreadsPerWarp_();
  assert(rank == sizePerThread.size() &&
         "unexpected rank in WarpEncodingAttr::getElemsPerThread");
  SmallVector<unsigned> elemsPerThread(rank);
  for (size_t i = 0; i < rank; ++i) {
    unsigned t = sizePerThread[i] * threadsPerWarp[i];
    elemsPerThread[i] = t;
  }
  return elemsPerThread;
}

unsigned WarpEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                  Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

SmallVector<unsigned> WarpEncodingAttr::getThreadsPerWarp() const {
  auto threadsPerWarp = getThreadsPerWarp_();
  return SmallVector<unsigned>{threadsPerWarp.begin(), threadsPerWarp.end()};
}

SmallVector<unsigned> WarpEncodingAttr::getSizePerThread() const {
  auto sizePerThread = getSizePerThread_();
  return SmallVector<unsigned>{sizePerThread.begin(), sizePerThread.end()};
}

SmallVector<unsigned> WarpEncodingAttr::getRepOrder() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getRepOrder");
}

SmallVector<unsigned> WarpEncodingAttr::getWarpsPerCTA() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getWarpsPerCTA");
}

LinearLayout WarpEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::toLinearLayout");
}

SmallVector<unsigned> WarpEncodingAttr::getDefaultOrder() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getDefaultOrder");
}

SmallVector<unsigned> WarpEncodingAttr::getDefaultThreadOrder() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getDefaultThreadOrder");
}

SmallVector<unsigned> WarpEncodingAttr::getDefaultWarpOrder() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getDefaultWarpOrder");
}

SmallVector<unsigned> WarpEncodingAttr::getCTAsPerCGA() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getCTAsPerCGA");
}

SmallVector<unsigned> WarpEncodingAttr::getCTAOrder() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getCTAOrder");
}

SmallVector<unsigned> WarpEncodingAttr::getCTASplitNum() const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::getCTASplitNum");
}

Attribute WarpEncodingAttr::parse(AsmParser &parser, Type type) {
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
  SmallVector<unsigned> order;

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
    } else if (attr.getName() == "order") {
      if (parseIntArrayAttr(parser, attr, order, "order").failed())
        return {};
    } else {
      parser.emitError(parser.getNameLoc(), "unexpected key: ")
          << attr.getName().strref();
      return {};
    }
  }
  return parser.getChecked<WarpEncodingAttr>(parser.getContext(), sizePerThread,
                                             threadsPerWarp, order);
}

void WarpEncodingAttr::print(mlir::AsmPrinter &printer) const {
  ArrayRef<unsigned> threadsPerWarp = getThreadsPerWarp_();
  ArrayRef<unsigned> sizePerThread = getSizePerThread_();
  printer << "<{"
          << "sizePerThread = [" << sizePerThread << "]"
          << ", threadsPerWarp = [" << threadsPerWarp << "]"
          << ", order = [" << getOrder_() << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//
// Dialect Interface
//===----------------------------------------------------------------------===//

struct TritonIntelGPUInferLayoutInterface
    : public triton::gpu::TritonGPUInferLayoutInterface {
  using TritonGPUInferLayoutInterface::TritonGPUInferLayoutInterface;

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const final {
    auto mmaRetEncoding = dyn_cast<DpasEncodingAttr>(retEncoding);
    if (mmaRetEncoding) {
      auto dotOpEnc = dyn_cast<DotOperandEncodingAttr>(operandEncoding);
      if (!(dotOpEnc && dotOpEnc.getOpIdx() == opIdx &&
            isa<DpasEncodingAttr>(dotOpEnc.getParent())))
        return emitOptionalError(location,
                                 "unexpected operand layout for DPAS");
    } else if (auto dotOpEnc =
                   dyn_cast<DotOperandEncodingAttr>(operandEncoding)) {
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
                                   Attribute operandEncodingB) const final {

    auto aEncoding =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingA);
    auto bEncoding =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingB);
    if (!aEncoding && !bEncoding)
      return mlir::success();
    // Verify that the encodings are valid.
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");

    auto dpasEncoding = dyn_cast<DpasEncodingAttr>(aEncoding.getParent());
    if (dpasEncoding) {
      if (dpasEncoding != bEncoding.getParent())
        return op->emitError(
            "mismatching parent encoding between A and B operands");

      auto opsPerChannel = dpasEncoding.getOpsPerChannel();
      if (opsPerChannel == 1) {
        if (aEncoding.getKWidth() != opsPerChannel)
          return op->emitError("mismatching kWidth of A operands");
      } else {
        if (aEncoding.getKWidth() != opsPerChannel / 2)
          return op->emitError("mismatching kWidth of A operands");
      }

      if (opsPerChannel != bEncoding.getKWidth())
        return op->emitError("mismatching kWidth of B operands");
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//

void TritonIntelGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"
      >();

  addInterfaces<TritonIntelGPUInferLayoutInterface>();

  addOperations<
#define GET_OP_LIST
#include "intel/include/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"
      >();
}
