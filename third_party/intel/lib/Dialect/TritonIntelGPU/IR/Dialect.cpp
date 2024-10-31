#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.cpp.inc"

#include "llvm/ADT/TypeSwitch.h"

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
  auto instShapeA = getDPASInstShapeA();
  auto repCluster = getRepCluster();
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeA[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeA[1];
  return resShape;
}

SmallVector<unsigned> DpasEncodingAttr::getShapeB() const {
  auto instShapeB = getDPASInstShapeB();
  auto repCluster = getRepCluster();
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeB[0];
  resShape[rank - 1] = instShapeB[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> DpasEncodingAttr::getShapeC() const {
  auto instShapeC = getDPASInstShapeC();
  auto repCluster = getRepCluster();
  size_t rank = repCluster.size();
  SmallVector<unsigned> resShape(rank, 1);
  resShape[rank - 2] = instShapeC[0] * repCluster[rank - 2];
  resShape[rank - 1] = instShapeC[1] * repCluster[rank - 1];
  return resShape;
}

SmallVector<unsigned> DpasEncodingAttr::getSizePerThread() const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  unsigned threadsPerWarp = getSubGroupSize();
  auto shapeC = getDPASInstShapeC();
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

SmallVector<unsigned>
DpasEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  auto shapeC = getShapeC();
  SmallVector<unsigned> warpsPerCTA = getWarpsPerCTA();
  size_t rank = shapeC.size();
  SmallVector<unsigned> shapePerCTATile(rank);
  llvm::transform(
      llvm::zip_equal(shapeC, warpsPerCTA), shapePerCTATile.begin(),
      [](auto entry) { return std::get<0>(entry) * std::get<1>(entry); });
  return shapePerCTATile;
}

SmallVector<unsigned>
DpasEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
  size_t rank = shape.size();
  assert((rank == 2 || rank == 3) && "Unexpected rank of mma layout");

  SmallVector<unsigned> elemsPerThread(rank, 1);
  auto shapePerCTATile = getShapePerCTATile(shape);
  unsigned tilesRow =
      ceil<unsigned>(shape[rank - 2], shapePerCTATile[rank - 2]);
  unsigned tilesCol =
      ceil<unsigned>(shape[rank - 1], shapePerCTATile[rank - 1]);
  auto sizePerThread = getSizePerThread();
  if (rank == 3)
    elemsPerThread[0] =
        sizePerThread[0] * ceil<unsigned>(shape[0], shapePerCTATile[0]);
  elemsPerThread[rank - 2] = sizePerThread[rank - 2] * tilesRow;
  elemsPerThread[rank - 1] = sizePerThread[rank - 1] * tilesCol;

  return elemsPerThread;
}

unsigned DpasEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                  Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
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
DpasEncodingAttr::getDPASRepetitions(ArrayRef<int64_t> shape, int opIdx) const {
  // Always return a 3D shape repetitions for the ease of value handling, same
  // to mma.
  auto warpsPerCTA = getWarpsPerCTA();
  int rank = shape.size();
  SmallVector<int64_t> rep(3, 1);
  if (opIdx == 0) {
    auto shapePerWarp = getShapeA();
    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                    warpsPerCTA[rank - 2])),
            std::max<int64_t>(1, shape[rank - 1] / shapePerWarp[rank - 1])};
  }

  if (opIdx == 1) {
    auto shapePerWarp = getShapeB();
    int64_t numRepBatch =
        rank == 3 ? std::max<int64_t>(1, shape[0] /
                                             (shapePerWarp[0] * warpsPerCTA[0]))
                  : 1;
    return {numRepBatch,
            std::max<int64_t>(1, shape[rank - 2] / shapePerWarp[rank - 2]),
            std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                    warpsPerCTA[rank - 1]))};
  }

  assert(opIdx == 2 && "Unexpected operand id (valid ids are 0, 1 or 2)");
  auto shapePerWarp = getShapeC();
  int64_t numRepBatch =
      rank == 3
          ? std::max<int64_t>(1, shape[0] / (shapePerWarp[0] * warpsPerCTA[0]))
          : 1;
  return {numRepBatch,
          std::max<int64_t>(1, shape[rank - 2] / (shapePerWarp[rank - 2] *
                                                  warpsPerCTA[rank - 2])),
          std::max<int64_t>(1, shape[rank - 1] / (shapePerWarp[rank - 1] *
                                                  warpsPerCTA[rank - 1]))};
}

unsigned DpasEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, mlir::Type eltTy, int kWidth, int opIdx) const {
  auto shapePerCTA = getShapePerCTA(*this, shape);
  auto rep = getDPASRepetitions(shapePerCTA, opIdx);
  auto threadsPerWar = getSubGroupSize();
  size_t rank = shape.size();
  if (opIdx == 0) {
    auto shapeA = getShapeA();
    auto totalElem = product<unsigned>(shapeA);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * product<int64_t>(rep);
  }
  if (opIdx == 1) {
    auto shapeB = getShapeB();
    auto totalElem = product<unsigned>(shapeB);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * product<int64_t>(rep);
  }
  llvm_unreachable("DpasEncodingAttr opIdx must be 0 or 1");
}

SmallVector<unsigned> DpasEncodingAttr::getWarpOrder() const {
  size_t rank = getWarpsPerCTA().size();
  return llvm::to_vector(llvm::reverse(llvm::seq<unsigned>(rank)));
}

SmallVector<unsigned> DpasEncodingAttr::getThreadOrder() const {
  size_t rank = getWarpsPerCTA().size();
  return llvm::to_vector(llvm::reverse(llvm::seq<unsigned>(rank)));
}

SmallVector<unsigned> DpasEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__().begin(),
                               getWarpsPerCTA__().end());
}

SmallVector<unsigned> DpasEncodingAttr::getThreadsPerWarp() const {
  size_t rank = getWarpsPerCTA().size();
  SmallVector<unsigned> res(rank, 1);
  auto executionSize = getExecutionSize();
  auto subGroupSize = getSubGroupSize();
  if (subGroupSize < executionSize) {
    llvm::report_fatal_error("DpasEncodingAttr sub-group size could not be "
                             "smaller than the execution size");
  }
  res[rank - 2] = subGroupSize / executionSize;
  res[rank - 1] = executionSize;
  return res;
}

SmallVector<unsigned>
DpasEncodingAttr::getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape,
                                                   int opIdx) const {
  auto parentShapePerCTATile = getShapePerCTATile(shape);
  size_t rank = parentShapePerCTATile.size();
  assert((rank == 2 || rank == 3) && "unexpected rank number for Dpas layout");
  if (opIdx == 0) {
    auto shapeA = getShapeA();
    return (rank == 2)
               ? SmallVector<unsigned>{parentShapePerCTATile[0], shapeA[1]}
               : SmallVector<unsigned>{parentShapePerCTATile[0],
                                       parentShapePerCTATile[rank - 2],
                                       shapeA[rank - 1]};
  }

  if (opIdx == 1) {
    auto shapeB = getShapeB();
    return (rank == 2)
               ? SmallVector<unsigned>{shapeB[0], parentShapePerCTATile[1]}
               : SmallVector<unsigned>{parentShapePerCTATile[0],
                                       shapeB[rank - 2],
                                       parentShapePerCTATile[rank - 1]};
  }

  llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
}

SmallVector<unsigned>
DpasEncodingAttr::getSizePerThreadForOperands(unsigned opIdx) const {
  ArrayRef<unsigned> repCluster = getRepCluster();
  size_t rank = repCluster.size();
  assert((rank == 2 || rank == 3) && "unexpected rank number for Dpas layout");
  if (opIdx == 0) {
    SmallVector<unsigned> shapeA = getDPASInstShapeA();
    unsigned subGroupSize = getSubGroupSize();
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
  }

  if (opIdx == 1) {
    auto shapeB = getShapeB();
    auto subGroupSize = getSubGroupSize();
    auto executionSize = getExecutionSize();
    if (subGroupSize < executionSize) {
      llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                               "be smaller than the execution size");
    }
    SmallVector<unsigned, 2> threadsPerWarp = {subGroupSize / executionSize,
                                               executionSize};
    return {shapeB[rank - 2] / threadsPerWarp[0],
            shapeB[rank - 1] / threadsPerWarp[1] * repCluster[rank - 1]};
  }

  llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
}

SmallVector<unsigned> DpasEncodingAttr::getElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, Type eltTy, unsigned opIdx) const {
  SmallVector<unsigned> sizePerThread = getSizePerThreadForOperands(opIdx);
  SmallVector<int64_t> repetitions = getDPASRepetitions(shape, opIdx);

  size_t rank = shape.size();
  SmallVector<unsigned> elemsPerThread(rank);
  if (rank == 3)
    elemsPerThread[0] = repetitions[0];
  elemsPerThread[rank - 2] = sizePerThread[0] * repetitions[1];
  elemsPerThread[rank - 1] = sizePerThread[1] * repetitions[2];

  return elemsPerThread;
};

SmallVector<unsigned> DpasEncodingAttr::getContigPerThread() {
  size_t rank = getWarpsPerCTA().size();
  assert(rank == 2 || rank == 3);
  SmallVector<unsigned> contigPerThread(rank, 1);

  unsigned threadsPerWarp = getSubGroupSize();
  auto instShapeC = getDPASInstShapeC();
  // The software vectorization vectorized the value as C array: int a[N] -> int
  // a[N][threadsPerWarp]
  if (threadsPerWarp > instShapeC[1]) {
    return contigPerThread;
  }

  if (threadsPerWarp == instShapeC[1]) {
    auto repCluster = getRepCluster();
    contigPerThread[rank - 2] = instShapeC[0] * repCluster[rank - 2];
    return contigPerThread;
  }

  // threadsPerWarp < shapeC[1]
  llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                           "be smaller than the threads required per row.");
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
  auto shapeA = getShapeA();
  llvm::ArrayRef<unsigned> rA = shapeA;
  auto shapeB = getShapeB();
  llvm::ArrayRef<unsigned> rB = shapeB;
  auto shapeC = getShapeC();
  llvm::ArrayRef<unsigned> rC = shapeC;
  auto warpsPerCTA = getWarpsPerCTA();
  auto repCluster = getRepCluster();
  printer << "<{"
          << "repeatCount = " << getRepeatCount() << ", "
          << "systolicDepth = " << getSystolicDepth() << ", "
          << "executionSize = " << getExecutionSize() << ", "
          << "opsPerChan = " << getOpsPerChannel() << ", "
          << "threadsPerWarp = " << getSubGroupSize() << ", "
          << "warpsPerCTA = [" << llvm::ArrayRef<unsigned>(warpsPerCTA) << "], "
          << "repCluster = [" << repCluster << "], "
          << "A = [" << rA << "], "
          << "B = [" << rB << "], "
          << "C = [" << rC << "]"
          << "}>";
}

std::optional<LinearLayout>
DpasEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  return DPAStoLinearLayout(shape, *this);
}

//===----------------------------------------------------------------------===//
// WarpEncodingAttr
//===----------------------------------------------------------------------===//

SmallVector<unsigned>
WarpEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
  size_t rank = shape.size();
  auto sizePerThread = getSizePerThread();
  auto threadsPerWarp = getThreadsPerWarp();
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
  auto threadsPerWarp = getThreadsPerWarp();
  auto sizePerThread = getSizePerThread();
  printer << "<{"
          << "sizePerThread = [" << llvm::ArrayRef<unsigned>(sizePerThread)
          << "]"
          << ", threadsPerWarp = [" << llvm::ArrayRef<unsigned>(threadsPerWarp)
          << "]"
          << ", order = [" << getOrder() << "]"
          << "}>";
}

//===----------------------------------------------------------------------===//
// Dialect Interface
//===----------------------------------------------------------------------===//

struct TritonIntelGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding = mlir::triton::gpu::SliceEncodingAttr::get(
        getDialect()->getContext(), axis, operandEncoding);
    return success();
  }

  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     ArrayRef<int32_t> order, // trans order
                                     Attribute &resultEncoding) const override {
    // Not support TransOp on DPAS layout.
    return failure();
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    // Not support ExpandDimsOp on DPAS layout.
    return failure();
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> location) const override {
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
                                   Attribute operandEncodingB) const override {
    auto aEncoding =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingA);
    auto bEncoding =
        dyn_cast<triton::gpu::DotOperandEncodingAttr>(operandEncodingB);
    if (!aEncoding && !bEncoding)
      return mlir::success();
    // Verify that the encodings are valid.
    if (!aEncoding || !bEncoding)
      return op->emitError("mismatching encoding between A and B operands");
    if (aEncoding.getKWidth() != bEncoding.getKWidth())
      return op->emitError("mismatching kWidth between A and B operands");
    return success();
  }

  LogicalResult
  inferReshapeOpNoReorderEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                                  ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                                  std::optional<Location> loc) const override {
    // TODO
    return failure();
  }

  LogicalResult
  inferJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                      std::optional<Location> loc) const override {
    // TODO
    return failure();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       std::optional<Location> loc) const override {
    // TODO
    return failure();
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
