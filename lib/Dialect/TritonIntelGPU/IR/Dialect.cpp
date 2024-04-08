#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#include "triton/Dialect/TritonIntelGPU/IR/Dialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::gpu::intel;

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {}
} // namespace gpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// DpasEncodingAttr
//===----------------------------------------------------------------------===//

SmallVector<unsigned> DpasEncodingAttr::getShapeA() const {
  return {getRepeatCount(), getSystolicDepth() * getOpsPerChannel()};
}

SmallVector<unsigned> DpasEncodingAttr::getShapeB() const {
  return {getSystolicDepth() * getOpsPerChannel(), getExecutionSize()};
}

SmallVector<unsigned> DpasEncodingAttr::getShapeC() const {
  return {getRepeatCount(), getExecutionSize()};
}

SmallVector<unsigned> DpasEncodingAttr::getSizePerThread() const {
  unsigned threadsPerWarp = getSubGroupSize();
  auto shapeC = getShapeC();
  unsigned elemsNum = product<unsigned>(shapeC);
  unsigned elemsPerThread = elemsNum / threadsPerWarp;
  // The Value is shard per col to threads.
  return {elemsPerThread, 1};
}

SmallVector<unsigned>
DpasEncodingAttr::getShapePerCTATile(ArrayRef<int64_t> tensorShape) const {
  auto shapeC = getShapeC();
  return {shapeC[0] * getWarpsPerCTA()[0], shapeC[1] * getWarpsPerCTA()[1]};
}

SmallVector<unsigned>
DpasEncodingAttr::getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const {
  size_t rank = shape.size();
  assert(rank == 2 && "Unexpected rank of mma layout");

  SmallVector<unsigned> elemsPerThread(rank);
  auto shapePerCTATile = getShapePerCTATile(shape);
  unsigned tilesRow = ceil<unsigned>(shape[0], shapePerCTATile[0]);
  unsigned tilesCol = ceil<unsigned>(shape[1], shapePerCTATile[1]);
  auto sizePerThread = getSizePerThread();
  elemsPerThread[0] = sizePerThread[0] * tilesRow;
  elemsPerThread[1] = sizePerThread[1] * tilesCol;

  return elemsPerThread;
}

unsigned DpasEncodingAttr::getTotalElemsPerThread(ArrayRef<int64_t> shape,
                                                  Type eltTy) const {
  return product<unsigned>(getElemsPerThread(shape, eltTy));
}

SmallVector<unsigned> DpasEncodingAttr::getCTASplitNum() const {
  SmallVector<unsigned> res{1, 1};
  return res;
}

SmallVector<unsigned> DpasEncodingAttr::getCTAOrder() const {
  SmallVector<unsigned> res{1, 0};
  return res;
}

SmallVector<unsigned> DpasEncodingAttr::getCTAsPerCGA() const {
  SmallVector<unsigned> res{1, 1};
  return res;
}

SmallVector<int64_t>
DpasEncodingAttr::getDPASRepetitions(ArrayRef<int64_t> shape, int opIdx) const {
  auto warpsPerCTA = getWarpsPerCTA();
  if (opIdx == 0) {
    auto shapePerWarp = getShapeA();
    return {std::max<int64_t>(1, shape[0] / (shapePerWarp[0] * warpsPerCTA[0])),
            std::max<int64_t>(1, shape[1] / shapePerWarp[1])};
  } else {
    assert(opIdx == 1);
    auto shapePerWarp = getShapeB();
    return {
        std::max<int64_t>(1, shape[0] / shapePerWarp[0]),
        std::max<int64_t>(1, shape[1] / (shapePerWarp[1] * warpsPerCTA[1]))};
  }
}

unsigned DpasEncodingAttr::getTotalElemsPerThreadForOperands(
    ArrayRef<int64_t> shape, mlir::Type eltTy, int kWidth, int opIdx) const {
  auto shapePerCTA = getShapePerCTA(*this, shape);
  int warpsPerCTAM = getWarpsPerCTA()[0];
  int warpsPerCTAN = getWarpsPerCTA()[1];
  auto rep = getDPASRepetitions(shapePerCTA, opIdx);
  auto threadsPerWar = getSubGroupSize();
  if (opIdx == 0) {
    auto instrShapeA = getShapeA();
    auto totalElem = product<unsigned>(instrShapeA);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * rep[0] * rep[1];
  } else { // if (opIdx == 1)
    auto instrShapeB = getShapeB();
    auto totalElem = product<unsigned>(instrShapeB);
    // dpas operands scalar are evenly sharded to each work item.
    return (totalElem / threadsPerWar) * rep[0] * rep[1];
  }
}

SmallVector<unsigned> DpasEncodingAttr::getWarpOrder() const { return {1, 0}; }

SmallVector<unsigned> DpasEncodingAttr::getThreadOrder() const {
  return {1, 0};
}

SmallVector<unsigned> DpasEncodingAttr::getWarpsPerCTA() const {
  return SmallVector<unsigned>(getWarpsPerCTA__().begin(),
                               getWarpsPerCTA__().end());
}

SmallVector<unsigned> DpasEncodingAttr::getThreadsPerWarp() const {
  auto executionSize = getExecutionSize();
  auto subGroupSize = getSubGroupSize();
  if (subGroupSize < executionSize) {
    llvm::report_fatal_error("DpasEncodingAttr sub-group size could not be "
                             "smaller than the execution size");
  }
  return {subGroupSize / executionSize, executionSize};
}

SmallVector<unsigned>
DpasEncodingAttr::getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape,
                                                   int opIdx) const {
  auto parentShapePerCTATile = getShapePerCTATile(shape);
  auto threadsPerWarp = getThreadsPerWarp();
  if (opIdx == 0) {
    auto shapeA = getShapeA();
    return {parentShapePerCTATile[0], shapeA[1]};
  } else if (opIdx == 1) {
    auto shapeB = getShapeB();
    return {shapeB[0], parentShapePerCTATile[1]};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
  }
}

SmallVector<unsigned>
DpasEncodingAttr::getSizePerThreadForOperands(unsigned opIdx) const {
  if (opIdx == 0) {
    SmallVector<unsigned> shapeA = getShapeA();
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
    unsigned rowsPerWarp = subGroupSize / packedColNum;
    return {shapeA[0] / rowsPerWarp, packedOpsPerLane};
  } else if (opIdx == 1) {
    auto shapeB = getShapeB();
    auto subGroupSize = getSubGroupSize();
    auto executionSize = getExecutionSize();
    if (subGroupSize < executionSize) {
      llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                               "be smaller than the execution size");
    }
    SmallVector<unsigned, 2> threadsPerWarp = {subGroupSize / executionSize,
                                               executionSize};
    return {shapeB[0] / threadsPerWarp[0], shapeB[1] / threadsPerWarp[1]};
  } else {
    llvm::report_fatal_error("DotOperandEncodingAttr opIdx must be 0 or 1");
    return {};
  }
}

SmallVector<unsigned> DpasEncodingAttr::getContigPerThread() {
  unsigned threadsPerWarp = getSubGroupSize();
  auto shapeC = getShapeC();
  // The software vectorization vetorize the value as C array: int a[N] -> int
  // a[N][threadsPerWarp]
  if (threadsPerWarp > shapeC[1]) {
    return {1, 1};
  } else if (threadsPerWarp == shapeC[1]) {
    return {shapeC[0], 1};
  } else {
    // threadsPerWarp < shapeC[1]
    llvm::report_fatal_error("DpasEncodingAttr sub-group size could not "
                             "be smaller than the threads required per row.");
  }
}

LogicalResult DpasEncodingAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    unsigned repeatCount, unsigned systolicDepth, unsigned executionSize,
    unsigned opsPerChan, ::llvm::ArrayRef<unsigned> warpsPerCTA__,
    unsigned sugGroupSize) {
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

  SmallVector<unsigned> warpsPerCTA;
  unsigned repeatCount;
  unsigned systolicDepth;
  unsigned executionSize;
  unsigned opsPerChan;
  unsigned threadsPerWarp;

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
    if (attr.getName() == "threadsPerWarp") {
      if (parseUInt(parser, attr, threadsPerWarp, "threadsPerWarp").failed())
        return {};
    }
  }

  return parser.getChecked<DpasEncodingAttr>(
      parser.getContext(), repeatCount, systolicDepth, executionSize,
      opsPerChan, warpsPerCTA, threadsPerWarp);
}

void DpasEncodingAttr::print(AsmPrinter &printer) const {
  auto shapeA = getShapeA();
  llvm::ArrayRef<unsigned> rA = shapeA;
  auto shapeB = getShapeB();
  llvm::ArrayRef<unsigned> rB = shapeB;
  auto shapeC = getShapeC();
  llvm::ArrayRef<unsigned> rC = shapeC;
  auto warpsPerCTA = getWarpsPerCTA();
  printer << "<{"
          << "repeatCount = " << getRepeatCount() << ", "
          << "systolicDepth = " << getSystolicDepth() << ", "
          << "executionSize = " << getExecutionSize() << ", "
          << "opsPerChan = " << getOpsPerChannel() << ", "
          << "threadsPerWarp = " << getSubGroupSize() << ", "
          << "warpsPerCTA = [" << llvm::ArrayRef<unsigned>(warpsPerCTA) << "], "
          << "A = [" << rA << "], "
          << "B = [" << rB << "], "
          << "C = [" << rC << "]"
          << "}>";
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
    auto mmaRetEncoding = retEncoding.dyn_cast<DpasEncodingAttr>();
    if (mmaRetEncoding) {
      auto dotOpEnc = operandEncoding.dyn_cast<DotOperandEncodingAttr>();
      if (!(dotOpEnc && dotOpEnc.getOpIdx() == opIdx &&
            dotOpEnc.getParent().isa<DpasEncodingAttr>()))
        return emitOptionalError(location,
                                 "unexpected operand layout for DPAS");
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
#include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.cpp.inc"
      >();

  addInterfaces<TritonIntelGPUInferLayoutInterface>();

  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"
      >();
}
