#include "triton/Dialect/Triton/IR/Dialect.h"

#include "Dialect/TritonIntelGPU/IR/Attributes.h"
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

SmallVector<unsigned> DpasEncodingAttr::getRepOrder() const {
  auto rank = getWarpsPerCTA().size();
  return getMatrixOrder(rank, /*rowMajor*/ true);
}

SmallVector<unsigned>
DpasEncodingAttr::getRepOrderForOperand(OpIdx opIdx) const {
  size_t rank = getWarpsPerCTA().size();
  return getOrderForDotOperand(unsigned(opIdx), rank, /*kMajor*/ true);
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
  auto warpsPerCTA = getWarpsPerCTA();
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
  unsigned threadsPerWar = getThreadsPerWarp();
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

SmallVector<unsigned> DpasEncodingAttr::getContigPerThread() const {
  size_t rank = getWarpsPerCTA().size();
  assert(rank == 2 || rank == 3);
  SmallVector<unsigned> contigPerThread(rank, 1);

  unsigned threadsPerWarp = getThreadsPerWarp();
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
    unsigned opsPerChan, ::llvm::ArrayRef<unsigned> warpsPerCTA,
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
  auto warpsPerCTA = getWarpsPerCTA();
  ArrayRef<unsigned> repCluster = getRepCluster();
  printer << "<{"
          << "repeatCount = " << getRepeatCount() << ", "
          << "systolicDepth = " << getSystolicDepth() << ", "
          << "executionSize = " << getExecutionSize() << ", "
          << "opsPerChan = " << getOpsPerChannel() << ", "
          << "threadsPerWarp = " << getThreadsPerWarp() << ", "
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

LinearLayout WarpEncodingAttr::toLinearLayout(ArrayRef<int64_t> shape) const {
  llvm::report_fatal_error("NYI. WarpEncodingAttr::toLinearLayout");
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
    : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding) const override {
    resultEncoding =
        SliceEncodingAttr::get(getDialect()->getContext(), axis,
                               cast<DistributedEncodingTrait>(operandEncoding));
    return success();
  }

  LogicalResult inferTransOpEncoding(Attribute operandEncoding,
                                     ArrayRef<int64_t> shape,
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

  // Given a src shape + encoding and a dst shape, our goal is to compute a dst
  // encoding that makes the reshape a "nop".  That is, if GPU thread [x,y,z]
  // contains elements [a,b,c,d] before the reshape, it contains those same
  // elements after the reshape, they're just "renamed".
  //
  // Using legacy layouts, a dst encoding that satisfies this property may not
  // exist.  Here are some positive and negative examples.
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
  // With linear layouts, we can always find a dst encoding that satisfies
  // this property. See inferReshapeOpEncoding.
  //
  // Users of this function require that it is symmetrical: if
  // (srcShape,srcEnc,dstShape) => dstEnc, then (dstShape,dstEnc,srcShape) =>
  // srcEnc.
  LogicalResult inferReshapeOpLegacyEncoding(ArrayRef<int64_t> srcShape,
                                             Attribute srcEnc,
                                             ArrayRef<int64_t> dstShape,
                                             Attribute &dstEnc) const {
    auto src = mlir::dyn_cast<BlockedEncodingAttr>(srcEnc);
    if (!src) {
      return failure();
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
      return failure();
    }

    // Cowardly refuse to handle encodings where shape[dim] is not divisible by
    // sizePerThread[dim], threadsPerWarp[dim], and warpsPerCTA[dim].  (We make
    // an exception if the block is larger than the shape.)
    auto checkDivisibility = [&](StringRef name, ArrayRef<unsigned> subblock) {
      for (int dim = 0; dim < srcShape.size(); dim++) {
        if (srcShape[dim] >= subblock[dim] &&
            srcShape[dim] % subblock[dim] != 0) {
          return failure();
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
        return failure();
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
              return failure();
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
            return failure();
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
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (expected == got) {
      return success();
    }
    // Check whether the encodings are structurally the same.
    const auto &expectedLL = triton::gpu::toLinearLayout(shape, expected);
    const auto &gotLL = triton::gpu::toLinearLayout(shape, got);
    if (expectedLL != gotLL) {
      return emitOptionalError(loc, "Expected result encoding ", expected,
                               " but was ", got);
    }
    return success();
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    auto result =
        inferReshapeOpLegacyEncoding(srcShape, srcEnc, dstShape, dstEnc);
    if (succeeded(result)) {
      return result;
    }

    // If the legacy encoding failed use LinearLayouts.
    // Once LinearLayouts are more widely used, we can remove
    // inferReshapeOpLegacyEncoding and simply use LLs.
    auto *ctx = getContext();
    auto src = toLinearLayout(srcShape, srcEnc);

    if (product(srcShape) != product(dstShape)) {
      return emitOptionalError(loc, "numel of dst shape does not match "
                                    "numel of src shape");
    }

    auto newRank = dstShape.size();
    SmallVector<std::pair<StringAttr, int32_t>> newOutDims;
    for (auto [dim, size] :
         llvm::zip(standardOutDimNames(ctx, newRank), dstShape)) {
      newOutDims.emplace_back(dim, size);
    }
    auto srcOutDims = to_vector(src.getOutDimNames());
    // reshapeOp assumes minor-to-major, so we need to transpose the out dims
    // before the reshape
    std::reverse(srcOutDims.begin(), srcOutDims.end());
    std::reverse(newOutDims.begin(), newOutDims.end());
    auto dst = src.transposeOuts(srcOutDims)
                   .reshapeOuts(newOutDims)
                   .transposeOuts(standardOutDimNames(ctx, newRank));
    dstEnc = LinearEncodingAttr::get(ctx, dst);
    return success();
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    // TODO
    return failure();
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    // TODO
    return failure();
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    auto *ctx = getContext();

    if (getOrder(cast<DistributedEncodingTrait>(inEnc), shape)[axis] == 0) {
      // Dot operand: double kWidth if kDim == axis.
      if (auto dotEnc = mlir::dyn_cast<DotOperandEncodingAttr>(inEnc)) {
        auto dpasEnc = mlir::dyn_cast<DpasEncodingAttr>(dotEnc.getParent());
        int opsPerChan = dpasEnc.getOpsPerChannel();
        auto kWidth = dotEnc.getKWidth();
        if (fwdInference) {
          kWidth *= 2;
          opsPerChan *= 2;
        } else {
          if (kWidth > 1) {
            // bwd inference
            kWidth /= 2;
            opsPerChan /= 2;
          } else {
            return emitOptionalError(loc,
                                     "Fp4ToFpOp requires at least 2 elements "
                                     "per thread in the axis dimension");
          }
        }
        auto newDpasEnc = DpasEncodingAttr::get(
            ctx, dpasEnc.getRepeatCount(), dpasEnc.getSystolicDepth(),
            dpasEnc.getExecutionSize(), opsPerChan, dpasEnc.getWarpsPerCTA(),
            dpasEnc.getRepCluster(), dpasEnc.getThreadsPerWarp());
        outEnc = DotOperandEncodingAttr::get(ctx, dotEnc.getOpIdx(), newDpasEnc,
                                             kWidth);
        return success();
      }
    }
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
