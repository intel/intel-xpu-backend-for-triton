#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

namespace mlir::triton::gpu {

LogicalResult UpcastMXFPOp::verify() {
  auto fpType = getFpType();

  auto xTy = getSrc().getType();
  auto scaleTy = getScale().getType();

  if (xTy.getElementType() != FloatType::getBF16(getContext()) &&
      xTy.getElementType() != IntegerType::get(getContext(), 8)) {
    return emitOpError("element type of the first operand must be bf16 or i8");
  }

  if (scaleTy.getElementType() != IntegerType::get(getContext(), 8)) {
    return emitOpError("element type of the second operand must be uint8");
  }

  auto xShape = xTy.getShape();
  auto scaleShape = scaleTy.getShape();

  if (xShape.size() != scaleShape.size() || xShape.size() < 2) {
    return emitOpError(
        "operands must have the same number of dimensions, at least 2");
  }

  if (!(fpType == ScaleDotElemType::E2M1 || fpType == ScaleDotElemType::E4M3 ||
        fpType == ScaleDotElemType::E5M2)) {
    return emitOpError("NYI: fpType must be E2M1, E4M3, or E5M2");
  }

  auto layoutX = xTy.getEncoding();
  auto layoutScale = scaleTy.getEncoding();
  if (bool(layoutX) != bool(layoutScale)) {
    return emitOpError(
        "Expected either both or neither operands to have an encoding");
  }
  // Nothing to check if no encoding. This is used to infer the return type in
  // AccelerateMatmul.cpp
  if (!layoutX) {
    return success();
  }

  auto dotEncoding = dyn_cast<DotOperandEncodingAttr>(layoutX);
  if (!dotEncoding) {
    return emitOpError("Expected a DotOperandEncodingAttr for values");
  }
  auto blockedScale = dyn_cast<BlockedEncodingAttr>(layoutScale);
  if (!blockedScale) {
    return emitOpError("Expected a BlockOperandEncoding for scales");
  }

  if (isa<NvidiaMmaEncodingAttr>(dotEncoding.getParent())) {
    // Necessary to keep all of the scales of a given block of values in the
    // same warp
    auto threadsPerWarp = blockedScale.getThreadsPerWarp();
    if (threadsPerWarp != ArrayRef<unsigned>({16, 2})) {
      return emitOpError("Expected threads per warp to be {16, 2}");
    }
  }

  // Change to support fp8 types
  const auto elemsPacked = fpType == ScaleDotElemType::E2M1 ? 2 : 1;
  // Figure out the K dimension for the input A/B. For A/B scale, the K
  // dimension is always the last dimension.
  const int opIdx = dotEncoding.getOpIdx();
  const bool hasBatch = xShape.size() == 3;
  const int kIdx = (opIdx == 0 ? 1 : 0) + hasBatch;

  if (xShape[kIdx] != (32 / elemsPacked) * scaleShape.back()) {
    return emitOpError("K dimension of first operand must be 16 times "
                       "larger than last/K dimension of the second operand");
  }

  // Check other dimensions match too. For input A/B, we need to figure out the
  // index for the M/N dimension. For scale, it's always {(batch), M/N, K}.
  const int mnIdx = (opIdx == 0 ? 0 : 1) + hasBatch;
  if (hasBatch && xShape[0] != scaleShape[0])
    return emitOpError("batch dimension must match between operands");
  if (xShape[mnIdx] != scaleShape[hasBatch]) {
    return emitOpError("M/N dimension must match between operands");
  }

  return success();
}

LogicalResult UpcastMXFPOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties opaqueProperties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xTy = cast<RankedTensorType>(operands[0].getType());
  auto properties = opaqueProperties.as<const Properties *>();
  auto typeEncoded = properties->fp_type.getValue();
  auto xShape = xTy.getShape();

  auto encoding = xTy.getEncoding();

  if (typeEncoded == ScaleDotElemType::E2M1) {
    RankedTensorType retTy;

    auto newShape = SmallVector<int64_t>(xShape);
    newShape.back() *= 2;
    if (!encoding) {
      retTy = RankedTensorType::get(xShape, FloatType::getBF16(ctx));
    } else {
      auto oldEncoding = cast<DotOperandEncodingAttr>(encoding);

      // Note: For Intel the dot operands layout's kWidth parameter must
      // match the parent's dpas layout opsPerChannel. Given that the kWidth
      // parameter for the result dot layout is going to be twice the kWidth
      // parameter of the operand, we cannot reuse the operand's parent dpas
      // layout and we need to materialize a new dpas encoding.
      auto parentEncoding = oldEncoding.getParent();
      if (auto dpasEncoding = dyn_cast<intel::DpasEncodingAttr>(parentEncoding))
        parentEncoding = intel::DpasEncodingAttr::get(
            ctx, dpasEncoding.getRepeatCount(), dpasEncoding.getSystolicDepth(),
            dpasEncoding.getExecutionSize(),
            dpasEncoding.getOpsPerChannel() * 2, dpasEncoding.getWarpsPerCTA(),
            dpasEncoding.getRepCluster(), dpasEncoding.getSubGroupSize());

      auto newVEncoding = DotOperandEncodingAttr::get(
          ctx, oldEncoding.getOpIdx(), parentEncoding,
          oldEncoding.getKWidth() * 2);
      retTy = RankedTensorType::get(newShape, FloatType::getBF16(ctx),
                                    newVEncoding);
    }
    inferredReturnTypes.push_back(retTy);
  } else {
    inferredReturnTypes.push_back(xTy);
  }

  return success();
}

} // namespace mlir::triton::gpu
