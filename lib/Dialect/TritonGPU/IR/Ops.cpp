#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Types.h"
#include "llvm/Support/raw_ostream.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

namespace mlir::triton::gpu {

LogicalResult UpcastMXFPOp::verify() {
  auto fpType = getFpType();

  auto xTy = getSrc().getType();
  auto scaleTy = getScale().getType();

  if (xTy.getElementType() != FloatType::getBF16(getContext())) {
    return emitOpError("element type of the first operand must be bf16");
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

  if (!(fpType == F8F6F4Type::E2M1 || fpType == F8F6F4Type::E4M3 ||
        fpType == F8F6F4Type::E5M2)) {
    return emitOpError("NYI: fpType must be E2M1, E4M3, or E5M2");
  }

  // Change to support fp8 types
  const auto elems_packed = fpType == F8F6F4Type::E2M1 ? 2 : 1;

  if (xShape.back() != (32 / elems_packed) * scaleShape.back()) {
    return emitOpError("last dimension of first operand must be 16 times "
                       "larger than that of the second operand");
  }

  if (!std::equal(xShape.begin(), xShape.end() - 1, scaleShape.begin())) {
    return emitOpError(
        "all dimensions except the last must match between operands");
  }

  auto layoutX = xTy.getEncoding();
  if (!layoutX || !isa<DotOperandEncodingAttr>(layoutX)) {
    return emitOpError("Expected a DotOperandEncodingAttr for values");
  }
  auto layoutScale = scaleTy.getEncoding();
  if (!layoutScale || !isa<BlockedEncodingAttr>(layoutScale)) {
    return emitOpError("Expected a BlockOperandEncoding for scales");
  }
  auto blockedScale = cast<BlockedEncodingAttr>(layoutScale);

  // Necessary to keep all of the scales of a given block of values in the same
  // warp
  auto threadsPerWarp = blockedScale.getThreadsPerWarp();
  if (threadsPerWarp != ArrayRef<unsigned>({16, 2})) {
    return emitOpError("Expected threads per warp to be {16, 2}");
  }

  return success();
}

LogicalResult UpcastMXFPOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties opaqueProperties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto xTy = cast<RankedTensorType>(operands[0].getType());
  auto properties = opaqueProperties.as<const Properties *>();
  auto typeEncoded = properties->fp_type.getValue();
  auto xShape = xTy.getShape();

  auto encoding = xTy.getEncoding();
  if (!encoding) {
    return emitOptionalError(location, "expected an encoding");
  }
  if (!mlir::isa<DotOperandEncodingAttr>(encoding)) {
    return emitOptionalError(location, "expected an mma layout encoding");
  }
  if (xShape.size() < 2) {
    return emitOptionalError(location, "tensor rank must be at least 2");
  }

  // For now we just return the input encoding. For fp4 we'll need to cast from
  // tf32 to fp16 encoding and multiply the shape by two
  assert((typeEncoded == F8F6F4Type::E4M3 || typeEncoded == F8F6F4Type::E5M2) &&
         "NYI: only fp8e4m3 and fp8e5m2 are supported");

  inferredReturnTypes.push_back(xTy);
  return success();
}

} // namespace mlir::triton::gpu
