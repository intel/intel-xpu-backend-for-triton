//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;
  Type a = inputs.front(), b = outputs.front();
  auto aT = llvm::dyn_cast<TensorType>(a);
  auto bT = llvm::dyn_cast<TensorType>(b);
  if (!aT || !bT)
    return false;

  if (aT.getElementType() != bT.getElementType())
    return false;

  return succeeded(verifyCompatibleShape(aT, bT));
}

LogicalResult ConcatOp::verify() {
  if (getOperands().size() < 2)
    return emitOpError("requires at least 2 operands");

  auto getRank = [](Type type) {
    return TypeSwitch<Type, unsigned>(type)
        .Case<RankedTensorType>([](auto ty) { return ty.getRank(); })
        .Case<triton::PointerType>([](auto ty) {
          assert(isa<RankedTensorType>(ty.getPointeeType()) &&
                 "Expecting ptr to tensor");
          return cast<RankedTensorType>(ty.getPointeeType()).getRank();
        })
        .Default([](auto) {
          llvm_unreachable("Unexpected type");
          return 0;
        });
  };

  SmallVector<Type> inputTypes;
  for (auto input : getOperands())
    inputTypes.push_back(input.getType());

  Type resultType = getRes().getType();
  unsigned resultRank = getRank(resultType);
  if (llvm::any_of(inputTypes,
                   [&](Type type) { return getRank(type) != resultRank; }))
    return emitOpError("rank of concatenated operands must match result rank");

  unsigned dim = getDim();
  if (dim >= resultRank)
    return emitOpError("concatenation dim must be less than the tensor rank");

  auto getElementType = [](Type type) {
    return TypeSwitch<Type, Type>(type)
        .Case<RankedTensorType>([](auto ty) { return ty.getElementType(); })
        .Case<triton::PointerType>([](auto ty) {
          assert(isa<RankedTensorType>(ty.getPointeeType()) &&
                 "Expecting ptr to tensor");
          return cast<RankedTensorType>(ty.getPointeeType()).getElementType();
        })
        .Default([](auto) {
          llvm_unreachable("Unexpected type");
          return Type();
        });
  };

  Type resultElementType = getElementType(resultType);
  if (llvm::any_of(inputTypes, [&](Type type) {
        return getElementType(type) != resultElementType;
      }))
    return emitOpError("operands and result element type must match");

  auto getDimSize = [](Type type, unsigned dim) {
    return TypeSwitch<Type, unsigned>(type)
        .Case<RankedTensorType>([dim](auto ty) { return ty.getDimSize(dim); })
        .Case<triton::PointerType>([dim](auto ty) {
          assert(isa<RankedTensorType>(ty.getPointeeType()) &&
                 "Expecting ptr to tensor");
          return cast<RankedTensorType>(ty.getPointeeType()).getDimSize(dim);
        })
        .Default([](auto) {
          llvm_unreachable("Unexpected type");
          return 0;
        });
  };

  SmallVector<int64_t> sizes(resultRank);
  for (unsigned i = 0; i < resultRank; ++i) {
    if (i == dim)
      continue;
    SaturatedInteger size;
    for (auto inputType : inputTypes) {
      FailureOr<SaturatedInteger> maybeSize =
          size.desaturate(SaturatedInteger::wrap(getDimSize(inputType, i)));
      if (failed(maybeSize))
        return emitOpError("static concatenation size mismatch along ")
               << "non-concatenated dimension " << i;
      size = *maybeSize;
    }
    sizes[i] = size.asInteger();
  }

  return success();
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
