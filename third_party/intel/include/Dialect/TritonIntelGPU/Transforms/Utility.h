//===- Utility.h - Triton Intel GPU utilities -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H
#define TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

namespace mlir {
class ConversionPatternRewriter;
}

namespace mlir::triton::gpu::intel {

// If the given type is a pointer of tensors, return the pointee type.
// Otherwise, attempt to cast the given type to a ranked tensor and return the
// dynamic cast result.
RankedTensorType getRankedTensorType(Type type);

// Check if given value is divisible by the divisor.
bool isDivisible(Value value, unsigned divisor);

// Infers the encoding of the source of op given the result encoding.
std::optional<Attribute> inferSrcEncoding(Operation *op, Attribute encoding);

// Retuns true if the operation is an expensive load or store operation.
bool isExpensiveLoadOrStore(Operation *op);

// Returns true if the tensor type has a dot dpas encoding.
bool hasDotDpasEncoding(RankedTensorType tensorType);

// Returns true if the tensor type has a dpas encoding.
bool hasDpasEncoding(RankedTensorType tensorType);

// Returns the dot encoding of the tensor type or std::nullopt.
std::optional<DotOperandEncodingAttr>
getDotEncoding(RankedTensorType tensorType);

// Get backward slice of tensor values starting from the root node along with
// encoding propagation.
LogicalResult getConvertBackwardSlice(
    Value root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr);

LLVM::LLVMFuncOp lookupOrCreateSPIRVFn(Operation *symbolTable, StringRef name,
                                       ArrayRef<Type> paramTypes,
                                       Type resultType);

LLVM::CallOp createSPIRVBuiltinCall(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    LLVM::LLVMFuncOp func, ValueRange args);

// This function folds the `op` operation and returns the constant value if it
// has successfully folded to a constant. Otherwise, it returns `std::nullopt`.
std::optional<int64_t> getFoldedConstantValue(Operation *op);

// Return true if the `val` value is a constant containing a value equal to
// expected.
bool isConstant(Value val, const unsigned expected);

} // namespace mlir::triton::gpu::intel

#endif // TRITON_DIALECT_TRITONINTELGPU_TRANSFORMS_UTILITY_H
