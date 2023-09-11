#include "TypeConverter.h"
#include "Utility.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::MmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;

/// Mapping between SPIR-V storage classes to Triton memory spaces.
///
#define STORAGE_SPACE_MAP_LIST(MAP_FN)                                         \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 1)                               \
  MAP_FN(spirv::StorageClass::Workgroup, 3)

#if 0
MAP_FN(spirv::StorageClass::StorageBuffer, 0)                                \
  MAP_FN(spirv::StorageClass::Uniform, 4)                                      \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::PushConstant, 7)                                 \
  MAP_FN(spirv::StorageClass::UniformConstant, 8)                              \
  MAP_FN(spirv::StorageClass::Input, 9)                                        \
  MAP_FN(spirv::StorageClass::Output, 10)                                      \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 11)                              \
  MAP_FN(spirv::StorageClass::AtomicCounter, 12)                               \
  MAP_FN(spirv::StorageClass::Image, 13)                                       \
  MAP_FN(spirv::StorageClass::CallableDataKHR, 14)                             \
  MAP_FN(spirv::StorageClass::IncomingCallableDataKHR, 15)                     \
  MAP_FN(spirv::StorageClass::RayPayloadKHR, 16)                               \
  MAP_FN(spirv::StorageClass::HitAttributeKHR, 17)                             \
  MAP_FN(spirv::StorageClass::IncomingRayPayloadKHR, 18)                       \
  MAP_FN(spirv::StorageClass::ShaderRecordBufferKHR, 19)                       \
  MAP_FN(spirv::StorageClass::PhysicalStorageBuffer, 20)                       \
  MAP_FN(spirv::StorageClass::CodeSectionINTEL, 21)                            \
  MAP_FN(spirv::StorageClass::DeviceOnlyINTEL, 22)                             \
  MAP_FN(spirv::StorageClass::HostOnlyINTEL, 23)
#endif

std::optional<spirv::StorageClass>
getStorageClassForMemorySpace(unsigned space) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case space:                                                                  \
    return storage;

  switch (space) {
    STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    return std::nullopt;
  }
#undef STORAGE_SPACE_MAP_FN
}

TritonGPUToSPIRVTypeConverter::TritonGPUToSPIRVTypeConverter(
    spirv::TargetEnvAttr &targetAttr, SPIRVConversionOptions &option)
    : SPIRVTypeConverter(targetAttr, option) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([&](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  addConversion([&](mlir::VectorType type) -> std::optional<Type> {
    // Recursively translate vector type
    return mlir::VectorType::get(type.getShape(),
                                 convertType(type.getElementType()));
  });
  // Internally store float8 as int8
  addConversion([&](mlir::Float8E4M3B11FNUZType type) -> std::optional<Type> {
    llvm::report_fatal_error("SPIRV doesn't support fp8 type");
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNType type) -> std::optional<Type> {
    llvm::report_fatal_error("SPIRV doesn't support fp8 type");
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E4M3FNUZType type) -> std::optional<Type> {
    llvm::report_fatal_error("SPIRV doesn't support fp8 type");
    return IntegerType::get(type.getContext(), 8);
  });
  addConversion([&](mlir::Float8E5M2Type type) -> std::optional<Type> {
    llvm::report_fatal_error("SPIRV doesn't support fp8 type");
    return IntegerType::get(type.getContext(), 8);
  });
  // Internally store bfloat16 as int16
  addConversion([&](BFloat16Type type) -> std::optional<Type> {
    return IntegerType::get(type.getContext(), 16);
  });
  addConversion(
      [&](IndexType type) -> std::optional<Type> { return getIndexType(); });

  // Add generic source and target materializations to handle cases where
  // non-SPIRV types persist after an SPIRV conversion.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs,
                               Location loc) -> std::optional<Value> {
    if (inputs.size() != 1)
      return std::nullopt;

    return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  });
}

Type TritonGPUToSPIRVTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  // Recursively translate pointee type
  std::optional<spirv::StorageClass> storageClass =
      getStorageClassForMemorySpace(type.getAddressSpace());
  assert(storageClass && "uncompatible pointer address type in SPIRV");
  return spirv::PointerType::get(convertType(type.getPointeeType()),
                                 *storageClass);
}

Value TritonGPUToSPIRVTypeConverter::packLLElements(
    Location loc, ValueRange resultVals, ConversionPatternRewriter &rewriter,
    Type type) {
  auto structType = this->convertType(type).dyn_cast<spirv::StructType>();
  if (!structType) {
    assert(resultVals.size() == 1);
    return *resultVals.begin();
  }

  auto elementTypes = structType.getElementTypes();
  if (elementTypes.size() != resultVals.size()) {
    emitError(loc) << " size mismatch when packing elements for LLVM struct"
                   << " expected " << elementTypes.size() << " but got "
                   << resultVals.size();
  }
  Value spirvStruct = rewriter.create<spirv::UndefOp>(loc, structType);
  for (const auto &v : llvm::enumerate(resultVals)) {
    if (!v.value()) {
      emitError(loc)
          << "cannot insert null values into struct, but tried to insert"
          << v.value();
    }
    if (v.value().getType() != elementTypes[v.index()]) {
      emitError(loc) << "invalid element type in packLLEElements. Expected "
                     << elementTypes[v.index()] << " but got "
                     << v.value().getType();
    }
    spirvStruct = insert_val(structType, v.value(), spirvStruct,
                             rewriter.getI32ArrayAttr(v.index()));
  }
  return spirvStruct;
}

SmallVector<Value> TritonGPUToSPIRVTypeConverter::unpackLLElements(
    Location loc, Value spirvStruct, ConversionPatternRewriter &rewriter,
    Type type) {
  assert(bool(spirvStruct) && "can not unpack null values");
  if (spirvStruct.getType().isIntOrIndexOrFloat() ||
      spirvStruct.getType().isa<triton::PointerType>() ||
      spirvStruct.getType().isa<spirv::PointerType>())
    return {spirvStruct};
  auto types =
      spirvStruct.getType().cast<spirv::StructType>().getElementTypes();
  SmallVector<Value> results(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    results[i] = extract_val(type, spirvStruct, rewriter.getI32ArrayAttr(i));
  }
  return results;
}

Type TritonGPUToSPIRVTypeConverter::getElementTypeForStruct(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  Type elemTy = convertType(type.getElementType());
  auto dotOpLayout = layout.dyn_cast<DotOperandEncodingAttr>();
  if (!dotOpLayout)
    return elemTy;
  auto mmaParent = dotOpLayout.getParent().dyn_cast<MmaEncodingAttr>();
  if (!mmaParent)
    return elemTy;
  if (mmaParent.isAmpere()) {
    int bitwidth = elemTy.getIntOrFloatBitWidth();
    assert(bitwidth <= 32);
    return IntegerType::get(ctx, 32);
  } else {
    assert(mmaParent.isVolta());
    return vec_ty(elemTy, 2);
  }
}

Type TritonGPUToSPIRVTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  auto ctx = type.getContext();
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> shape(type.getShape().begin(), type.getShape().end());
  Type eltType = getElementTypeForStruct(type);

  if (auto shared_layout = layout.dyn_cast<SharedEncodingAttr>()) {
    SmallVector<Type, 4> types;
    // base ptr
    auto ptrType =
        spirv::PointerType::get(eltType, spirv::StorageClass::Workgroup);
    types.push_back(ptrType);
    // shape dims
    auto rank = type.getRank();
    // offsets + strides
    for (auto i = 0; i < rank * 2; i++) {
      types.push_back(IntegerType::get(ctx, 32));
    }
    return spirv::StructType::get(types);
  }

  unsigned numElementsPerThread = getTotalElemsPerThread(type);
  SmallVector<Type, 4> types(numElementsPerThread, eltType);
  return spirv::StructType::get(types);
}
