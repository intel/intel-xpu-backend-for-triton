//===- Utility.cpp - Code generation utilities ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/LinearLayoutConversions.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::LLVM::intel {

static Type findShuffleType(RewriterBase &rewriter, Type valType) {
  if (valType.isBF16())
    return rewriter.getI16Type();

  unsigned bitWidth = valType.getIntOrFloatBitWidth();
  if (bitWidth < 8)
    return rewriter.getI8Type();

  assert((valType.isInteger(8) || valType.isInteger(16) ||
          valType.isInteger(32) || valType.isInteger(64) || valType.isF16() ||
          valType.isF32() || valType.isF64()) &&
         "Invalid Shuffle Type");
  return valType;
}

static Value shuffleCommon(Location loc, RewriterBase &rewriter, Value val,
                           Value i, mlir::gpu::ShuffleMode mode) {
  Type valType = val.getType();
  Type shuffleType = findShuffleType(rewriter, valType);

  const unsigned bitWidth = valType.getIntOrFloatBitWidth();
  if (shuffleType != valType) {
    assert(shuffleType.isInteger() &&
           "expected to bitcast to an integer for unsupported shuffles");
    if (!valType.isInteger()) {
      val = bitcast(val, int_ty(bitWidth));
    }
    if (bitWidth < shuffleType.getIntOrFloatBitWidth()) {
      val = zext(shuffleType, val);
    }
  }

  int width = TritonGEN::getSubgroupSize(i.getDefiningOp());
  Value widthConstant = i32_val(width);
  Value result =
      rewriter.create<mlir::gpu::ShuffleOp>(loc, val, i, widthConstant, mode)
          .getShuffleResult();

  if (shuffleType != valType) {
    if (bitWidth < shuffleType.getIntOrFloatBitWidth()) {
      result = trunc(int_ty(bitWidth), result);
    }
    if (!valType.isInteger()) {
      result = bitcast(result, valType);
    }
  }

  return result;
}

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred) {
  assert(cast<LLVMPointerType>(ptr.getType()).getAddressSpace() == 3 &&
         "Invalid addr space for loadShared");
  Value undef = undef(elemTy);
  Block &endBlock = createPredicatedBlock(rewriter, loc, pred,
                                          SmallVector<Value, 1>{undef}, [&] {
                                            Value ret = load(elemTy, ptr);
                                            return SmallVector<Value, 1>{ret};
                                          });
  return *endBlock.args_begin();
}

Value shuffleXor(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i),
                       mlir::gpu::ShuffleMode::XOR);
}

Value shuffleUp(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleCommon(loc, rewriter, val, i32_val(i),
                       mlir::gpu::ShuffleMode::UP);
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, int i) {
  return shuffleIdx(loc, rewriter, val, i32_val(i));
}

Value shuffleIdx(Location loc, RewriterBase &rewriter, Value val, Value i) {
  return shuffleCommon(loc, rewriter, val, i, mlir::gpu::ShuffleMode::IDX);
}

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content, unsigned addressSpace) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr), /*alignment=*/0, addressSpace);
  }

  Value zero = i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart = gep(ptr_ty(ctx, global.getAddrSpace()), i8_ty, globalPtr,
                          SmallVector<Value>({zero}));
  return stringStart;
}

// declare __spirv_ocl_printf(i8*, ...) as external function
LLVM::LLVMFuncOp getSpirvPrintfDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName("_Z18__spirv_ocl_printf");
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  MLIRContext *context = rewriter.getContext();
  auto ptrTy = LLVM::LLVMPointerType::get(
      context, TritonGEN::TritonGENMemorySpace::kUniformConstant);
  SmallVector<Type> argsType{ptrTy};
  auto retType = i32_ty;
  auto funcType =
      LLVM::LLVMFunctionType::get(retType, argsType, /*isVarArg*/ true);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto printFunc = rewriter.create<LLVM::LLVMFuncOp>(
      UnknownLoc::get(context), funcName, funcType, LLVM::Linkage::External,
      /*dsoLocal*/ false, LLVM::CConv::SPIR_FUNC, /*comdat=*/SymbolRefAttr{});
  printFunc->setAttr("nounwind", rewriter.getUnitAttr());

  return printFunc;
}

static std::string getFormatSubstr(Value value, bool hex = false,
                                   std::optional<int> width = std::nullopt,
                                   bool isSigned = false) {
  Type type = value.getType();
  // If the `value` is a pointer, just return %p.
  if (isa<LLVM::LLVMPointerType>(type)) {
    return "%p";
  }
  // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in the
  // type (so 4 for fp16, 8 for int32, 16 for int64).
  if (hex) {
    // Ignore `width` for `hex` values, pad to typeWidth.
    std::string ret = "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
    if (type.getIntOrFloatBitWidth() > 32) {
      ret += "ll";
    }
    ret += "x";
    return ret;
  }

  std::string prefix = "%";
  if (width.has_value()) {
    prefix += std::to_string(*width);
  }

  if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
    return prefix + "f";
  } else if (type.isInteger()) {
    if (type.getIntOrFloatBitWidth() == 64)
      return prefix + (isSigned ? "lli" : "llu");
    else
      return prefix + (isSigned ? "i" : "u");
  }
  assert(false && "not supported type");
  return "";
}

// count for the string to |formatStrByteCount| if not null.
Value llPrintf(StringRef msg, ValueRange args,
               ConversionPatternRewriter &rewriter,
               const TargetInfoBase &targetInfo,
               int *formatStrByteCount = nullptr) {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue = LLVM::intel::addStringToModule(
      UnknownLoc::get(rewriter.getContext()), rewriter, "printfFormat_",
      msgNewline, TritonGEN::TritonGENMemorySpace::kUniformConstant);
  targetInfo.printf(rewriter, msgValue, msgNewline.size_in_bytes(), args);
  if (formatStrByteCount)
    *formatStrByteCount = msgNewline.size_in_bytes();
  return msgValue;
}

void printTensor(StringRef msg, Value tensor, Type tensorTy,
                 ConversionPatternRewriter &rewriter,
                 const TargetInfoBase &targetInfo) {
  auto *context = rewriter.getContext();

  // Elements of the tensor that are resident in this GPU thread.
  auto elems = unpackLLElements(UnknownLoc::get(context), tensor, rewriter);

  // Get the indices of `elems` within the tensor.  Note that if `elems`
  // has an "interesting" layout, then these will not be in any
  // particularly nice order.

  // Extract the shape of the tensor being printed and use it to figure
  // out how many digits we need for each of the dimensions.
  SmallVector<int, 8> dimWidths;
  SmallVector<SmallVector<Value>> indices;
  if (auto rankedTy = dyn_cast<RankedTensorType>(tensorTy)) {
    indices =
        ::intel::emitIndices(UnknownLoc::get(context), rewriter, targetInfo,
                             rankedTy.getEncoding(), rankedTy, true);
    for (int64_t dim : rankedTy.getShape()) {
      if (dim > 0) {
        dimWidths.push_back(static_cast<int>(std::ceil(std::log10(dim))));
      } else {
        dimWidths.push_back(0);
      }
    }
  } else {
    // We're printing a scalar.
    assert(elems.size() == 1);
    indices.push_back({});
  }

  size_t rank = dimWidths.size();

  auto getPid = [&](int axis) {
    return targetInfo.programId(
        rewriter, UnknownLoc::get(context),
        rewriter.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>(),
        axis);
  };
  std::array<Value, 3> pid = {getPid(0), getPid(1), getPid(2)};
  // Format is:
  //   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>
  // where we leave off "(operand <n>)" if there's only one operand.
  //
  // The Python wrapper munges `prefix` so that it prints nicely (e.g. starts
  // with " " and ends with ": ").

  Value formatStrValue;
  int formatStrByteCount = 0;
  for (int i = 0; i < elems.size(); i++) {
    std::string formatStr;
    llvm::raw_string_ostream os(formatStr);

    // Device printf can only accept 32 args; if we pass more than that, it
    // will print garbage for the trailing args.
    constexpr int kMaxPrintfOperands = 32;
    SmallVector<Value, kMaxPrintfOperands> printfOperands;

    // If `rank` is large enough, we could end up exceeding
    // kMaxPrintfOperands.  In that case, just truncate the index.
    // (Subtract 2 because we're going to add two operands after the index.)
    int maxAllowedRank = kMaxPrintfOperands - printfOperands.size() - 2;

    os << "idx (";
    const auto &index = indices[i];
    for (size_t dim = 0; dim < index.size(); dim++) {
      if (dim != 0) {
        os << ", ";
      }
      if (dim == maxAllowedRank) {
        os << "... (truncated)";
        break;
      }
      os << getFormatSubstr(index[dim], /*hex=*/false,
                            /*width=*/dimWidths[dim]);
      printfOperands.push_back(index[dim]);
    }
    os << ")";

    // TODO(jlebar): We really should pad the pid, but because the max pid is
    // not known at compile-time, this would require nontrivial device-side
    // work.
    os << " pid (";
    for (int j = 0; j < pid.size(); j++) {
      if (j != 0) {
        os << ", ";
      }
      os << getFormatSubstr(pid[j]);
      printfOperands.push_back(pid[j]);
    }
    os << ") ";

    auto loc = UnknownLoc::get(rewriter.getContext());
    auto mod =
        rewriter.getInsertionBlock()->getParent()->getParentOfType<ModuleOp>();
    unsigned iWarpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    Value threadId = getThreadId(rewriter, loc);
    Value warpSize = i32_val(iWarpSize);
    Value warpId = udiv(threadId, warpSize);
    Value laneId = urem(threadId, warpSize);
    os << "warp " << getFormatSubstr(warpId);
    printfOperands.push_back(warpId);
    os << " lane " << getFormatSubstr(laneId);
    printfOperands.push_back(laneId);

    os << msg;

    auto elem = elems[i];
    os << getFormatSubstr(elem, false);
    printfOperands.push_back(elem);

    // It's the same format string each iteration, but it's a lot easier if we
    // construct the format string at the same time as we populate
    // printfOperands.  But we don't want to create BLOCK_SIZE duplicate
    // strings, so we cache the Value.
    if (i == 0) {
      formatStrValue = llPrintf(formatStr, printfOperands, rewriter, targetInfo,
                                &formatStrByteCount);
    } else {
      targetInfo.printf(rewriter, formatStrValue, formatStrByteCount,
                        printfOperands);
    }
  }
}

} // namespace mlir::LLVM::intel
