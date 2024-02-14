#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

// The input print op contains:
//  - a "prefix" (string) specified by the user, and
//  - one or more "operands" (tensors).
//
// For each operand, we print all of the values contained in this GPU thread,
// one per line, along with the index of the value in its tensor.
struct PrintOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::PrintOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::PrintOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto typeConverter = getTypeConverter();
    auto loc = op->getLoc();
    Value prefixStr = LLVM::addStringToModule(
        loc, rewriter, "printfPrefix_", op.getPrefix(),
        (target == Target::GENX) ? GENX::GENXMemorySpace::kUniformConstant : 0);

    auto getPid = [&](int axis) {
      return llGetPid(axis, loc, op->getParentOfType<ModuleOp>(), rewriter,
                      target);
    };
    std::array<Value, 3> pid = {getPid(0), getPid(1), getPid(2)};

    // Simple printf of a string without any tensors.
    if (op.getNumOperands() == 0) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);
      os << "pid (" << getFormatSubstr(pid[0]) << ", "
         << getFormatSubstr(pid[1]) << ", " << getFormatSubstr(pid[2]) << ")%s";
      llPrintf(formatStr, {pid[0], pid[1], pid[2], prefixStr}, rewriter,
               target);
    } else {
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        // Elements of the tensor that are resident in this GPU thread.
        auto elems = typeConverter->unpackLLElements(
            loc, adaptor.getOperands()[i], rewriter);

        // Get the indices of `elems` within the tensor.  Note that if `elems`
        // has an "interesting" layout, then these will not be in any
        // particularly nice order.

        // Extract the shape of the tensor being printed and use it to figure
        // out how many digits we need for each of the dimensions.
        SmallVector<int, 8> dimWidths;
        SmallVector<SmallVector<Value>> indices;
        if (auto rankedTy =
                op.getOperand(i).getType().dyn_cast<RankedTensorType>()) {
          indices = emitIndices(loc, rewriter, rankedTy.getEncoding(), rankedTy,
                                true);
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

        if (!elems.empty()) {
          printTensor(prefixStr, /*operand=*/i,
                      /*numOperands=*/op.getNumOperands(), elems, pid, indices,
                      dimWidths, rewriter);
        }
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  void printTensor(Value prefixStr, size_t operand, size_t numOperands,
                   ArrayRef<Value> elems, std::array<Value, 3> pid,
                   ArrayRef<SmallVector<Value>> indices,
                   ArrayRef<int> dimWidths,
                   ConversionPatternRewriter &rewriter) const {
    assert(!elems.empty());
    assert(elems.size() == indices.size());
    assert(dimWidths.size() == indices.front().size());

    size_t rank = dimWidths.size();

    // Format is:
    //   pid (<x>, <y>, <z>) idx (<i1>, <i2>, ...)<prefix> (operand <n>) <elem>
    // where we leave off "(operand <n>)" if there's only one operand.
    //
    // The Python wrapper munges `prefix` so that it prints nicely (e.g. starts
    // with " " and ends with ": ").

    Value formatStrValue;
    for (int i = 0; i < elems.size(); i++) {
      std::string formatStr;
      llvm::raw_string_ostream os(formatStr);

      // nvptx printf can only accept 32 args; if we pass more than that, it
      // will print garbage for the trailing args.
      constexpr int kMaxPrintfOperands = 32;
      SmallVector<Value, kMaxPrintfOperands> printfOperands;

      // TODO(jlebar): We really should pad the pid, but because the max pid is
      // not known at compile-time, this would require nontrivial device-side
      // work.
      os << "pid (";
      for (int j = 0; j < pid.size(); j++) {
        if (j != 0) {
          os << ", ";
        }
        os << getFormatSubstr(pid[j]);
        printfOperands.push_back(pid[j]);
      }
      os << ") ";

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
        os << getFormatSubstr(index[dim], /*width=*/dimWidths[dim]);
        printfOperands.push_back(index[dim]);
      }
      os << ")";

      os << "%s";
      printfOperands.push_back(prefixStr);

      if (numOperands > 1) {
        os << "(operand " << operand << ") ";
      }

      auto elem = elems[i];
      os << getFormatSubstr(elem);
      printfOperands.push_back(elem);

      // It's the same format string each iteration, but it's a lot easier if we
      // construct the format string at the same time as we populate
      // printfOperands.  But we don't want to create BLOCK_SIZE duplicate
      // strings, so we cache the Value.
      if (i == 0) {
        formatStrValue = llPrintf(formatStr, printfOperands, rewriter, target);
      } else {
        llPrintf(formatStrValue, printfOperands, rewriter, target);
      }
    }
  }

  std::string getFormatSubstr(Value value,
                              std::optional<int> width = std::nullopt) const {
    std::string prefix = "%";
    if (width.has_value()) {
      prefix += std::to_string(*width);
    }

    Type type = value.getType();
    if (type.isa<LLVM::LLVMPointerType>()) {
      return prefix + "p";
    } else if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
      return prefix + "f";
    } else if (type.isSignedInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "lli";
      else
        return prefix + "i";
    } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return prefix + "llu";
      else
        return prefix + "u";
    }
    assert(false && "not supported type");
    return "";
  }

  // declare __spirv_ocl_printf(i8*, ...) as external function
  static LLVM::LLVMFuncOp
  getSpirvPrintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("_Z18__spirv_ocl_printf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    MLIRContext *context = rewriter.getContext();
    auto ptrTy = LLVM::LLVMPointerType::get(
        context, GENX::GENXMemorySpace::kUniformConstant);
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

  // declare vprintf(i8*, i8*) as external function
  static LLVM::LLVMFuncOp
  getVprintfDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName("vprintf");
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    auto *context = rewriter.getContext();

    SmallVector<Type> argsType{ptr_ty(context), ptr_ty(context)};
    auto funcType = LLVM::LLVMFunctionType::get(i32_ty, argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(context), funcName,
                                             funcType);
  }

  // extend integer to int32, extend float to float64
  // this comes from vprintf alignment requirements.
  static std::pair<Type, Value>
  promoteValue(ConversionPatternRewriter &rewriter, Value value) {
    auto *context = rewriter.getContext();
    auto type = value.getType();
    Value newOp = value;
    Type newType = type;
    auto loc = UnknownLoc::get(context);

    bool bUnsigned = type.isUnsignedInteger();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      if (bUnsigned) {
        newType = ui32_ty;
        newOp = zext(newType, value);
      } else {
        newType = i32_ty;
        newOp = sext(newType, value);
      }
    } else if (type.isBF16() || type.isF16() || type.isF32()) {
      newType = f64_ty;
      newOp = fpext(newType, value);
    }

    return {newType, newOp};
  }

  // Returns a Value for the format string, which you can reuse.
  static Value llPrintf(StringRef msg, ValueRange args,
                        ConversionPatternRewriter &rewriter, Target target) {
    assert(!msg.empty() && "printf with empty string not supported");
    llvm::SmallString<64> msgNewline(msg);
    msgNewline.push_back('\n');
    Value msgValue = LLVM::addStringToModule(
        UnknownLoc::get(rewriter.getContext()), rewriter, "printfFormat_",
        msgNewline,
        (target == Target::GENX) ? GENX::GENXMemorySpace::kUniformConstant : 0);
    llPrintf(msgValue, args, rewriter, target);
    return msgValue;
  }

  static void llPrintf(Value msg, ValueRange args,
                       ConversionPatternRewriter &rewriter, Target target) {
    auto *ctx = rewriter.getContext();
    Type ptr = ptr_ty(ctx);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    auto funcOp = (target == Target::GENX) ? getSpirvPrintfDeclaration(rewriter)
                                           : getVprintfDeclaration(rewriter);
    auto loc = UnknownLoc::get(ctx);

    Value one = i32_val(1);
    Value zero = i32_val(0);

    Value bufferPtr = null(ptr);

    SmallVector<Value> operands;
    operands.push_back(msg);
    if (target == Target::GENX) {
      // __spirv_ocl_printf expects the value instead of pointer to value
      for (auto arg : args) {
        operands.push_back(arg);
      }
    } else {
      SmallVector<Value, 16> newArgs;
      if (args.size() >= 1) {
        SmallVector<Type> argTypes;
        for (auto arg : args) {
          Type newType;
          Value newArg;
          std::tie(newType, newArg) = promoteValue(rewriter, arg);
          argTypes.push_back(newType);
          newArgs.push_back(newArg);
        }

        Type structTy = LLVM::LLVMStructType::getLiteral(ctx, argTypes);
        auto allocated =
            rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), structTy, one,
                                            /*alignment=*/0);

        for (const auto &entry : llvm::enumerate(newArgs)) {
          auto index = i32_val(entry.index());
          auto fieldPtr = gep(ptr_ty(ctx), structTy, allocated,
                              ArrayRef<Value>{zero, index});
          store(entry.value(), fieldPtr);
        }
        bufferPtr = bitcast(allocated, ptr);
      }
    }

    call(funcOp, operands);
  }
};

} // namespace

void mlir::triton::populatePrintOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit) {
  patterns.add<PrintOpConversion>(typeConverter, target, benefit);
}
