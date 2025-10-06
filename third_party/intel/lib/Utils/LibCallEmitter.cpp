#include "LibCallEmitter.h"

#include "Dialect/TritonIntelGPU/IR/Utils.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace mlir::triton::gpu::intel {

static Value printfPromoteValue(RewriterBase &rewriter, Value value,
                                bool isSigned) {
  auto type = value.getType();
  if (isa<IntegerType>(type) && type.getIntOrFloatBitWidth() == 1) {
    // FIXME: There is some problem when using i1 type now,
    // remove this code once IGC fix the problem.
    TritonLLVMOpBuilder b(rewriter.getUnknownLoc(), rewriter);
    return b.zext(i8_ty, value);
  } else if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
    TritonLLVMOpBuilder b(rewriter.getUnknownLoc(), rewriter);
    if (isSigned) {
      return b.sext(i32_ty, value);
    } else {
      return b.zext(i32_ty, value);
    }
  } else {
    return value;
  }
}

// declare __spirv_ocl_printf(i8*, ...) as external function
static LLVM::LLVMFuncOp getSpirvPrintfDeclaration(RewriterBase &rewriter) {
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

static LLVM::LLVMFuncOp getAssertfailDeclaration(RewriterBase &rewriter) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName = "__assert_fail";
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  // void __assert_fail(const char * assertion, const char * file, unsigned
  // int line, const char * function);
  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType;
  argsType = {ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric),
              ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), i32_ty,
              ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric)};
  auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);

  RewriterBase::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  auto func = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                                funcType);
  func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
  return func;
}

Value LibCallEmitter::getGlobalStringStart(Location loc, RewriterBase &rewriter,
                                           StringRef name, StringRef value,
                                           unsigned addressSpace) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  LLVM::GlobalOp global =
      getGlobalString(loc, rewriter, name, value, addressSpace);
  MLIRContext *ctx = rewriter.getContext();
  Type globalPtrType = ptr_ty(ctx, addressSpace);
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(loc, global);
  return b.gep(globalPtrType, i8_ty, globalPtr, LLVM::GEPArg{0});
}

LLVM::GlobalOp LibCallEmitter::getGlobalString(Location loc,
                                               RewriterBase &rewriter,
                                               StringRef name, StringRef value,
                                               unsigned addressSpace) const {
  StringAttr valueAttr = rewriter.getStringAttr(value);
  std::pair<unsigned, StringAttr> cacheKey{addressSpace, valueAttr};
  auto pos = globals.find(cacheKey);
  if (pos != globals.end())
    return pos->second;

  ModuleOp moduleOp =
      rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();

  llvm::SmallString<64> contentStr(value);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  auto createGlobal = [&](StringRef name) {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    return rewriter.create<LLVM::GlobalOp>(
        rewriter.getUnknownLoc(), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, name, valueAttr,
        /*alignment=*/0, addressSpace);
  };

  LLVM::GlobalOp global =
      moduleOp.lookupSymbol(name)
          ? createGlobal(Twine{name}.concat(Twine{globals.size()}).str())
          : createGlobal(name);

  globals.try_emplace(cacheKey, global);

  return global;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

void LibCallEmitter::printf(RewriterBase &rewriter, Value formatStrStart,
                            int /*formatStrByteCount*/, ValueRange args,
                            ArrayRef<bool> isSigned) const {
  auto *ctx = rewriter.getContext();
  Type ptr = ptr_ty(ctx);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto funcOp = getSpirvPrintfDeclaration(rewriter);
  auto loc = UnknownLoc::get(ctx);
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  SmallVector<Value> operands;
  operands.push_back(formatStrStart);
  for (auto [i, arg] : llvm::enumerate(args)) {
    operands.push_back(printfPromoteValue(
        rewriter, arg, isSigned.empty() ? true : isSigned[i]));
  }
  auto callOp = b.call(funcOp, operands);
  callOp.setCConv(triton::gpu::intel::getRequiredCConv(callOp));
}

void LibCallEmitter::printf(RewriterBase &rewriter, StringRef msg,
                            ValueRange args, ArrayRef<bool> isSigned) const {
  assert(!msg.empty() && "printf with empty string not supported");
  llvm::SmallString<64> msgNewline(msg);
  msgNewline.push_back('\n');
  msgNewline.push_back('\0');
  Value msgValue = getGlobalStringStart(
      rewriter.getUnknownLoc(), rewriter, "printfFormat_", msgNewline,
      /*addressSpace=*/TritonGEN::kUniformConstant);
  printf(rewriter, msgValue, msgNewline.size_in_bytes(), args, isSigned);
}

void LibCallEmitter::assertFail(RewriterBase &rewriter, Location loc,
                                StringRef message, StringRef file,
                                StringRef func, int line) const {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto funcOp = getAssertfailDeclaration(rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  unsigned addrSpace = TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
  llvm::SmallString<64> messageString(message), fileString(file),
      funcString(func);
  messageString.push_back('\0');
  fileString.push_back('\0');
  funcString.push_back('\0');
  Value messageStringVal =
      getGlobalStringStart(loc, rewriter, "assertMessage_", messageString,
                           /*addressSpace=*/TritonGEN::kCrossWorkgroup);
  Value fileStringVal =
      getGlobalStringStart(loc, rewriter, "assertFile_", fileString,
                           /*addressSpace=*/TritonGEN::kCrossWorkgroup);
  Value funcStringVal =
      getGlobalStringStart(loc, rewriter, "assertFunc_", funcString,
                           /*addressSpace=*/TritonGEN::kCrossWorkgroup);
  Value lineNumber = b.i32_val(line);

  auto *ctx = rewriter.getContext();
  SmallVector<Value> operands;
  Value messageStringPtr = b.addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), messageStringVal);
  Value fileStringPtr = b.addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), fileStringVal);
  Value funcStringPtr = b.addrspacecast(
      ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), funcStringVal);
  operands = {messageStringPtr, fileStringPtr, lineNumber, funcStringPtr};
  auto ret = b.call(funcOp, operands);
  ret.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
}

} // namespace mlir::triton::gpu::intel
