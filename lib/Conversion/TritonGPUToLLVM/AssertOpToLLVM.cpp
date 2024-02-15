#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct AssertOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::AssertOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::AssertOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();
    auto elems = unpackLLElements(loc, adaptor.getCondition(), rewriter);
    auto elemTy = elems[0].getType();
    Value condition = int_val(elemTy.getIntOrFloatBitWidth(), 0);
    for (auto elem : elems) {
      if (elemTy.isSignedInteger() || elemTy.isSignlessInteger()) {
        condition =
            or_(condition,
                icmp_eq(elem, rewriter.create<LLVM::ConstantOp>(
                                  loc, elemTy, rewriter.getZeroAttr(elemTy))));
      } else {
        assert(false && "Unsupported type for assert");
        return failure();
      }
    }
    llAssert(op, condition, adaptor.getMessage(), adaptor.getFile(),
             adaptor.getFunc(), adaptor.getLine(), rewriter, target);
    rewriter.eraseOp(op);
    return success();
  }

  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void llAssert(Operation *op, Value condition, StringRef message,
                       StringRef file, StringRef func, int line,
                       ConversionPatternRewriter &rewriter, Target target) {
    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    auto ctx = rewriter.getContext();
    auto loc = op->getLoc();

    // #block1
    // if (condition) {
    //   #block2
    //   __assertfail(message);
    // }
    // #block3
    Block *prevBlock = op->getBlock();
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    auto funcOp = getAssertfailDeclaration(rewriter, target);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    unsigned addrSpace =
        (target == Target::GENX) ? GENX::GENXMemorySpace::kCrossWorkgroup : 0;
    Value messageString = LLVM::addStringToModule(
        loc, rewriter, "assertMessage_", message, addrSpace);
    Value fileString =
        LLVM::addStringToModule(loc, rewriter, "assertFile_", file, addrSpace);
    Value funcString =
        LLVM::addStringToModule(loc, rewriter, "assertFunc_", func, addrSpace);
    Value lineNumber = i32_val(line);

    SmallVector<Value> operands;
    if (target == Target::GENX) {
      Value messageStringPtr = addrspacecast(
          ptr_ty(ctx, GENX::GENXMemorySpace::kGeneric), messageString);
      Value fileStringPtr = addrspacecast(
          ptr_ty(ctx, GENX::GENXMemorySpace::kGeneric), fileString);
      Value funcStringPtr = addrspacecast(
          ptr_ty(ctx, GENX::GENXMemorySpace::kGeneric), funcString);
      operands = {messageStringPtr, fileStringPtr, lineNumber, funcStringPtr};
    } else {
      Value charSize = int_val(sizeof(size_t) * 8, sizeof(char));
      operands = {messageString, fileString, lineNumber, funcString,
                  int_val(sizeof(size_t) * 8, sizeof(char))};
    }
    auto ret = call(funcOp, operands);

    if (target == Target::GENX) {
      ret.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    }

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }

  static LLVM::LLVMFuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter, Target target) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    StringRef funcName =
        (target == Target::GENX) ? "__assert_fail" : "__assertfail";
    Operation *funcOp = moduleOp.lookupSymbol(funcName);
    if (funcOp)
      return cast<LLVM::LLVMFuncOp>(*funcOp);

    // void __assert_fail(const char * assertion, const char * file, unsigned
    // int line, const char * function);
    auto *ctx = rewriter.getContext();
    SmallVector<Type> argsType;
    if (target == Target::GENX) {
      argsType = {ptr_ty(ctx, GENX::GENXMemorySpace::kGeneric),
                  ptr_ty(ctx, GENX::GENXMemorySpace::kGeneric), i32_ty,
                  ptr_ty(ctx, GENX::GENXMemorySpace::kGeneric)};
    } else {
      argsType = {ptr_ty(ctx), ptr_ty(ctx), i32_ty, ptr_ty(ctx),
                  rewriter.getIntegerType(sizeof(size_t) * 8)};
    }
    auto funcType = LLVM::LLVMFunctionType::get(void_ty(ctx), argsType);

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto func = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx),
                                                  funcName, funcType);
    if (target == Target::GENX) {
      func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    }
    return func;
  }
};

} // namespace

void mlir::triton::populateAssertOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit) {
  patterns.add<AssertOpConversion>(typeConverter, target, benefit);
}
