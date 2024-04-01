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
             adaptor.getFunc(), adaptor.getLine(), rewriter);
    rewriter.eraseOp(op);
    return success();
  }

  // op: the op at which the assert is inserted. Unlike printf, we need to
  // know about the op to split the block.
  static void llAssert(Operation *op, Value condition, StringRef message,
                       StringRef file, StringRef func, int line,
                       ConversionPatternRewriter &rewriter) {
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

    auto funcOp = getAssertfailDeclaration(rewriter);
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
    unsigned addrSpace = TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
    Value messageString = LLVM::Intel::addStringToModule(
        loc, rewriter, "assertMessage_", message, addrSpace);
    Value fileString = LLVM::Intel::addStringToModule(
        loc, rewriter, "assertFile_", file, addrSpace);
    Value funcString = LLVM::Intel::addStringToModule(
        loc, rewriter, "assertFunc_", func, addrSpace);
    Value lineNumber = i32_val(line);

    SmallVector<Value> operands;
    Value messageStringPtr = addrspacecast(
        ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), messageString);
    Value fileStringPtr = addrspacecast(
        ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), fileString);
    Value funcStringPtr = addrspacecast(
        ptr_ty(ctx, TritonGEN::TritonGENMemorySpace::kGeneric), funcString);
    operands = {messageStringPtr, fileStringPtr, lineNumber, funcStringPtr};
    auto ret = call(funcOp, operands);
    ret.setCConv(LLVM::cconv::CConv::SPIR_FUNC);

    // Split a block after the call.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, condition, ifBlock, thenBlock);
  }

  static LLVM::LLVMFuncOp
  getAssertfailDeclaration(ConversionPatternRewriter &rewriter) {
    auto moduleOp =
        rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
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

    ConversionPatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());

    auto func = rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx),
                                                  funcName, funcType);
    func.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    return func;
  }
};

} // namespace

void mlir::triton::intel::populateAssertOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<AssertOpConversion>(typeConverter, benefit);
}
