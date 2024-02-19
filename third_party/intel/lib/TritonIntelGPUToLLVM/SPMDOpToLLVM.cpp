#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

using ::intel::ConvertTritonGPUOpToLLVMPattern;
using ::intel::ConvertTritonGPUOpToLLVMPatternBase;
using ::intel::TritonGPUToLLVMTypeConverter;

struct GetProgramIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetProgramIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetProgramIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId =
        llGetPid(op.getAxisAsInt(), op->getLoc(),
                 op->getParentOfType<ModuleOp>(), rewriter, target);
    rewriter.replaceOp(op, programId);
    return success();
  }
};

struct GetNumProgramsOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (target == triton::Target::GENX) {
      Location loc = op->getLoc();
      assert(op.getAxis() < 3);

      Value blockId =
          rewriter.create<::mlir::gpu::GridDimOp>(loc, dims[op.getAxis()]);
      rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, i32_ty, blockId);

      return success();
    }

    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    assert(op.getAxis() < 3);
    std::string sreg = numCTAs == 1 ? "%nctaid." : "%nclusterid.";
    sreg.append(1, 'x' + op.getAxis()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    Value numPrograms = LLVM::getSRegValue(rewriter, loc, sreg);
    rewriter.replaceOp(op, numPrograms);
    return success();
  }

  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};
};

struct GetClusterCTAIdOpConversion
    : public ConvertTritonGPUOpToLLVMPattern<
          triton::nvidia_gpu::GetClusterCTAIdOp> {
  using ConvertTritonGPUOpToLLVMPattern<
      triton::nvidia_gpu::GetClusterCTAIdOp>::ConvertTritonGPUOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::GetClusterCTAIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, getClusterCTAId(rewriter, op->getLoc()));
    return success();
  }
};

} // namespace

void mlir::triton::populateSPMDOpToLLVMPattern(
    TritonGPUToLLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    Target target, PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, target, benefit);
  patterns.add<GetNumProgramsOpConversion>(typeConverter, target, benefit);
  patterns.add<GetClusterCTAIdOpConversion>(typeConverter, target, benefit);
}
