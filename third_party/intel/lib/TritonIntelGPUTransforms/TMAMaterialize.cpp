#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#include "include/triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include <memory>

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUTMALOWERINGPASS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::gpu::intel;

/// Holds the values related to a block pointer.
/// It includes the base pointer, base width and height, row and column
/// stride, and offset base for X and Y.
struct BlockPointerValues {
  Value base;
  Value baseWidth;
  Value baseHeight;
  Value rowStride;
  Value colStride;
};

// FIXME: Only supports 2D matrices for now.
BlockPointerValues getValuesFromTMADescStruct(Location loc, Value tmaDescPtr,
                                              PatternRewriter &rewriter,
                                              Type baseType) {
  MLIRContext *ctx = rewriter.getContext();
  //    struct TMA_desc {
  //      void* base;
  //      uint64_t shape[5];
  //      uint64_t strides[5];
  //    };

  triton::PointerType tmaPtrType =
      cast<triton::PointerType>(tmaDescPtr.getType());
  auto addressSpace = tmaPtrType.getAddressSpace();

  auto rank = 5;
  SmallVector<Type, 4> types;
  // base ptr
  auto basePtrType = LLVM::LLVMPointerType::get(ctx, 1);
  types.push_back(basePtrType);
  // offsets + strides
  for (auto i = 0; i < rank * 2; i++) {
    types.push_back(IntegerType::get(ctx, 64));
  }
  auto TMA_desc = LLVM::LLVMStructType::getLiteral(ctx, types);
  auto descLLVMPtrType = LLVM::LLVMPointerType::get(ctx, addressSpace);
  auto castPtr = rewriter.create<UnrealizedConversionCastOp>(
      loc, descLLVMPtrType, tmaDescPtr);
  SmallVector<LLVM::GEPArg> indices{0, 0};
  Value basePtr = gep(descLLVMPtrType, TMA_desc, castPtr.getResult(0), indices);
  indices[1] = LLVM::GEPArg{1 + 3}; // shape[3];
  Value baseHeightPtr =
      gep(descLLVMPtrType, TMA_desc, castPtr.getResult(0), indices);
  indices[1] = LLVM::GEPArg{1 + 4}; // shape[4];
  Value baseWidthPtr =
      gep(descLLVMPtrType, TMA_desc, castPtr.getResult(0), indices);
  indices[1] = LLVM::GEPArg{1 + 5 + 3}; // strides[3];
  Value rowStridePtr =
      gep(descLLVMPtrType, TMA_desc, castPtr.getResult(0), indices);
  Value base = load(basePtrType, basePtr);
  auto tritonBase =
      rewriter.create<UnrealizedConversionCastOp>(loc, baseType, base);

  BlockPointerValues values{
      .base = tritonBase.getResult(0),
      .baseWidth = load(int_ty(64), baseHeightPtr),
      .baseHeight = load(int_ty(64), baseWidthPtr),
      .rowStride = load(int_ty(64), rowStridePtr),
      .colStride = rewriter.create<arith::ConstantIntOp>(
          loc, 1, 64), // TMA descriptor is always row major.
  };
  return values;
}

class TMALoadLowering : public OpRewritePattern<ExperimentalDescriptorLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExperimentalDescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto tensorType = op.getResult().getType();
    auto blockPtrType = triton::PointerType::get(tensorType, 1);
    auto tritonPtrType =
        triton::PointerType::get(tensorType.getElementType(), 1);

    auto [base, baseWidth, baseHeight, rowStride, colStride] =
        getValuesFromTMADescStruct(loc, op.getDescPtr(), rewriter,
                                   tritonPtrType);

    auto rank = tensorType.getRank();
    std::vector<Value> shape;
    std::vector<Value> strides;
    DenseI32ArrayAttr order;
    if (rank == 1) {
      shape = {baseWidth};
      strides = {colStride};
      order = rewriter.getDenseI32ArrayAttr({0});
    } else {
      shape = {baseHeight, baseWidth};
      strides = {rowStride, colStride};
      order = rewriter.getDenseI32ArrayAttr({1, 0});
    }

    Value newBlockPtr = rewriter.create<MakeTensorPtrOp>(
        loc, blockPtrType, base, shape, strides, op.getIndices(), order);
    rewriter.replaceOpWithNewOp<LoadOp>(op, newBlockPtr, CacheModifier::NONE,
                                        EvictionPolicy::NORMAL, false);

    return success();
  }
};

class TMAStoreLowering
    : public OpRewritePattern<ExperimentalDescriptorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExperimentalDescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    auto loc = op.getLoc();
    auto tensorType = op.getSrc().getType();
    auto blockPtrType = triton::PointerType::get(tensorType, 1);
    auto tritonPtrType =
        triton::PointerType::get(tensorType.getElementType(), 1);

    auto [base, baseWidth, baseHeight, rowStride, colStride] =
        getValuesFromTMADescStruct(loc, op.getDescPtr(), rewriter,
                                   tritonPtrType);

    auto rank = tensorType.getRank();
    std::vector<Value> shape;
    std::vector<Value> strides;
    DenseI32ArrayAttr order;
    if (rank == 1) {
      shape = {baseWidth};
      strides = {colStride};
      order = rewriter.getDenseI32ArrayAttr({0});
    } else {
      shape = {baseHeight, baseWidth};
      strides = {rowStride, colStride};
      order = rewriter.getDenseI32ArrayAttr({1, 0});
    }

    auto newBlockPtr = rewriter.create<MakeTensorPtrOp>(
        loc, blockPtrType, base, shape, strides, op.getIndices(), order);
    rewriter.replaceOpWithNewOp<StoreOp>(op, newBlockPtr, op.getSrc(),
                                         CacheModifier::NONE,
                                         EvictionPolicy::NORMAL);

    return success();
  }
};

class TritonIntelGPUTMALoweringPass
    : public triton::gpu::intel::impl::TritonIntelGPUTMALoweringPassBase<
          TritonIntelGPUTMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<TMALoadLowering, TMAStoreLowering>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace
