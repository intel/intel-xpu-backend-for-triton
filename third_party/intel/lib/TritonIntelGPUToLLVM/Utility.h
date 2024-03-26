#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_UTILITY_H

#include "triton/Conversion/TritonGPUToLLVM/Utility2.h"
#include "intel/include/TritonIntelGPUToLLVM/PTXAsmFormat.h"

#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#undef sub
#define sub(...) rewriter.create<LLVM::SubOp>(loc, __VA_ARGS__)
#undef store
#define store(...) rewriter.create<LLVM::StoreOp>(loc, __VA_ARGS__)
#define barSync(rewriter, op, bar, numThreads)                                 \
  do {                                                                         \
    ::mlir::triton::intel::PTXBuilder ptxBuilder;                              \
    auto &barSyncOp = *ptxBuilder.create<>("bar.sync");                        \
    barSyncOp(ptxBuilder.newConstantOperand(bar),                              \
              ptxBuilder.newConstantOperand(numThreads));                      \
    auto voidTy = void_ty(op->getContext());                                   \
    ptxBuilder.launch(rewriter, op->getLoc(), voidTy);                         \
  } while (0)

// #undef call
#define call(...) rewriter.create<LLVM::CallOp>(loc, __VA_ARGS__)
#undef addrspacecast
#define addrspacecast(...)                                                     \
  rewriter.create<LLVM::AddrSpaceCastOp>(loc, __VA_ARGS__)

// Constants
#define f16_val(...) LLVM::utils::createConstantF16(loc, rewriter, __VA_ARGS__)
#undef f32_val
#define f32_val(...) LLVM::utils::createConstantF32(loc, rewriter, __VA_ARGS__)
#undef f64_val
#define f64_val(...) LLVM::utils::createConstantF64(loc, rewriter, __VA_ARGS__)
#undef i32_val
#define i32_val(...) LLVM::utils::createConstantI32(loc, rewriter, __VA_ARGS__)
#define i64_val(...) LLVM::utils::createConstantI64(loc, rewriter, __VA_ARGS__)
#undef int_val
#define int_val(width, val)                                                    \
  LLVM::utils::createLLVMIntegerConstant(rewriter, loc, width, val)

namespace mlir {
namespace triton {
// namespace intel {

// } // namespace intel
} // namespace triton

namespace LLVM {
namespace utils {
using namespace mlir::triton;

/// Create a predicated block, using \p cond as the condition and \p ops for the
/// values supplied by the conditional branch to the exit block. The \p
/// thenOpsFn function is used to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2(%ops)
///   ^br1:
///     %then_ops = `thenOpsFn()`
///     cf.br ^br2(%then_ops)
///   ^br2(%block_ops):
template <typename ThenOpsFn>
Block &createPredicatedBlock(ConversionPatternRewriter &rewriter, Location loc,
                             Value cond, ArrayRef<Value> ops,
                             ThenOpsFn &&thenOpsFn) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *thenBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *endBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());

  rewriter.setInsertionPointToEnd(insertionBlock);
  rewriter.create<cf::CondBranchOp>(loc, cond, thenBlock, endBlock, ops);

  rewriter.setInsertionPointToStart(thenBlock);
  auto thenOps = thenOpsFn();
  assert(thenOps.size() == ops.size() && "Inconsistent size");
  assert(llvm::all_of(llvm::enumerate(ops, thenOps),
                      [](const auto &enumerator) {
                        auto [index, op, thenOp] = enumerator;
                        return op.getType() == thenOp.getType();
                      }) &&
         "type mismatch found");

  if (thenOps.empty())
    rewriter.create<cf::BranchOp>(loc, endBlock);
  else
    rewriter.create<cf::BranchOp>(loc, endBlock, thenOps);

  for (Value op : thenOps)
    endBlock->addArgument(op.getType(), op.getLoc());

  rewriter.setInsertionPointToStart(endBlock);
  return *endBlock;
}

/// Create a predicated block, using \p cond as the condition and \p thenOpsFn
/// to inject operations in the 'then' branch:
///   cf.cond_br %cond, ^br1, ^br2
///   ^br1:
///     `thenOpsFn()`
///     cf.br ^br2
///   ^br2:
template <typename ThenOpsFn>
Block &createPredicatedBlock(ConversionPatternRewriter &rewriter, Location loc,
                             Value cond, ThenOpsFn &&thenOpsFn) {
  return createPredicatedBlock(rewriter, loc, cond, {}, thenOpsFn);
}

/// Create a 32-bit integer constant.
Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v);

/// Create a 64-bit integer constant.
Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v);

/// Create a 16-bit float constant.
Value createConstantF16(Location loc, OpBuilder &rewriter, float v);

/// Create a 32-bit float constant.
Value createConstantF32(Location loc, OpBuilder &rewriter, float v);

/// Create a 64-bit float constant.
Value createConstantF64(Location loc, OpBuilder &rewriter, double v);

/// Create NaN constant of specified type.
Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type);

/// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          TypeConverter *converter, int64_t value);

/// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value);

/// Usage of macro load_dsmem
/// (1) load_dsmem(addr, ctaId)
/// (2) load_dsmem(addr, ctaId, vec)
Value createLoadDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Type elemTy);
SmallVector<Value> createLoadDSmem(Location loc, PatternRewriter &rewriter,
                                   Value addr, Value ctaId, unsigned vec,
                                   Type elemTy);

/// Usage of macro store_dsmem
/// (1) store_dsmem(addr, ctaId, value, pred)
/// (2) store_dsmem(addr, ctaId, value)
/// (3) store_dsmem(addr, ctaId, values, pred)
/// (4) store_dsmem(addr, ctaId, values)
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value, Value pred);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, Value value);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values, Value pred);
void createStoreDSmem(Location loc, PatternRewriter &rewriter, Value addr,
                      Value ctaId, ArrayRef<Value> values);

/// Helper function to get strides from a given shape and its order
SmallVector<Value> getStridesFromShapeAndOrder(ArrayRef<int64_t> shape,
                                               ArrayRef<unsigned> order,
                                               Location loc,
                                               RewriterBase &rewriter);

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape,
                ArrayRef<unsigned> order);

Value linearize(ConversionPatternRewriter &rewriter, Location loc,
                ArrayRef<Value> multiDim, ArrayRef<unsigned> shape);

Value storeShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                  Value val, Value pred);

Value loadShared(ConversionPatternRewriter &rewriter, Location loc, Value ptr,
                 Type elemTy, Value pred);

Value shflSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
               int i);
Value shflUpSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                 int i);
Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  int i);
Value shflIdxSync(Location loc, ConversionPatternRewriter &rewriter, Value val,
                  Value i);

Value llGetPid(Location loc, ConversionPatternRewriter &rewriter,
               ModuleOp moduleOp, int axis);

Value addStringToModule(Location loc, ConversionPatternRewriter &rewriter,
                        StringRef key, StringRef content,
                        unsigned addressSpace);

static bool isKernel(FunctionOpInterface funcOp) {
  return funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

static Value getStackPointer(PatternRewriter &rewriter,
                             FunctionOpInterface funcOp) {
  auto mod = funcOp->getParentOfType<ModuleOp>();
  LLVM::LLVMPointerType ptrTy = ptr_ty(
      rewriter.getContext(), TritonGEN::TritonGENMemorySpace::kWorkgroup);
  if (mod->getAttrOfType<IntegerAttr>("triton_gpu.shared").getInt() == 0)
    return rewriter.create<LLVM::UndefOp>(funcOp.getLoc(), ptrTy);
  return funcOp.getArgument(funcOp.getNumArguments() - 1);
}

static Value getSharedMemoryBase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Operation *op) {
  auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
  FunctionOpInterface func =
      op->template getParentOfType<FunctionOpInterface>();
  assert(op->hasAttr("allocation.offset"));
  size_t offset = op->getAttr("allocation.offset")
                      .cast<IntegerAttr>()
                      .getValue()
                      .getZExtValue();
  Value offVal = i32_val(offset);
  Value base =
      gep(ptrTy, i8_ty, LLVM::utils::getStackPointer(rewriter, func), offVal);
  return base;
}

// Returns a Value for the format string, which you can reuse.
Value llPrintf(ConversionPatternRewriter &rewriter, StringRef msg,
               ValueRange args);

void llPrintf(ConversionPatternRewriter &rewriter, Value msg, ValueRange args);

} // namespace utils
} // namespace LLVM

static Value getModuleWarpSize(RewriterBase &rewriter, Location loc) {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  return i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
}

// -----------------------------------------------------------------------
// Shared memory utilities
// -----------------------------------------------------------------------
// using ::mlir::LLVM::utils::delinearize;
// using ::mlir::LLVM::utils::SharedMemoryObject;
using ::mlir::triton::getMultiDimIndex;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type);

static void
emitOffsetForDpasLayoutPerCTA(const DpasEncodingAttr &dpasLayout,
                              SmallVector<SmallVector<unsigned>> &offsets,
                              unsigned ctaOffsetX, unsigned ctaOffsetY) {
  SmallVector<unsigned> sizePerThreads = getSizePerThread(dpasLayout);
  uint32_t elemsPerThreadPerGroup = product<unsigned>(sizePerThreads);
  uint32_t rowsPerWarp =
      dpasLayout.getSubGroupSize() / dpasLayout.getExecutionSize();
  SmallVector<unsigned> shapePerCTA =
      triton::gpu::getShapePerCTATile(dpasLayout);

  for (unsigned elem = 0; elem < elemsPerThreadPerGroup; elem++) {
    uint32_t elemRowIndex = (elem / sizePerThreads[1]) * rowsPerWarp;
    uint32_t elemColIndex = elem % sizePerThreads[1];
    offsets.push_back({ctaOffsetX + elemRowIndex, ctaOffsetY + elemColIndex});
  }
}

// -----------------------------------------------------------------------
// Dpas layout indices
// -----------------------------------------------------------------------

static SmallVector<Value>
emitBaseIndexForDpasLayout(Location loc, RewriterBase &rewriter,
                           const DpasEncodingAttr &dpasLayout,
                           RankedTensorType type) {
  Value threadId = getThreadId(rewriter, loc);
  Value warpSize = i32_val(triton::gpu::getWarpSize(dpasLayout));
  Value warpId = udiv(threadId, warpSize);
  Value laneId = urem(threadId, warpSize);

  auto warpsPerCTA = dpasLayout.getWarpsPerCTA();
  ArrayRef<int64_t> shape = type.getShape();

  auto order = triton::gpu::getOrder(dpasLayout);
  SmallVector<Value> multiDimWarpId =
      delinearize(rewriter, loc, warpId, warpsPerCTA, order);

  // Compute the 2-dim coordinates of the warp containing the tensor element
  // operated on by this thread.
  SmallVector<unsigned> warpShape = dpasLayout.getShapeC();
  Value rowWarpId =
      urem(multiDimWarpId[0], i32_val(std::ceil(shape[0] / warpShape[0])));
  Value colWarpId =
      urem(multiDimWarpId[1], i32_val(std::ceil(shape[1] / warpShape[1])));
  Value rowWarpOffset = mul(rowWarpId, i32_val(warpShape[0]));
  Value colWarpOffset = mul(colWarpId, i32_val(warpShape[1]));

  // Compute the 2-dim coordinates of the first element in the warp operated
  // on by this thread.
  SmallVector<unsigned> threadsPerWarp = getThreadsPerWarp(dpasLayout);
  SmallVector<Value> multiDimBase = {
      add(udiv(laneId, i32_val(threadsPerWarp[1])), rowWarpOffset),
      add(urem(laneId, i32_val(threadsPerWarp[1])), colWarpOffset)};
  return multiDimBase;
}

} // namespace mlir

#endif
