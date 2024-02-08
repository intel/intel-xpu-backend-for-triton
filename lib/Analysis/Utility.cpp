#include "triton/Analysis/Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <deque>

namespace mlir {

namespace {

int getParentAxis(Attribute layout, int axis) {
  if (auto sliceEncoding = layout.dyn_cast<triton::gpu::SliceEncodingAttr>()) {
    axis = axis < sliceEncoding.getDim() ? axis : axis + 1;
    return getParentAxis(sliceEncoding.getParent(), axis);
  }
  return axis;
}

SmallVector<unsigned> getParentOrder(Attribute layout) {
  if (auto sliceEncoding = layout.dyn_cast<triton::gpu::SliceEncodingAttr>()) {
    return getParentOrder(sliceEncoding.getParent());
  }
  return triton::gpu::getOrder(layout);
}

} // namespace

bool ReduceOpHelper::isReductionOnLayoutFastAxis() {
  return getParentAxis(getSrcLayout(), axis) ==
         getParentOrder(getSrcLayout())[0];
}

SmallVector<unsigned> ReduceOpHelper::getOrderWithAxisAtBeginning() {
  auto srcLayout = getSrcLayout();
  auto order = triton::gpu::getOrder(srcLayout);
  auto it = std::find(order.begin(), order.end(), axis);
  // delete the axis from order
  order.erase(it);
  // insert axis at the beginning of order
  order.insert(order.begin(), axis);
  return order;
}

// Thread offset is the thread index offset of two adjacent threads on the
// reduction axis within the warp.
unsigned ReduceOpHelper::getThreadOffsetOnReductionAxis() {
  auto srcLayout = getSrcLayout();

  // If the reduction axis is the fast axis of the parent layout
  if (isReductionOnLayoutFastAxis()) {
    return 1;
  }

  unsigned threadOffset = 1;
  if (auto sliceLayout =
          srcLayout.dyn_cast<mlir::triton::gpu::SliceEncodingAttr>()) {
    auto parentLayout = sliceLayout.getParent();
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(parentLayout);
    threadOffset = threadsPerWarp[sliceLayout.getDim()];
  } else {
    auto threadsPerWarp = triton::gpu::getThreadsPerWarp(srcLayout);
    auto order = triton::gpu::getOrder(srcLayout);
    for (unsigned i = 0; i < order.size(); i++) {
      if (order[i] == axis)
        break;
      threadOffset *= threadsPerWarp[order[i]];
    }
  }
  return threadOffset;
}

// Cases where distributed shared memory is not required in ConvertLayout:
// (1) numCTAs == 1
// (2) numCTAs > 1 but srcCTALayout == dstCTALayout
// TODO: Case with SliceLayout as srcLayout and numCTAs > 1 is to be implemented
// in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout) {
  unsigned numCTAs = triton::gpu::getNumCTAs(srcLayout);
  assert(numCTAs == triton::gpu::getNumCTAs(dstLayout) &&
         "Invalid layout conversion: the numbers of CTAs of src and dst "
         "layouts are different");

  // Case (1): Never use dsmem when numCTAs == 1
  if (numCTAs == 1)
    return false;

  // Case where CTAsPerCGA of srcLayout in the sliced dim is not 1 is not
  // implemented yet
  if (auto sliceLayout = srcLayout.dyn_cast<triton::gpu::SliceEncodingAttr>()) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      llvm::report_fatal_error("Layout conversion to be implemented");
  }

  // Case where CTAsPerCGA of dstLayout in the sliced dim is not 1 is supported
  if (auto sliceLayout = dstLayout.dyn_cast<triton::gpu::SliceEncodingAttr>()) {
    auto dim = sliceLayout.getDim();
    auto CTAsPerCGA = triton::gpu::getCTAsPerCGA(sliceLayout.getParent());
    if (CTAsPerCGA[dim] != 1)
      return true;
  }

  // The above two branches make sure that it is legal to call getCTALayout of
  // srcLayout and dstLayout

  // Case (2): Do not use dsmem when srcCTALayout == dstCTALayout
  auto srcCTALayout = triton::gpu::getCTALayout(srcLayout);
  auto dstCTALayout = triton::gpu::getCTALayout(dstLayout);
  if (srcCTALayout == dstCTALayout)
    return false;

  // Dsmem access is required when srcCTALayout != dstCTALayout
  return true;
}

unsigned ReduceOpHelper::getInterWarpSize() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned sizeIntraWarps = getIntraWarpSize();
  return std::min(srcReduceDimSize / sizeIntraWarps,
                  triton::gpu::getWarpsPerCTA(getSrcLayout())[axis]);
}

unsigned ReduceOpHelper::getIntraWarpSize() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  return std::min(srcReduceDimSize,
                  triton::gpu::getThreadsPerWarp(getSrcLayout())[axis]);
}

unsigned ReduceOpHelper::getInterWarpSizeWithUniqueData() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned sizeIntraWarps = getIntraWarpSizeWithUniqueData();
  return std::min(srcReduceDimSize / sizeIntraWarps,
                  triton::gpu::getWarpsPerCTAWithUniqueData(
                      getSrcLayout(), getSrcShape())[axis]);
}

unsigned ReduceOpHelper::getIntraWarpSizeWithUniqueData() {
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned elementPerThreads = triton::gpu::getUniqueContigPerThread(
      getSrcLayout(), getSrcShape())[axis];
  return std::min(srcReduceDimSize / elementPerThreads,
                  triton::gpu::getThreadsPerWarpWithUniqueData(
                      getSrcLayout(), getSrcShape())[axis]);
}

unsigned ReduceOpHelper::getThreadsReductionAxis() {
  auto srcLayout = getSrcLayout();
  auto srcShape = getSrcShape();
  return triton::gpu::getThreadsPerWarpWithUniqueData(srcLayout,
                                                      srcShape)[axis] *
         triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape)[axis];
}

bool ReduceOpHelper::isWarpSynchronous() {
  auto srcLayout = getSrcLayout();
  auto srcShape = getSrcShape();
  return triton::gpu::getWarpsPerCTAWithUniqueData(srcLayout, srcShape)[axis] ==
         1;
}

SmallVector<unsigned> ReduceOpHelper::getScratchConfig() {
  SmallVector<unsigned> smemShape;
  // that case doesn't need inter-warp communication
  if (isWarpSynchronous())
    return {0, 0};

  smemShape = convertType<unsigned>(getSrcShape());
  smemShape[axis] = getInterWarpSizeWithUniqueData();

  return smemShape;
}

unsigned ReduceOpHelper::getScratchSizeInBytes() {
  auto smemShape = getScratchConfig();
  auto elems = product<unsigned>(smemShape);

  unsigned bytesPerElem = 0;
  for (const auto &ty : srcElementTypes) {
    bytesPerElem += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return bytesPerElem * elems;
}

bool ReduceOpHelper::isReduceWithinCTA() {
  auto axis = getAxis();
  auto srcLayout = getSrcLayout();
  auto CTASplitNum = mlir::triton::gpu::getCTASplitNum(srcLayout);
  assert(axis < CTASplitNum.size());
  return CTASplitNum[axis] == 1;
}

bool ReduceOpHelper::isSupportedLayout() {
  // Layout optimization passes such as PlanCTAPass and
  // RemoveLayoutConversionPass should avoid cross-CTA reduction
  if (!isReduceWithinCTA()) {
    return false;
  }

  auto srcLayout = getSrcLayout();
  if (srcLayout.isa<triton::gpu::BlockedEncodingAttr>()) {
    return true;
  }
  if (auto mmaLayout = srcLayout.dyn_cast<triton::gpu::MmaEncodingTrait>()) {
    return mmaLayout.supportReduction();
  }
  if (auto sliceLayout = srcLayout.dyn_cast<triton::gpu::SliceEncodingAttr>()) {
    return true;
  }
  return false;
}

unsigned ScanLoweringHelper::getAxisNumElementsPerThread() {
  return getEncoding().getSizePerThread()[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumElementsPerThread() {
  SmallVector<unsigned> sizePerThreads =
      triton::gpu::getContigPerThread(getEncoding());
  sizePerThreads[getAxis()] = 1;
  return product<unsigned>(sizePerThreads);
}

Region &ScanLoweringHelper::getCombineOp() { return scanOp.getCombineOp(); }

unsigned ScanLoweringHelper::getAxisNumThreadsPerWarp() {
  return triton::gpu::getThreadsPerWarp(getEncoding())[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumThreadsPerWarpWithUniqueData() {
  return triton::gpu::getThreadsPerWarpWithUniqueData(getEncoding(),
                                                      getShape())[getAxis()];
}

unsigned ScanLoweringHelper::getNonAxisNumThreadsPerWarp() {
  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(getEncoding());
  threadsPerWarp[getAxis()] = 1;
  return product<unsigned>(threadsPerWarp);
}

// Return the flat numbers of threads computing independent scan results.
unsigned ScanLoweringHelper::getNonAxisNumThreadsPerCTA() {
  unsigned numParallelThreadsPerWarp = getNonAxisNumThreadsPerWarp();
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(getEncoding());
  warpsPerCTA[getAxis()] = 1;
  unsigned numParallelWarpsPerCTA = product<unsigned>(warpsPerCTA);
  return numParallelThreadsPerWarp * numParallelWarpsPerCTA;
}

unsigned ScanLoweringHelper::getAxisNumWarps() {
  return triton::gpu::getWarpsPerCTA(getEncoding())[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumWarpsWithUniqueData() {
  return triton::gpu::getWarpsPerCTAWithUniqueData(getEncoding(),
                                                   getShape())[getAxis()];
}

unsigned ScanLoweringHelper::getAxisNumBlocks() {
  auto sizePerThreads = triton::gpu::getSizePerThread(getEncoding());
  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(getEncoding());
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(getEncoding());
  unsigned axis = getAxis();
  return ceil<unsigned>(
      getShape()[axis],
      (sizePerThreads[axis] * threadsPerWarp[axis] * warpsPerCTA[axis]));
}

unsigned ScanLoweringHelper::getNonAxisNumBlocks() {
  auto sizePerThreads = triton::gpu::getSizePerThread(getEncoding());
  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(getEncoding());
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(getEncoding());
  unsigned axis = getAxis();
  unsigned numBlocks = 1;
  for (unsigned i = 0; i < sizePerThreads.size(); i++) {
    if (i == axis)
      continue;
    numBlocks *=
        ceil<unsigned>(getShape()[i], (sizePerThreads[i] * threadsPerWarp[i] *
                                       warpsPerCTA[i]));
  }
  return numBlocks;
}

bool ScanLoweringHelper::isSupported() {
  // TODO: Support the following cases:
  // 1. Scan on non-blocking encodings
  if (!isa<triton::gpu::BlockedEncodingAttr>(getEncoding()))
    return false;
  return true;
}

unsigned ScanLoweringHelper::getScratchSizeInElems() {
  auto mod = scanOp->getParentOfType<ModuleOp>();
  unsigned numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
  unsigned numNonAxisElementsPerWarp =
      getNonAxisNumThreadsPerWarp() * getNonAxisNumElementsPerThread();
  unsigned numElements = numWarps * numNonAxisElementsPerWarp *
                         getAxisNumBlocks() * getNonAxisNumBlocks();
  return numElements;
}

unsigned ScanLoweringHelper::getScratchSizeInBytes() {
  unsigned axisNumWarps = getAxisNumWarpsWithUniqueData();
  if (axisNumWarps == 1)
    return 0;
  unsigned elementSizeInBytes = 0;
  for (const auto &ty : srcElementTypes) {
    elementSizeInBytes += ceil<unsigned>(ty.getIntOrFloatBitWidth(), 8);
  }
  return elementSizeInBytes * getScratchSizeInElems();
}

triton::gpu::BlockedEncodingAttr ScanLoweringHelper::getEncoding() {
  return srcEncoding.cast<triton::gpu::BlockedEncodingAttr>();
}

unsigned ScanLoweringHelper::getAxisElementStride() {
  auto order = triton::gpu::getOrder(getEncoding());
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= getContigPerThread(getEncoding())[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisThreadStride() {
  auto order = triton::gpu::getOrder(getEncoding());
  unsigned stride = 1;
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= getEncoding().getThreadsPerWarp()[dim];
  }
  llvm_unreachable("Axis not found in order");
}

unsigned ScanLoweringHelper::getAxisBlockStride() {
  auto order = triton::gpu::getOrder(getEncoding());
  unsigned stride = 1;
  auto sizePerThreads = triton::gpu::getSizePerThread(getEncoding());
  auto threadsPerWarp = triton::gpu::getThreadsPerWarp(getEncoding());
  auto warpsPerCTA = triton::gpu::getWarpsPerCTA(getEncoding());
  for (unsigned dim : order) {
    if (dim == getAxis())
      return stride;
    stride *= ceil<unsigned int>(getShape()[dim], sizePerThreads[dim] *
                                                      threadsPerWarp[dim] *
                                                      warpsPerCTA[dim]);
  }
  llvm_unreachable("Axis not found in order");
}

bool maybeSharedAllocationOp(Operation *op) {
  // TODO(Keren): This function can be replaced by adding
  // MemoryEffectOpInterface. We can then use the MemoryEffectOpInterface to
  // query the memory effects of the op.
  auto *dialect = op->getDialect();
  return dialect &&
         (dialect->getTypeID() ==
              mlir::TypeID::get<triton::gpu::TritonGPUDialect>() ||
          dialect->getTypeID() ==
              mlir::TypeID::get<triton::nvidia_gpu::TritonNvidiaGPUDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<triton::TritonDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<arith::ArithDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<tensor::TensorDialect>());
}

bool maybeAliasOp(Operation *op) {
  return isa<triton::gpu::ExtractSliceOp, triton::TransOp,
             triton::gpu::InsertSliceAsyncOp,
             triton::nvidia_gpu::InsertSliceTMAOp,
             triton::nvidia_gpu::StoreAsyncTMAOp, tensor::InsertSliceOp>(op);
}

static bool supportMFMAGranularity(int m, int n, int k) {
  // these limitations are dtype dependent, in future we may relax them
  const static std::pair<int, int> mfmaTypes[2] = {{32, 8}, {16, 16}};
  for (const auto &mfmaType : mfmaTypes) {
    auto [granularityMN, granularityK] = mfmaType;
    if (m % granularityMN != 0 || n % granularityMN != 0)
      continue;
    if (k % granularityK != 0)
      continue;
    return true;
  }
  return false;
}

bool supportMFMATypes(Type a, Type b) {
  if (a.getIntOrFloatBitWidth() != b.getIntOrFloatBitWidth())
    return false;

  auto F8E4M3FNUZ = TypeID::get<mlir::Float8E4M3FNUZType>();
  auto F8E5M2FNUZ = TypeID::get<mlir::Float8E5M2FNUZType>();
  auto F16 = TypeID::get<mlir::Float16Type>();
  auto BF16 = TypeID::get<mlir::BFloat16Type>();
  auto F32 = TypeID::get<mlir::Float32Type>();
  auto Int = TypeID::get<mlir::IntegerType>();
  DenseSet<std::pair<mlir::TypeID, mlir::TypeID>> supportedTypes = {
      {F32, F32},
      {F16, F16},
      {BF16, BF16},
      {F8E4M3FNUZ, F8E4M3FNUZ},
      {F8E4M3FNUZ, F8E5M2FNUZ},
      {F8E5M2FNUZ, F8E4M3FNUZ},
      {F8E5M2FNUZ, F8E5M2FNUZ},
      {Int, Int}};

  if (!supportedTypes.contains({a.getTypeID(), b.getTypeID()}))
    return false;

  if (a.isIntOrIndex() && a.getIntOrFloatBitWidth() != 8)
    return false;
  return true;
}

bool supportMFMA(triton::DotOp op) {
  auto aTy = op.getA().getType().cast<RankedTensorType>();
  auto bTy = op.getB().getType().cast<RankedTensorType>();

  auto aElemTy = aTy.getElementType();
  auto bElemTy = bTy.getElementType();

  if (!supportMFMATypes(aElemTy, bElemTy))
    return false;

  auto aShape = aTy.getShape();
  auto bShape = bTy.getShape();

  assert(aShape[1] == bShape[0]);
  if (!supportMFMAGranularity(aShape[0], bShape[1], aShape[1]))
    return false;

  return true;
}

bool supportMMA(triton::DotOp op, int version) {
  // Refer to mma section for the data type supported by Volta and Hopper
  // Tensor Core in
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
  auto aElemTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bElemTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  if (version == 3) {
    if (::triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return false;
    auto retType = op.getResult().getType().cast<RankedTensorType>();
    auto retShapePerCTA = triton::gpu::getShapePerCTA(retType);
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
    if (!(numWarps % 4 == 0 && retShapePerCTA[0] % 64 == 0 &&
          retShapePerCTA[1] % 8 == 0 &&
          (aElemTy.isFloat8E5M2() || aElemTy.isFloat8E4M3FNUZ() ||
           aElemTy.isInteger(8) || aElemTy.isF16() || aElemTy.isBF16() ||
           aElemTy.isF32()))) {
      return false;
    }
    // We cannot use MMA_V3 if we need to accumulate in F32 within the MMA op.
    if (op.getMaxNumImpreciseAcc() < 32 &&
        (aElemTy.isFloat8E5M2() || aElemTy.isFloat8E4M3FNUZ()) &&
        op.getType().cast<RankedTensorType>().getElementType().isF32()) {
      return false;
    }
  }
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getAllowTF32() && version >= 2;
  }
  return supportMMA(op.getA(), version) && supportMMA(op.getB(), version);
}

bool supportMMA(Value value, int version) {
  // Tell whether a DotOp support MMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2 || version == 3) &&
         "Unexpected MMA layout version found");
  auto elemTy = value.getType().cast<RankedTensorType>().getElementType();
  // FP8 is not natively supported on all mma versions but it can always be
  // promoted to fp16 therefore we can always support it.
  bool isFP8 = elemTy.isFloat8E5M2() || elemTy.isFloat8E4M3FN() ||
               elemTy.isFloat8E5M2FNUZ() || elemTy.isFloat8E4M3FNUZ() ||
               elemTy.isFloat8E4M3B11FNUZ();
  return isFP8 || elemTy.isF16() || elemTy.isBF16() ||
         (elemTy.isF32() && version >= 2) ||
         (elemTy.isInteger(8) && version >= 2);
}

bool supportDPAS(triton::DotOp op) {
  auto ATy = op.getA().getType().cast<RankedTensorType>();
  auto BTy = op.getB().getType().cast<RankedTensorType>();
  auto CTy = op.getC().getType().cast<RankedTensorType>();
  auto DTy = op.getResult().getType().cast<RankedTensorType>();

  Type AElemTy = ATy.getElementType(), BElemTy = BTy.getElementType(),
       CElemTy = CTy.getElementType(), DElemTy = DTy.getElementType();

  if (AElemTy != BElemTy || CElemTy != DElemTy)
    return false;

  if (!AElemTy.isF16() && !AElemTy.isBF16() && !AElemTy.isF32() &&
      !AElemTy.isInteger(8))
    return false;

  if (!CElemTy.isF32() && !CElemTy.isInteger(32))
    return false;

  // Supported types element types combinations are:
  //   Operand  A     B     C    D
  //  -------------------------------
  //   elemTy   f16   f16   f32  f32
  //   elemTy   bf16  bf16  f32  f32
  //   elemTy   i8    i8    i32  i32
  //   elemTy   tf32  tf32  f32  f32  Note: tf32 represented via f32

  if ((AElemTy.isF16() || AElemTy.isBF16()) && !CElemTy.isF32())
    return false;
  if (AElemTy.isInteger(8) && !CElemTy.isInteger(32))
    return false;
  if (AElemTy.isF32() && !CElemTy.isF32())
    return false;

  return true;
}

bool isMfmaToDotShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy) {
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mfmaLayout = srcLayout.cast<triton::gpu::MfmaEncodingAttr>();
  auto dotOperandLayout = dstLayout.cast<triton::gpu::DotOperandEncodingAttr>();
  // TODO: Remove the restriction on the warpsPerCTA once chain dot testing is
  // improved. In addition, we can enable this shortcut for regular MFMA
  // layout when opIdx == 1.
  return mfmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 &&
         dotOperandLayout.getKWidth() == 4 &&
         dotOperandLayout.getParent() == mfmaLayout &&
         mfmaLayout.getNonKDim() == 32 && mfmaLayout.getIsTransposed() &&
         (srcTy.getElementType().isF16() || srcTy.getElementType().isBF16());
}

static bool isMmaToMmaShortcut(Attribute srcEncoding, Attribute dstEncoding) {
  auto src = srcEncoding.dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>();
  auto dst = dstEncoding.dyn_cast<triton::gpu::NvidiaMmaEncodingAttr>();
  if (!src || !dst)
    return false;
  // when #mma = MmaEncoding<version=3, warpsPerCTA=[..., 1]>
  return src && dst && src.getVersionMajor() == 3 &&
         src.getWarpsPerCTA()[1] == 1 && dst.getVersionMajor() == 3 &&
         dst.getWarpsPerCTA()[1] == 1;
}

bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  return isMmaToMmaShortcut(srcTy.getEncoding(), dstTy.getEncoding());
}

// For MMAV3 dotOperand layout matches mma operand for f16 and bf16 cases.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy) {
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mmaLayout = srcLayout.cast<triton::gpu::NvidiaMmaEncodingAttr>();
  auto dotOperandLayout = dstLayout.cast<triton::gpu::DotOperandEncodingAttr>();
  auto ans =
      mmaLayout.getVersionMajor() == 3 && dotOperandLayout.getOpIdx() == 0 &&
      isMmaToMmaShortcut(dotOperandLayout.getParent(), srcLayout) &&
      (srcTy.getElementType().isF16() || srcTy.getElementType().isBF16());
  return ans;
}

bool isMmaToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  if (matchMmaV3AndDotOperandLayout(srcTy, dstTy))
    return true;
  // dot_op<opIdx=0, parent=#mma> = #mma
  // when #mma = MmaEncoding<version=2, warpsPerCTA=[..., 1]>
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mmaLayout = srcLayout.cast<triton::gpu::NvidiaMmaEncodingAttr>();
  auto dotOperandLayout = dstLayout.cast<triton::gpu::DotOperandEncodingAttr>();
  return mmaLayout.getVersionMajor() == 2 &&
         mmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 &&
         dotOperandLayout.getParent() == mmaLayout &&
         !srcTy.getElementType().isF32();
}

namespace {

/// A data structure similar to SetVector but maintains
/// a deque instead of a vector to allow for efficient
/// push_back and pop_front operations.
/// Using SetVector doesn't suffice our needs because
/// it only pushes and pops from the back.
/// For example, if we have a queue like this:
/// 0->4 1->2->3
///    ^--------
/// where 3 depends on 4, once we pop 3, we found
/// 4 is not ready, so we check 2 and push 3 back
/// to the queue.
struct DFSSubgraphState {
  DFSSubgraphState() : set(), deque() {}
  DenseSet<Operation *> set;
  std::deque<Operation *> deque;

  bool push_back(Operation *op) {
    if (set.insert(op).second) {
      deque.push_back(op);
      return true;
    }
    return false;
  }

  Operation *pop_front() {
    Operation *op = deque.front();
    deque.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return deque.empty(); }
};

/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set) : toSort(set), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;

  /// We mark each op as ready if all its operands and parents ops are seen. If
  /// an op is ready, we add it to the queue. Otherwise, we keep adding its
  /// operands to the ancestors set.
  /// We always want an op to be scheduled after all its parents to handle
  /// correctly cases with scf operations.
  void addToReadyQueue(Operation *op, DFSSubgraphState &subGraph,
                       SmallVector<Operation *, 4> &readyQueue) {
    bool ready = true;
    for (Value operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (def && !seen.count(def)) {
        subGraph.push_back(def);
        ready = false;
      }
    }
    Operation *parent = op->getParentOp();
    while (parent) {
      if (!seen.count(parent)) {
        subGraph.push_back(parent);
        ready = false;
      }
      parent = parent->getParentOp();
    }
    if (ready)
      readyQueue.push_back(op);
  }
};

void dfsPostorder(Operation *root, DFSState *state) {
  DFSSubgraphState subGraph;
  subGraph.push_back(root);
  SmallVector<Operation *> ops;
  while (!subGraph.empty()) {
    // Nodes in the ready queue are ready to be processed.
    // Meaning that either their operands are all seen or they have null
    // operands.
    SmallVector<Operation *, 4> readyQueue;
    auto *current = subGraph.pop_front();
    state->addToReadyQueue(current, subGraph, readyQueue);
    while (!readyQueue.empty()) {
      Operation *current = readyQueue.pop_back_val();
      if (!state->seen.insert(current).second)
        continue;
      ops.push_back(current);
      for (Value result : current->getResults()) {
        for (Operation *op : result.getUsers())
          state->addToReadyQueue(op, subGraph, readyQueue);
      }
      for (Region &region : current->getRegions()) {
        for (Operation &op : region.getOps())
          state->addToReadyQueue(&op, subGraph, readyQueue);
      }
    }
  }

  for (Operation *op : llvm::reverse(ops)) {
    if (state->toSort.count(op) > 0)
      state->topologicalCounts.push_back(op);
  }
}

} // namespace

SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter backwardFilter,
                                         TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    mlir::BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    getBackwardSlice(currentOp, &backwardSlice, opt);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

namespace {
// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
// interacts with constant propagation, but SparseConstantPropagation
// doesn't seem to be sufficient.
class ConstantAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    WalkResult result = top->walk([&](Operation *op) {
      if (failed(visit(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult visit(ProgramPoint point) override {
    Operation *op = point.get<Operation *>();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(
          op->getResult(0));
      propagateIfChanged(constant, constant->join(dataflow::ConstantValue(
                                       value, op->getDialect())));
      return success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // https://github.com/llvm/llvm-project/blob/2ec1aba2b69faa1de5f71832a48e25aa3b5d5314/mlir/lib/Analysis/DataFlow/DeadCodeAnalysis.cpp#L322
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks())
        setAllToUnknownConstants(block.getArguments());
    }
    return success();
  }

private:
  /// Set all given values as not constants.
  void setAllToUnknownConstants(ValueRange values) {
    dataflow::ConstantValue unknownConstant(nullptr, nullptr);
    for (Value value : values) {
      auto *constant =
          getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(value);
      propagateIfChanged(constant, constant->join(unknownConstant));
    }
  }
};
} // namespace

std::unique_ptr<DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<DataFlowSolver>();
  solver->load<dataflow::DeadCodeAnalysis>();
  solver->load<ConstantAnalysis>();
  return solver;
}

static triton::MakeTensorPtrOp getMakeTensorPtrOpImpl(Operation *op, Value v) {

  if (auto makeTensorPtrOp = dyn_cast<triton::MakeTensorPtrOp>(op)) {
    return makeTensorPtrOp;
  }

  if (auto advanceOp = dyn_cast<triton::AdvanceOp>(op)) {
    return getMakeTensorPtrOp(advanceOp.getPtr());
  }

  if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
    auto idx = v.cast<OpResult>().getResultNumber();
    llvm::SmallVector<scf::YieldOp> yieldOps;
    op->walk([&](Operation *op) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op))
        yieldOps.push_back(yieldOp);
    });

    // benzh@ if multi yields, all yields operand should come from same arg.
    Value newValue = yieldOps[0].getOperands()[idx];
    return getMakeTensorPtrOp(newValue);
  }

  llvm_unreachable("Unable to getMakeTensorPtr()");
}

triton::MakeTensorPtrOp getMakeTensorPtrOp(Value v) {
  using BranchOps = llvm::SetVector<std::pair<Operation *, int>>;
  llvm::DenseMap<Block *, BranchOps> blockToCFOps;
  auto moduleOp =
      v.getParentBlock()->getParentOp()->getParentOfType<ModuleOp>();

  moduleOp.walk([&](Operation *op) {
    if (auto br = dyn_cast<cf::BranchOp>(op)) {
      Block *block = br.getDest();
      blockToCFOps[block].insert({op, -1});
    }
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op)) {
      Block *blockT = condBr.getTrueDest();
      Block *blockF = condBr.getFalseDest();
      blockToCFOps[blockT].insert({condBr, 1});
      blockToCFOps[blockF].insert({condBr, 0});
    }
  });

  if (Operation *definingOp = v.getDefiningOp())
    return getMakeTensorPtrOpImpl(definingOp, v);

  // If there is no defining op, v must be a BlockArgument.
  BlockArgument arg = v.cast<BlockArgument>();
  unsigned argNum = arg.getArgNumber();
  Operation *argOwner = arg.getOwner()->getParentOp();

  if (auto forOp = dyn_cast<scf::ForOp>(argOwner))
    return getMakeTensorPtrOp(
        forOp.getOperand(argNum + forOp.getNumControlOperands() - 1));
  if (auto funcOp = dyn_cast<mlir::triton::FuncOp>(argOwner)) {
    Block *block = arg.getOwner();
    Operation *op;
    int tOrF;
    std::tie(op, tOrF) = blockToCFOps[block][0];
    if (auto br = dyn_cast<cf::BranchOp>(op))
      return getMakeTensorPtrOp(br.getDestOperands()[argNum]);
    if (auto condBr = dyn_cast<cf::CondBranchOp>(op))
      return getMakeTensorPtrOp(tOrF ? condBr.getTrueDestOperands()[argNum]
                                     : condBr.getFalseDestOperands()[argNum]);
    return getMakeTensorPtrOp(argOwner->getOperand(argNum));
  }
  llvm_unreachable("Unable to getMakeTensorPtr()");
}

} // namespace mlir
