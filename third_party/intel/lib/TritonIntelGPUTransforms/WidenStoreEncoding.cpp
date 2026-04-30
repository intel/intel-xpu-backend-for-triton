#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonintelgpu-widen-store-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUWIDENSTOREENCODING
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Compute a widened BlockedEncodingAttr for `op`'s value tensor, or return
// nullptr if the store does not qualify (not a ranked tensor, not blocked,
// current per-thread width != 128 bits, insufficient alignment/contiguity,
// or the widened tile would not fit in the CTA shape).
Attribute tryWidenStoreEncoding(tt::StoreOp op,
                                tt::intel::ModuleAxisInfoAnalysis &axisInfo) {
  Value val = op.getValue();
  auto valTy = dyn_cast<RankedTensorType>(val.getType());
  if (!valTy)
    return nullptr;

  auto enc = dyn_cast<ttg::BlockedEncodingAttr>(valTy.getEncoding());
  if (!enc)
    return nullptr;

  ArrayRef<unsigned> order = enc.getOrder();
  unsigned fastAxis = order[0];
  unsigned elemBits = mlir::getElementBitWidth(valTy);
  ArrayRef<unsigned> sizePerThread = enc.getSizePerThread();
  int64_t curSizePerThread = sizePerThread[fastAxis];

  // Widen only from 128b to 256b per thread.
  if (curSizePerThread * elemBits != 128) {
    LDBG("skip: per-thread bits = " << curSizePerThread * elemBits
                                    << " (not 128b)");
    return nullptr;
  }

  int64_t newSizePerThread = 2 * curSizePerThread;

  // getDivisibility and getContiguity are in elements, not bytes.
  Value ptr = op.getPtr();
  tt::AxisInfo *info = axisInfo.getAxisInfo(ptr);
  if (!info)
    return nullptr;
  if (info->getDivisibility(fastAxis) < newSizePerThread) {
    LDBG("skip: divisibility " << info->getDivisibility(fastAxis) << " < "
                               << newSizePerThread << " needed");
    return nullptr;
  }
  if (info->getContiguity(fastAxis) < newSizePerThread) {
    LDBG("skip: contiguity " << info->getContiguity(fastAxis) << " < "
                             << newSizePerThread << " needed");
    return nullptr;
  }

  // Shape gate: the widened tile must fit in the per-CTA shape along
  // `fastAxis`.
  SmallVector<int64_t> shapePerCTA = ttg::getShapePerCTA(valTy);
  unsigned threadsPerWarp = enc.getThreadsPerWarp()[fastAxis];
  unsigned warpsPerCTA = enc.getWarpsPerCTA()[fastAxis];
  int64_t needed = newSizePerThread * threadsPerWarp * warpsPerCTA;
  if (shapePerCTA[fastAxis] < needed) {
    LDBG("skip: shapePerCTA " << shapePerCTA[fastAxis] << " < " << needed
                              << " needed");
    return nullptr;
  }

  SmallVector<unsigned> widenedSizePerThread(sizePerThread.begin(),
                                             sizePerThread.end());
  widenedSizePerThread[fastAxis] = newSizePerThread;
  LDBG("widening store encoding on op: " << *op << " sizePerThread[" << fastAxis
                                         << "] " << curSizePerThread << " -> "
                                         << newSizePerThread);
  return ttg::BlockedEncodingAttr::get(
      op.getContext(), widenedSizePerThread, enc.getThreadsPerWarp(),
      enc.getWarpsPerCTA(), order, enc.getCGALayout());
}

struct TritonIntelGPUWidenStoreEncodingPass
    : public ttgi::impl::TritonIntelGPUWidenStoreEncodingBase<
          TritonIntelGPUWidenStoreEncodingPass> {
  using ttgi::impl::TritonIntelGPUWidenStoreEncodingBase<
      TritonIntelGPUWidenStoreEncodingPass>::
      TritonIntelGPUWidenStoreEncodingBase;

  void runOnOperation() final {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport256bLoadStoreAttrName()))
      return;

    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Collect candidates first: convertDistributedOpEncoding rewrites the
    // store in place (erasing the old op), which would invalidate a walk.
    SmallVector<std::pair<Operation *, Attribute>> rewrites;
    mod.walk([&](tt::StoreOp op) {
      if (Attribute newEnc = tryWidenStoreEncoding(op, axisInfoAnalysis))
        rewrites.emplace_back(op, newEnc);
    });

    for (auto &[op, newEnc] : rewrites)
      mlir::convertDistributedOpEncoding(newEnc, op);
  }
};

} // namespace
