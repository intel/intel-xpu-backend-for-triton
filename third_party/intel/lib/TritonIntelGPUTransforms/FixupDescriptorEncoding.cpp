//===- FixupDescriptorEncoding.cpp - Fix blocked encoding for desc ops ----===//
//
// Descriptor load/store operations require a row-major blocked encoding with
// sizePerThread=[1,...,1] so that MaterializeBlockPointer can later lower them
// to 2D block I/O instructions.  Earlier passes (e.g. Coalesce) may assign a
// different blocked encoding based on pointer-access analysis that doesn't
// apply to descriptor ops.  This pass rewrites those encodings and inserts
// ConvertLayoutOps to bridge the old and new layouts.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUFIXUPDESCRIPTORENCODING
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-fixup-descriptor-encoding"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct FixupDescriptorEncodingPass
    : public ttgi::impl::TritonIntelGPUFixupDescriptorEncodingBase<
          FixupDescriptorEncodingPass> {

  void runOnOperation() final {
    ModuleOp mod = getOperation();

    // Opt-in only. Current PVC measurements show that rewriting descriptor
    // encodings to a 2D-block-IO-compatible form and paying the resulting
    // ttg.convert_layout cost does not beat main's vectorized scalar gather
    // path for this workload. Keep the pass available for future revisits
    // (e.g. when IGC or hardware changes shift the perf trade-off) behind
    // TRITON_INTEL_FIXUP_DESCRIPTOR_ENCODING=1.
    if (!tt::tools::getBoolEnv("TRITON_INTEL_FIXUP_DESCRIPTOR_ENCODING"))
      return;

    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return;

    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
    MLIRContext *ctx = &getContext();

    SmallVector<tt::DescriptorLoadOp> loadsToFix;
    mod.walk([&](tt::DescriptorLoadOp op) {
      auto tensorType = dyn_cast<RankedTensorType>(op.getType());
      if (!tensorType)
        return;
      if (!dyn_cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding()))
        return;
      // Only rewrite ops that will actually lower to 2D block IO.
      // MaterializeBlockPointer tags exactly those ops with ttig.block_io after
      // running all alignment/pitch/padding/DPAS-transpose checks.
      if (!op->hasAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName()))
        return;
      loadsToFix.push_back(op);
    });

    SmallVector<tt::DescriptorStoreOp> storesToFix;
    mod.walk([&](tt::DescriptorStoreOp op) {
      auto tensorType = dyn_cast<RankedTensorType>(op.getSrc().getType());
      if (!tensorType)
        return;
      if (!dyn_cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding()))
        return;
      // Only rewrite ops that will actually lower to 2D block IO.
      // MaterializeBlockPointer tags exactly those ops with ttig.block_io after
      // running all alignment/pitch/padding/DPAS-transpose checks.
      if (!op->hasAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName()))
        return;
      storesToFix.push_back(op);
    });

    if (loadsToFix.empty() && storesToFix.empty())
      return;

    for (tt::DescriptorLoadOp op : loadsToFix) {
      auto tensorType = cast<RankedTensorType>(op.getType());
      auto oldEnc = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
      ArrayRef<int64_t> shape = tensorType.getShape();
      unsigned rank = shape.size();

      int numWarps = ttg::lookupNumWarps(op);
      SmallVector<unsigned> sizePerThread(rank, 1);
      SmallVector<unsigned> order =
          ttg::getMatrixOrder(rank, /*rowMajor=*/true);
      auto cgaLayout = oldEnc.getCGALayout();

      auto newEnc =
          ttg::BlockedEncodingAttr::get(ctx, shape, sizePerThread, order,
                                        numWarps, threadsPerWarp, cgaLayout);

      if (newEnc == oldEnc)
        continue;

      LDBG("Fixing descriptor load: " << *op);
      LDBG("  old encoding: " << oldEnc);
      LDBG("  new encoding: " << newEnc);

      auto newType =
          RankedTensorType::get(shape, tensorType.getElementType(), newEnc);
      op.getResult().setType(newType);

      OpBuilder builder(op);
      builder.setInsertionPointAfter(op);
      auto oldType =
          RankedTensorType::get(shape, tensorType.getElementType(), oldEnc);
      auto cvt = ttg::ConvertLayoutOp::create(builder, op.getLoc(), oldType,
                                              op.getResult());
      op.getResult().replaceAllUsesExcept(cvt, cvt);
    }

    for (tt::DescriptorStoreOp op : storesToFix) {
      auto tensorType = cast<RankedTensorType>(op.getSrc().getType());
      auto oldEnc = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
      ArrayRef<int64_t> shape = tensorType.getShape();
      unsigned rank = shape.size();

      int numWarps = ttg::lookupNumWarps(op);
      SmallVector<unsigned> sizePerThread(rank, 1);
      SmallVector<unsigned> order =
          ttg::getMatrixOrder(rank, /*rowMajor=*/true);
      auto cgaLayout = oldEnc.getCGALayout();

      auto newEnc =
          ttg::BlockedEncodingAttr::get(ctx, shape, sizePerThread, order,
                                        numWarps, threadsPerWarp, cgaLayout);

      if (newEnc == oldEnc)
        continue;

      LDBG("Fixing descriptor store: " << *op);
      LDBG("  old encoding: " << oldEnc);
      LDBG("  new encoding: " << newEnc);

      auto newType =
          RankedTensorType::get(shape, tensorType.getElementType(), newEnc);
      OpBuilder builder(op);
      builder.setInsertionPoint(op);
      auto cvt = ttg::ConvertLayoutOp::create(builder, op.getLoc(), newType,
                                              op.getSrc());
      op.getSrcMutable().assign(cvt);
    }
  }
};

} // namespace
