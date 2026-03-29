#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-optimize-dot-operands"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUOPTIMIZEDOTOPERANDS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Fuse tt.trans into tt.descriptor_load for dot operands.
//
// The descriptor is always row-major (stride-1 on the last dimension) and is
// never modified. Instead, the DescriptorLoadOp is replaced with one whose
// result type matches the tt.trans result type. The "column_major" block_io
// attribute signals to the lowering that the result dimensions are transposed
// relative to the descriptor's block shape, so it must swap indices and request
// a hardware-transposed 2D block load.
//
// Any single-use chain of ops (e.g. tt.fp_to_fp, ttg.convert_layout) between
// tt.trans and the consuming tt.dot/tt.dot_scaled is preserved intact.
//
// Transform:
//   %desc = tt.make_tensor_desc %base, [%N, %K], [%K_stride, %1]
//         : <tensor<BNxBKxf16>>
//   %load = tt.descriptor_load %desc[%n, %k] {ttig.block_io = "row_major"}
//         : !tt.tensordesc<tensor<BNxBKxf16>> -> tensor<BNxBKxf16>
//   %trans = tt.trans %load : tensor<BKxBN, #blocked>
//   %cvt = ttg.convert_layout %trans : tensor<BKxBN, #dotEnc>
//   tt.dot(%a, %cvt)
// into:
//   %desc = tt.make_tensor_desc %base, [%N, %K], [%K_stride, %1]
//         : <tensor<BNxBKxf16>> (unchanged)
//   %load = tt.descriptor_load %desc[%n, %k] {block_io = "column_major"}
//         : !tt.tensordesc<tensor<BNxBKxf16>> -> tensor<BKxBN, #blocked>
//   %cvt = ttg.convert_layout %load : tensor<BKxBN, #dotEnc>
//   tt.dot(%a, %cvt)
class FuseTransWithDescriptorLoad {
public:
  void run(ModuleOp moduleOp) {
    moduleOp.walk([&](tt::TransOp transOp) {
      if (isCandidate(transOp))
        fuse(transOp);
    });
    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;

  bool isCandidate(tt::TransOp transOp) const {
    assert(transOp && "Expecting a valid transpose operation");

    ModuleOp mod = transOp->getParentOfType<ModuleOp>();
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return false;

    if (transOp->getParentOfType<scf::WhileOp>())
      return false;

    if (!transOp->hasOneUse())
      return false;

    // Walk through an arbitrary single-use chain until a dot operation is
    // reached. This rewrite only replaces the transpose result directly, so
    // intermediate ops such as `ttg.convert_layout` and `tt.fp_to_fp` can
    // remain untouched.
    Operation *user = *transOp->getUsers().begin();
    while (!isa<tt::DotOp, tt::DotScaledOp>(user)) {
      if (!user->hasOneUse())
        return false;
      user = *user->getUsers().begin();
    }

    // Source must be DescriptorLoadOp with single use and rank >= 2.
    auto descLoadOp = dyn_cast_or_null<tt::DescriptorLoadOp>(
        transOp.getSrc().getDefiningOp());
    if (!descLoadOp || !descLoadOp->hasOneUse())
      return false;

    if (cast<RankedTensorType>(descLoadOp.getType()).getRank() < 2)
      return false;

    // Validate that the transpose only swaps the innermost 2 dimensions
    // (identity on outer dims). block_io = "column_major" in the lowering
    // only handles this inner-2-dim swap.
    // TODO: Support general permutations when a richer block_io encoding is
    //       available.
    {
      ArrayRef<int32_t> order = transOp.getOrder();
      unsigned rank = order.size();
      for (unsigned i = 0; i + 2 < rank; ++i) {
        if (order[i] != static_cast<int32_t>(i))
          return false;
      }
      if (order[rank - 2] != static_cast<int32_t>(rank - 1) ||
          order[rank - 1] != static_cast<int32_t>(rank - 2))
        return false;
    }

    // Must be able to find the defining MakeTensorDescOp.
    auto makeTensorDescOp =
        tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
            descLoadOp.getDesc());
    if (!makeTensorDescOp.has_value())
      return false;

    // Only fuse if the descriptor load carries block_io = "row_major", which
    // MaterializeBlockPointer sets to confirm the load is a 2D block IO
    // candidate.
    StringRef blockIOAttrName =
        ttgi::TritonIntelGPUDialect::getBlockIOAttrName();
    StringAttr attr = descLoadOp->getAttrOfType<StringAttr>(blockIOAttrName);
    if (!attr || ttgi::symbolizeBlockIOMode(attr.getValue()) !=
                     ttgi::BlockIOMode::RowMajor)
      return false;

    return true;
  }

  void fuse(tt::TransOp transOp) {
    auto descLoadOp =
        cast<tt::DescriptorLoadOp>(transOp.getSrc().getDefiningOp());

    // Keep the original descriptor — do NOT reverse it.
    // The descriptor is always row-major (stride-1 on last dim).
    auto descType = cast<tt::TensorDescType>(descLoadOp.getDesc().getType());
    RankedTensorType blockType = descType.getBlockType();
    ArrayRef<int64_t> origShape = blockType.getShape();
    ArrayRef<int> perm = transOp.getOrder();
    SmallVector<int64_t> transposedShape(origShape.size());
    for (unsigned i = 0; i < origShape.size(); ++i)
      transposedShape[i] = origShape[perm[i]];

    // Create a new DescriptorLoadOp that directly produces the transpose
    // result type. The verifier allows result shape to differ from the
    // descriptor block shape as long as the total element count matches.
    OpBuilder builder(descLoadOp);
    auto transposedType = cast<RankedTensorType>(transOp.getType());
    auto newResultType =
        RankedTensorType::get(transposedShape, transposedType.getElementType(),
                              transposedType.getEncoding());
    auto newLoad = tt::DescriptorLoadOp::create(
        builder, descLoadOp.getLoc(), newResultType, descLoadOp.getDesc(),
        descLoadOp.getIndices(), descLoadOp.getCache(), descLoadOp.getEvict());

    // Copy any discardable attributes from the original load,
    // except block_io which we set explicitly below.
    StringRef blockIOName = ttgi::TritonIntelGPUDialect::getBlockIOAttrName();
    for (auto attr : descLoadOp->getDiscardableAttrs())
      if (attr.getName() != blockIOName)
        newLoad->setDiscardableAttr(attr.getName(), attr.getValue());

    // Set block_io = column_major: signals that the result type's inner two
    // dimensions are transposed relative to the descriptor's block shape.
    // Outer (batch) dimensions are preserved unchanged.
    // TODO: To support general permutations (not just inner-2-dim swap),
    //       replace column_major with a richer encoding carrying the full
    //       permutation (e.g., an ArrayAttr).
    newLoad->setAttr(blockIOName,
                     StringAttr::get(transOp.getContext(),
                                     ttgi::stringifyBlockIOMode(
                                         ttgi::BlockIOMode::ColumnMajor)));

    // Replace the transpose directly and keep the rest of the single-use chain
    // intact. This is sufficient to support intermediate ops such as
    // `tt.fp_to_fp` without rebuilding them here.
    transOp.replaceAllUsesWith(newLoad.getResult());

    cleanUp.insert(transOp);
    cleanUp.insert(descLoadOp);
    // Do NOT clean up MakeTensorDescOp — it's unchanged, may have other uses
  }
};

} // namespace

class TritonIntelGPUOptimizeDotOperandsPass
    : public ttgi::impl::TritonIntelGPUOptimizeDotOperandsBase<
          TritonIntelGPUOptimizeDotOperandsPass> {

public:
  using ttgi::impl::TritonIntelGPUOptimizeDotOperandsBase<
      TritonIntelGPUOptimizeDotOperandsPass>::
      TritonIntelGPUOptimizeDotOperandsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    FuseTransWithDescriptorLoad descFuser;
    descFuser.run(moduleOp);
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};
