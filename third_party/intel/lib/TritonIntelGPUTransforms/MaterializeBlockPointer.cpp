#include "intel/include/Analysis/AxisInfo.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <optional>

#define DEBUG_TYPE "tritonintelgpu-materialize-block-pointer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUMATERIALIZEBLOCKPOINTER
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

class BlockedToSubgroup2DBlock : public mlir::OpRewritePattern<tt::LoadOp> {
  private:
    tt::intel::ModuleAxisInfoAnalysis &axisAnalysisPass;

public:
  BlockedToSubgroup2DBlock(mlir::MLIRContext *context,
                           tt::intel::ModuleAxisInfoAnalysis &axisAnalysisPass,
                           int benefit)
      : OpRewritePattern<tt::LoadOp>(context, benefit), axisAnalysisPass(axisAnalysisPass) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {
    LDBG("Considering op: " << loadOp);

    Value ptr = loadOp.getPtr();
    if (!tt::isTensorPointerType(ptr.getType()))
      return MaterializeTensorOfPointers(loadOp);

    if (!isa<RankedTensorType>(loadOp.getResult().getType())) {
      loadOp.emitError() << "Expected 'loadOp' to load a tensor value.";
      return failure();
    }

    // Find the make tensor ptr operation that created the base ptr.
    tt::MakeTensorPtrOp makeTensorPtrOp = tt::getMakeTensorPtrOp(ptr);
    if (!makeTensorPtrOp) {
      LDBG("Could not find make tensor ptr op for: " << loadOp);
      return success();
    }
    LDBG("Make tensor ptr op: " << makeTensorPtrOp);

    Operation::operand_range shape = makeTensorPtrOp.getShape();
    unsigned rank = shape.size();
    LDBG("Rank: " << rank);
    if (rank == 1)
      return success();

    if (!satisfies2DBlockReadAlignment(loadOp)) {
      LDBG("Alignment checks failed for: " << loadOp);
      return success();
    }

    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    LDBG("elementWidth: " << elementWidth);

    Operation::operand_range strides = makeTensorPtrOp.getStrides();
    std::optional<unsigned> strideOneDim = getStrideOneDim(makeTensorPtrOp);
    assert((strideOneDim && strideOneDim.value() < strides.size()) &&
           "Expected strideOneDim to be set and less than strides.size()");
    unsigned strideOneDimVal = strideOneDim.value();

    if (strideOneDimVal == rank - 2 && elementWidth == 8) {
      // TODO: column major layout w/ fp8 has performance regression
      return success();
    }

    if (strideOneDimVal >= (rank - 2)) {
      // HW 2D block read instruction only supports contiguous access.
      Value fastChangeStride = strides[strideOneDimVal];
      if (!tt::intel::isConstant(fastChangeStride, 1))
        return success();

      // Across Intel platforms, the strictest pitch restriction is to be a
      // multiple of OWord(128 bits).
      Value pitch =
          strides[(strideOneDimVal == rank - 1) ? rank - 2 : rank - 1];
      LDBG("Pitch: " << pitch);
      if (!ttgi::isDivisible(pitch, 128 / elementWidth))
        return success();

      const bool isRowMajor = (strideOneDimVal == rank - 1);
      std::optional<ttg::DotOperandEncodingAttr> dotLayout =
          getDotLayout(loadOp);
      if (dotLayout) {
        // Check if the load is being used by a tt.dot operation, and if so is
        // this the first operand and is it a transposed row major matrix. If
        // so, skip the block ptr attribute as performance is worse than if we
        // remove the tensor pointer.
        LDBG("dotLayout: " << *dotLayout);
        auto opIdx =
            static_cast<ttgi::DpasEncodingAttr::OpIdx>(dotLayout->getOpIdx());
        auto dotOrder = tt::gpu::getThreadOrder(tensorType);
        const bool valueRowMajor = (dotOrder[0] == 1 && dotOrder[1] == 0);
        if (opIdx == ttgi::DpasEncodingAttr::OpIdx::OperandA &&
            valueRowMajor ^ isRowMajor) {
          LDBG("Skipping block pointer attribute for transposed A matrix in "
               "dot operation");
          return success();
        }
      } else {
        llvm::errs() << "loadOp: " << loadOp << "\n";
        assert(false && "missing dot layout?");
      }

      llvm::errs() << "dotLayout: " << dotLayout << "\n";
      auto dpasLayout =
          dyn_cast<ttgi::DpasEncodingAttr>(dotLayout->getParent());
      if (dpasLayout) {
        llvm::errs() << "dpasLayout: " << dpasLayout << "\n";

        Type eltTy = tensorType.getElementType();
        unsigned elemSizeInBits = eltTy.getIntOrFloatBitWidth();

        auto tileParams =
            ttgi::Subgroup2DBlockEncodingAttr::getInstrShapeForLayout(
                *dotLayout, tensorType.getShape(), isRowMajor,
                elemSizeInBits / 8, loadOp.getContext());
        auto subgroup2DBlockLayout = ttgi::Subgroup2DBlockEncodingAttr::get(
            loadOp.getContext(), dpasLayout.getWarpsPerCTA(),
            mlir::triton::gpu::getCTALayout(dpasLayout),
            {tileParams[0], tileParams[1]}, tileParams[2],
            mlir::triton::gpu::getOrderForDotOperand(dotLayout->getOpIdx(),
                                                     /*rank*/ 2,
                                                     /*kContig*/ true),
            elemSizeInBits / 8, dpasLayout.getThreadsPerWarp());
        llvm::errs() << "subgroup2DBlockLayout: " << subgroup2DBlockLayout
                     << "\n";
        auto attrs = loadOp->getAttrs();
        llvm::errs() << "oldLoad: " << loadOp << "\n";
        // TODO: replace the load or add a layout conversion? 
        // auto newLoad = rewriter.create<LoadOp>(loadOp.getLoc(),
        // loadOp.getArgs(), attrs); llvm::errs() << "newLoad: " << newLoad <<
        // "\n";
      } else {
        llvm::errs() << "Failed to find dpas layout for : " << dotLayout << "\n";
      }
      #if 0
      loadOp->setAttr(
          ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
          StringAttr::get(loadOp.getContext(), isRowMajor ? "row_major" : "column_major"));
      #endif 
    }
    return success();
  }

private:
  mlir::LogicalResult MaterializeTensorOfPointers(
      tt::LoadOp loadOp) const {
    MLIRContext *context = loadOp.getContext();
    Value ptr = loadOp.getPtr();
    assert(!tt::isTensorPointerType(ptr.getType()) &&
           "Expected 'loadOp' to load a tensor value.");

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy)
      return success();

    LDBG("Considering tensor of pointer load op: " << loadOp);

    if (loadOp.getMask()) {
      LDBG("Load op has mask, skip block IO attribute");
      return success();
    }

    // The axis info gives the information about the value of the indices
    // tensor. For example, if the indices tensor is tensor<8x16xi32> and
    // its value is:
    //   [[ 0,  1,  2,  3, ..., 12, 13, 14, 15],
    //    [16, 17, 18, 19, ..., 28, 29, 30, 31],
    //    ...
    //    [ 96,  97,  98,  99, ..., 108, 109, 110, 111],
    //    [112, 113, 114, 115, ..., 124, 125, 126, 127]]
    // Then the global memory refer by the tensor pointer is row-major
    // contiguous. And the axis info will be: stride: [16, 1],
    // contiguity: [1, 16], divisibility: [1, 16], constancy: [1, 1].
    const tt::AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
    unsigned rank = axisInfo->getRank();
    if (rank != 2) {
      LDBG("Rank is not 2, skip block IO attribute");
      return success();
    }

    // Determine if LoadOp is row-major or column-major.
    auto isMajor = [&](unsigned fastChangeDim) {
      assert((fastChangeDim == 0 || fastChangeDim == 1) &&
             "fastChangeDim is expected to be 0 or 1");
      const unsigned otherDim = !fastChangeDim;
      // Limit to full row being contiguous.
      if (axisInfo->getContiguity(fastChangeDim) !=
          tensorTy.getDimSize(fastChangeDim)) {
        LDBG("Found non-contiguous row: "
             << axisInfo->getContiguity(fastChangeDim));
        return false;
      }

      // Value -1 is used to represent the unknown stride.
      if (axisInfo->getStride(otherDim) <= 0) {
        LDBG("Found unknown or non positive stride: "
             << axisInfo->getStride(otherDim));
        return false;
      }

      // Surface pitch is required to be 16 bytes aligned.
      Type elemTy =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned elemSizeInBytes = elemTy.getIntOrFloatBitWidth() / 8;
      if ((axisInfo->getStride(otherDim) * elemSizeInBytes) % 16 != 0) {
        LDBG("Found Non 16 bytes aligned stride: "
             << axisInfo->getStride(otherDim));
        return false;
      }

      // Base pointer can be compensate by the offset and base width, where they
      // each has restriction that it has to be 4 bytes aligned.
      if (axisInfo->getDivisibility(fastChangeDim) % 4 != 0) {
        LDBG(
            "Found Non 4 bytes aligned base: " << axisInfo->getDivisibility(1));
        return false;
      }

      return true;
    };

    // Check if loadOp is row major, i.e., fast changing dimension is one.
    if (isMajor(1 /*fastChangeDim*/)) {
      LDBG("Setting row_major attribute\n");
      loadOp->setAttr(ttgi::TritonIntelGPUDialect::getBlockIOAttrName(),
                      StringAttr::get(context, "row_major"));
    }

    // TODO: set column_major attribute
    return success();
  }

  // Return the load layout if it is a dot layout. If it is not, check if the
  // load result is converted to a dot layout. If so, return the dot layout,
  // otherwise return nullopt.
  std::optional<ttg::DotOperandEncodingAttr>
  getDotLayout(tt::LoadOp loadOp) const {
    Value ptr = loadOp.getPtr();
    if (!tt::isTensorPointerType(ptr.getType()))
      return std::nullopt;

    RankedTensorType tensorType = ttgi::getRankedTensorType(ptr.getType());
    if (!tensorType)
      return std::nullopt;

    auto dotLayout = ttgi::getDotEncoding(tensorType);
    if (dotLayout)
      return dotLayout;

    auto allUsersAreConvertOps = [](Operation::user_range users) {
      return llvm::all_of(users, [](Operation *user) {
        return isa<ttg::ConvertLayoutOp>(user);
      });
    };

    auto allUserHaveIdenticalLayout = [](Operation::user_range users) {
      Attribute firstUserLayout =
          cast<ttg::ConvertLayoutOp>(*users.begin()).getType().getEncoding();
      return llvm::all_of(users, [&firstUserLayout](Operation *user) {
        return firstUserLayout ==
               cast<ttg::ConvertLayoutOp>(user).getType().getEncoding();
      });
    };

    Operation::user_range users = loadOp->getUsers();
    if (!users.empty() && allUsersAreConvertOps(users) &&
        allUserHaveIdenticalLayout(users)) {
      Attribute firstUserLayout =
          cast<ttg::ConvertLayoutOp>(*users.begin()).getType().getEncoding();
      if (isa<ttg::DotOperandEncodingAttr>(firstUserLayout))
        return dyn_cast<ttg::DotOperandEncodingAttr>(firstUserLayout);
      return std::nullopt;
    }

    return std::nullopt;
  }

  std::optional<unsigned>
  getStrideOneDim(tt::MakeTensorPtrOp makeTensorPtrOp) const {
    assert(makeTensorPtrOp && "Expected a make tensor ptr op.");
    Operation::operand_range strides = makeTensorPtrOp.getStrides();
    std::optional<unsigned> strideOneDim{std::nullopt};
    for (auto [idx, stride] : llvm::enumerate(strides)) {
      if (!tt::intel::isConstant(stride, 1))
        continue;
      strideOneDim = idx;
      break;
    }
    return strideOneDim;
  }

  bool satisfies2DBlockReadAlignment(
      tt::LoadOp loadOp) const {
    Value ptr = loadOp.getPtr();
    assert(tt::isTensorPointerType(ptr.getType()) &&
           "Expected a ptr to a tensor of ptrs.");
    assert(isa<RankedTensorType>(loadOp.getResult().getType()) &&
           "Expected 'loadOp' to load a ranked tensor value.");

    // Find the make tensor ptr operation that created the base ptr for the load
    // operation.
    tt::MakeTensorPtrOp makeTensorPtrOp = tt::getMakeTensorPtrOp(ptr);
    assert(makeTensorPtrOp && "Expected a make tensor ptr op.");

    Operation::operand_range shape = makeTensorPtrOp.getShape();
    if (shape.size() == 1)
      return false;

    std::optional<unsigned> strideOneDim = getStrideOneDim(makeTensorPtrOp);
    if (!strideOneDim) {
      LDBG("Could not find stride one dimension in: " << makeTensorPtrOp);
      return false;
    }

    auto ptrType = cast<tt::PointerType>(makeTensorPtrOp.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
    unsigned elementWidth = tensorType.getElementTypeBitWidth();
    unsigned strideOneDimVal = strideOneDim.value();
    LDBG("strideOneDim: " << strideOneDimVal);

    // Ensure the base ptr is 4-byte aligned.
    // Note: the HW requires the address to be 64-byte aligned, however we will
    // compensate by imposing restrictions on the offsetX and baseWidth.
    const tt::AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(ptr);
    if (axisInfo->getDivisibility(strideOneDimVal) % 4 != 0) {
      LDBG("Found non 4 bytes aligned base: "
           << axisInfo->getDivisibility(strideOneDimVal));
      return false;
    }

    // Analyze the shape of the stride one dimension to ensure it satisfies HW
    // constraints.
    Value baseWidth = tt::intel::getFinalValue(shape[strideOneDimVal]);
    unsigned divisor = std::ceil(32 / elementWidth);
    if (!ttgi::isDivisible(baseWidth, divisor)) {
      LLVM_DEBUG({
        llvm::dbgs() << "baseWidth does not satisfies HW constraint: ";
        baseWidth.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\ndivisor: " << divisor << "\n";
      });
      return false;
    }
    LDBG("baseWidth: " << baseWidth);

    // Analyze the initial offset corresponding to the stride one dimension to
    // ensure it satisfies HW constraints.
    Value offset =
        tt::intel::getFinalValue(makeTensorPtrOp.getOffsets()[strideOneDimVal]);
    if (!ttgi::isDivisible(offset, divisor)) {
      LLVM_DEBUG({
        llvm::dbgs() << "offset does not satisfies HW constraints: ";
        offset.printAsOperand(llvm::dbgs(), {});
        llvm::dbgs() << "\ndivisor: " << divisor << "\n";
      });
      return false;
    }
    LDBG("offset: " << offset);

    Region *loadRgn = loadOp->getParentRegion();
    Region *makeTensorPtrRgn = makeTensorPtrOp->getParentRegion();
    bool inSameRegion = (loadRgn == makeTensorPtrRgn);
    if (inSameRegion)
      return satisfies2DBlockReadAlignment(offset, divisor);

    // TODO: analyze tt.advance (issue #3762).

    return true;
  }

  bool satisfies2DBlockReadAlignment(Value offset, unsigned divisor) const {
    assert(divisor != 0 && "Expected divisor to be non-zero");

    auto checkUsers = [&](Value::user_range users) {
      return llvm::all_of(users, [&](Operation *user) {
        if (isa<tt::MakeTensorPtrOp>(user))
          return true;
        if (Operation *addOp = dyn_cast<arith::AddIOp>(user)) {
          auto other = llvm::find_if(addOp->getOperands(),
                                     [&](Value op) { return op != offset; });
          if (!ttgi::isDivisible(*other, divisor)) {
            LDBG("Found a non-divisible increment: " << *addOp);
            return false;
          }
          return true;
        }
        LDBG("Unhandled user kind: " << user);
        return false;
      });
    };

    // Ensure that the offset is incremented by a multiple of the divisor.
    if (auto blockArg = dyn_cast<BlockArgument>(offset))
      return checkUsers(blockArg.getUsers());

    return true;
  }
};

struct TritonIntelGPUMaterializeBlockPointerPass
    : public triton::gpu::intel::impl::
          TritonIntelGPUMaterializeBlockPointerBase<
              TritonIntelGPUMaterializeBlockPointerPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUMaterializeBlockPointerBase<
      TritonIntelGPUMaterializeBlockPointerPass>::
      TritonIntelGPUMaterializeBlockPointerBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!mod->hasAttr(
            ttgi::TritonIntelGPUDialect::getSupportSG2DBlockAttrName()))
      return;
    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    MLIRContext *context = &getContext();
    constexpr int benefitDefault = 1;

    mlir::RewritePatternSet patterns(context);
    patterns.add<BlockedToSubgroup2DBlock>(context, axisInfoAnalysis,
                                           benefitDefault);
    if (applyPatternsGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace
