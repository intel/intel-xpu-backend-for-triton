#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-tdesc-to-block-pointer"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELTENSORDESCTOBLOCKPOINTER
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

constexpr unsigned offsetBitwidth = 32u;
constexpr unsigned shapeAndStridesBitwidth = 64u;

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

struct TritonIntelTensorDescToBlockPointer
    : tt::intel::impl::TritonIntelTensorDescToBlockPointerBase<
          TritonIntelTensorDescToBlockPointer> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    moduleOp->walk<WalkOrder::PreOrder>([&](tt::DescriptorLoadOp loadOp) {
      // Retrieve the padding option from the MakeTensorDescOp.
      std::optional<tt::MakeTensorDescOp> makeTensorDescOp =
          tt::intel::findDefiningOpOfType<tt::MakeTensorDescOp>(
              loadOp.getDesc());
      assert(makeTensorDescOp.has_value() &&
             "Expecting to find the defining MakeTensorDescOp");
      OpToPaddingMap[loadOp] = makeTensorDescOp->getPadding();
      return WalkResult::advance();
    });

    moduleOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
      assert(!isa<tt::DescriptorGatherOp>(op) &&
             !isa<tt::DescriptorScatterOp>(op) &&
             !isa<tt::DescriptorReduceOp>(op) &&
             "Expecting no gather/scatter/reduce ops at this stage");
      TypeSwitch<Operation *>(op)
          .Case<tt::MakeTensorDescOp>([&](auto makeTensorDescOp) {
            [[maybe_unused]] LogicalResult res =
                rewriteMakeTensorDescriptorOp(makeTensorDescOp);
            assert(succeeded(res) &&
                   "Failed to rewrite make_tensor_descriptor op");
          })
          .Case<tt::DescriptorLoadOp, tt::DescriptorStoreOp>(
              [&](auto loadOrStoreOp) {
                rewriteDescriptorLoadOrStoreOp(loadOrStoreOp);
              })
          .Default([&](auto) {});
      return WalkResult::advance();
    });

    if (!cleanUp.empty())
      tt::intel::eraseOperations(cleanUp);

    LLVM_DEBUG(llvm::dbgs()
                   << "After TDesc to block_ptr: " << moduleOp << "\n";);
    moduleOp->walk([&](Operation *op) {
      assert(!hasATensorDescriptorType(op->getOperandTypes()) &&
             !hasATensorDescriptorType(op->getResultTypes()) &&
             "Expecting no tensor descriptor types after conversion");
    });
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }

private:
  // Create a new block pointer if a suitable one doesn't already exist.
  // Otherwise, return the existing one. The function takes the base, shape,
  // strides, offsets, sizes of the block pointer to create/lookup and its
  // tensor element type (to ensure the block pointer has the tensor layout).
  tt::MakeTensorPtrOp
  findOrCreateMakeTensorPtr(Location loc, Value base, ValueRange shape,
                            ValueRange strides, ValueRange offsets,
                            ArrayRef<int64_t> sizes, Attribute layout,
                            OpBuilder &builder) {
    Block *block = builder.getInsertionBlock();
    const Block::iterator insertPoint = builder.getInsertionPoint();
    auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
      if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        triton::PointerType resType = makeTensorPtrOp.getResult().getType();
        auto tensorType = cast<RankedTensorType>(resType.getPointeeType());
        auto sameShape = [](ArrayRef<int64_t> arr1, ArrayRef<int64_t> arr2) {
          for (auto [dim1, dim2] : llvm::zip(arr1, arr2)) {
            if (dim1 != dim2)
              return false;
          }
          return true;
        };

        return makeTensorPtrOp.getBase() == base &&
               makeTensorPtrOp.getShape() == shape &&
               makeTensorPtrOp.getStrides() == strides &&
               makeTensorPtrOp.getOffsets() == offsets &&
               sameShape(tensorType.getShape(), sizes) &&
               tensorType.getEncoding() == layout;
      }
      return false;
    });

    auto makeTensorPtrOp = [&]() {
      auto pointerType = cast<tt::PointerType>(base.getType());
      auto tensorType =
          RankedTensorType::get(sizes, pointerType.getPointeeType(), layout);
      auto tensorPtrType =
          tt::PointerType::get(tensorType, pointerType.getAddressSpace());
      auto makeTensorPtr = tt::MakeTensorPtrOp::create(
          builder, loc, tensorPtrType, base, shape, strides, offsets,
          builder.getDenseI32ArrayAttr({1, 0}));
      return makeTensorPtr;
    };

    return (it != insertPoint) ? cast<tt::MakeTensorPtrOp>(*it)
                               : makeTensorPtrOp();
  }

  void propagateToLoops(Operation *op) {
    auto loopOp = dyn_cast<LoopLikeOpInterface>(op);
    if (!loopOp)
      return;

    bool updated = false;
    for (auto [initArg, rgnInitArg, yieldVal, loopRes] :
         llvm::zip(loopOp.getInits(), loopOp.getRegionIterArgs(),
                   loopOp.getYieldedValues(), loopOp->getResults())) {
      Type initArgType = initArg.getType();
      Type rgnInitArgType = rgnInitArg.getType();
      assert(rgnInitArgType == loopRes.getType() &&
             rgnInitArgType == yieldVal.getType() && "Type mismatch");
      if (rgnInitArgType != initArgType) {
        rgnInitArg.setType(initArgType);
        yieldVal.setType(initArgType);
        loopRes.setType(initArgType);
        updated = true;
      }
    }
    if (!updated)
      return;

    // For while loops we also need to update the "after" region arguments.
    if (auto loopOp = dyn_cast<scf::WhileOp>(op)) {
      for (auto [initArg, rgnAfterArg] :
           llvm::zip(loopOp.getInits(), loopOp.getAfterArguments())) {
        Type initArgType = initArg.getType();
        if (rgnAfterArg.getType() != initArgType)
          rgnAfterArg.setType(initArgType);
      }
    }

    // Propagate the loop results to their users.
    for (Operation *user : loopOp->getUsers())
      propagateToLoops(user);
  }

  LogicalResult rewriteMakeTensorDescriptorOp(tt::MakeTensorDescOp op) {
    assert(op && "Expecting a valid operation");
    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    tt::TensorDescType tDescType = op.getType();

    // Create a new block pointer if a suitable one doesn't already exist.
    SmallVector<Value> shapes, strides, offsets;
    SmallVector<int64_t> sizes;
    for (const auto [shape, stride, size] :
         llvm::zip(op.getShape(), op.getStrides(),
                   tDescType.getBlockType().getShape())) {
      shapes.push_back(tt::intel::findOrCreateCastOp(
          shape, builder.getIntegerType(shapeAndStridesBitwidth)));
      strides.push_back(tt::intel::findOrCreateCastOp(
          stride, builder.getIntegerType(shapeAndStridesBitwidth)));
      Value zero =
          tt::intel::findOrCreateIntConstant(loc, 0, offsetBitwidth, builder);
      offsets.push_back(zero);
      sizes.push_back(size);
    }

    Attribute layout = ttg::getDefaultBlockedEncoding(
        builder.getContext(), sizes, ttg::lookupNumWarps(op),
        ttg::lookupThreadsPerWarp(builder), ttg::lookupNumCTAs(builder));
    auto tensorPtr = findOrCreateMakeTensorPtr(
        loc, op.getBase(), shapes, strides, offsets, sizes, layout, builder);
    LLVM_DEBUG({
      llvm::dbgs() << "With:\n";
      llvm::dbgs().indent(2) << tensorPtr << "\n";
    });

    op->replaceAllUsesWith(tensorPtr);
    cleanUp.insert(op);

    // Propagate the `tensorPtr` type to loops init args, yielded values,
    // results, ... (if necessary).
    for (Operation *user : tensorPtr->getUsers())
      propagateToLoops(user);

    return success();
  }

  template <typename OpTy,
            std::enable_if_t<llvm::is_one_of<OpTy, tt::DescriptorLoadOp,
                                             tt::DescriptorStoreOp>::value,
                             bool> = true>
  void rewriteDescriptorLoadOrStoreOp(OpTy op) {
    assert(op && "Expecting a valid operation");

    // At this point we expect to have transformed `make_tensor_descriptor` into
    // a `make_block_ptr` operation, except when the tensor descriptor is
    // allocated on the host and passed to the kernel as an argument.
    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    Value operand = op.getOperand(0);
    assert(triton::isTensorPointerType(operand.getType()) &&
           "Expecting a block ptr");
    auto ptrType = cast<tt::PointerType>(operand.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());

    // FIXME: If we want to move TensorDescToBlockPointer pass further down in
    // the pipeline, then we need to handle also non-default layouts.
    [[maybe_unused]] Attribute defaultLayout = ttg::getDefaultBlockedEncoding(
        builder.getContext(), tensorType.getShape(), ttg::lookupNumWarps(op),
        ttg::lookupThreadsPerWarp(builder), ttg::lookupNumCTAs(builder));
    assert(tensorType.getEncoding() == defaultLayout &&
           "Expecting the default blocked encoding");

    Value ptr =
        tt::AdvanceOp::create(builder, loc, ptrType, operand, op.getIndices());

    SmallVector<int32_t> boundaryCheck;
    for (size_t i = 0; i < tensorType.getRank(); ++i)
      boundaryCheck.push_back(i);

    constexpr bool isLoad = std::is_same_v<OpTy, tt::DescriptorLoadOp>;
    if constexpr (isLoad) {
      // Default to PAD_ZERO as this is the expected padding behavior for
      // descriptor loads. It should be specified in the tt.make_tensor_desc if
      // it is retrieved.
      triton::PaddingOption padding = triton::PaddingOption::PAD_ZERO;
      if (OpToPaddingMap.contains(op))
        padding = OpToPaddingMap[op];

      auto loadOp = builder.createOrFold<tt::LoadOp>(
          loc, ptr, boundaryCheck, padding, op.getCache(), op.getEvict(),
          /*volatile*/ false);

      RankedTensorType loadType = cast<RankedTensorType>(loadOp.getType());
      RankedTensorType resType = cast<RankedTensorType>(op.getType());
      if (loadType == resType) {
        LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n");
        op.replaceAllUsesWith(loadOp);
      } else {
        // Note: the Triton combine pass might 'combine' a reshape op with the
        // descriptor load op by changing the result yielded by the descriptor
        // load op (see RankedReduceDescriptorLoads). In this case we need to
        // insert a reshape op to ensure the load op result has the expected
        // shape for subsequent operations.
        ArrayRef<int64_t> resShape = resType.getShape();
        assert(loadType.getShape() != resShape && "Expecting different shapes");
        assert(loadType.getElementType() == resType.getElementType() &&
               "Expecting the same element type");
        auto reshapeOp =
            builder.createOrFold<tt::ReshapeOp>(loc, resShape, loadOp);
        LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n";
                   llvm::dbgs().indent(2) << reshapeOp << "\n");
        op.replaceAllUsesWith(reshapeOp);
      }
    } else {
      [[maybe_unused]] auto storeOp = builder.createOrFold<tt::StoreOp>(
          loc, ptr, op.getSrc(), boundaryCheck, tt::CacheModifier::NONE,
          tt::EvictionPolicy::NORMAL);
      LLVM_DEBUG(llvm::dbgs().indent(2) << storeOp << "\n");
    }

    cleanUp.insert(op);
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
  llvm::SmallDenseMap<Operation *, tt::PaddingOption, 8> OpToPaddingMap;
};

} // namespace
