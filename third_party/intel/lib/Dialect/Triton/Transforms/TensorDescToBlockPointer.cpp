#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>

#define DEBUG_TYPE "triton-intel-tdesc-to-block-pointer"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELTENSORDESCTOBLOCKPOINTER
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute>
filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs) {
  mlir::SmallVector<NamedAttribute> ret;
  llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
    auto attrName = attr.getName().getValue();
    return attrName != "operandSegmentSizes";
  });
  return ret;
}

struct Descriptor {
  Value blockPointer;
  Value paddingOption;
};

Descriptor unpackDescriptor(tt::TensorDescType type, ValueRange pack) {
  assert(pack.size() == 2 &&
         "Expected tensor descriptors to consist of a block pointer followed "
         "by a padding option value.");

  Descriptor res;
  res.blockPointer = pack[0];
  res.paddingOption = pack[1];
  return res;
}

Value generateOther(OpBuilder &builder, Location loc, Type scalarTy,
                    ArrayRef<int64_t> blockShape,
                    Value paddingOption = nullptr) {
  auto blockTy = RankedTensorType::get(blockShape, scalarTy);
  if (paddingOption && mlir::isa<FloatType>(scalarTy)) {
    auto floatTy = mlir::cast<FloatType>(scalarTy);
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto nanValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getFloatAttr(floatTy, nan)));
    auto zeroValue = arith::ConstantOp::create(
        builder, loc,
        SplatElementsAttr::get(blockTy, builder.getZeroAttr(floatTy)));
    return mlir::arith::SelectOp::create(builder, loc, paddingOption, nanValue,
                                         zeroValue);
  } else {
    auto attr = builder.getZeroAttr(blockTy);
    return arith::ConstantOp::create(builder, loc, attr);
  }
}

Value generateOther(OpBuilder &builder, Location loc, tt::TensorDescType descTy,
                    Value paddingOption = nullptr) {
  auto blockTy = descTy.getSignlessBlockType();
  return generateOther(builder, loc, blockTy.getElementType(),
                       blockTy.getShape(), paddingOption);
}

constexpr unsigned offsetBitwidth = 32u;
constexpr unsigned shapeAndStridesBitwidth = 64u;

Value findOrCreateCast(Location loc, Value val, Type tgtType,
                       OpBuilder &builder) {
  assert(isa<IntegerType>(tgtType) && isa<IntegerType>(val.getType()) &&
         "Expecting integer types");
  assert(val.getType().getIntOrFloatBitWidth() <=
             tgtType.getIntOrFloatBitWidth() &&
         "Expecting smaller type");

  if (val.getType() == tgtType)
    return val;

  Block *block = builder.getInsertionBlock();
  const Block::iterator insertPoint = builder.getInsertionPoint();

  auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
    if (auto castOp = dyn_cast<arith::ExtSIOp>(op))
      return castOp.getIn() == val && castOp.getType() == tgtType;
    return false;
  });

  return (it != insertPoint)
             ? cast<arith::ExtSIOp>(*it)
             : getValueOrCreateCastToIndexLike(builder, loc, tgtType, val);
}

struct RewriteMakeTensorDesc : OpConversionPattern<triton::MakeTensorDescOp> {
  using OpConversionPattern<triton::MakeTensorDescOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    tt::TensorDescType tDescType = op.getType();
    Location loc = op.getLoc();

    SmallVector<Value> shapes, strides, offsets;
    SmallVector<int32_t> sizes, orders;
    unsigned rank = tDescType.getBlockType().getRank();
    for (const auto [dim, shape, stride, size] :
         llvm::enumerate(op.getShape(), op.getStrides(),
                         tDescType.getBlockType().getShape())) {
      shapes.push_back(findOrCreateCast(
          loc, shape, rewriter.getIntegerType(shapeAndStridesBitwidth),
          rewriter));
      strides.push_back(findOrCreateCast(
          loc, stride, rewriter.getIntegerType(shapeAndStridesBitwidth),
          rewriter));
      Value zero =
          tt::intel::findOrCreateIntConstant(loc, 0, offsetBitwidth, rewriter);
      offsets.push_back(zero);
      sizes.push_back(static_cast<int32_t>(size));
      orders.push_back(rank - (1 + dim)); // the fastest change dim first.
    }

    tt::MakeTensorPtrOp makeTensorPtr = tt::MakeTensorPtrOp::create(
        rewriter, op.getLoc(), adaptor.getBase(), shapes, strides, offsets,
        sizes, rewriter.getDenseI32ArrayAttr(orders));

    SmallVector<mlir::Value> blockPopinterAndPaddingOption;
    llvm::append_values(blockPopinterAndPaddingOption, makeTensorPtr);
    auto paddingOption = mlir::arith::ConstantOp::create(
        rewriter, op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(adaptor.getPadding() ==
                             triton::PaddingOption::PAD_NAN));
    llvm::append_values(blockPopinterAndPaddingOption, paddingOption);
    rewriter.replaceOpWithMultiple(op, {blockPopinterAndPaddingOption});

    return mlir::success();
  }
};

struct RewriteLoadPattern : OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto blockType = descTy.getSignlessBlockType();
    auto ptrType = cast<tt::PointerType>(descTy.getSignlessBlockType());
    Value ptr = tt::AdvanceOp::create(rewriter, loc, ptrType, desc.blockPointer,
                                      op.getIndices());

    SmallVector<int32_t> boundaryCheck;
    for (size_t i = 0; i < blockType.getRank(); ++i)
      boundaryCheck.push_back(i);

    auto other = generateOther(rewriter, loc, descTy, desc.paddingOption);

    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, ptr, boundaryCheck, other, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, /*volatile*/ false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

struct RewriteStorePattern : OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // auto loc = op.getLoc();
    // auto descTy = op.getDesc().getType();
    // const auto blockShape = descTy.getBlockType().getShape();
    // auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    // auto offsets = castToI64(rewriter, op.getIndices());
    //
    // auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
    //     op, generatePtr(rewriter, loc, blockShape, desc, offsets),
    //     op.getSrc(), generateMask(rewriter, loc, blockShape, desc, offsets),
    //     triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    // newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

struct TritonIntelTensorDescToBlockPointer
    : tt::intel::impl::TritonIntelTensorDescToBlockPointerBase<
          TritonIntelTensorDescToBlockPointer> {
public:
  using Base::Base;
  using IndexMapSet = std::map<int, std::set<int>>;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    WalkResult res = moduleOp->walk<WalkOrder::PreOrder>([=](Operation *op) {
      if (isa<tt::DescriptorGatherOp>(op) || isa<tt::DescriptorScatterOp>(op) ||
          isa<tt::DescriptorReduceOp>(op)) {
        op->emitRemark(
            "TritonIntelTensorDescToBlockPointer: Failed to rewrite");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (res.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "TritonIntelTensorDescToBlockPointer: Skipping module - "
                    "contains unsupported operations\n");
      return;
    }

    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      mlir::scf::SCFDialect,
                                      mlir::triton::TritonDialect>(
        [](mlir::Operation *op) {
          return !hasATensorDescriptorType(op->getOperandTypes()) &&
                 !hasATensorDescriptorType(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    mlir::TypeConverter converter;

    converter.addConversion([](mlir::Type t) {
      // Most types don't require any conversion
      return t;
    });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into a block pointer and padding option.
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });

    mlir::RewritePatternSet patterns(moduleOp->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    patterns
        .add<RewriteMakeTensorDesc, RewriteLoadPattern, RewriteStorePattern>(
            converter, &getContext());

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            moduleOp, target, std::move(patterns), config))) {
      signalPassFailure();
    }
  }

private:
  // Create a new block pointer if a suitable one doesn't already exist.
  // Otherwise, return the existing one. The function takes the base, shape,
  // strides, offsets, sizes of the block pointer to create/lookup and its
  // tensor element type (to ensure the block pointer has the tensor layout).
  tt::MakeTensorPtrOp
  findOrCreateMakeTensorPtr(Location loc, Value base, ValueRange shape,
                            ValueRange strides, ValueRange offsets,
                            ArrayRef<int32_t> sizes, OpBuilder &builder) {
    Block *block = builder.getInsertionBlock();
    const Block::iterator insertPoint = builder.getInsertionPoint();
    auto it = std::find_if(block->begin(), insertPoint, [&](Operation &op) {
      if (auto makeTensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        triton::PointerType resType = makeTensorPtrOp.getResult().getType();
        auto tensorType = cast<RankedTensorType>(resType.getPointeeType());
        auto sameShape = [](ArrayRef<int64_t> arr1, ArrayRef<int32_t> arr2) {
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
               sameShape(tensorType.getShape(), sizes);
      }
      return false;
    });

    auto makeTensorPtrOp = [&]() {
      auto makeTensorPtr = tt::MakeTensorPtrOp::create(
          builder, loc, base, shape, strides, offsets, sizes,
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
    SmallVector<int32_t> sizes;
    for (const auto [shape, stride, size] :
         llvm::zip(op.getShape(), op.getStrides(),
                   tDescType.getBlockType().getShape())) {
      shapes.push_back(findOrCreateCast(
          loc, shape, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      strides.push_back(findOrCreateCast(
          loc, stride, builder.getIntegerType(shapeAndStridesBitwidth),
          builder));
      Value zero =
          tt::intel::findOrCreateIntConstant(loc, 0, offsetBitwidth, builder);
      offsets.push_back(zero);
      sizes.push_back(static_cast<int32_t>(size));
    }

    auto tensorPtr = findOrCreateMakeTensorPtr(
        loc, op.getBase(), shapes, strides, offsets, sizes, builder);
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
  LogicalResult rewriteDescriptorLoadOrStoreOp(OpTy op) {
    assert(op && "Expecting a valid operation");

    // At this point we expect to have transformed `make_tensor_descriptor` into
    // a `make_block_ptr` operation, except when the tensor descriptor is
    // allocated on the host and passed to the kernel as an argument.
    Value operand = op.getOperand(0);
    if (isa<tt::TensorDescType>(operand.getType()))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Rewriting: " << op << "\n");

    OpBuilder builder(op);
    Location loc = op.getLoc();
    assert(triton::isTensorPointerType(operand.getType()) &&
           "Expecting a block ptr");
    auto ptrType = cast<tt::PointerType>(operand.getType());
    auto tensorType = cast<RankedTensorType>(ptrType.getPointeeType());
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
      if (paddingInfo.contains(op)) {
        padding = paddingInfo[op];
      }
      auto loadOp = builder.createOrFold<tt::LoadOp>(
          loc, ptr, boundaryCheck,
          /*padding*/ padding, op.getCache(), op.getEvict(),
          /*volatile*/ false);
      LLVM_DEBUG(llvm::dbgs().indent(2) << loadOp << "\n");
      op.replaceAllUsesWith(loadOp);
    } else {
      [[maybe_unused]] auto storeOp = builder.createOrFold<tt::StoreOp>(
          loc, ptr, op.getSrc(), boundaryCheck, tt::CacheModifier::NONE,
          tt::EvictionPolicy::NORMAL);
      LLVM_DEBUG(llvm::dbgs().indent(2) << storeOp << "\n");
    }

    cleanUp.insert(op);

    return success();
  }

private:
  SmallPtrSet<Operation *, 8> cleanUp;
  llvm::SmallDenseMap<Operation *, triton::PaddingOption, 8> paddingInfo;
};

} // namespace
