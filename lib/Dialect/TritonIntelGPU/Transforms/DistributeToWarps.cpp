//===- DistributeToWarps.cpp - Distribute block workload to warp -*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-distribute-to-warps"
namespace {

/// Add named attrs contained in \p dictAttrs to the given operation \p op.
void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

/// FIXME: maybe set sizePerWarp in the attr directly.
SmallVector<int64_t> getSizePerWarp(RankedTensorType type, Attribute layout) {
  llvm::SmallVector<long> sizePerWarp;
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    const SmallVector<unsigned> &sizePerThread =
        blockedLayout.getSizePerThread();
    const SmallVector<unsigned> &threadsPerWarp =
        blockedLayout.getThreadsPerWarp();
    for (auto [lhs, rhs] : llvm::zip(sizePerThread, threadsPerWarp)) {
      sizePerWarp.push_back(lhs * rhs);
    }
  } else if (auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    int idx = dotLayout.getOpIdx();
    assert(isa<ttg::BlockedEncodingAttr>(dotLayout.getParent()) &&
           "at this stage, parent layout should be blocked layout.");
    const SmallVector<int64_t> &parentSizePerWarp = getSizePerWarp(
        type, cast<ttg::BlockedEncodingAttr>(dotLayout.getParent()));
    if (idx == 0) { // dot operand A
      sizePerWarp.assign({parentSizePerWarp[0], type.getShape()[1]});
    } else { // idx == 1, dot operand B
      sizePerWarp.assign({type.getShape()[0], parentSizePerWarp[1]});
    }
  } else {
    llvm::report_fatal_error(
        "getSizePerWarp not implemented for this attribute");
  }
  return sizePerWarp;
}

Attribute getWarpLayout(Attribute layout) {
  MLIRContext *ctx = layout.getContext();
  if (auto blockedLayout = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    auto warpLayout = ttg::WarpEncodingAttr::get(
        ctx, blockedLayout.getSizePerThread(),
        blockedLayout.getThreadsPerWarp(), blockedLayout.getOrder());
    return warpLayout;
  } else if (auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    Attribute parentLayout = getWarpLayout(dotLayout.getParent());
    auto newDotLayout = ttg::DotOperandEncodingAttr::get(
        ctx, dotLayout.getOpIdx(), parentLayout, dotLayout.getKWidth());
    return newDotLayout;
  }
  return layout;
}

template <typename T> static T convertType(T type) { return type; }

template <>
RankedTensorType convertType<RankedTensorType>(RankedTensorType type) {
  auto layout = type.getEncoding();
  auto sizePerWarp = getSizePerWarp(type, layout);
  auto warpLayout = getWarpLayout(layout);
  auto newType =
      RankedTensorType::get(sizePerWarp, type.getElementType(), warpLayout);
  return newType;
}

template <> tt::PointerType convertType<tt::PointerType>(tt::PointerType type) {
  auto pointeeType = type.getPointeeType();
  auto tensorType = dyn_cast<RankedTensorType>(pointeeType);
  if (!tensorType)
    return type;
  auto newTensorType = convertType(tensorType);
  auto newType = tt::PointerType::get(newTensorType, type.getAddressSpace());
  return newType;
}

/// @brief get each warp's offset
/// warpsPerDim = blockShape / warpShape
/// assert(warpsPerDim <= warpsPerCTA)
/// warpId.x = warpId % warpPerCTA.x
/// warpId.y = warpId / warpPerCTA.x
/// incX = (warpId.x % warpsPerDim.x) * SizePerWarp.x
/// incY = (warpId.y % warpsPerDim.y) * SizePerWarp.y
/// newX = oldX + incX
/// newY = oldY + incY
SmallVector<Value> distributeOffset(SmallVector<Value> oldOffsets,
                                    RankedTensorType tensorType, Value warpId,
                                    OpBuilder b, Location loc) {
  Attribute layout = tensorType.getEncoding();
  const SmallVector<unsigned> &warpsPerCTA = ttg::getWarpsPerCTA(layout);
  size_t dims = warpsPerCTA.size();
  assert(dims <= 2 && "no more than 2D shape");
  // same to the module attribute num-warps
  unsigned numWarps = product<unsigned>(warpsPerCTA);
  RankedTensorType newTensorType = convertType(tensorType);
  ArrayRef<int64_t> blockShape = tensorType.getShape();
  ArrayRef<int64_t> warpShape = newTensorType.getShape();
  SmallVector<unsigned> warpsPerDim;
  for (auto [lhs, rhs] : llvm::zip(blockShape, warpShape)) {
    if (lhs % rhs != 0)
      return oldOffsets;
    warpsPerDim.push_back(lhs / rhs);
  }
  SmallVector<Value> newOffsets;
  for (auto i = 0; i < dims; i++) {
    Value oldOffset = oldOffsets[i];
    Value warpIdPerDim;
    if (i == 1) { // warpId.x
      if (warpsPerCTA[dims - 1] == 1) {
        newOffsets.push_back(oldOffset);
        continue;
      } else if (warpsPerCTA[dims - 1] == numWarps) {
        warpIdPerDim = warpId;
      } else {
        warpIdPerDim = b.create<arith::RemSIOp>(
            loc, warpId,
            b.create<arith::ConstantIntOp>(loc, warpsPerCTA[dims - 1], 32));
      }
    } else { // i == 0, warpId.y
      if (warpsPerCTA[dims - 1] == 1) {
        warpIdPerDim = warpId;
      } else if (warpsPerCTA[dims - 1] == numWarps) {
        newOffsets.push_back(oldOffset);
        continue;
      } else {
        warpIdPerDim = b.create<arith::DivSIOp>(
            loc, warpId,
            b.create<arith::ConstantIntOp>(loc, warpsPerCTA[dims - 1], 32));
      }
    }

    Value step;
    if (warpsPerDim[i] == 1) {
      newOffsets.push_back(oldOffset);
      continue;
    } else if (warpsPerDim[i] == numWarps) {
      step = warpIdPerDim;
    } else {
      step = b.create<arith::RemSIOp>(
          loc, warpIdPerDim,
          b.create<arith::ConstantIntOp>(loc, warpsPerDim[i], 32));
    }

    auto inc = b.create<arith::MulIOp>(
        loc, step, b.create<arith::ConstantIntOp>(loc, warpShape[i], 32));
    auto newOffset = b.create<arith::AddIOp>(loc, inc, oldOffset);
    newOffsets.push_back(newOffset);
  }

  return newOffsets;
}

void distributeGenericOp(Operation *op) {
  OpBuilder b(op);
  Operation *newOp = b.clone(*op);
  for (OpResult result : newOp->getResults()) {
    if (auto castType = dyn_cast<RankedTensorType>(result.getType()))
      result.setType(convertType(castType));
    else if (auto castType = dyn_cast<tt::PointerType>(result.getType()))
      result.setType(convertType(castType));
  }
  op->replaceAllUsesWith(newOp->getResults());
  op->erase();
  return;
}

void distributeArithConstantOp(arith::ConstantOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  if (!type)
    return;

  RankedTensorType newType = convertType(type);
  auto value = cast<DenseElementsAttr>(op.getValue()).resizeSplat(newType);
  OpBuilder b(op);
  auto newOp = b.create<arith::ConstantOp>(op.getLoc(), newType, value);
  addNamedAttrs(newOp, op->getAttrDictionary());
  op->replaceAllUsesWith(newOp->getResults());
  op->erase();
}

void distributeMakeTensorPtrOp(tt::MakeTensorPtrOp op, Value warpId) {
  Location loc = op.getLoc();
  tt::PointerType type = op.getType();
  OpBuilder b(op);
  auto tensorType = dyn_cast<RankedTensorType>(type.getPointeeType());
  if (!tensorType)
    return;

  SmallVector<Value> newOffsets =
      distributeOffset(op.getOffsets(), tensorType, warpId, b, loc);
  Operation *newOp = b.clone(*op.getOperation());
  tt::PointerType newType = convertType(type);
  auto newPtrOp = cast<tt::MakeTensorPtrOp>(newOp);
  newPtrOp.getOffsetsMutable().assign(newOffsets);
  newPtrOp.getResult().setType(newType);
  op->replaceAllUsesWith(newPtrOp->getResults());
  op->erase();
}

void distributeConvertLayoutOp(ttg::ConvertLayoutOp op, Value warpId,
                               RankedTensorType oldSrcType) {
  Location loc = op.getLoc();
  OpBuilder b(op);
  auto dstType = cast<RankedTensorType>(op.getResult().getType());
  RankedTensorType convertedDstType = convertType(dstType);
  auto dstPtrType = tt::PointerType::get(convertedDstType, 3 /* shared mem*/);
  auto srcPtrType =
      tt::PointerType::get(op.getSrc().getType(), 3 /* shared mem*/);

  // fixme: allocOp may carry the size info, tt::PointerType::get(oldSrcType)
  // fixme: set addrspace 1 instead of 3 to avoid makeTensorOp type match
  // error
  auto baseType = tt::PointerType::get(oldSrcType.getElementType(), 1);
  auto base = b.create<ttgi::AllocOp>(loc, baseType);
  SmallVector<Value> shape;
  shape.push_back(
      b.create<arith::ConstantIntOp>(loc, oldSrcType.getShape()[0], 64));
  shape.push_back(
      b.create<arith::ConstantIntOp>(loc, oldSrcType.getShape()[1], 64));

  SmallVector<Value> strides;
  strides.push_back(
      b.create<arith::ConstantIntOp>(loc, oldSrcType.getShape()[1], 64));
  strides.push_back(b.create<arith::ConstantIntOp>(loc, 1, 64));

  SmallVector<Value> offsets;
  offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
  offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));

  SmallVector<Value> srcOffsets =
      distributeOffset(offsets, oldSrcType, warpId, b, loc);
  Value storePtr =
      b.create<tt::MakeTensorPtrOp>(loc, srcPtrType, base, shape, strides,
                                    srcOffsets, b.getDenseI32ArrayAttr({1, 0}));
  b.create<tt::StoreOp>(loc, storePtr, op.getSrc(), tt::CacheModifier::NONE,
                        tt::EvictionPolicy::NORMAL);
  b.create<gpu::BarrierOp>(loc);

  SmallVector<Value> dstOffsets =
      distributeOffset(offsets, dstType, warpId, b, loc);
  Value loadPtr =
      b.create<tt::MakeTensorPtrOp>(loc, dstPtrType, base, shape, strides,
                                    dstOffsets, b.getDenseI32ArrayAttr({1, 0}));
  auto load = b.create<tt::LoadOp>(loc, loadPtr, tt::CacheModifier::NONE,
                                   tt::EvictionPolicy::NORMAL, false);
  op->replaceAllUsesWith(load->getResults());
  op->erase();
}

void distributeScfForOp(scf::ForOp op) {
  Block *body = op.getBody();
  for (auto [lhs, rhs] :
       llvm::zip(body->getArguments().drop_front(1), op.getInitArgs()))
    lhs.setType(rhs.getType());
  for (OpResult result : op->getResults()) {
    if (auto castType = dyn_cast<RankedTensorType>(result.getType()))
      result.setType(convertType(castType));
    else if (auto castType = dyn_cast<tt::PointerType>(result.getType()))
      result.setType(convertType(castType));
  }
}

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class TritonIntelGPUDistributeToWarpsPass
    : public TritonIntelGPUDistributeToWarpsBase<
          TritonIntelGPUDistributeToWarpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    for (auto func : m.getOps<tt::FuncOp>()) {
      auto b = OpBuilder::atBlockBegin(&func.getBody().front());
      Location loc = func.getLoc();
      auto warpId = b.create<arith::IndexCastOp>(
          loc, b.getI32Type(), b.create<gpu::SubgroupIdOp>(loc));

      // record old type before transform
      DenseMap<Operation *, RankedTensorType> typeMap;
      func.walk([&](ttg::ConvertLayoutOp op) {
        typeMap[op] = op.getSrc().getType().cast<RankedTensorType>();
      });

      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (llvm::all_of(op->getResultTypes(), [](Type type) {
              return !isa<RankedTensorType>(type) &&
                     !isa<tt::PointerType>(type);
            }))
          return WalkResult::advance();

        TypeSwitch<Operation *>(op)
            .Case([&](scf::ForOp forOp) { distributeScfForOp(forOp); })
            .Case([&](tt::MakeTensorPtrOp ptrOp) {
              distributeMakeTensorPtrOp(ptrOp, warpId);
            })
            .Case([&](ttg::ConvertLayoutOp convertOp) {
              distributeConvertLayoutOp(convertOp, warpId, typeMap[convertOp]);
            })
            .Case([&](arith::ConstantOp cstOp) {
              distributeArithConstantOp(cstOp);
            })
            .Default([](auto op) {
              if (isa<tt::LoadOp, tt::DotOp, tt::AdvanceOp, arith::TruncFOp>(
                      op))
                distributeGenericOp(op);
              else
                assert(false && "Unexpected operation type");
            });

        return WalkResult::advance();
      });
    }
  }
};

std::unique_ptr<Pass>
mlir::triton::gpu::intel::createTritonIntelGPUDistributeToWarpsPass() {
  return std::make_unique<TritonIntelGPUDistributeToWarpsPass>();
}
