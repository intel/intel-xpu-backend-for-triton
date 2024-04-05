//===- DistributeToWarps.cpp - Distribute block workload to warp -*-C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace {

/// FIXME: maybe set sizePerWarp in the attr directly.
SmallVector<int64_t> getSizePerWarp(RankedTensorType type, Attribute layout) {
  SmallVector<int64_t> sizePerWarp;

  TypeSwitch<Attribute>(layout)
      .Case<ttg::BlockedEncodingAttr>([&](auto blockedLayout) {
        const SmallVector<unsigned> &sizePerThread =
            blockedLayout.getSizePerThread();
        const SmallVector<unsigned> &threadsPerWarp =
            blockedLayout.getThreadsPerWarp();
        for (auto [lhs, rhs] : llvm::zip(sizePerThread, threadsPerWarp))
          sizePerWarp.push_back(lhs * rhs);
      })
      .Case<ttg::DotOperandEncodingAttr>([&](auto dotLayout) {
        assert(isa<ttg::BlockedEncodingAttr>(dotLayout.getParent()) &&
               "at this stage, parent layout should be blocked layout");
        const SmallVector<int64_t> &parentSizePerWarp = getSizePerWarp(
            type, cast<ttg::BlockedEncodingAttr>(dotLayout.getParent()));
        if (dotLayout.getOpIdx() == 0) // dot operand A
          sizePerWarp.assign({parentSizePerWarp[0], type.getShape()[1]});
        else // dot operand B
          sizePerWarp.assign({type.getShape()[0], parentSizePerWarp[1]});
      })
      .Default([](auto) {
        llvm::report_fatal_error(
            "getSizePerWarp not implemented for this attribute");
      });

  return sizePerWarp;
}

Attribute getWarpLayout(Attribute layout) {
  return TypeSwitch<Attribute, Attribute>(layout)
      .Case<ttg::BlockedEncodingAttr>([&](auto blockedLayout) {
        return ttgi::WarpEncodingAttr::get(
            layout.getContext(), blockedLayout.getSizePerThread(),
            blockedLayout.getThreadsPerWarp(), blockedLayout.getOrder());
      })
      .Case<ttg::DotOperandEncodingAttr>([&](auto dotLayout) {
        return ttg::DotOperandEncodingAttr::get(
            layout.getContext(), dotLayout.getOpIdx(),
            getWarpLayout(dotLayout.getParent()), dotLayout.getKWidth());
      })
      .Default([&](auto) { return layout; });
}

RankedTensorType convertType(RankedTensorType type) {
  Attribute layout = type.getEncoding();
  SmallVector<int64_t> sizePerWarp = getSizePerWarp(type, layout);
  Attribute warpLayout = getWarpLayout(layout);
  return RankedTensorType::get(sizePerWarp, type.getElementType(), warpLayout);
}

tt::PointerType convertType(tt::PointerType type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type.getPointeeType()))
    return tt::PointerType::get(convertType(tensorType),
                                type.getAddressSpace());
  return type;
}

/// @brief get each warp's offset
/// warpsPerDim = blockShape / warpShape
/// assert(warpsPerDim <= warpsPerCTA)
/// warpId.x = warpId % warpPerCTA.x
/// warpId.y = warpId / warpPerCTA.x
/// incX = (warpId.x % warpsPerDim.x) * sizePerWarp.x
/// incY = (warpId.y % warpsPerDim.y) * sizePerWarp.y
/// newX = oldX + incX
/// newY = oldY + incY
SmallVector<Value> distributeOffset(const SmallVector<Value> &oldOffsets,
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
  for (int32_t i = 0; i < dims; i++) {
    Value oldOffset = oldOffsets[i];
    Value warpIdPerDim;

    switch (i) {
    case 0: {
      if (warpsPerCTA[dims - 1] == numWarps) {
        newOffsets.push_back(oldOffset);
        continue;
      }
      warpIdPerDim =
          (warpsPerCTA[dims - 1] == 1)
              ? warpId
              : b.create<arith::DivSIOp>(loc, warpId,
                                         b.create<arith::ConstantIntOp>(
                                             loc, warpsPerCTA[dims - 1], 32));
    } break;
    case 1: {
      if (warpsPerCTA[dims - 1] == 1) {
        newOffsets.push_back(oldOffset);
        continue;
      }
      warpIdPerDim =
          (warpsPerCTA[dims - 1] == numWarps)
              ? warpId
              : b.create<arith::RemSIOp>(loc, warpId,
                                         b.create<arith::ConstantIntOp>(
                                             loc, warpsPerCTA[dims - 1], 32));
    } break;
    }

    if (warpsPerDim[i] == 1) {
      newOffsets.push_back(oldOffset);
      continue;
    }

    Value step = (warpsPerDim[i] == numWarps)
                     ? warpIdPerDim
                     : b.create<arith::RemSIOp>(loc, warpIdPerDim,
                                                b.create<arith::ConstantIntOp>(
                                                    loc, warpsPerDim[i], 32));
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
}

void distributeArithConstantOp(arith::ConstantOp op) {
  auto type = dyn_cast<RankedTensorType>(op.getType());
  if (!type)
    return;

  RankedTensorType newType = convertType(type);
  auto value = cast<DenseElementsAttr>(op.getValue()).resizeSplat(newType);
  OpBuilder b(op);
  auto newOp = b.create<arith::ConstantOp>(op.getLoc(), newType, value);

  for (const NamedAttribute &attr : op->getAttrDictionary().getValue())
    if (!newOp->hasAttr(attr.getName()))
      newOp->setAttr(attr.getName(), attr.getValue());

  op->replaceAllUsesWith(newOp->getResults());
  op->erase();
}

void distributeMakeTensorPtrOp(tt::MakeTensorPtrOp op, Value warpId) {
  tt::PointerType type = op.getType();
  auto tensorType = dyn_cast<RankedTensorType>(type.getPointeeType());
  if (!tensorType)
    return;

  OpBuilder b(op);
  SmallVector<Value> newOffsets =
      distributeOffset(op.getOffsets(), tensorType, warpId, b, op.getLoc());
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
  auto dstType = cast<RankedTensorType>(op.getResult().getType());
  RankedTensorType convertedDstType = convertType(dstType);
  auto dstPtrType = tt::PointerType::get(convertedDstType, 3 /*shared mem*/);
  auto srcPtrType =
      tt::PointerType::get(op.getSrc().getType(),
                           triton::TritonGEN::TritonGENMemorySpace::kWorkgroup);

  // FIXME: allocOp may carry the size info.
  OpBuilder b(op);
  auto baseType = tt::PointerType::get(
      oldSrcType.getElementType(),
      triton::TritonGEN::TritonGENMemorySpace::kCrossWorkgroup);
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
    ModuleOp mod = getOperation();

    for (auto func : mod.getOps<tt::FuncOp>()) {
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
            .Case<scf::ForOp>([](auto forOp) { distributeScfForOp(forOp); })
            .Case<tt::MakeTensorPtrOp>(
                [&](auto ptrOp) { distributeMakeTensorPtrOp(ptrOp, warpId); })
            .Case<ttg::ConvertLayoutOp>([&](auto convertOp) {
              distributeConvertLayoutOp(convertOp, warpId, typeMap[convertOp]);
            })
            .Case<arith::ConstantOp>(
                [](auto cstOp) { distributeArithConstantOp(cstOp); })
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
