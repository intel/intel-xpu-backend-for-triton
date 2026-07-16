//===- InsertGlobalBarrier.cpp - Barrier cross-thread global exchange -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sometimes a kernel writes a global buffer and then reads it back with a
// different layout. When that happens one thread can read a value another
// thread wrote. Membar only checks shared memory, so it does not catch this.
// This pass puts a barrier before that read.
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUINSERTGLOBALBARRIER
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace {

// Follow a load/store pointer back through splat and addptr to find which
// kernel argument it points into. Returns null if it is not one.
BlockArgument getKernelArgBase(Value ptr) {
  while (Operation *def = ptr.getDefiningOp()) {
    if (auto addPtr = dyn_cast<tt::AddPtrOp>(def))
      ptr = addPtr.getPtr();
    else if (auto splat = dyn_cast<tt::SplatOp>(def))
      ptr = splat.getSrc();
    else
      return nullptr;
  }
  return dyn_cast<BlockArgument>(ptr);
}

struct TritonIntelGPUInsertGlobalBarrierPass
    : public triton::gpu::intel::impl::TritonIntelGPUInsertGlobalBarrierBase<
          TritonIntelGPUInsertGlobalBarrierPass> {
  void runOnOperation() override {
    // Only look at the function's top block. If we put a barrier inside an
    // scf.if or scf.for, some threads may skip it and the kernel hangs. As we
    // go, keep the layout of the last store to each kernel argument.
    getOperation().walk([](Block *block) {
      if (!isa<tt::FuncOp>(block->getParentOp()))
        return;
      llvm::DenseMap<Value, Attribute> storedEnc;
      for (Operation &op : *block) {
        if (auto store = dyn_cast<tt::StoreOp>(&op)) {
          auto ty = dyn_cast<RankedTensorType>(store.getValue().getType());
          if (BlockArgument base =
                  ty ? getKernelArgBase(store.getPtr()) : nullptr)
            storedEnc[base] = ty.getEncoding();
        } else if (auto load = dyn_cast<tt::LoadOp>(&op)) {
          auto ty = dyn_cast<RankedTensorType>(load.getType());
          BlockArgument base = ty ? getKernelArgBase(load.getPtr()) : nullptr;
          auto it = base ? storedEnc.find(base) : storedEnc.end();
          // Same layout: each thread reads back its own data, so it is safe.
          // Different layout: a thread reads another thread's data, so we
          // need a barrier first.
          if (it != storedEnc.end() && it->second != ty.getEncoding()) {
            OpBuilder builder(&op);
            ttg::BarrierOp::create(builder, op.getLoc(), ttg::AddrSpace::All);
            storedEnc.erase(it);
          }
        }
      }
    });
  }
};

} // namespace
