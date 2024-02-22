/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "DumpLayout.h"

#include "../../../lib/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "../../../lib/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

namespace {

//===----------------------------------------------------------------------===//
// IndexEmitter
//===----------------------------------------------------------------------===//

class IndexEmitter {
public:
  IndexEmitter(MLIRContext *context_)
      : context(context_), option(context), typeConverter(context, option),
        rewriter(context), loc(UnknownLoc::get(context)) {
    rewriter.setInsertionPointToStart(&block);
  }

  llvm::SmallVector<llvm::SmallVector<Value>>
  emitIndices(Attribute layout, llvm::ArrayRef<int64_t> shape,
              bool withCTAOffset) {
    auto type = RankedTensorType::get(shape, rewriter.getF16Type(), layout);
    return mlir::emitIndices(loc, rewriter, layout, type, withCTAOffset);
  }

  llvm::DenseMap<unsigned, Value>
  emitDistributedToShared(Attribute srcLayout, SharedEncodingAttr sharedLayout,
                          Type elemTy, llvm::ArrayRef<int64_t> shape,
                          bool withCTAOffset) {
    auto srcTy = RankedTensorType::get(shape, elemTy, srcLayout);
    SharedMemoryObject smemObj(getMockSmemBase(), elemTy, shape,
                               sharedLayout.getOrder(), loc, rewriter);
    return getSwizzledSharedPtrs(loc, /*inVec=*/1, srcTy, sharedLayout, elemTy,
                                 smemObj, rewriter, smemObj.offsets,
                                 smemObj.strides);
  }

private:
  Value getMockSmemBase() {
    Value mockSmemBase =
        mlir::LLVM::getSRegValue(rewriter, loc, "%mock_smem_base");
    auto llPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        loc, TypeRange{llPtrTy}, ValueRange{mockSmemBase});
    return cast.getResult(0);
  }

  // Non-static members are initialized in declaration order
  MLIRContext *context;
  LowerToLLVMOptions option;
  TritonGPUToLLVMTypeConverter typeConverter;
  Block block;
  ConversionPatternRewriter rewriter;
  Location loc;
};

//===----------------------------------------------------------------------===//
// MLIR expression evaluation
//===----------------------------------------------------------------------===//

int eval(Value value, int ctaid, int tid);

int evalThreadIdOp(mlir::gpu::ThreadIdOp threadIdOp, int ctaid, int tid) {
  auto dim = threadIdOp.getDimension();
  if (dim == mlir::gpu::Dimension::x)
    return tid;
  else if (dim == mlir::gpu::Dimension::y)
    return 0;
  else if (dim == mlir::gpu::Dimension::z)
    return 0;
  else
    llvm::report_fatal_error("Invalid thread dim");
  return 0;
}

int evalInlineAsmOp(mlir::LLVM::InlineAsmOp asmOp, int ctaid, int tid) {
  std::string asmStr = asmOp.getAsmString().str();
  if (asmStr.find("%cluster_ctaid.x") != std::string::npos)
    return ctaid;
  else if (asmStr.find("%cluster_ctaid.y") != std::string::npos)
    return 0;
  else if (asmStr.find("%cluster_ctaid.z") != std::string::npos)
    return 0;
  else if (asmStr.find("%cluster_nctaid.x") != std::string::npos)
    llvm::report_fatal_error("%cluster_nctaid.x not supported");
  else if (asmStr.find("%cluster_nctaid.y") != std::string::npos)
    return 1;
  else if (asmStr.find("%cluster_nctaid.z") != std::string::npos)
    return 1;
  else if (asmStr.find("%mock_smem_base") != std::string::npos)
    return 0;
  else
    llvm::report_fatal_error("Unrecognized ASM string");
  return 0;
}

int evalGEPOp(mlir::LLVM::GEPOp gepOp, int ctaid, int tid) {
  assert(gepOp.getNumOperands() == 2 && "Unrecognized format of GEPOp");
  int base = eval(gepOp.getBase(), ctaid, tid);
  int offset = eval(gepOp.getOperand(1), ctaid, tid);
  auto llPtrTy = gepOp.getRes().getType().cast<LLVM::LLVMPointerType>();
  int bytesPerElem = llPtrTy.getIntOrFloatBitWidth() / 8;
  return base + offset * bytesPerElem;
}

int eval(Value value, int ctaid, int tid) {
  Operation *op = value.getDefiningOp();
  assert(op && "Unrecognized source value in the index expression");
  if (auto constantOp = llvm::dyn_cast<mlir::LLVM::ConstantOp>(op)) {
    auto attr = constantOp.getValue();
    return attr.cast<mlir::IntegerAttr>().getInt();
  } else if (auto addOp = llvm::dyn_cast<mlir::LLVM::AddOp>(op)) {
    return eval(addOp.getLhs(), ctaid, tid) + eval(addOp.getRhs(), ctaid, tid);
  } else if (auto mulOp = llvm::dyn_cast<mlir::LLVM::MulOp>(op)) {
    return eval(mulOp.getLhs(), ctaid, tid) * eval(mulOp.getRhs(), ctaid, tid);
  } else if (auto udivOp = llvm::dyn_cast<mlir::LLVM::UDivOp>(op)) {
    return eval(udivOp.getLhs(), ctaid, tid) /
           eval(udivOp.getRhs(), ctaid, tid);
  } else if (auto uremOp = llvm::dyn_cast<mlir::LLVM::URemOp>(op)) {
    return eval(uremOp.getLhs(), ctaid, tid) %
           eval(uremOp.getRhs(), ctaid, tid);
  } else if (auto xorOp = llvm::dyn_cast<mlir::LLVM::XOrOp>(op)) {
    return eval(xorOp.getLhs(), ctaid, tid) ^ eval(xorOp.getRhs(), ctaid, tid);
  } else if (auto trunciOp = llvm::dyn_cast<arith::TruncIOp>(op)) {
    return eval(trunciOp.getIn(), ctaid, tid);
  } else if (auto castOp = llvm::dyn_cast<UnrealizedConversionCastOp>(op)) {
    return eval(castOp.getOperand(0), ctaid, tid);
  } else if (auto threadOp = llvm::dyn_cast<mlir::gpu::ThreadIdOp>(op)) {
    return evalThreadIdOp(threadOp, ctaid, tid);
  } else if (auto asmOp = llvm::dyn_cast<mlir::LLVM::InlineAsmOp>(op)) {
    return evalInlineAsmOp(asmOp, ctaid, tid);
  } else if (auto gepOp = llvm::dyn_cast<mlir::LLVM::GEPOp>(op)) {
    return evalGEPOp(gepOp, ctaid, tid);
  } else {
    llvm::report_fatal_error("Unrecognized op type in the index expression");
    return 0;
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Dump Distributed Layout
//===----------------------------------------------------------------------===//

std::string dumpDistributedLayout(Attribute layout,
                                  llvm::ArrayRef<int64_t> shape,
                                  bool multiCTA) {
  assert(isaDistributedLayout(layout) &&
         "Unsupported layout type for dumpDistributedLayout");

  assert(shape.size() > 0 && "Empty shape");
  assert(shape.size() <= 2 &&
         "High order tensor is not supported in dumpLayout");

  int numThreads = 32 * getNumWarpsPerCTA(layout);
  int numCTAs = getNumCTAs(layout);
  auto f16Ty = FloatType::getF16(layout.getContext());
  int numElems = getTotalElemsPerThread(layout, shape, f16Ty);

  if (!multiCTA)
    assert(numCTAs == 1 && "numCTAs must be 1 when multiCTA is false");

  IndexEmitter emitter(layout.getContext());
  auto indices = emitter.emitIndices(layout, shape, multiCTA);
  assert(indices.size() == numElems && "Incorrect number of indices emitted");

  auto genStr = [multiCTA](int ctaid, int tid, int idx) -> std::string {
    std::ostringstream oss;
    if (multiCTA)
      oss << "CTA" << ctaid << ":";
    oss << "T" << tid << ":" << idx;
    return oss.str();
  };

  std::ostringstream oss;

  auto dumpLayout1d = [&]() {
    for (int idx = 0; idx < numElems; ++idx)
      assert(indices[idx].size() == 1 && "Incorrect rank of indices emitted");

    int size = shape[0];
    std::vector<std::string> mapping(size);

    for (int ctaid = 0; ctaid < numCTAs; ++ctaid) {
      for (int tid = 0; tid < numThreads; ++tid) {
        for (int idx = 0; idx < numElems; ++idx) {
          int i = eval(indices[idx][0], ctaid, tid);
          assert(i >= 0 && i < size && "Invalid index emitted");
          std::string &value = mapping[i];
          if (value.empty())
            value = genStr(ctaid, tid, idx);
          else
            value = value + "|" + genStr(ctaid, tid, idx);
        }
      }
    }

    for (int i = 0; i < size; ++i) {
      if (i > 0)
        oss << ",";
      oss << mapping[i];
    }
    oss << "\n";
  };

  auto dumpLayout2d = [&]() {
    for (int idx = 0; idx < numElems; ++idx)
      assert(indices[idx].size() == 2 && "Incorrect rank of indices emitted");

    int row = shape[0], col = shape[1];
    std::vector<std::vector<std::string>> mapping(
        row, std::vector<std::string>(col));

    for (int ctaid = 0; ctaid < numCTAs; ++ctaid) {
      for (int tid = 0; tid < numThreads; ++tid) {
        for (int idx = 0; idx < numElems; ++idx) {
          int r = eval(indices[idx][0], ctaid, tid);
          int c = eval(indices[idx][1], ctaid, tid);
          assert(r >= 0 && r < row && c >= 0 && c < col &&
                 "Invalid index emitted");
          std::string &value = mapping[r][c];
          if (value.empty())
            value = genStr(ctaid, tid, idx);
          else
            value = value + "|" + genStr(ctaid, tid, idx);
        }
      }
    }

    for (int r = 0; r < row; ++r) {
      for (int c = 0; c < col; ++c) {
        if (c > 0)
          oss << ",";
        oss << mapping[r][c];
      }
      oss << "\n";
    }
  };

  if (shape.size() == 1)
    dumpLayout1d();
  else
    dumpLayout2d();

  return oss.str();
}

//===----------------------------------------------------------------------===//
// Dump Shared Layout
//===----------------------------------------------------------------------===//

std::string dumpSharedLayout(Attribute layout, llvm::ArrayRef<int64_t> shape,
                             Type elemTy, bool multiCTA) {
  assert(shape.size() == 2 && "Only 2d shape supported in dumpSharedLayout");
  int row = shape[0], col = shape[1];
  int size = row * col;
  int bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
  int totalBytes = size * bytesPerElem;

  int numWarps = 1;
  int numThreads = 32 * numWarps;
  int numCTAs = getNumCTAs(layout);

  if (!multiCTA)
    assert(numCTAs == 1 && "numCTAs must be 1 when multiCTA is false");

  auto sharedLayout = layout.cast<SharedEncodingAttr>();
  auto blockedLayout = BlockedEncodingAttr::get(
      /*context=*/layout.getContext(), /*shape=*/shape,
      /*sizePerThread=*/{1, 1}, /*order=*/sharedLayout.getOrder(),
      /*numWarps=*/numWarps, 32, /*CTALayout=*/sharedLayout.getCTALayout());

  int numElems = getTotalElemsPerThread(blockedLayout, shape, elemTy);

  IndexEmitter emitter(layout.getContext());
  auto blockedIndices = emitter.emitIndices(blockedLayout, shape, multiCTA);
  auto sharedPtrs = emitter.emitDistributedToShared(blockedLayout, sharedLayout,
                                                    elemTy, shape, multiCTA);

  assert(blockedIndices.size() == numElems &&
         "Incorrect number of indices emitted by blockedLayout");
  assert(sharedPtrs.size() == numElems &&
         "Incorrect number of pointers emitted by sharedLayout");

  for (int idx = 0; idx < numElems; ++idx)
    assert(blockedIndices[idx].size() == 2 &&
           "Incorrect rank of indices emitted by blockedLayout");

  auto genStr = [](int r, int c) -> std::string {
    std::ostringstream oss;
    oss << "(" << r << ":" << c << ")";
    return oss.str();
  };

  std::vector<std::string> mapping(size);
  for (int ctaid = 0; ctaid < numCTAs; ++ctaid) {
    for (int tid = 0; tid < numThreads; ++tid) {
      for (int idx = 0; idx < numElems; ++idx) {
        int r = eval(blockedIndices[idx][0], ctaid, tid);
        int c = eval(blockedIndices[idx][1], ctaid, tid);
        assert(r >= 0 && r < row && c >= 0 && c < col &&
               "Invalid index emitted");
        int ptr = eval(sharedPtrs[idx], ctaid, tid);
        assert(ptr % bytesPerElem == 0 && ptr < totalBytes &&
               "Invalid pointer emitted");
        std::string &value = mapping[ptr / bytesPerElem];
        if (value.empty())
          value = genStr(r, c);
        else
          value = value + "|" + genStr(r, c);
      }
    }
  }

  const int bytesPerBank = 4;
  const int totalBanks = 32;
  const int bytesPerLine =
      std::min(col * bytesPerElem, bytesPerBank * totalBanks);
  int elemsPerLine = bytesPerLine / bytesPerElem;

  std::ostringstream oss;

  for (int i = 0; i < size; ++i) {
    int r = i / elemsPerLine;
    int c = i % elemsPerLine;
    if (c > 0)
      oss << ",";
    oss << mapping[i];
    if (c == elemsPerLine - 1)
      oss << "\n";
  }

  return oss.str();
}

} // namespace gpu
} // namespace triton
} // namespace mlir
