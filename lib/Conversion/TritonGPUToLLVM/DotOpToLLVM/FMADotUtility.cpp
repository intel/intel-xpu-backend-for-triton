#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Value.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

/// OperandValueKey structure represents compile time part
/// of spatial coordinates of a value in a tensor.
///
/// Every Value spatial coordinates(i.e. [batch;nonK;k]) in tensor can be
/// defined as:
///
/// batch = (bRepIdx * CTABSize + bIdx) + (laneBCoord + warpBCoord)
/// nonK = (nonKRepIdx * CTANKSize + nonKIdx) + (laneNonKCoord + warpNonKCoord)
/// k = kIdx
///
/// Where:
/// CTABSize, CTANKSize: constants;
/// laneBCoord, warpBCoord, laneNonKCoord, warpNonKCoord: runtime components;
/// bRepIdx, nonKRepIdx, bIdx, nonKIdx, kIdx: compile time components.
struct OperandValueKey {
  unsigned bRepIdx, nonKRepIdx;
  unsigned bIdx, nonKIdx, kIdx;

  bool operator==(const OperandValueKey &other) const {
    return (bRepIdx == other.bRepIdx && nonKRepIdx == other.nonKRepIdx &&
            bIdx == other.bIdx && nonKIdx == other.nonKIdx &&
            kIdx == other.kIdx);
  }
};

} // namespace

template <> struct std::hash<OperandValueKey> {
  std::size_t operator()(const OperandValueKey &k) const {
    return llvm::hash_combine(k.bRepIdx, k.nonKRepIdx, k.bIdx, k.nonKIdx,
                              k.kIdx);
  }
};

namespace {

using ValueTableFMA = std::unordered_map<OperandValueKey, Value>;

ValueTableFMA getValueTableFromStructFMA(
    Value val, ArrayRef<unsigned> perRepShape, ArrayRef<unsigned> repetitions,
    unsigned kDim, unsigned nonKDim, ConversionPatternRewriter &rewriter,
    Location loc, ArrayRef<unsigned> inRepOrder, ArrayRef<unsigned> repOrder) {
  ValueTableFMA res;
  auto elems = unpackLLElements(loc, val, rewriter);
  assert(perRepShape.size() == 3);
  auto numElemsRep = product(perRepShape);
  assert(elems.size() == numElemsRep * product(repetitions));
  assert(kDim == 1 || kDim == 2);
  assert(nonKDim == 1 || nonKDim == 2);
  const unsigned bDim = 0;

  for (unsigned idx = 0; idx < elems.size(); ++idx) {
    auto inRepLinearIdx = idx % numElemsRep;
    auto repLinearIdx = idx / numElemsRep;
    auto inRepSpatialIdx =
        mlir::LLVM::delinearize(inRepLinearIdx, perRepShape, inRepOrder);
    auto repSpatialIdx =
        mlir::LLVM::delinearize(repLinearIdx, repetitions, repOrder);
    OperandValueKey key{repSpatialIdx[0], repSpatialIdx[nonKDim],
                        inRepSpatialIdx[0], inRepSpatialIdx[nonKDim],
                        inRepSpatialIdx[kDim]};
    res[key] = elems[idx];
  }
  return res;
}

struct LoopInfo {
  Block *header;
  Block *body;
  Block *end;
};

/// Creates an empty loop structure with a header, body, and end block. The
/// loop is initialized with an induction variable and an initial argument.
/// - Parameters:
///   - `iv`: The induction variable (already initialized to the lower bound).
///   - `ub`: The upper bound for the loop.
///   - `step`: The step for the induction variable.
///   - `initArg`: The initial argument passed to the loop.
/// - Returns a `LoopInfo` structure containing the header, body, and end///
///   blocks.
LoopInfo createEmptyLoop(Value iv, Value ub, Value step, Value initArg,
                         ConversionPatternRewriter &rewriter, Location loc) {
  // Create loop harness.
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  Block *insertionBlock = rewriter.getInsertionBlock();
  Block *headerBlock =
      rewriter.splitBlock(insertionBlock, rewriter.getInsertionPoint());
  Block *bodyBlock = rewriter.splitBlock(headerBlock, headerBlock->begin());
  Block *endBlock = rewriter.splitBlock(bodyBlock, bodyBlock->begin());

  Type ivTy = iv.getType();
  Type initArgTy = initArg.getType();
  headerBlock->addArguments({ivTy, initArgTy}, {loc, loc});
  bodyBlock->addArguments({ivTy, initArgTy}, {loc, loc});
  endBlock->addArgument(initArgTy, loc);

  // Loop header.
  rewriter.setInsertionPointToEnd(insertionBlock);
  rewriter.create<cf::BranchOp>(loc, headerBlock,
                                SmallVector<Value>{iv, initArg});
  rewriter.setInsertionPointToStart(headerBlock);

  auto cond = b.icmp_slt(headerBlock->getArgument(0), ub);
  auto headerArgs = headerBlock->getArguments();
  rewriter.create<cf::CondBranchOp>(loc, cond, bodyBlock, headerArgs, endBlock,
                                    headerArgs.drop_front());
  rewriter.setInsertionPointToStart(bodyBlock);

  // Loop body.
  auto nextIV = b.add(bodyBlock->getArgument(0), step);
  SmallVector<Value> args1{nextIV};
  args1.push_back(bodyBlock->getArgument(1));
  rewriter.create<cf::BranchOp>(loc, headerBlock, args1);
  rewriter.setInsertionPointToStart(endBlock);

  return {headerBlock, bodyBlock, endBlock};
}

/// Initializes a variable to a given value and returns the loaded value.
/// - Parameters:
///   - `init`: The initial value to assign to the variable.
/// - Returns: The loaded value of the initialized variable.
Value createIV(Value init, ConversionPatternRewriter &rewriter, Location loc) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ptr = rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(rewriter.getContext()),
                                             init.getType(), b.i32_val(1));
  rewriter.create<LLVM::StoreOp>(loc, init, ptr);
  return rewriter.create<LLVM::LoadOp>(loc, init.getType(), ptr);
}

} // namespace

namespace mlir::triton::gpu {

enum class CodeGenMode {
  Unroll,
  Loop,
} codeGenMode = CodeGenMode::Loop;

LogicalResult genFMALoop(DotOp, ValueTableFMA &, ValueTableFMA &,
                         ArrayRef<Value>, ArrayRef<unsigned>,
                         ArrayRef<unsigned>, unsigned, Type,
                         ConversionPatternRewriter &);

LogicalResult parametricConvertFMADot(DotOp op, DotOp::Adaptor adaptor,
                                      const LLVMTypeConverter *typeConverter,
                                      ConversionPatternRewriter &rewriter,
                                      FMAVectorMultiplier &multiplier) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  auto A = op.getA();
  auto D = op.getResult();

  auto aTensorTy = cast<RankedTensorType>(A.getType());
  auto dTensorTy = cast<RankedTensorType>(D.getType());

  SmallVector<int64_t> aShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(aTensorTy)));
  auto dShapePerCTA =
      expandMatrixShapeWithBatch(ArrayRef(getShapePerCTA(dTensorTy)));

  BlockedEncodingAttr dLayout =
      cast<BlockedEncodingAttr>(dTensorTy.getEncoding());
  // TODO process A and B operand separately
  auto inRepOrder = expandMatrixOrderWithBatch(dLayout.getOrder());
  auto repOrder = expandMatrixOrderWithBatch(dLayout.getRepOrder());
  auto cc = unpackLLElements(loc, adaptor.getC(), rewriter);

  Value llA = adaptor.getA();
  Value llB = adaptor.getB();

  auto sizePerThread = getContigPerThread(dTensorTy);
  auto numElemsPerThread = product(sizePerThread);
  SmallVector<unsigned> shapePerCTATile;
  for (auto [reg, thread, warp] :
       llvm::zip(sizePerThread, dLayout.getThreadsPerWarp(),
                 dLayout.getWarpsPerCTA())) {
    shapePerCTATile.push_back(reg * thread * warp);
  }
  shapePerCTATile = expandMatrixShapeWithBatch(ArrayRef(shapePerCTATile));
  sizePerThread = expandMatrixShapeWithBatch(ArrayRef(sizePerThread));

  unsigned K = aShapePerCTA[2];

  unsigned threadTileShape[3];
  unsigned repetitions[3];
  for (int i = 0; i < 3; ++i) {
    repetitions[i] =
        ceil(dShapePerCTA[i], static_cast<int64_t>(shapePerCTATile[i]));
  }

  auto has = getValueTableFromStructFMA(
      llA, {sizePerThread[0], sizePerThread[1], K},
      {repetitions[0], repetitions[1], 1},
      /*kDim*/ 2, /*nonKDim*/ 1, rewriter, loc, inRepOrder, repOrder);
  auto hbs = getValueTableFromStructFMA(
      llB, {sizePerThread[0], K, sizePerThread[2]},
      {repetitions[0], 1, repetitions[2]},
      /*kDim*/ 1, /*nonKDim*/ 2, rewriter, loc, inRepOrder, repOrder);

  SmallVector<Value> acc = cc;

  if (codeGenMode == CodeGenMode::Loop) {
    Type dType = typeConverter->convertType(dTensorTy);
    return genFMALoop(op, has, hbs, acc, sizePerThread, repetitions, K, dType,
                      rewriter);
  }

  for (unsigned bRep = 0; bRep < repetitions[0]; ++bRep)
    for (unsigned mRep = 0; mRep < repetitions[1]; ++mRep)
      for (unsigned nRep = 0; nRep < repetitions[2]; ++nRep)
        for (unsigned b = 0; b < sizePerThread[0]; ++b)
          for (unsigned m = 0; m < sizePerThread[1]; ++m)
            for (unsigned n = 0; n < sizePerThread[2]; ++n) {
              SmallVector<unsigned> multiDimAccumIdx = {b, m, n};
              unsigned linearInRepIdx =
                  LLVM::linearize(multiDimAccumIdx, sizePerThread, inRepOrder);
              SmallVector<unsigned> multiDimRepIdx = {bRep, mRep, nRep};
              unsigned linearRepIdx =
                  LLVM::linearize(multiDimRepIdx, repetitions, repOrder);
              unsigned linearAccumIdx =
                  linearInRepIdx + linearRepIdx * numElemsPerThread;

              SmallVector<Value> aOpVector;
              SmallVector<Value> bOpVector;

              for (unsigned k = 0; k < K; ++k) {
                aOpVector.push_back(has.at({bRep, mRep, b, m, k}));
                bOpVector.push_back(hbs.at({bRep, nRep, b, n, k}));
              }

              acc[linearAccumIdx] = multiplier.multiplyVectors(
                  aOpVector, bOpVector, acc[linearAccumIdx]);
            }

  auto res = packLLElements(loc, typeConverter, acc, rewriter, dTensorTy);
  rewriter.replaceOp(op, res);

  return success();
}

LogicalResult genFMALoop(DotOp op, ValueTableFMA &has, ValueTableFMA &hbs,
                         ArrayRef<Value> acc, ArrayRef<unsigned> sizePerThread,
                         ArrayRef<unsigned> repetitions, unsigned K, Type dType,
                         ConversionPatternRewriter &rewriter) {
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  MLIRContext *ctx = rewriter.getContext();
  Location loc = op.getLoc();

  // Copy struct into vector for operand A.
  SmallVector<Value> v1;
  for (unsigned bRep = 0; bRep < repetitions[0]; ++bRep)
    for (unsigned mRep = 0; mRep < repetitions[1]; ++mRep)
      for (unsigned b = 0; b < sizePerThread[0]; ++b)
        for (unsigned m = 0; m < sizePerThread[1]; ++m)
          for (unsigned k = 0; k < K; ++k)
            v1.push_back(has.at({bRep, mRep, b, m, k}));
  Value vecA = packLLVector(loc, v1, rewriter);

  // Copy struct into vector for operand B.
  SmallVector<Value> v2;
  for (unsigned bRep = 0; bRep < repetitions[0]; ++bRep)
    for (unsigned nRep = 0; nRep < repetitions[2]; ++nRep)
      for (unsigned b = 0; b < sizePerThread[0]; ++b)
        for (unsigned n = 0; n < sizePerThread[2]; ++n)
          for (unsigned k = 0; k < K; ++k)
            v2.push_back(hbs.at({bRep, nRep, b, n, k}));
  Value vecB = packLLVector(loc, v2, rewriter);

  // Copy struct into vector for operand C.
  Value vecC = packLLVector(loc, acc, rewriter);

  const unsigned len = acc.size();
  Type elemType = acc.front().getType();
  auto builder = TritonLLVMOpBuilder(loc, rewriter);

  Value vecD;
  Value zero = builder.i32_val(0), one = builder.i32_val(1);
  for (unsigned bRep = 0; bRep < repetitions[0]; ++bRep)
    for (unsigned mRep = 0; mRep < repetitions[1]; ++mRep)
      for (unsigned nRep = 0; nRep < repetitions[2]; ++nRep)
        for (unsigned b = 0; b < sizePerThread[0]; ++b) {
          // Generate the outer loop.
          Value outerIV = createIV(zero, rewriter, loc);
          Value outerUB = builder.i32_val(sizePerThread[1]);
          Value outerStep = builder.i32_val(sizePerThread[2]);
          LoopInfo outerLoopInfo = createEmptyLoop(outerIV, outerUB, outerStep,
                                                   {vecC}, rewriter, loc);
          Block *outerBody = outerLoopInfo.body;
          Block *outerEnd = outerLoopInfo.end;
          auto outerLatch = cast<cf::BranchOp>(outerBody->getTerminator());
          vecD = outerEnd->getArgument(0);
          auto afterOuterLoop = rewriter.saveInsertionPoint();
          rewriter.setInsertionPointToStart(outerLoopInfo.body);

          // Get the values for operand A.
          SmallVector<Value> AElems;
          for (unsigned i = 0; i < sizePerThread[2]; ++i) {
            Value idx = builder.add(outerIV, builder.i32_val(i));
            AElems.push_back(builder.extract_element(vecA, idx));
          }

          // Generate the inner loop.
          Value innerIV = createIV(zero, rewriter, loc);
          Value innerUB = outerStep;
          Value innerStep = one;
          Value initArg =
              outerBody->getArgument(outerBody->getNumArguments() - 1);
          LoopInfo innerLoopInfo = createEmptyLoop(innerIV, innerUB, innerStep,
                                                   initArg, rewriter, loc);
          Block *innerBody = innerLoopInfo.body;
          Block *innerEnd = innerLoopInfo.end;
          rewriter.setInsertionPointToStart(innerLoopInfo.body);

          // Get the values for operand B.
          SmallVector<Value> BElems;
          for (unsigned j = 0; j < sizePerThread[2]; ++j) {
            Value idx =
                builder.add(innerIV, builder.i32_val(sizePerThread[2] * j));
            BElems.push_back(builder.extract_element(vecB, idx));
          }

          // Get the value for operand C.
          Value accIdx = builder.add(builder.mul(innerUB, outerIV), innerIV);
          Value innerInitArg =
              innerBody->getArgument(innerBody->getNumArguments() - 1);
          Value acc = builder.extract_element(innerInitArg, accIdx);

          // Perform the FMAs.
          for (unsigned k = 0; k < sizePerThread[2]; ++k) {
            TypeSwitch<Type>(elemType)
                .Case<FloatType>([&](auto) {
                  acc = rewriter.create<LLVM::FMulAddOp>(loc, AElems[k],
                                                         BElems[k], acc);
                })
                .Case<IntegerType>([&](auto) {
                  acc = builder.fma(AElems[k], BElems[k], acc);
                });
          }

          // Store the result.
          innerInitArg = builder.insert_element(innerInitArg, acc, accIdx);
          rewriter.restoreInsertionPoint(afterOuterLoop);

          // Pass the result to the next inner loop iteration.
          auto innerLatch = cast<cf::BranchOp>(innerBody->getTerminator());
          innerLatch->setOperand(innerLatch->getNumOperands() - 1,
                                 innerInitArg);

          // Pass the result of the inner loop to the next outer loop iteration.
          Value innerEndArg = innerEnd->getArgument(0);
          outerLatch->setOperand(outerLatch->getNumOperands() - 1, innerEndArg);
        }

  // Create a loop to copy the result into a struct.
  Value ub = builder.i32_val(len);
  auto structPtr =
      rewriter.create<LLVM::AllocaOp>(loc, ptr_ty(ctx), elemType, ub);
  Value iv = createIV(zero, rewriter, loc);
  LoopInfo loopInfo = createEmptyLoop(iv, ub, one, vecD, rewriter, loc);
  Block *body = loopInfo.body;
  auto afterLoop = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(body);
  Value val = builder.extract_element(
      body->getArgument(body->getNumArguments() - 1), iv);
  Value ptr = builder.gep(ptr_ty(ctx), val.getType(), structPtr, iv);
  rewriter.create<LLVM::StoreOp>(loc, val, ptr);

  // Load the struct and replace the original op.
  rewriter.restoreInsertionPoint(afterLoop);
  auto loadVal = rewriter.create<LLVM::LoadOp>(loc, dType, structPtr);
  rewriter.replaceOp(op, loadVal);

  llvm::errs() << "at line: " << __LINE__ << "\n";
  llvm::errs() << "Module after:\n";
  mod->dumpPretty();
  llvm::errs() << "\n";

  return success();
}

} // namespace mlir::triton::gpu
