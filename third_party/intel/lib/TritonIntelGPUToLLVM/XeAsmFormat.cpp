#include "intel/include/TritonIntelGPUToLLVM/XeAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/FormatVariadic.h>
#include <sstream>

namespace mlir {
namespace triton {

XeInstr::Operand *
XeBuilder::newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int)> formatter) {
  argArchive.emplace_back(std::make_unique<Operand>(value, constraint));
  auto *opr = argArchive.back().get();
  opr->repr = formatter;
  opr->idx = oprCounter++;
  return opr;
}

void XeBuilder::initOperand(Operand *opr) {
  auto numBits = 0;
  // Derive numBits from the constraint.
  if (opr->constraint[1] == 'c' || opr->constraint[1] == 'h')
    numBits = 16;
  else if (opr->constraint[1] == 'r')
    numBits = 32;
  else if (opr->constraint[1] == 'l')
    numBits = 64;
  else
    llvm_unreachable(("Unknown constraint: " + opr->constraint).c_str());
  // If numBits is less than 16, we use 16 as default because Xe does not
  // support 8-bit mov.
  numBits = numBits < 16 ? 16 : numBits;
  auto *zero = newConstantOperand(0);
  auto &init = create<>("mov")->o("u" + std::to_string(numBits));
  init(opr, zero);
}

XeBuilder::Operand *XeBuilder::newOperand(StringRef constraint, bool init) {
  // Constraint should be something like "=rw"
  assert(constraint[0] == '=');
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = constraint;
  if (init) {
    initOperand(opr);
  }
  return opr;
}

XeBuilder::Operand *XeBuilder::newOperand(unsigned operandIndex) {
  assert(operandIndex < oprCounter && "operand index out of range");
  auto *opr = newOperand();
  opr->idx = oprCounter++;
  opr->constraint = std::to_string(operandIndex);
  return opr;
}

XeBuilder::Operand *XeBuilder::newConstantOperand(const std::string &v) {
  argArchive.emplace_back(std::make_unique<Operand>());
  argArchive.back()->repr = [v](int idx) { return v; };
  return argArchive.back().get();
}

XeBuilder::Operand *XeBuilder::newConstantOperand(int64_t v) {
  std::stringstream ss;
  ss << "0x" << std::hex << v;
  return newConstantOperand(ss.str());
}

std::string XeBuilder::getConstraints() const {
  auto args = getAllArgs();
  llvm::SmallVector<std::string, 4> argReprs;
  for (auto arg : args)
    argReprs.push_back(arg->constraint);
  return strJoin(argReprs, ",");
}

llvm::SmallVector<Value, 4> XeBuilder::getAllMLIRArgs() const {
  llvm::SmallVector<Value, 4> res;
  for (auto &arg : argArchive) {
    if (!arg->isList() && arg->value)
      res.push_back(arg->value);
  }
  return res;
}

SmallVector<XeBuilder::Operand *, 4> XeBuilder::getAllArgs() const {
  llvm::SmallVector<Operand *, 4> res;
  for (auto &x : argArchive)
    if (!x->isList())
      res.push_back(x.get());
  return res;
}

mlir::Value XeBuilder::launch(OpBuilder &rewriter, Location loc, Type resTy,
                              bool hasSideEffect, bool isAlignStack,
                              ArrayRef<Attribute> attrs) const {
  auto *ctx = rewriter.getContext();
  auto inlineAsm = LLVM::InlineAsmOp::create(
      rewriter, loc, resTy, getAllMLIRArgs(), // operands
      dump(),                                 // asm_string
      getConstraints(),                       // constraints
      hasSideEffect,                          // has_side_effects
      isAlignStack,                           // is_align_stack
      LLVM::TailCallKind::None,
      /*convergent=*/false,
      LLVM::AsmDialectAttr::get(ctx,
                                LLVM::AsmDialect::AD_ATT), // asm_dialect
      ArrayAttr::get(ctx, attrs)                           // operand_attrs
  );

  return inlineAsm.getRes();
}

std::string XeInstr::Operand::dump() const {
  if (repr)
    return repr(idx);
  if (!isList())
    return "$" + std::to_string(idx);

  llvm::SmallVector<std::string> oprs;
  for (auto *opr : list)
    oprs.push_back(opr->dump());
  return "{ " + strJoin(oprs, ", ") + " }";
}

XeInstr::Operand *XeBuilder::newAddrOperand(mlir::Value addr,
                                            StringRef constraint, int off) {
  auto *opr = newOperand(addr, constraint);
  opr->repr = [off](int idx) -> std::string {
    std::stringstream ss;
    ss << "[ $" << idx << " + " << off << " ]";
    return ss.str();
  };

  return opr;
}

std::string XeBuilder::dump() const {
  llvm::SmallVector<std::string> lines;
  for (auto &exec : executions) {
    lines.push_back(exec->dump());
  }

  return strJoin(lines, "\n\t");
}

XeInstrExecution &XeInstrCommon::call(ArrayRef<Operand *> oprs,
                                      bool onlyAttachMLIRArgs) {
  if (onlyAttachMLIRArgs) {
    // Nearly impossible to make the $0,$1 in two Xe code snippets to point to
    // the same MLIR values in onlyAttachMLIRArgs mode.
    assert(builder->executions.empty() &&
           "builder can only hold a single execution when onlyAttachMIIRArgs "
           "is true.");
    builder->reorderArgArchive(oprs);
  }

  builder->executions.emplace_back(
      std::make_unique<XeInstrExecution>(this, oprs, onlyAttachMLIRArgs));

  return *builder->executions.back();
}

XeInstrExecution &XeInstrCommon::operator()(ArrayRef<Operand *> oprs,
                                            bool onlyAttachMLIRArgs) {
  return call(oprs, onlyAttachMLIRArgs);
}

std::string XeInstrExecution::dump() const {
  std::string osStr;
  llvm::raw_string_ostream os(osStr);

  if (pred) {
    if (!pred->repr)
      os << "@" << pred->dump() << " ";
    else
      os << pred->repr(pred->idx) << " ";
  }

  std::string instrRepr = strJoin(instr->instrParts, ".");
  if (onlyAttachMLIRArgs) {
    os << instrRepr;
    os.flush();
    return osStr;
  }

  llvm::SmallVector<std::string, 4> argReprs;
  for (auto *arg : argsInOrder) {
    argReprs.push_back(arg->dump());
  }

  std::string argsRepr = strJoin(argReprs, ", ");

  os << instrRepr << " " << argsRepr << ";";
  os.flush();
  return osStr;
}

SmallVector<XeInstrExecution::Operand *> XeInstrExecution::getArgList() const {
  SmallVector<Operand *> args;
  for (auto *arg : argsInOrder) {
    if (arg->isList())
      args.insert(args.end(), arg->list.begin(), arg->list.end());
    else
      args.push_back(arg);
  }
  return args;
}

XeInstr &XeInstr::global() {
  o("global");
  return *this;
}

XeInstr &XeInstr::shared() {
  o("shared");
  return *this;
}

XeInstr &XeInstr::v(int vecWidth, bool predicate) {
  if (vecWidth > 1) {
    o("v" + std::to_string(vecWidth), predicate);
  }
  return *this;
}

XeInstr &XeInstr::b(int width) {
  o("b" + std::to_string(width));
  return *this;
}

std::optional<std::string> XeVISAInstr::getTypeName(Type scalarTy) {
  std::string typeSyntax;
  TypeSwitch<Type>(scalarTy)
      .Case<Float32Type>([&](auto) { typeSyntax = "f"; })
      .Case<Float16Type>([&](auto) { typeSyntax = "hf"; })
      .Case<FloatType, IntegerType>([&](auto type) {
        unsigned bitWidth = type.getIntOrFloatBitWidth();
        switch (bitWidth) {
        case 8:
          typeSyntax = "ub";
          break;
        case 16:
          typeSyntax = "uw";
          break;
        case 32:
          typeSyntax = "ud";
          break;
        case 64:
          typeSyntax = "uq";
          break;
        }
      })
      .Default([&](auto) { typeSyntax = ""; });

  if (!typeSyntax.empty())
    return typeSyntax;
  return std::nullopt;
}

unsigned XeVISAInstr::getGRFSizeInBytes(XeArch arch) {
  switch (arch) {
  case Xe:
    return 8 * 4;
  case Xe2:
  case Xe3:
  default:
    return 16 * 4;
  }
}

std::string simdReduceAsm(std::string binOp, unsigned warpSize,
                          unsigned numLaneToReduce, unsigned interleave,
                          unsigned accSize, Type elemTy, XeArch arch) {

  assert(interleave > 0 && "interleave must be > 0");
  assert((interleave & (interleave - 1)) == 0 &&
         "interleave must be a power of two");
  assert((numLaneToReduce & (numLaneToReduce - 1)) == 0 &&
         "numLaneToReduce must be a power of two");

  // TODO: implement more variants.
  assert(arch != Xe ||
         warpSize == 16 && "only support warpSize=16 for on Xe arch fow now");
  unsigned numElem = accSize * warpSize;
  unsigned tempResultSize = numElem / 2;

  // Interleaved reductions can be viewed as reducing a virtual contiguous
  // group after removing the per-lane stride. In other words,
  // numLaneToReduce is the number of contiguous logica lanes reduced together.
  // The logica lanes is partitioned into clusters of size (warpSize /
  // interleave) / numLaneToReduce to do all reduce. Lanes are first interleaved
  // into groups, and then the reduction happens within each group across
  // numLaneToReduce lanes.
  //
  // The current lowering assumes the reduction stays within a single
  // interleaved group after factoring out the lane stride.
  assert(numLaneToReduce <= warpSize / interleave &&
         "numLaneToReduce must fit within one interleaved group");

  unsigned logicalReduceWidth = std::max(1u, warpSize / interleave);
  unsigned numLogicalResults = logicalReduceWidth / numLaneToReduce;
  unsigned pairDistance =
      std::max(1u, logicalReduceWidth / ((1 << numLogicalResults)));
  unsigned numResultPerInput = interleave * numLogicalResults;
  unsigned numResult = accSize * numResultPerInput;

  auto typeSyntax = XeVISAInstr::getTypeName(elemTy);
  if (!typeSyntax)
    llvm_unreachable("Unsupported scalar type");

  unsigned grfSizeInBytes = XeVISAInstr::getGRFSizeInBytes(arch);
  unsigned grfElemsPerRow =
      grfSizeInBytes / (elemTy.getIntOrFloatBitWidth() / 8);

  constexpr StringLiteral reduceBuff = R"({
  .decl temp_result v_type=G type={0} num_elts={1} align=GRF
  )";

  std::string simdAsm =
      llvm::formatv(reduceBuff.data(), *typeSyntax, tempResultSize).str();

  constexpr StringLiteral reduceBinOp =
      R"({0} (M1_NM, {1}) {2}({3}, {4})<1>  {5}({6}, {7})<{11};{12},1> {8}({9}, {10})<{11};{12},1>
  )";

  for (unsigned n = numElem, logicalRowStride = numLaneToReduce,
                logicalColNum = pairDistance;
       n > numResult;
       n >>= 1, logicalRowStride = std::max(1u, logicalRowStride >> 1),
                logicalColNum = std::max(1u, logicalColNum >> 1)) {
    unsigned physicalRowStride = logicalRowStride * interleave;
    unsigned physicalColNum = logicalColNum * interleave;
    unsigned elemPerRow = std::min(n, warpSize);
    unsigned rowNum = n / elemPerRow;
    unsigned reduceNum = (rowNum + 1) / 2;

    for (unsigned r = 0; r < reduceNum; r++) {
      unsigned dstOffset = r * elemPerRow;
      unsigned dstOffsetM = dstOffset / grfElemsPerRow;
      unsigned dstOffsetN = dstOffset % grfElemsPerRow;
      unsigned srcAOffset = r * 2 * elemPerRow;
      unsigned srcBOffset = srcAOffset + physicalColNum;
      unsigned srcAOffM = srcAOffset / grfElemsPerRow;
      unsigned srcAOffN = srcAOffset % grfElemsPerRow;
      unsigned srcBOffM = srcBOffset / grfElemsPerRow;
      unsigned srcBOffN = srcBOffset % grfElemsPerRow;
      std::string simdOps =
          llvm::formatv(
              reduceBinOp.data(), binOp, std::min(n / 2, warpSize),
              /*dst*/ logicalColNum == 1 ? "$0" : "temp_result", dstOffsetM,
              dstOffsetN,
              /*srcA*/ logicalColNum == numLaneToReduce / 2 ? "$1"
                                                            : "temp_result",
              srcAOffM, srcAOffN,
              /*srcB*/ logicalColNum == numLaneToReduce / 2 ? "$1"
                                                            : "temp_result",
              srcBOffM, srcBOffN, physicalRowStride, physicalColNum)
              .str();
      simdAsm += simdOps;
    }
  }
  simdAsm += "}";
  return simdAsm;
}

std::string simdLaneShuffleAsm(unsigned warpSize, bool packOrUnpack,
                               unsigned packSize, unsigned vecSize, Type elemTy,
                               XeArch arch) {
  assert((warpSize & (warpSize - 1)) == 0 && "warpSize must be a power of two");
  assert((packSize & (packSize - 1)) == 0 && "packSize must be a power of two");
  assert((vecSize & (vecSize - 1)) == 0 && "vecSize must be a power of two");
  assert(packSize <= vecSize && "packSize must be <= vecSize");
  assert(arch != Xe && "Doesn't support Xe arch fow now");

  auto typeSyntax = XeVISAInstr::getTypeName(elemTy);
  if (!typeSyntax)
    llvm_unreachable("Unsupported scalar type");

  constexpr unsigned maxDstHorStride = 4;
  assert(packSize <= maxDstHorStride * maxDstHorStride &&
         "pack only support 2 step at most 16 pack size.");

  bool use2StepMove = packOrUnpack ? packSize > maxDstHorStride : false;

  constexpr StringLiteral shuffleBuff = R"({
  .decl src v_type=G type={0} num_elts={1} alias=<$1, 0>
  .decl dst v_type=G type={0} num_elts={1} alias=<$0, 0>
  )";

  std::string simdAsm =
      llvm::formatv(shuffleBuff.data(), *typeSyntax, vecSize * warpSize).str();

  unsigned step2ElemSize;
  if (use2StepMove) {
    constexpr StringLiteral tmpDecl = R"(
    .decl _tmp0 v_type=G type={0} num_elts={1} align=GRF
    .decl _tmp1 v_type=G type={2} num_elts={3} alias=<_tmp0, 0>
    )";
    unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
    std::optional<std::string> aliasTypeSyntax;
    step2ElemSize = bitWidth * (packSize / maxDstHorStride);
    switch (step2ElemSize) {
    case 8:
      aliasTypeSyntax = "ub";
      break;
    case 16:
      aliasTypeSyntax = "uw";
      break;
    case 32:
      aliasTypeSyntax = "ud";
      break;
    case 64:
      aliasTypeSyntax = "uq";
      break;
    }
    if (!aliasTypeSyntax)
      llvm_unreachable("Unsupported scalar type");
    simdAsm += llvm::formatv(tmpDecl.data(), *typeSyntax, vecSize * warpSize,
                             aliasTypeSyntax,
                             vecSize * warpSize / (packSize / maxDstHorStride))
                   .str();
  }

  auto emitMove = [&](std::string dstName, std::string srcName,
                      unsigned elemSIze, unsigned packSize) {
    unsigned grfSizeInBytes = XeVISAInstr::getGRFSizeInBytes(arch);
    unsigned grfElemsPerRow = grfSizeInBytes / (elemSIze / 8);
    for (unsigned i = 0; i < vecSize; i += packSize) {
      unsigned srcOffset = i * warpSize;
      unsigned dstOffset = i * warpSize;
      for (unsigned j = 0; j < packSize; ++j) {
        unsigned packSrcOffset = srcOffset + (packOrUnpack ? j * warpSize : j);
        unsigned packDstOffset = dstOffset + (packOrUnpack ? j : j * warpSize);
        unsigned srcOffsetM = packSrcOffset / grfElemsPerRow;
        unsigned srcOffsetN = packSrcOffset % grfElemsPerRow;
        unsigned dstOffsetM = packDstOffset / grfElemsPerRow;
        unsigned dstOffsetN = packDstOffset % grfElemsPerRow;
        simdAsm +=
            llvm::formatv(
                "mov (M1_NM, {0}) {1}({2}, {3})<{4}> {5}({6}, {7})<{8};1,0>\n",
                warpSize, dstName, dstOffsetM, dstOffsetN,
                packOrUnpack ? packSize : 1, srcName, srcOffsetM, srcOffsetN,
                packOrUnpack ? 1 : packSize)
                .str();
      }
    }
  };

  if (use2StepMove) {
    emitMove("_tmp0", "src", elemTy.getIntOrFloatBitWidth(),
             packSize / maxDstHorStride);
    emitMove("dst", "_tmp1", step2ElemSize, maxDstHorStride);
  } else
    emitMove("dst", "src", elemTy.getIntOrFloatBitWidth(), packSize);

  simdAsm += "}";
  return simdAsm;
}

} // namespace triton
} // namespace mlir
