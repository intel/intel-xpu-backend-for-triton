#include "intel/include/TritonIntelGPUToLLVM/XeAsmFormat.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/AsmFormat.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/ADT/TypeSwitch.h>
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
  auto inlineAsm = rewriter.create<LLVM::InlineAsmOp>(
      loc, resTy, getAllMLIRArgs(), // operands
      dump(),                       // asm_string
      getConstraints(),             // constraints
      hasSideEffect,                // has_side_effects
      isAlignStack,                 // is_align_stack
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

std::string simdReduceAsm(std::string binOp, int warpSize, int accSize,
                          Type elemTy) {
  // TODO: implement more varients.
  assert(warpSize == 16 && "only suppor warpSize=16 for now");
  assert(accSize == warpSize && "The acc size has to be equal to size");
  unsigned numElem = accSize * warpSize;
  unsigned tempResultSize = numElem / 2;
  // TODO: to support Xe (ATS) and Xe3 (BMG).
  // PVC is 16 words per GRF.
  static constexpr unsigned align = 16;

  std::string typeSyntax;
  // TODO: support all possible reduction modes
  TypeSwitch<Type>(elemTy)
      .Case<Float32Type>([&](auto) { typeSyntax = "f"; })
      .Case<Float16Type>([&](auto) { typeSyntax = "hf"; })
      .Default([&](auto) { llvm_unreachable("Unsupported scalar type"); });

  constexpr StringLiteral reduceBuff = R"({
  .decl temp_result v_type=G type={0} num_elts={1} align=wordx{2}
  )";

  std::string simdAsm =
      llvm::formatv(reduceBuff.data(), typeSyntax, tempResultSize, align).str();

  constexpr StringLiteral reduceBinOp =
      R"({0} (M1_NM, {1}) {5}({2}, 0)<1>  {6}({7}, 0)<{3};{4},1> {6}({7}, {8})<{3};{4},1>
  )";

  for (unsigned n = numElem, rowStride = warpSize, colNum = warpSize / 2;
       n > accSize; n >>= 1, rowStride >>= 1, colNum >>= 1) {
    unsigned rowNum = n / warpSize;
    unsigned reduceNum = rowNum / 2;
    for (unsigned r = 0; r < reduceNum; r++) {
      unsigned dstOffset = r;
      unsigned srcOffM = r * 2;
      unsigned srcOffN = colNum;
      std::string simdOps =
          llvm::formatv(reduceBinOp.data(), binOp, warpSize, dstOffset,
                        rowStride, colNum, colNum == 1 ? "$0" : "temp_result",
                        colNum == warpSize / 2 ? "$1" : "temp_result", srcOffM,
                        srcOffN)
              .str();
      simdAsm += simdOps;
    }
  }
  simdAsm += "}";
  llvm::outs() << "johnlu the asm code:" << simdAsm << "\n";

  constexpr StringLiteral asmTemplate = R"({
  .decl temp_result v_type=G type=f num_elts={2} align=wordx32
  {0} (M1_NM, {1}) temp_result(0, 0)<1>  $1(0, 0)<16;8,1> $1(0, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(1, 0)<1>  $1(2, 0)<16;8,1> $1(2, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(2, 0)<1>  $1(4, 0)<16;8,1> $1(4, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(3, 0)<1>  $1(6, 0)<16;8,1> $1(6, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(4, 0)<1>  $1(8, 0)<16;8,1> $1(8, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(5, 0)<1>  $1(10, 0)<16;8,1> $1(10, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(6, 0)<1>  $1(12, 0)<16;8,1> $1(12, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(7, 0)<1>  $1(14, 0)<16;8,1> $1(14, 8)<16;8,1>
  {0} (M1_NM, {1}) temp_result(0, 0)<1>  temp_result(0, 0)<8;4,1> temp_result(0, 4)<8;4,1>
  {0} (M1_NM, {1}) temp_result(1, 0)<1>  temp_result(2, 0)<8;4,1> temp_result(2, 4)<8;4,1>
  {0} (M1_NM, {1}) temp_result(2, 0)<1>  temp_result(4, 0)<8;4,1> temp_result(4, 4)<8;4,1>
  {0} (M1_NM, {1}) temp_result(3, 0)<1>  temp_result(6, 0)<8;4,1> temp_result(6, 4)<8;4,1>
  {0} (M1_NM, {1}) temp_result(0, 0)<1>  temp_result(0, 0)<4;2,1> temp_result(0, 2)<4;2,1>
  {0} (M1_NM, {1}) temp_result(1, 0)<1>  temp_result(2, 0)<4;2,1> temp_result(2, 2)<4;2,1>
  {0} (M1_NM, {1}) $0(0, 0)<1>  temp_result(0, 0)<2;1,1> temp_result(0, 1)<2;1,1>
  })";
  std::string simdAsmExample =
      llvm::formatv(asmTemplate.data(), binOp, warpSize, tempResultSize).str();

  llvm::outs() << "johnlu the simdAsmExample code:" << simdAsmExample << "\n";
  return simdAsm;
}

} // namespace triton
} // namespace mlir
