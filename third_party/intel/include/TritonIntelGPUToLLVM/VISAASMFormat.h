#ifndef TRITON_CONVERSION_TRITON_GPU_TO_LLVM_VISA_ASM_FORMAT_H_
#define TRITON_CONVERSION_TRITON_GPU_TO_LLVM_VISA_ASM_FORMAT_H_

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
class ConversionPatternRewriter;
class Location;

namespace triton {
using llvm::StringRef;

struct VISAInstr;
struct VISAInstrCommon;
struct VISAInstrExecution;

// VISABuilder helps to manage a VISA asm program.
//
// A helper for building an ASM program, the objective of VISABuilder is to give
// a thin encapsulation and make the ASM code for MLIR LLVM Dialect clearer.
// Currently, several factors are introduced to reduce the need for mixing
// string and C++ if-else code.
//
// Usage:
// To build:
// ```
// @$3 asm("@%3 add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k), "b"(p));
// ```
// ```cpp
// VISABuilder builder;
// auto& add = builder.create<>();
// add.predicate(pVal).o("lo").o("u32"); // add any suffix
// // predicate here binds %0 to pVal, pVal is a mlir::Value
//
// auto* iOpr = builder.newOperand(iVal, "r"); // %1 bind to iVal
// auto* jOpr = builder.newOperand(jVal, "r"); // %2 bind to jVal
// auto* kOpr = builder.newOperand(kVal, "r"); // %3 bind to kVal
// add(iOpr, jOpr, kOpr).predicate(predVal);   // set operands and predicate
//
// To get the asm code:
// builder.dump()
// ```
// To get all the `mlir::Value` used in the VISA code,
// ```cpp
// builder.getAllMlirArgs() // get {pVal, iVal, jVal, kVal}
// ```
// To get the string containing all the constraints separated by a comma:
// ```cpp
// builder.getConstraints() // get "=r,r,k"
// ```
// VISABuilder can build a VISA asm with multiple instructions, sample code:
// ```cpp
// VISABuilder builder;
// auto& mov = builder.create("mov");
// auto& cp = builder.create("cp");
// mov(...);
// cp(...);
// ```
// This will get a VISA code with two instructions.
//
// Similar to a C function, a declared VISAInstr instance can be launched
// multiple times with different operands, e.g.
// ```cpp
//   auto& mov = builder.create("mov");
//   mov(... some operands ...);
//   mov(... some different operands ...);
// ```
// Finally, we will get a VISA code with two mov instructions.
//
// There are several derived instruction type for typical instructions, for
// example, the PtxIOInstr for ld and st instructions.
struct VISABuilder {
  struct Operand {
    std::string constraint;
    Value value;
    int idx{-1};
    llvm::SmallVector<Operand *> list;
    std::function<std::string(int idx)> repr;

    // for list
    Operand() = default;
    Operand(const Operation &) = delete;
    Operand(Value value, StringRef constraint)
        : constraint(constraint), value(value) {}

    bool isList() const { return !value && constraint.empty(); }

    Operand *listAppend(Operand *arg) {
      list.push_back(arg);
      return this;
    }

    Operand *listGet(size_t nth) const {
      assert(nth < list.size());
      return list[nth];
    }

    std::string dump() const;
  };

  template <typename INSTR = VISAInstr, typename... Args>
  INSTR *create(Args &&...args) {
    instrs.emplace_back(std::make_unique<INSTR>(this, args...));
    return static_cast<INSTR *>(instrs.back().get());
  }

  // Create a list of operands.
  Operand *newListOperand() { return newOperand(); }

  Operand *newListOperand(ArrayRef<std::pair<mlir::Value, std::string>> items) {
    auto *list = newOperand();
    for (auto &item : items) {
      list->listAppend(newOperand(item.first, item.second));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, mlir::Value val,
                          const std::string &constraint) {
    auto *list = newOperand();
    for (unsigned i = 0; i < count; ++i) {
      list->listAppend(newOperand(val, constraint));
    }
    return list;
  }

  Operand *newListOperand(unsigned count, const std::string &constraint) {
    auto *list = newOperand();
    for (unsigned i = 0; i < count; ++i) {
      list->listAppend(newOperand(constraint));
    }
    return list;
  }

  // Create a new operand. It will not add to operand list.
  // @value: the MLIR value bind to this operand.
  // @constraint: ASM operand constraint, .e.g. "=r"
  // @formatter: extra format to represent this operand in ASM code, default is
  //             "%{0}".format(operand.idx).
  Operand *newOperand(mlir::Value value, StringRef constraint,
                      std::function<std::string(int idx)> formatter = nullptr);

  // Create a new operand which is written to, that is, the constraint starts
  // with "=", e.g. "=r".
  // If the operand will be used in predicated execution,
  // users may want to initialize it before use.
  // Otherwise if the register is only used in the true branch or the false
  // branch but not both, the register is undefined and ptxas can perform
  // aggressive optimizations that may lead to incorrect results.
  Operand *newOperand(StringRef constraint, bool init = false);

  // Create a new operand that is tied to a previous operand. In this case the
  // asm would be permitted to write to an input register. Instead of providing
  // constraint code for this operand, the constraint code of the tied operand
  // is used.
  Operand *newOperand(unsigned operandIndex);

  // Create a constant integer operand.
  Operand *newConstantOperand(int64_t v);
  // Create a constant operand with explicit code specified.
  Operand *newConstantOperand(const std::string &v);

  Operand *newAddrOperand(mlir::Value addr, StringRef constraint, int off = 0);

  llvm::SmallVector<Operand *, 4> getAllArgs() const;

  llvm::SmallVector<Value, 4> getAllMLIRArgs() const;

  std::string getConstraints() const;

  std::string dump() const;

  mlir::Value launch(OpBuilder &rewriter, Location loc, Type resTy,
                     bool hasSideEffect = true, bool isAlignStack = false,
                     ArrayRef<Attribute> attrs = {}) const;

private:
  Operand *newOperand() {
    argArchive.emplace_back(std::make_unique<Operand>());
    return argArchive.back().get();
  }

  void initOperand(Operand *opr);

  // Make the operands in argArchive follow the provided \param order.
  void reorderArgArchive(ArrayRef<Operand *> order) {
    assert(order.size() == argArchive.size());
    // The order in argArchive is unnecessary when onlyAttachMLIRArgs=false, but
    // it does necessary when onlyAttachMLIRArgs is true for the $0, $1... are
    // determined by VISA code snippet passed from external.
    sort(argArchive.begin(), argArchive.end(),
         [&](std::unique_ptr<Operand> &a, std::unique_ptr<Operand> &b) {
           auto ida = std::find(order.begin(), order.end(), a.get());
           auto idb = std::find(order.begin(), order.end(), b.get());
           assert(ida != order.end());
           assert(idb != order.end());
           return ida < idb;
         });
  }

  friend struct VISAInstr;
  friend struct VISAInstrCommon;

protected:
  llvm::SmallVector<std::unique_ptr<Operand>, 6> argArchive;
  llvm::SmallVector<std::unique_ptr<VISAInstrCommon>, 2> instrs;
  llvm::SmallVector<std::unique_ptr<VISAInstrExecution>, 4> executions;
  int oprCounter{};
};

// VISA instruction common interface.
// Put the generic logic for all the instructions here.
struct VISAInstrCommon {
  explicit VISAInstrCommon(VISABuilder *builder) : builder(builder) {}

  using Operand = VISABuilder::Operand;

  // Set operands of this instruction.
  VISAInstrExecution &operator()(llvm::ArrayRef<Operand *> oprs,
                                 bool onlyAttachMLIRArgs = false);

protected:
  // "Call" the instruction with operands.
  // \param oprs The operands of this instruction.
  // \param onlyAttachMLIRArgs Indicate that it simply attach the MLIR Arguments
  // to the inline Asm without generating the operand ids(such as $0, $1) in
  // VISA code.
  VISAInstrExecution &call(llvm::ArrayRef<Operand *> oprs,
                           bool onlyAttachMLIRArgs = false);

  VISABuilder *builder{};
  llvm::SmallVector<std::string, 4> instrParts;

  friend struct VISAInstrExecution;
};

template <class ConcreteT> struct VISAInstrBase : public VISAInstrCommon {
  using Operand = VISABuilder::Operand;

  explicit VISAInstrBase(VISABuilder *builder, const std::string &name)
      : VISAInstrCommon(builder) {
    o(name);
  }

  // Append a suffix to the instruction.
  // e.g. VISAInstr("add").o("s32") get a add.s32.
  // A predicate is used to tell whether to apply the suffix, so that no if-else
  // code needed. e.g. `VISAInstr("add").o("s32", isS32).o("u32", !isS32);` will
  // get a `add.s32` if isS32 is true.
  ConcreteT &o(const std::string &suffix, bool predicate = true) {
    if (predicate)
      instrParts.push_back(suffix);
    return *static_cast<ConcreteT *>(this);
  }
};

struct VISAInstr : public VISAInstrBase<VISAInstr> {
  using VISAInstrBase<VISAInstr>::VISAInstrBase;

  // Append a ".global" to the instruction.
  VISAInstr &global();

  // Append a ".shared" to the instruction.
  VISAInstr &shared();

  // Append a ".v[0-9]+" to the instruction
  VISAInstr &v(int vecWidth, bool predicate = true);

  // Append a".b[0-9]+" to the instruction
  VISAInstr &b(int width);
};

// Record the operands and context for "launching" a PtxInstr.
struct VISAInstrExecution {
  using Operand = VISABuilder::Operand;

  llvm::SmallVector<Operand *> argsInOrder;

  VISAInstrExecution() = default;
  explicit VISAInstrExecution(VISAInstrCommon *instr,
                              llvm::ArrayRef<Operand *> oprs,
                              bool onlyAttachMLIRArgs)
      : argsInOrder(oprs.begin(), oprs.end()), instr(instr),
        onlyAttachMLIRArgs(onlyAttachMLIRArgs) {}

  // Prefix a predicate to the instruction.
  VISAInstrExecution &predicate(mlir::Value value, StringRef constraint = "b") {
    pred = instr->builder->newOperand(value, constraint);
    return *this;
  }

  // Prefix a !predicate to the instruction.
  VISAInstrExecution &predicateNot(mlir::Value value, StringRef constraint) {
    pred = instr->builder->newOperand(value, constraint);
    pred->repr = [](int idx) { return "@!$" + std::to_string(idx); };
    return *this;
  }

  std::string dump() const;

  SmallVector<Operand *> getArgList() const;

  VISAInstrCommon *instr{};
  Operand *pred{};
  bool onlyAttachMLIRArgs{};
};

/// ====== Some instruction wrappers ======
// We add the wrappers to make the usage more intuitive by avoiding mixing the
// VISA code with some trivial C++ code.

struct VISACpAsyncLoadInstr : VISAInstrBase<VISACpAsyncLoadInstr> {
  explicit VISACpAsyncLoadInstr(VISABuilder *builder,
                                triton::CacheModifier modifier)
      : VISAInstrBase(builder, "cp.async") {
    o(triton::stringifyCacheModifier(modifier).str());
    o("shared");
    o("global");
  }
};

} // namespace triton
} // namespace mlir

#endif
