﻿#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/GENXDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// A custom op builder that keeps track of the last location
class TritonOpBuilder {
public:
  TritonOpBuilder(mlir::MLIRContext *context) {
    builder = std::make_unique<mlir::OpBuilder>(context);
    lastLoc = std::make_unique<mlir::Location>(builder->getUnknownLoc());
  }

  mlir::OpBuilder &getBuilder() { return *builder; }

  bool isLineInfoEnabled() { return lineInfoEnabled; }

  void setLastLoc(mlir::Location loc) {
    if (lineInfoEnabled)
      lastLoc = std::make_unique<mlir::Location>(loc);
  }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(mlir::FileLineColLoc::get(context, fileName, line, column));
  }

  mlir::Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(mlir::Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(mlir::Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(mlir::Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(mlir::OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(),
                   mlir::Value>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::Location> lastLoc;
  bool lineInfoEnabled = !triton::tools::getBoolEnv("TRITON_DISABLE_LINE_INFO");
};

static std::string locationToString(mlir::Location loc) {
  std::string str;
  llvm::raw_string_ostream os(str);
  loc.print(os);
  os.flush(); // Make sure all the content is dumped into the 'str' string
  return str;
}

static void outputWarning(mlir::Location loc, const std::string &msg) {
  std::string locStr = locationToString(loc);

  PyErr_WarnEx(PyExc_UserWarning, (locStr + ": " + msg).c_str(),
               /*stack_level=*/2);
}

/*****************************************************************************/
/* Python bindings for triton::ir                                            */
/*****************************************************************************/

void init_triton_ir(py::module &&m) {
  using ret = py::return_value_policy;
  using namespace pybind11::literals;

  py::enum_<mlir::triton::PaddingOption>(m, "PADDING_OPTION",
                                         py::module_local())
      .value("PAD_ZERO", mlir::triton::PaddingOption::PAD_ZERO)
      .value("PAD_NAN", mlir::triton::PaddingOption::PAD_NAN)
      .export_values();

  py::enum_<mlir::triton::CacheModifier>(m, "CACHE_MODIFIER",
                                         py::module_local())
      .value("NONE", mlir::triton::CacheModifier::NONE)
      .value("CA", mlir::triton::CacheModifier::CA)
      .value("CG", mlir::triton::CacheModifier::CG)
      .value("WB", mlir::triton::CacheModifier::WB)
      .value("CS", mlir::triton::CacheModifier::CS)
      .value("WT", mlir::triton::CacheModifier::WT)
      .export_values();

  py::enum_<mlir::triton::MemSemantic>(m, "MEM_SEMANTIC", py::module_local())
      .value("ACQUIRE_RELEASE", mlir::triton::MemSemantic::ACQUIRE_RELEASE)
      .value("ACQUIRE", mlir::triton::MemSemantic::ACQUIRE)
      .value("RELEASE", mlir::triton::MemSemantic::RELEASE)
      .value("RELAXED", mlir::triton::MemSemantic::RELAXED)
      .export_values();

  py::enum_<mlir::triton::MemSyncScope>(m, "MEM_SYNC_SCOPE", py::module_local())
      .value("GPU", mlir::triton::MemSyncScope::GPU)
      .value("CTA", mlir::triton::MemSyncScope::CTA)
      .value("SYSTEM", mlir::triton::MemSyncScope::SYSTEM)
      .export_values();

  py::enum_<mlir::triton::EvictionPolicy>(m, "EVICTION_POLICY",
                                          py::module_local())
      .value("NORMAL", mlir::triton::EvictionPolicy::NORMAL)
      .value("EVICT_FIRST", mlir::triton::EvictionPolicy::EVICT_FIRST)
      .value("EVICT_LAST", mlir::triton::EvictionPolicy::EVICT_LAST)
      .export_values();

  py::enum_<mlir::triton::RMWOp>(m, "ATOMIC_OP", py::module_local())
      .value("ADD", mlir::triton::RMWOp::ADD)
      .value("FADD", mlir::triton::RMWOp::FADD)
      .value("AND", mlir::triton::RMWOp::AND)
      .value("OR", mlir::triton::RMWOp::OR)
      .value("XOR", mlir::triton::RMWOp::XOR)
      .value("XCHG", mlir::triton::RMWOp::XCHG)
      .value("MAX", mlir::triton::RMWOp::MAX)
      .value("MIN", mlir::triton::RMWOp::MIN)
      .value("UMIN", mlir::triton::RMWOp::UMIN)
      .value("UMAX", mlir::triton::RMWOp::UMAX);

  py::enum_<mlir::triton::RoundingMode>(m, "ROUNDING_MODE", py::module_local())
      .value("RTZ", mlir::triton::RoundingMode::RTZ)
      .value("RTNE", mlir::triton::RoundingMode::RTNE);

  py::class_<mlir::MLIRContext>(m, "context", py::module_local())
      .def(py::init<>())
      .def("load_triton", [](mlir::MLIRContext &self) {
        self.getOrLoadDialect<mlir::triton::TritonDialect>();
        self.getOrLoadDialect<mlir::index::IndexDialect>();
        self.getOrLoadDialect<mlir::triton::TritonDialect>();
        self.getOrLoadDialect<mlir::gpu::GPUDialect>();
        // we load LLVM because the frontend uses LLVM.undef for
        // some placeholders
        self.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
        self.getOrLoadDialect<mlir::tensor::TensorDialect>();
      });

  py::class_<mlir::Type>(m, "type", py::module_local())
      .def("is_integer", &mlir::Type::isInteger)
      .def("is_fp16", &mlir::Type::isF16)
      .def("__str__", [](mlir::Type &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<mlir::FunctionType>(m, "function_type", py::module_local())
      .def("param_types", [](mlir::FunctionType &self) {
        return std::vector<mlir::Type>(self.getInputs().begin(),
                                       self.getInputs().end());
      });

  py::class_<mlir::Location>(m, "location", py::module_local())
      .def("__str__", [](mlir::Location &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<mlir::Value>(m, "value", py::module_local())
      .def("set_attr",
           [](mlir::Value &self, std::string &name,
              mlir::Attribute &attr) -> void {
             if (mlir::Operation *definingOp = self.getDefiningOp())
               definingOp->setAttr(name, attr);
             else {
               auto arg = self.cast<mlir::BlockArgument>();
               int id = arg.getArgNumber();
               std::string attrName = name + "_arg" + std::to_string(id);
               mlir::Block *owner = arg.getOwner();
               if (owner->isEntryBlock() &&
                   !mlir::isa<mlir::triton::FuncOp>(owner->getParentOp())) {
                 owner->getParentOp()->setAttr(attrName, attr);
               }
             }
           })
      .def("get_context", &mlir::Value::getContext)
      .def("replace_all_uses_with",
           [](mlir::Value &self, mlir::Value &newValue) {
             self.replaceAllUsesWith(newValue);
           })
      .def("get_type", &mlir::Value::getType);

  py::class_<mlir::BlockArgument, mlir::Value>(m, "block_argument",
                                               py::module_local());

  py::class_<mlir::Region>(m, "region", py::module_local())
      .def("get_parent_region", &mlir::Region::getParentRegion, ret::reference)
      .def("size", [](mlir::Region &self) { return self.getBlocks().size(); })
      .def("empty", &mlir::Region::empty);

  py::class_<mlir::Block>(m, "block", py::module_local())
      .def("arg",
           [](mlir::Block &self, int index) -> mlir::BlockArgument {
             return self.getArgument(index);
           })
      .def("add_argument",
           [](mlir::Block &self, mlir::Type ty) {
             auto loc = mlir::UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &mlir::Block::getNumArguments)
      .def("dump", &mlir::Block::dump)
      .def("move_before", &mlir::Block::moveBefore)
      .def("insert_before", &mlir::Block::insertBefore)
      .def("get_parent", &mlir::Block::getParent, ret::reference)
      .def("merge_block_before",
           [](mlir::Block &self, mlir::Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error(
                   "This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](mlir::Block &self, mlir::Value &v, mlir::Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](mlir::OpOperand &operand) {
               mlir::Operation *user = operand.getOwner();
               mlir::Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock =
                     currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](mlir::Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator",
           [](mlir::Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<mlir::OpTrait::IsTerminator>();
           })
      .def("has_return",
           [](mlir::Block &self) {
             return !self.empty() &&
                    self.back().hasTrait<mlir::OpTrait::ReturnLike>();
           })
      .def("erase", [](mlir::Block &self) { self.erase(); });

  py::class_<mlir::Attribute>(m, "attribute", py::module_local());
  py::class_<mlir::IntegerAttr, mlir::Attribute>(m, "integer_attr",
                                                 py::module_local());
  py::class_<mlir::BoolAttr, mlir::Attribute>(m, "bool_attr",
                                              py::module_local());

  // Ops
  py::class_<mlir::OpState>(m, "OpState", py::module_local())
      .def("set_attr",
           [](mlir::OpState &self, std::string &name,
              mlir::Attribute &attr) -> void { self->setAttr(name, attr); })
      .def(
          "get_num_results",
          [](mlir::OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](mlir::OpState &self, unsigned idx) -> mlir::Value {
             return self->getResult(idx);
           })
      .def(
          "get_region",
          [](mlir::OpState &self, unsigned idx) -> mlir::Region & {
            return self->getRegion(idx);
          },
          ret::reference)
      .def(
          "get_body",
          [](mlir::scf::ForOp &self, unsigned idx) -> mlir::Block * {
            return self.getBody(idx);
          },
          ret::reference)
      .def("dump", [](mlir::OpState &self) { self->dump(); })
      .def("__str__",
           [](mlir::OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self->print(os, printingFlags);
             return str;
           })
      .def("append_operand",
           [](mlir::OpState &self, mlir::Value &val) {
             self->insertOperands(self->getNumOperands(), val);
           })
      .def("verify", [](mlir::OpState &self) -> bool {
        return mlir::succeeded(mlir::verify(self.getOperation()));
      });
  // scf Ops
  py::class_<mlir::scf::ForOp, mlir::OpState>(m, "ForOp", py::module_local())
      .def("get_induction_var", &mlir::scf::ForOp::getInductionVar);

  py::class_<mlir::scf::IfOp, mlir::OpState>(m, "IfOp", py::module_local())
      .def("get_then_block", &mlir::scf::IfOp::thenBlock, ret::reference)
      .def("get_else_block", &mlir::scf::IfOp::elseBlock, ret::reference)
      .def("get_then_yield", &mlir::scf::IfOp::thenYield)
      .def("get_else_yield", &mlir::scf::IfOp::elseYield);
  py::class_<mlir::scf::YieldOp, mlir::OpState>(m, "YieldOp",
                                                py::module_local());
  py::class_<mlir::scf::WhileOp, mlir::OpState>(m, "WhileOp",
                                                py::module_local())
      .def("get_before", &mlir::scf::WhileOp::getBefore, ret::reference)
      .def("get_after", &mlir::scf::WhileOp::getAfter, ret::reference);
  py::class_<mlir::scf::ConditionOp, mlir::OpState>(m, "ConditionOp",
                                                    py::module_local());

  // dynamic_attr is used to transfer ownership of the MLIR context to the
  // module
  py::class_<mlir::ModuleOp, mlir::OpState>(m, "module", py::module_local(),
                                            py::dynamic_attr())
      .def("dump", &mlir::ModuleOp::dump)
      .def("str",
           [](mlir::ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self.print(os, printingFlags);
             return str;
           })
      .def("push_back",
           [](mlir::ModuleOp &self, mlir::triton::FuncOp &funcOp) -> void {
             self.push_back(funcOp);
           })
      .def("has_function",
           [](mlir::ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](mlir::ModuleOp &self,
              std::string &funcName) -> mlir::triton::FuncOp {
             return self.lookupSymbol<mlir::triton::FuncOp>(funcName);
           })
      .def("get_int_attr",
           [](mlir::ModuleOp &self, std::string name) -> py::object {
             auto ret = self->getAttrOfType<mlir::IntegerAttr>(name);
             if (!ret)
               return py::none();
             return py::int_(ret.getInt());
           });

  m.def("make_attr",
        [](const std::vector<int> &values, mlir::MLIRContext &context) {
          return mlir::DenseIntElementsAttr::get(
                     mlir::RankedTensorType::get(
                         {static_cast<int64_t>(values.size())},
                         mlir::IntegerType::get(&context, 32)),
                     values)
              .cast<mlir::Attribute>();
        });

  m.def(
      "parse_mlir_module",
      [](const std::string &inputFilename, mlir::MLIRContext &context) {
        // initialize registry
        // note: we initialize llvm for undef
        mlir::DialectRegistry registry;
        registry.insert<
            mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
            mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
            mlir::triton::nvgpu::NVGPUDialect, mlir::math::MathDialect,
            mlir::arith::ArithDialect, mlir::index::IndexDialect,
            mlir::scf::SCFDialect, mlir::cf::ControlFlowDialect,
            mlir::LLVM::LLVMDialect>();
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        // parse module
        mlir::OwningOpRef<mlir::ModuleOp> module =
            mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, &context);
        if (!module)
          throw std::runtime_error("Parse MLIR file failed.");
        // locations are incompatible with ptx < 7.5 !
        module->walk([](mlir::Operation *op) {
          op->setLoc(mlir::UnknownLoc::get(op->getContext()));
        });

        return module->clone();
      },
      ret::take_ownership);

  py::class_<mlir::triton::FuncOp, mlir::OpState>(m, "function",
                                                  py::module_local())
      // .def_property_readonly("attrs", &ir::function::attrs)
      // .def("add_attr", &ir::function::add_attr);
      .def("args",
           [](mlir::triton::FuncOp &self, unsigned idx) -> mlir::BlockArgument {
             return self.getArgument(idx);
           })
      .def(
          "add_entry_block",
          [](mlir::triton::FuncOp &self) -> mlir::Block * {
            return self.addEntryBlock();
          },
          ret::reference)
      .def(
          "set_arg_attr",
          [](mlir::triton::FuncOp &self, int arg_no, const std::string &name,
             int val) {
            // set arg attributes "name" to value "val"
            auto attrTy = mlir::IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, mlir::IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      //  .def("has_attr", &mlir::::FuncOp::hasAttr)
      .def("finalize",
           [](mlir::triton::FuncOp &self) -> void {
             // Remove dead code
             // 1. Unreachable code after return
             self.walk([&](mlir::Block *block) {
               mlir::Operation *retOp = nullptr;
               // It's better to not use walk here because we only want to
               // check operations in the current block
               for (auto &op : block->getOperations()) {
                 if (mlir::isa<mlir::triton::ReturnOp>(op))
                   if (retOp == nullptr) {
                     retOp = &op;
                     break;
                   }
               }
               if (retOp && retOp != &block->back()) {
                 auto pos = retOp->getIterator();
                 pos++;
                 auto *newBlock = block->splitBlock(pos);
                 newBlock->erase();
               }
             });
             // 2. Check if the result of tl.advance is used
             self.walk([&](mlir::Operation *op) {
               if (mlir::isa<mlir::triton::AdvanceOp>(op) &&
                   op->getResult(0).use_empty())
                 outputWarning(op->getLoc(), "The result of tl.advance is not "
                                             "being used. Note that tl.advance "
                                             "does not have any side effects. "
                                             "To move the block pointer, you "
                                             "need to assign the result of "
                                             "tl.advance to a variable.");
             });
           })
      .def_property_readonly("type", &mlir::triton::FuncOp::getFunctionType)
      .def("reset_type", &mlir::triton::FuncOp::setType);

  py::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint",
                                           py::module_local());

  py::class_<TritonOpBuilder>(m, "builder", py::module_local(),
                              py::dynamic_attr())
      .def(py::init<mlir::MLIRContext *>())
      // getters
      .def("create_module",
           [](TritonOpBuilder &self) -> mlir::ModuleOp {
             return self.create<mlir::ModuleOp>();
           })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](TritonOpBuilder &self, mlir::Block &block) -> void {
             self.setInsertionPointToStart(block);
           })
      .def("set_insertion_point_to_end",
           [](TritonOpBuilder &self, mlir::Block &block) {
             self.setInsertionPointToEnd(block);
           })
      .def("set_insertion_point_after",
           [](TritonOpBuilder &self, mlir::Operation &op) {
             self.setInsertionPointAfter(op);
           })
      .def(
          "get_insertion_block",
          [](TritonOpBuilder &self) -> mlir::Block * {
            return self.getBuilder().getInsertionBlock();
          },
          ret::reference)
      .def("get_insertion_point",
           [](TritonOpBuilder &self) {
             return self.getBuilder().saveInsertionPoint();
           })
      .def("restore_insertion_point",
           [](TritonOpBuilder &self, mlir::OpBuilder::InsertPoint pt) {
             self.restoreInsertionPoint(pt);
           })
      // Attr
      .def("get_bool_attr",
           [](TritonOpBuilder &self, bool value) {
             return self.getBuilder().getBoolAttr(value);
           })
      .def("get_int32_attr",
           [](TritonOpBuilder &self, int32_t value) {
             return self.getBuilder().getI32IntegerAttr(value);
           })
      // Use arith.ConstantOp to create constants
      // Constants
      .def("get_int1",
           [](TritonOpBuilder &self, bool v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI1Type()));
           })
      .def("get_int8",
           [](TritonOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_int16",
           [](TritonOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_int32",
           [](TritonOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_int64",
           [](TritonOpBuilder &self, int64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_uint8",
           [](TritonOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI8Type()));
           })
      .def("get_uint16",
           [](TritonOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI16Type()));
           })
      .def("get_uint32",
           [](TritonOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI32Type()));
           })
      .def("get_uint64",
           [](TritonOpBuilder &self, uint64_t v) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ConstantIntOp>(
                 v, self.getBuilder().getI64Type()));
           })
      .def("get_bf16",
           [](TritonOpBuilder &self, float v) -> mlir::Value {
             auto type = self.getBuilder().getBF16Type();
             return self.create<mlir::arith::ConstantFloatOp>(
                 mlir::APFloat(type.getFloatSemantics(), std::to_string(v)),
                 type);
           })
      .def("get_fp16",
           [](TritonOpBuilder &self, float v) -> mlir::Value {
             return self.create<mlir::arith::ConstantOp>(
                 self.getBuilder().getF16FloatAttr(v));
           })
      .def("get_fp32",
           [](TritonOpBuilder &self, float v) -> mlir::Value {
             return self.create<mlir::arith::ConstantOp>(
                 self.getBuilder().getF32FloatAttr(v));
           })
      .def("get_fp64",
           [](TritonOpBuilder &self, double v) -> mlir::Value {
             return self.create<mlir::arith::ConstantOp>(
                 self.getBuilder().getF64FloatAttr(v));
           })
      .def("get_null_value",
           [](TritonOpBuilder &self, mlir::Type type) -> mlir::Value {
             if (auto floatTy = type.dyn_cast<mlir::FloatType>())
               return self.create<mlir::arith::ConstantFloatOp>(
                   mlir::APFloat(floatTy.getFloatSemantics(), 0), floatTy);
             else if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(0, intTy);
             else
               throw std::runtime_error("Not implemented");
           })
      .def("get_all_ones_value",
           [](TritonOpBuilder &self, mlir::Type type) -> mlir::Value {
             uint64_t val = 0xFFFFFFFFFFFFFFFF;
             if (auto intTy = type.dyn_cast<mlir::IntegerType>())
               return self.create<mlir::arith::ConstantIntOp>(val, intTy);
             else
               throw std::runtime_error("Not implemented");
           })

      // Types
      .def("get_void_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getNoneType();
           })
      .def("get_int1_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI1Type();
           }) // or ret::copy?
      .def("get_int8_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI8Type();
           })
      .def("get_int16_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getType<mlir::IntegerType>(16);
           })
      .def("get_int32_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI32Type();
           })
      .def("get_int64_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getI64Type();
           })
      .def("get_fp8e4nv_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getType<mlir::Float8E4M3FNUZType>();
           })
      .def("get_fp8e4b15_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             // TODO: upstream FP8E4B15 into MLIR, or find a way to externally
             // have a float-like type compatible with float only native ops
             return self.getBuilder().getType<mlir::Float8E4M3B11FNUZType>();
           })
      .def("get_fp8e4b15x4_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             // TODO: upstream FP8E4B15 into MLIR, or find a way to externally
             // have a float-like type compatible with float only native ops
             return self.getBuilder().getType<mlir::Float8E4M3FNType>();
           })
      .def("get_fp8e5_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getType<mlir::Float8E5M2Type>();
           })
      .def("get_half_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getF16Type();
           })
      .def("get_bf16_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getBF16Type();
           })
      .def("get_float_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getF32Type();
           })
      .def("get_double_ty",
           [](TritonOpBuilder &self) -> mlir::Type {
             return self.getBuilder().getF64Type();
           })
      .def("get_ptr_ty",
           [](TritonOpBuilder &self, mlir::Type &type,
              int addrSpace) -> mlir::Type {
             return mlir::triton::PointerType::get(type, addrSpace);
           })
      .def("get_block_ty",
           [](TritonOpBuilder &self, mlir::Type &elementType,
              std::vector<int64_t> &shape) -> mlir::Type {
             return mlir::RankedTensorType::get(shape, elementType);
           })
      .def("get_function_ty",
           [](TritonOpBuilder &self, std::vector<mlir::Type> inTypes,
              std::vector<mlir::Type> outTypes) -> mlir::Type {
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })
      // locs
      .def("set_loc", [](TritonOpBuilder &self,
                         mlir::Location loc) { self.setLastLoc(loc); })
      .def("set_loc",
           [](TritonOpBuilder &self, const std::string &fileName, int line,
              int column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc",
           [](TritonOpBuilder &self) -> mlir::Location {
             return self.getLastLoc();
           })

      // Ops
      .def("get_or_insert_function",
           [](TritonOpBuilder &self, mlir::ModuleOp &module,
              std::string &funcName, mlir::Type &funcType,
              std::string &visibility, bool noinline) -> mlir::triton::FuncOp {
             if (mlir::Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<mlir::triton::FuncOp>(funcOperation);
             if (auto funcTy = funcType.dyn_cast<mlir::FunctionType>()) {
               llvm::SmallVector<mlir::NamedAttribute> attrs = {
                   mlir::NamedAttribute(
                       self.getBuilder().getStringAttr("sym_visibility"),
                       self.getBuilder().getStringAttr(visibility)),
                   mlir::NamedAttribute(
                       self.getBuilder().getStringAttr("noinline"),
                       self.getBuilder().getBoolAttr(noinline))};
               return self.create<mlir::triton::FuncOp>(funcName, funcTy,
                                                        attrs);
             }
             throw std::runtime_error("invalid function type");
           })
      .def(
          "create_block",
          [](TritonOpBuilder &self) -> mlir::Block * {
            mlir::Region *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          ret::reference)
      .def(
          "create_block_with_parent",
          [](TritonOpBuilder &self, mlir::Region &parent,
             std::vector<mlir::Type> &argTypes) -> mlir::Block * {
            // TODO: update arg loc
            auto loc = self.getBuilder().getUnknownLoc();
            llvm::SmallVector<mlir::Location, 8> argLocs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes,
                                                 argLocs);
          },
          ret::reference)
      .def(
          "new_block",
          [](TritonOpBuilder &self) -> mlir::Block * {
            return new mlir::Block();
          },
          ret::reference)
      // Function
      .def("ret",
           [](TritonOpBuilder &self,
              std::vector<mlir::Value> &vals) -> mlir::OpState {
             return self.create<mlir::triton::ReturnOp>(vals);
           })
      .def("call",
           [](TritonOpBuilder &self, mlir::triton::FuncOp &func,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             return self.create<mlir::triton::CallOp>(func, args);
           })
      // Unstructured control flow
      .def("create_cond_branch",
           [](TritonOpBuilder &self, mlir::Value condition,
              mlir::Block *trueDest, mlir::Block *falseDest) -> mlir::OpState {
             return self.create<mlir::cf::CondBranchOp>(condition, trueDest,
                                                        falseDest);
           })
      .def("create_branch",
           [](TritonOpBuilder &self, mlir::Block *dest,
              std::vector<mlir::Value> &args) -> mlir::OpState {
             return self.create<mlir::cf::BranchOp>(dest, args);
           })
      // Structured control flow
      .def("create_for_op",
           [](TritonOpBuilder &self, mlir::Value &lb, mlir::Value &ub,
              mlir::Value &step,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::ForOp {
             return self.create<mlir::scf::ForOp>(lb, ub, step, initArgs);
           })
      .def("create_if_op",
           [](TritonOpBuilder &self, std::vector<mlir::Type> &retTypes,
              mlir::Value &condition, bool withElse) -> mlir::scf::IfOp {
             return self.create<mlir::scf::IfOp>(retTypes, condition, withElse);
           })
      .def("create_yield_op",
           [](TritonOpBuilder &self,
              std::vector<mlir::Value> &yields) -> mlir::scf::YieldOp {
             return self.create<mlir::scf::YieldOp>(yields);
           })
      .def("create_while_op",
           [](TritonOpBuilder &self, std::vector<mlir::Type> &retTypes,
              std::vector<mlir::Value> &initArgs) -> mlir::scf::WhileOp {
             return self.create<mlir::scf::WhileOp>(retTypes, initArgs);
           })
      .def("create_condition_op",
           [](TritonOpBuilder &self, mlir::Value &cond,
              std::vector<mlir::Value> &args) -> mlir::scf::ConditionOp {
             return self.create<mlir::scf::ConditionOp>(cond, args);
           })

      // miscellaneous
      .def("create_make_range",
           [](TritonOpBuilder &self, int start, int end) -> mlir::Value {
             auto retType = mlir::RankedTensorType::get(
                 {end - start}, self.getBuilder().getI32Type());
             return self.create<mlir::triton::MakeRangeOp>(retType, start, end);
           })

      // Cast instructions
      // Conversions for custom FP types (FP8 and non-standard rounding modes)
      .def("create_fp_to_fp",
           [](TritonOpBuilder &self, mlir::Value &src, mlir::Type &dstType,
              std::optional<::mlir::triton::RoundingMode> roundingMode)
               -> mlir::Value {
             if (roundingMode.has_value())
               return self.create<mlir::triton::FpToFpOp>(
                   dstType, src,
                   mlir::triton::RoundingModeAttr::get(
                       self.getBuilder().getContext(), roundingMode.value()));
             else
               return self.create<mlir::triton::FpToFpOp>(dstType, src);
           })
      // Conversions for standard LLVM builtin types
      .def("create_bitcast",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::triton::BitcastOp>(dstType, src);
           })
      .def("create_si_to_fp",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::SIToFPOp>(dstType, src);
           })
      .def("create_ui_to_fp",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::UIToFPOp>(dstType, src);
           })
      .def("create_fp_to_si",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::FPToSIOp>(dstType, src);
           })
      .def("create_fp_to_ui",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::FPToUIOp>(dstType, src);
           })
      .def("create_fp_ext",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::ExtFOp>(dstType, src);
           })
      .def("create_fp_trunc",
           [](TritonOpBuilder &self, mlir::Value &src,
              mlir::Type &dstType) -> mlir::Value {
             return self.create<mlir::arith::TruncFOp>(dstType, src);
           })
      .def("create_int_cast",
           [](TritonOpBuilder &self, mlir::Value &src, mlir::Type &dstType,
              bool isSigned) -> mlir::Value {
             // get element type if necessary
             mlir::Type srcType = src.getType();
             auto srcTensorType = srcType.dyn_cast<mlir::RankedTensorType>();
             auto dstTensorType = dstType.dyn_cast<mlir::RankedTensorType>();
             mlir::Type srcEltType = srcType;
             mlir::Type dstEltType = dstType;
             if (dstTensorType && srcTensorType) {
               dstEltType = dstTensorType.getElementType();
               srcEltType = srcTensorType.getElementType();
             }
             unsigned srcWidth = srcEltType.getIntOrFloatBitWidth();
             unsigned dstWidth = dstEltType.getIntOrFloatBitWidth();
             if (srcWidth == dstWidth)
               return self.create<mlir::arith::BitcastOp>(dstType, src);
             else if (srcWidth > dstWidth)
               return self.create<mlir::arith::TruncIOp>(dstType, src);
             else if (isSigned)
               return self.create<mlir::arith::ExtSIOp>(dstType, src);
             else
               return self.create<mlir::arith::ExtUIOp>(dstType, src);
           })
      .def("create_to_index",
           [](TritonOpBuilder &self, mlir::Value &input) -> mlir::Value {
             return self.create<mlir::arith::IndexCastOp>(
                 self.getBuilder().getIndexType(), input);
           })
      .def("create_index_to_si",
           [](TritonOpBuilder &self, mlir::Value &input) -> mlir::Value {
             return self.create<mlir::arith::IndexCastOp>(
                 self.getBuilder().getI64Type(), input);
           })
      .def("create_fmul",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::MulFOp>(lhs, rhs);
           })
      .def("create_fdiv",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::DivFOp>(lhs, rhs);
           })
      .def("create_frem",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::RemFOp>(lhs, rhs);
           })
      .def("create_fadd",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::AddFOp>(lhs, rhs);
           })
      .def("create_fsub",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::SubFOp>(lhs, rhs);
           })
      .def("create_mul",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::MulIOp>(lhs, rhs);
           })
      .def("create_sdiv",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::DivSIOp>(lhs, rhs);
           })
      .def("create_udiv",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::DivUIOp>(lhs, rhs);
           })
      .def("create_srem",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::RemSIOp>(lhs, rhs);
           })
      .def("create_urem",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::RemUIOp>(lhs, rhs);
           })
      .def("create_add",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::AddIOp>(lhs, rhs);
           })
      .def("create_sub",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::SubIOp>(lhs, rhs));
           })
      .def("create_shl",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ShLIOp>(lhs, rhs));
           })
      .def("create_lshr",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ShRUIOp>(lhs, rhs));
           })
      .def("create_ashr",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::ShRSIOp>(lhs, rhs));
           })
      .def("create_minsi",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinSIOp>(lhs, rhs));
           })
      .def("create_minui",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinUIOp>(lhs, rhs));
           })
      // minimumf follows the torch.minimum convention and returns NaN if either
      // operand is NaN
      .def("create_minimumf",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinimumFOp>(lhs, rhs));
           })
      // minnumf follows the torch.fmin convention and returns the non-NaN
      // operand
      .def("create_minnumf",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MinNumFOp>(lhs, rhs));
           })
      .def("create_maxsi",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaxSIOp>(lhs, rhs));
           })
      .def("create_maxui",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaxUIOp>(lhs, rhs));
           })
      // maximumf follows the torch.maximum convention and returns NaN if either
      // operand is NaN
      .def("create_maximumf",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaximumFOp>(lhs, rhs));
           })
      // maxnumf follows the torch.fmax convention and returns the non-NaN
      // operand
      .def("create_maxnumf",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return mlir::Value(self.create<mlir::arith::MaxNumFOp>(lhs, rhs));
           })
      // AddPtr (similar to GEP)
      .def("create_addptr",
           [](TritonOpBuilder &self, mlir::Value &ptr,
              mlir::Value &offset) -> mlir::Value {
             return self.create<mlir::triton::AddPtrOp>(ptr.getType(), ptr,
                                                        offset);
           })
      // Comparison (int)
      .def("create_icmpSLE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::sle, lhs, rhs);
           })
      .def("create_icmpSLT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::slt, lhs, rhs);
           })
      .def("create_icmpSGE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::sge, lhs, rhs);
           })
      .def("create_icmpSGT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::sgt, lhs, rhs);
           })
      .def("create_icmpULE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ule, lhs, rhs);
           })
      .def("create_icmpULT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ult, lhs, rhs);
           })
      .def("create_icmpUGE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::uge, lhs, rhs);
           })
      .def("create_icmpUGT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ugt, lhs, rhs);
           })
      .def("create_icmpEQ",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::eq, lhs, rhs);
           })
      .def("create_icmpNE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpIOp>(
                 mlir::arith::CmpIPredicate::ne, lhs, rhs);
           })
      // Comparison (float)
      .def("create_fcmpOLT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OLT, lhs, rhs);
           })
      .def("create_fcmpOGT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OGT, lhs, rhs);
           })
      .def("create_fcmpOLE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OLE, lhs, rhs);
           })
      .def("create_fcmpOGE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OGE, lhs, rhs);
           })
      .def("create_fcmpOEQ",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
           })
      .def("create_fcmpONE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::ONE, lhs, rhs);
           })
      .def("create_fcmpULT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::ULT, lhs, rhs);
           })
      .def("create_fcmpUGT",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UGT, lhs, rhs);
           })
      .def("create_fcmpULE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::ULE, lhs, rhs);
           })
      .def("create_fcmpUGE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UGE, lhs, rhs);
           })
      .def("create_fcmpUEQ",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UEQ, lhs, rhs);
           })
      .def("create_fcmpUNE",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::CmpFOp>(
                 mlir::arith::CmpFPredicate::UNE, lhs, rhs);
           })
      // // Logical
      .def("create_and",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::AndIOp>(lhs, rhs);
           })
      .def("create_xor",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::XOrIOp>(lhs, rhs);
           })
      .def("create_or",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             return self.create<mlir::arith::OrIOp>(lhs, rhs);
           })
      // Input/Output
      .def("create_load",
           [](TritonOpBuilder &self, mlir::Value &ptrs,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             return self.create<mlir::triton::LoadOp>(
                 ptrs, cacheModifier, evictionPolicy, isVolatile);
           })
      .def("create_store",
           [](TritonOpBuilder &self, mlir::Value &ptrs, mlir::Value &value,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy) -> void {
             self.create<mlir::triton::StoreOp>(ptrs, value, cacheModifier,
                                                evictionPolicy);
           })
      .def("create_tensor_pointer_load",
           [](TritonOpBuilder &self, mlir::Value &ptr,
              std::vector<int32_t> &boundaryCheck,
              std::optional<mlir::triton::PaddingOption> paddingOption,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             return self.create<mlir::triton::LoadOp>(
                 ptr, boundaryCheck, paddingOption, cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_tensor_pointer_store",
           [](TritonOpBuilder &self, mlir::Value &ptr, mlir::Value &val,
              std::vector<int32_t> &boundaryCheck,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy) -> void {
             self.create<mlir::triton::StoreOp>(ptr, val, boundaryCheck,
                                                cacheModifier, evictionPolicy);
           })
      .def("create_masked_load",
           [](TritonOpBuilder &self, mlir::Value &ptrs, mlir::Value &mask,
              std::optional<mlir::Value> &other,
              mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy,
              bool isVolatile) -> mlir::Value {
             return self.create<mlir::triton::LoadOp>(
                 ptrs, mask, other.value_or(mlir::Value()), cacheModifier,
                 evictionPolicy, isVolatile);
           })
      .def("create_masked_store",
           [](TritonOpBuilder &self, mlir::Value &ptrs, mlir::Value &val,
              mlir::Value &mask, mlir::triton::CacheModifier cacheModifier,
              mlir::triton::EvictionPolicy evictionPolicy) -> void {
             self.create<mlir::triton::StoreOp>(ptrs, val, mask, cacheModifier,
                                                evictionPolicy);
           })
      .def("create_reshape",
           [](TritonOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape, bool allowReorder) -> mlir::Value {
             auto argType =
                 arg.getType().cast<mlir::RankedTensorType>().getElementType();
             return self.create<mlir::triton::ReshapeOp>(
                 mlir::RankedTensorType::get(shape, argType), arg,
                 allowReorder);
           })
      .def(
          "create_expand_dims",
          [](TritonOpBuilder &self, mlir::Value &arg, int axis) -> mlir::Value {
            auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
            auto argEltType = argType.getElementType();
            std::vector<int64_t> retShape = argType.getShape();
            retShape.insert(retShape.begin() + axis, 1);
            return self.create<mlir::triton::ExpandDimsOp>(
                mlir::RankedTensorType::get(retShape, argEltType), arg, axis);
          })
      .def("create_cat",
           [](TritonOpBuilder &self, mlir::Value &lhs,
              mlir::Value &rhs) -> mlir::Value {
             auto lhsType = lhs.getType().dyn_cast<mlir::RankedTensorType>();
             auto rhsType = rhs.getType().dyn_cast<mlir::RankedTensorType>();
             if (!(lhsType.getShape().size() == 1 &&
                   rhsType.getShape().size() == 1))
               throw std::runtime_error(
                   "shape not supported by cat. Expecting rank-1 inputs");
             std::vector<int64_t> shape{lhsType.getShape()[0] +
                                        rhsType.getShape()[0]};
             return self.create<mlir::triton::CatOp>(
                 mlir::RankedTensorType::get(shape, lhsType.getElementType()),
                 lhs, rhs);
           })
      .def("create_interleave",
           [](TritonOpBuilder &self, mlir::Value &a,
              mlir::Value &b) -> mlir::Value {
             auto aTy = a.getType().cast<mlir::RankedTensorType>();
             llvm::SmallVector<int64_t> shape(aTy.getShape().begin(),
                                              aTy.getShape().end());
             shape[shape.size() - 1] *= 2;
             return self.create<mlir::triton::ExperimentalInterleaveOp>(
                 mlir::RankedTensorType::get(shape, aTy.getElementType()), a,
                 b);
           })
      .def("create_trans",
           [](TritonOpBuilder &self, mlir::Value &arg) -> mlir::Value {
             auto argType = arg.getType().dyn_cast<mlir::RankedTensorType>();
             auto argEltType = argType.getElementType();
             std::vector<int64_t> retShape = argType.getShape();
             std::reverse(retShape.begin(), retShape.end());
             return self.create<mlir::triton::TransOp>(
                 mlir::RankedTensorType::get(retShape, argEltType), arg);
           })
      .def("create_broadcast",
           [](TritonOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             if (auto argType =
                     arg.getType().dyn_cast<mlir::RankedTensorType>())
               return self.createOrFold<mlir::triton::BroadcastOp>(
                   mlir::RankedTensorType::get(shape, argType.getElementType()),
                   arg);
             throw std::runtime_error(
                 "arg is not of RankedTensorType, use create_splat");
           })
      .def("create_splat",
           [](TritonOpBuilder &self, mlir::Value &arg,
              std::vector<int64_t> &shape) -> mlir::Value {
             auto argType = arg.getType();
             auto ret = self.createOrFold<mlir::triton::SplatOp>(
                 mlir::RankedTensorType::get(shape, argType), arg);
             return ret;
           })
      // // atomic
      .def("create_atomic_cas",
           [](TritonOpBuilder &self, mlir::Value &ptr, mlir::Value &cmp,
              mlir::Value &val, mlir::triton::MemSemantic sem,
              mlir::triton::MemSyncScope scope) -> mlir::Value {
             mlir::Type dstType;
             if (auto srcTensorType =
                     ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
               mlir::Type dstElemType = srcTensorType.getElementType()
                                            .cast<mlir::triton::PointerType>()
                                            .getPointeeType();
               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
                                                     dstElemType);
             } else {
               auto ptrType = mlir::getElementTypeOrSelf(ptr)
                                  .cast<mlir::triton::PointerType>();
               dstType = ptrType.getPointeeType();
             }
             return self.create<mlir::triton::AtomicCASOp>(dstType, ptr, cmp,
                                                           val, sem, scope);
           })
      .def("create_atomic_rmw",
           [](TritonOpBuilder &self, mlir::triton::RMWOp rmwOp,
              mlir::Value &ptr, mlir::Value &val, mlir::Value &mask,
              mlir::triton::MemSemantic sem,
              mlir::triton::MemSyncScope scope) -> mlir::Value {
             mlir::Type dstType;
             if (auto srcTensorType =
                     ptr.getType().dyn_cast<mlir::RankedTensorType>()) {
               mlir::Type dstElemType = srcTensorType.getElementType()
                                            .cast<mlir::triton::PointerType>()
                                            .getPointeeType();
               dstType = mlir::RankedTensorType::get(srcTensorType.getShape(),
                                                     dstElemType);
             } else {
               auto ptrType = mlir::getElementTypeOrSelf(ptr)
                                  .cast<mlir::triton::PointerType>();
               dstType = ptrType.getPointeeType();
             }
             return self.create<mlir::triton::AtomicRMWOp>(
                 dstType, rmwOp, ptr, val, mask, sem, scope);
           })
      // External
      .def("create_extern_elementwise",
           [](TritonOpBuilder &self, const std::string &libName,
              const std::string &libPath, const std::string &symbol,
              std::vector<mlir::Value> &argList, mlir::Type retType,
              bool isPure) -> mlir::Value {
             return self.create<mlir::triton::ExternElementwiseOp>(
                 retType, argList, libName, libPath, symbol, isPure);
           })
      // Built-in instruction
      .def("create_get_program_id",
           [](TritonOpBuilder &self, int axis) -> mlir::Value {
             if (axis < 0 || axis > 3)
               throw std::runtime_error("program_id must be in [0,3]");
             return self.create<mlir::triton::GetProgramIdOp>(
                 self.getBuilder().getI32Type(),
                 mlir::triton::ProgramIDDimAttr::get(
                     self.getBuilder().getContext(),
                     mlir::triton::ProgramIDDim(axis)));
           })
      .def("create_get_num_programs",
           [](TritonOpBuilder &self, int axis) -> mlir::Value {
             return self.create<mlir::triton::GetNumProgramsOp>(
                 self.getBuilder().getI32Type(),
                 self.getBuilder().getI32IntegerAttr(axis));
           })
      .def("create_dot",
           [](TritonOpBuilder &self, mlir::Value &a, mlir::Value &b,
              mlir::Value &c, bool allowTF32,
              int maxNumImpreciseAcc) -> mlir::Value {
             return self.create<mlir::triton::DotOp>(
                 c.getType(), a, b, c, allowTF32, maxNumImpreciseAcc);
           })
      .def("create_exp",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::ExpOp>(val);
           })
      .def("create_cos",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::CosOp>(val);
           })
      .def("create_sin",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::SinOp>(val);
           })
      .def("create_log",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::LogOp>(val);
           })
      .def("create_sqrt",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::SqrtOp>(val);
           })
      .def("create_fabs",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::AbsFOp>(val);
           })
      .def("create_iabs",
           [](TritonOpBuilder &self, mlir::Value &val) -> mlir::Value {
             return self.create<mlir::math::AbsIOp>(val);
           })
      .def("create_reduce",
           [](TritonOpBuilder &self, std::vector<mlir::Value> operands,
              int axis) -> mlir::OpState {
             return self.create<mlir::triton::ReduceOp>(operands, axis);
           })
      .def("create_reduce_ret",
           [](TritonOpBuilder &self, py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<mlir::Value>(arg));
             }
             return self.create<mlir::triton::ReduceReturnOp>(return_values);
           })
      .def("create_scan",
           [](TritonOpBuilder &self, std::vector<mlir::Value> operands,
              int axis) -> mlir::OpState {
             return self.create<mlir::triton::ScanOp>(operands, axis);
           })
      .def("create_scan_ret",
           [](TritonOpBuilder &self, py::args args) -> mlir::OpState {
             llvm::SmallVector<mlir::Value> return_values;
             for (const auto &arg : args) {
               return_values.push_back(py::cast<mlir::Value>(arg));
             }
             return self.create<mlir::triton::ScanReturnOp>(return_values);
           })
      .def("create_ptr_to_int",
           [](TritonOpBuilder &self, mlir::Value &val,
              mlir::Type &type) -> mlir::Value {
             return self.create<mlir::triton::PtrToIntOp>(type, val);
           })
      .def("create_int_to_ptr",
           [](TritonOpBuilder &self, mlir::Value &val,
              mlir::Type &type) -> mlir::Value {
             return self.create<mlir::triton::IntToPtrOp>(type, val);
           })
      .def("create_select",
           [](TritonOpBuilder &self, mlir::Value &condition,
              mlir::Value &trueValue, mlir::Value &falseValue) -> mlir::Value {
             return self.create<mlir::arith::SelectOp>(condition, trueValue,
                                                       falseValue);
           })
      .def("create_inline_asm",
           [](TritonOpBuilder &self, const std::string &inlineAsm,
              const std::string &constraints,
              const std::vector<mlir::Value> &values,
              const std::vector<mlir::Type> &types, bool isPure,
              int pack) -> mlir::OpState {
             return self.create<mlir::triton::ElementwiseInlineAsmOp>(
                 types, inlineAsm, constraints, isPure, pack, values);
           })
      .def("create_print",
           [](TritonOpBuilder &self, const std::string &prefix,
              const std::vector<mlir::Value> &values) -> void {
             self.create<mlir::triton::PrintOp>(
                 mlir::StringAttr::get(self.getBuilder().getContext(),
                                       llvm::StringRef(prefix)),
                 values);
           })
      .def("create_assert",
           [](TritonOpBuilder &self, mlir::Value &condition,
              const std::string &message, const std::string &fileName,
              const std::string &funcName, unsigned lineNo) -> void {
             auto messageAttr = mlir::StringAttr::get(
                 self.getBuilder().getContext(), llvm::StringRef(message));
             auto fileNameAttr = mlir::StringAttr::get(
                 self.getBuilder().getContext(), llvm::StringRef(fileName));
             auto funcNameAttr = mlir::StringAttr::get(
                 self.getBuilder().getContext(), llvm::StringRef(funcName));
             auto lineNoAttr = self.getBuilder().getI32IntegerAttr(lineNo);
             self.create<mlir::triton::AssertOp>(condition, messageAttr,
                                                 fileNameAttr, funcNameAttr,
                                                 lineNoAttr);
           })
      // Undef
      .def("create_undef",
           [](TritonOpBuilder &self, mlir::Type &type) -> mlir::Value {
             return self.create<::mlir::LLVM::UndefOp>(type);
           })
      // Force GPU barrier
      .def("create_barrier",
           [](TritonOpBuilder &self) { self.create<mlir::gpu::BarrierOp>(); })
      // Make a block pointer (tensor pointer in Triton IR)
      .def("create_make_block_ptr",
           [](TritonOpBuilder &self, mlir::Value &base,
              std::vector<mlir::Value> &shape,
              std::vector<mlir::Value> &strides,
              std::vector<mlir::Value> &offsets,
              std::vector<int32_t> &tensorShape,
              std::vector<int32_t> &order) -> mlir::Value {
             return self.create<mlir::triton::MakeTensorPtrOp>(
                 base, shape, strides, offsets, tensorShape, order);
           })
      // Advance a block pointer
      .def("create_advance",
           [](TritonOpBuilder &self, mlir::Value &ptr,
              std::vector<mlir::Value> &offsets) -> mlir::Value {
             return self.create<mlir::triton::AdvanceOp>(ptr.getType(), ptr,
                                                         offsets);
           });

  py::class_<mlir::PassManager>(m, "pass_manager", py::module_local())
      .def(py::init<mlir::MLIRContext *>())
      .def("enable_debug",
           [](mlir::PassManager &self) {
             if (!::triton::tools::getBoolEnv("MLIR_ENABLE_DUMP"))
               return;
             self.getContext()->disableMultithreading();
             auto printingFlags = mlir::OpPrintingFlags();
             printingFlags.elideLargeElementsAttrs(16);
             printingFlags.enableDebugInfo();
             auto print_always = [](mlir::Pass *, mlir::Operation *) {
               return true;
             };
             self.enableIRPrinting(
                 /*shouldPrintBeforePass=*/print_always,
                 /*shouldPrintAfterPass=*/print_always,
                 /*printModuleScope=*/true,
                 /*printAfterOnlyOnChange=*/false,
                 /*printAfterOnlyOnFailure*/ true, llvm::dbgs(), printingFlags);
           })
      .def("run", [](mlir::PassManager &self, mlir::ModuleOp &mod) {
        // TODO: maybe dump module to file and print error for better
        // diagnostics
        if (mlir::failed(self.run(mod.getOperation())))
          throw std::runtime_error("PassManager::run failed");
      });
}

void init_triton_env_vars(py::module &m) {
  m.def("get_env_vars", []() -> std::map<std::string, bool> {
    std::map<std::string, bool> envVars;
    for (const auto &envVar : triton::ENV_VARS) {
      envVars[envVar] = triton::tools::getBoolEnv(envVar);
    }
    return envVars;
  });
}
