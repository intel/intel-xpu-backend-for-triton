#include "intel/include/Analysis/StrideInfo.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "intel-stride-info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::intel {

// StrideInfo static methods
StrideInfo StrideInfo::getPessimisticValueState(Value value) {
  unsigned rank = 1;
  Type ty = value.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty))
    rank = tensorTy.getRank();
  else if (auto ptrTy = dyn_cast<triton::PointerType>(ty))
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrTy.getPointeeType()))
      rank = tensorTy.getRank();
  if (auto descTy = dyn_cast<triton::TensorDescInterface>(ty))
    rank = descTy.getBlockType().getRank();

  return StrideInfo(DimVectorT(rank, -1));
}

StrideInfo StrideInfo::join(const StrideInfo &lhs, const StrideInfo &rhs) {
  if (lhs.getRank() == 0)
    return rhs;
  if (rhs.getRank() == 0)
    return lhs;
  DimVectorT result;
  for (int d = 0; d < lhs.getRank(); ++d) {
    if (lhs.stride[d] == rhs.stride[d])
      result.push_back(lhs.stride[d]);
    else
      result.push_back(-1);
  }
  return StrideInfo(std::move(result));
}

namespace {

/// Try to extract a scalar integer constant from a Value.
/// Handles both IntegerAttr (scalar arith.constant) and SplatElementsAttr
/// (tensor constant like dense<64>).
static std::optional<int64_t> getScalarIntConstant(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;
  Attribute attr;
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp))
    attr = constOp.getValue();
  else if (auto constOp = dyn_cast<LLVM::ConstantOp>(defOp))
    attr = constOp.getValue();
  else
    return std::nullopt;

  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue().getSExtValue();
  if (auto splatAttr = dyn_cast<SplatElementsAttr>(attr)) {
    if (splatAttr.getElementType().isIntOrIndex())
      return splatAttr.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

class StrideInfoVisitor {
public:
  virtual ~StrideInfoVisitor() = default;
  virtual StrideInfo
  getStrideInfo(Operation *op,
                ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) = 0;
  virtual bool match(Operation *op) = 0;
};

template <typename OpTy>
class StrideInfoVisitorImpl : public StrideInfoVisitor {
public:
  StrideInfo getStrideInfo(
      Operation *op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) final {
    return getStrideInfo(cast<OpTy>(op), operands);
  }
  bool match(Operation *op) final { return isa<OpTy>(op); }
  virtual StrideInfo
  getStrideInfo(OpTy op,
                ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) = 0;
};

class StrideInfoVisitorList {
public:
  template <typename... Ts, typename = std::enable_if_t<sizeof...(Ts) != 0>>
  void append() {
    (visitors.emplace_back(std::make_unique<Ts>()), ...);
  }

  StrideInfo apply(Operation *op,
                   ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) {
    for (auto &v : visitors)
      if (v->match(op))
        return v->getStrideInfo(op, operands);
    return StrideInfo();
  }

private:
  std::vector<std::unique_ptr<StrideInfoVisitor>> visitors;
};

// PassThrough: stride passes from operand 0
template <typename OpTy>
class PassThroughStrideVisitor final : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class MakeRangeOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::MakeRangeOp> {
public:
  StrideInfo getStrideInfo(
      triton::MakeRangeOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return StrideInfo({1});
  }
};

class PoisonOpStrideVisitor final : public StrideInfoVisitorImpl<ub::PoisonOp> {
public:
  StrideInfo getStrideInfo(
      ub::PoisonOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

template <typename OpTy>
class ConstantOpStrideVisitor final : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    auto splatAttr = dyn_cast<SplatElementsAttr>(op.getValue());
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      TensorType ty = cast<TensorType>(splatAttr.getType());
      return StrideInfo(StrideInfo::DimVectorT(ty.getRank(), 0));
    }
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    if (intAttr) {
      return StrideInfo({0});
    }
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (boolAttr) {
      return StrideInfo({0});
    }
    return StrideInfo();
  }
};

template <typename OpTy>
class AddSubStrideVisitor final : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    const auto &lhs = operands[0]->getValue();
    const auto &rhs = operands[1]->getValue();
    auto rank = lhs.getRank();
    StrideInfo::DimVectorT stride;
    for (int d = 0; d < rank; ++d) {
      if (lhs.getStride(d) < 0 || rhs.getStride(d) < 0) {
        stride.push_back(-1);
      } else if constexpr (std::is_same_v<OpTy, arith::SubIOp>) {
        stride.push_back(
            std::max(lhs.getStride(d) - rhs.getStride(d), int64_t(-1)));
      } else {
        stride.push_back(lhs.getStride(d) + rhs.getStride(d));
      }
    }
    return StrideInfo(std::move(stride));
  }
};

class MulIOpStrideVisitor final : public StrideInfoVisitorImpl<arith::MulIOp> {
public:
  StrideInfo getStrideInfo(
      arith::MulIOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    const auto &lhs = operands[0]->getValue();
    const auto &rhs = operands[1]->getValue();
    auto rank = lhs.getRank();
    StrideInfo::DimVectorT stride;

    auto lhsConst = getScalarIntConstant(op.getLhs());
    auto rhsConst = getScalarIntConstant(op.getRhs());

    for (int d = 0; d < rank; ++d) {
      if (lhs.getStride(d) > 0 && rhsConst.has_value())
        stride.push_back(lhs.getStride(d) * rhsConst.value());
      else if (rhs.getStride(d) > 0 && lhsConst.has_value())
        stride.push_back(lhsConst.value() * rhs.getStride(d));
      else {
        auto strideZero = [&](const StrideInfo &si, Value v) {
          return getScalarIntConstant(v).has_value() || si.getStride(d) == 0 ||
                 !isa<TensorType>(op.getType());
        };
        if (strideZero(lhs, op.getLhs()) && strideZero(rhs, op.getRhs()))
          stride.push_back(0);
        else
          stride.push_back(-1);
      }
    }
    return StrideInfo(std::move(stride));
  }
};

template <typename OpTy>
class DivOpStrideVisitor final : public StrideInfoVisitorImpl<OpTy> {
public:
  StrideInfo getStrideInfo(
      OpTy op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    const auto &lhs = operands[0]->getValue();
    const auto &rhs = operands[1]->getValue();
    auto rank = lhs.getRank();
    StrideInfo::DimVectorT stride;

    auto rhsConst = getScalarIntConstant(op.getRhs());
    auto lhsConst = getScalarIntConstant(op.getLhs());

    for (int d = 0; d < rank; ++d) {
      if (lhs.getStride(d) > 0 && rhsConst.has_value() &&
          rhsConst.value() != 0 && lhs.getStride(d) % rhsConst.value() == 0)
        stride.push_back(lhs.getStride(d) / rhsConst.value());
      else if (rhs.getStride(d) > 0 && lhsConst.has_value() &&
               lhsConst.value() % rhs.getStride(d) == 0)
        stride.push_back(lhsConst.value() / rhs.getStride(d));
      else if (lhs.getStride(d) == 0)
        stride.push_back(0);
      else
        stride.push_back(-1);
    }
    return StrideInfo(std::move(stride));
  }
};

class SplatOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::SplatOp> {
public:
  StrideInfo getStrideInfo(
      triton::SplatOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    TensorType retTy = cast<TensorType>(*op->result_type_begin());
    return StrideInfo(StrideInfo::DimVectorT(retTy.getRank(), 0));
  }
};

class LoadOpStrideVisitor final : public StrideInfoVisitorImpl<triton::LoadOp> {
public:
  StrideInfo getStrideInfo(
      triton::LoadOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

class ExpandDimsOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::ExpandDimsOp> {
public:
  StrideInfo getStrideInfo(
      triton::ExpandDimsOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    auto opStride = operands[0]->getValue().getStride();
    StrideInfo::DimVectorT stride(opStride.begin(), opStride.end());
    stride.insert(stride.begin() + op.getAxis(), 0);
    return StrideInfo(std::move(stride));
  }
};

class BroadcastOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::BroadcastOp> {
public:
  StrideInfo getStrideInfo(
      triton::BroadcastOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class TransOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::TransOp> {
public:
  StrideInfo getStrideInfo(
      triton::TransOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    const auto &srcInfo = operands[0]->getValue();
    auto order = op.getOrder();
    StrideInfo::DimVectorT stride;
    for (int d = 0; d < srcInfo.getRank(); ++d) {
      stride.push_back(srcInfo.getStride(order[d]));
    }
    return StrideInfo(std::move(stride));
  }
};

class MakeTensorPtrOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::MakeTensorPtrOp> {
public:
  StrideInfo getStrideInfo(
      triton::MakeTensorPtrOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

class MakeTensorDescOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::MakeTensorDescOp> {
public:
  StrideInfo getStrideInfo(
      triton::MakeTensorDescOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

class DescriptorLoadOpStrideVisitor final
    : public StrideInfoVisitorImpl<triton::DescriptorLoadOp> {
public:
  StrideInfo getStrideInfo(
      triton::DescriptorLoadOp op,
      ArrayRef<const dataflow::Lattice<StrideInfo> *> operands) override {
    return StrideInfo::getPessimisticValueState(op.getResult());
  }
};

//===----------------------------------------------------------------------===//
// StrideAnalysis
//===----------------------------------------------------------------------===//

class StrideAnalysis : public dataflow::SparseForwardDataFlowAnalysis<
                           dataflow::Lattice<StrideInfo>> {
private:
  StrideInfoVisitorList visitors;

  void setToEntryState(dataflow::Lattice<StrideInfo> *lattice) override {
    propagateIfChanged(lattice,
                       lattice->join(StrideInfo::getPessimisticValueState(
                           lattice->getAnchor())));
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor & /*successor*/,
      ValueRange /*nonSuccessorInputs*/,
      ArrayRef<dataflow::Lattice<StrideInfo> *> argLattices) override {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Induction variable has stride 0 (scalar).
      auto iv = StrideInfo({0});
      (void)argLattices[0]->join(iv);
    } else {
      setAllToEntryStates(argLattices);
    }
  }

public:
  StrideAnalysis(DataFlowSolver &solver)
      : dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<StrideInfo>>(
            solver) {
    // PassThrough visitors
    visitors.append<PassThroughStrideVisitor<arith::ExtSIOp>,
                    PassThroughStrideVisitor<arith::ExtUIOp>,
                    PassThroughStrideVisitor<arith::TruncIOp>,
                    PassThroughStrideVisitor<arith::IndexCastOp>,
                    PassThroughStrideVisitor<triton::gpu::ConvertLayoutOp>,
                    PassThroughStrideVisitor<mlir::UnrealizedConversionCastOp>,
                    PassThroughStrideVisitor<triton::BitcastOp>,
                    PassThroughStrideVisitor<triton::AdvanceOp>>();
    visitors.append<MakeRangeOpStrideVisitor>();
    visitors.append<PoisonOpStrideVisitor>();
    visitors.append<ConstantOpStrideVisitor<arith::ConstantOp>,
                    ConstantOpStrideVisitor<LLVM::ConstantOp>>();
    visitors.append<AddSubStrideVisitor<triton::AddPtrOp>,
                    AddSubStrideVisitor<arith::AddIOp>,
                    AddSubStrideVisitor<arith::SubIOp>,
                    AddSubStrideVisitor<LLVM::AddOp>>();
    visitors.append<MulIOpStrideVisitor>();
    visitors.append<DivOpStrideVisitor<arith::DivSIOp>,
                    DivOpStrideVisitor<arith::DivUIOp>>();
    visitors.append<SplatOpStrideVisitor>();
    visitors.append<LoadOpStrideVisitor>();
    visitors.append<ExpandDimsOpStrideVisitor>();
    visitors.append<BroadcastOpStrideVisitor>();
    visitors.append<TransOpStrideVisitor>();
    visitors.append<MakeTensorPtrOpStrideVisitor>();
    visitors.append<MakeTensorDescOpStrideVisitor>();
    visitors.append<DescriptorLoadOpStrideVisitor>();
  }

  using dataflow::SparseForwardDataFlowAnalysis<
      dataflow::Lattice<StrideInfo>>::getLatticeElement;

  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const dataflow::Lattice<StrideInfo> *> operands,
                 ArrayRef<dataflow::Lattice<StrideInfo> *> results) override {
    // Initialize unresolved operands.
    for (auto op : operands)
      if (op->getValue().getRank() == 0)
        setToEntryState((dataflow::Lattice<StrideInfo> *)op);

    StrideInfo curr = visitors.apply(op, operands);
    if (curr.getRank() == 0) {
      setAllToEntryStates(results);
      return success();
    }
    for (auto *result : results)
      propagateIfChanged(result, result->join(curr));
    return success();
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// ModuleStrideAnalysis
//===----------------------------------------------------------------------===//

ModuleStrideAnalysis::ModuleStrideAnalysis(ModuleOp moduleOp)
    : CallGraph<StrideInfoMapT>(moduleOp) {
  SmallVector<FunctionOpInterface> funcs;
  walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
      [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
      [&](FunctionOpInterface funcOp) {
        funcs.push_back(funcOp);
        funcMap.try_emplace(funcOp, StrideInfoMapT{});
      });
  SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
  SymbolTableCollection symbolTable;
  for (auto funcOp : llvm::reverse(sortedFuncs)) {
    initialize(funcOp);
    funcOp.walk([&](CallOpInterface callOp) {
      auto callee = dyn_cast<FunctionOpInterface>(
          callOp.resolveCallableInTable(&symbolTable));
      update(callOp, callee);
    });
  }
}

StrideInfo *ModuleStrideAnalysis::getStrideInfo(Value value) {
  auto funcOp = value.getParentRegion()->getParentOfType<FunctionOpInterface>();
  auto *strideInfoMap = getFuncData(funcOp);
  if (!strideInfoMap)
    return nullptr;
  auto it = strideInfoMap->find(value);
  if (it == strideInfoMap->end())
    return nullptr;
  return &(it->second);
}

void ModuleStrideAnalysis::initialize(FunctionOpInterface funcOp) {
  std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
  StrideAnalysis *analysis = solver->load<StrideAnalysis>();
  if (failed(solver->initializeAndRun(funcOp)))
    return;
  auto *strideInfoMap = getFuncData(funcOp);
  auto updateMap = [&](Value value) {
    const auto &info = analysis->getLatticeElement(value)->getValue();
    StrideInfo curInfo;
    if (strideInfoMap->count(value))
      curInfo = StrideInfo::join(info, strideInfoMap->lookup(value));
    else
      curInfo = info;
    (*strideInfoMap)[value] = std::move(curInfo);
  };
  funcOp.walk([&](Operation *op) {
    for (auto value : op->getResults())
      updateMap(value);
  });
  funcOp.walk([&](Block *block) {
    for (auto value : block->getArguments())
      updateMap(value);
  });
}

void ModuleStrideAnalysis::update(CallOpInterface callOp,
                                  FunctionOpInterface callee) {
  // StrideInfo does not forward across call boundaries.
}

} // namespace mlir::triton::intel
