#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "intel-axis-info"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::intel {

namespace ttgi = mlir::triton::gpu::intel;
namespace {

/// Compute AxisInfo for ops that create a block pointer or tensor descriptor.
///
/// Both MakeTensorPtrOp and MakeTensorDescOp share the same operand layout:
///   operand 0            – base pointer
///   operands [1, rank)   – shape values  (rank operands)
///   operands [rank+1, 2*rank] – stride values (rank operands)
/// and the same stride / contiguity / divisibility / constancy logic.
/// The only difference between the two ops is how the block shape and rank are
/// extracted from the result type, which is passed in by the caller.
static AxisInfo
makeTensorPtrAxisInfo(ArrayRef<int64_t> blkShape, unsigned rank,
                      ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) {
  SmallVector<AxisInfo, 2> strideInfo;
  // Strides start after base (operand 0) and shape operands.
  for (unsigned i = rank + 1; i <= rank * 2; ++i)
    strideInfo.emplace_back(operands[i]->getValue());

  int64_t ptrDivisibility = operands[0]->getValue().getDivisibility(0);

  AxisInfo::DimVectorT contiguity, constancy, divisibility;
  for (unsigned dim = 0; dim < rank; ++dim) {
    contiguity.push_back(strideInfo[dim].getConstantValue() == 1 ? blkShape[dim]
                                                                 : 1);
    // For 2-D tensors the divisibility of dim d is bounded by the stride of
    // the *other* dimension; for 1-D tensors the single stride suffices.
    const AxisInfo &relevantStride =
        (rank == 2) ? strideInfo[dim == 0 ? 1 : 0] : strideInfo[dim];
    divisibility.push_back(
        contiguity[dim] > 1
            ? std::min(ptrDivisibility, relevantStride.getDivisibility()[0])
            : 1);
    constancy.push_back(1);
  }

  return AxisInfo(std::move(contiguity), std::move(divisibility),
                  std::move(constancy), std::nullopt);
}

class MakeTensorPtrOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto makePtrOp = cast<triton::MakeTensorPtrOp>(op);
    LDBG("MakeTensorPtrOpAxisInfoVisitor: " << *op);

    auto tensorType = cast<RankedTensorType>(
        cast<PointerType>(makePtrOp.getResult().getType()).getPointeeType());
    unsigned rank = makePtrOp.getShape().size();

    // TODO: Support higher rank tensors.
    if (rank > 2)
      return AxisInfo();

    auto axisInfo =
        makeTensorPtrAxisInfo(tensorType.getShape(), rank, operands);

    LLVM_DEBUG({
      std::string axisStr;
      llvm::raw_string_ostream os(axisStr);
      axisInfo.print(os);
      LDBG("-- " << axisStr);
    });

    return axisInfo;
  }

  bool match(Operation *op) override {
    return isa<triton::MakeTensorPtrOp>(op);
  }
};

class AdvanceOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }

  bool match(Operation *op) override { return isa<triton::AdvanceOp>(op); }
};

class MakeTensorDescOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto descOp = cast<triton::MakeTensorDescOp>(op);
    LDBG("MakeTensorDescOpAxisInfoVisitor: " << *op);

    RankedTensorType tensorType =
        cast<TensorDescType>(descOp.getResult().getType()).getBlockType();
    unsigned rank = descOp.getShape().size();

    // TODO: Support higher rank tensors.
    if (rank > 2) {
      LDBG("Unsupported tensor rank > 2, returning default AxisInfo");
      return AxisInfo();
    }

    assert(operands.size() >= rank * 2 + 1 &&
           "Insufficient operands for MakeTensorDescOp AxisInfo analysis");

    auto axisInfo =
        makeTensorPtrAxisInfo(tensorType.getShape(), rank, operands);

    LLVM_DEBUG({
      std::string axisStr;
      llvm::raw_string_ostream os(axisStr);
      axisInfo.print(os);
      LDBG("-- " << axisStr);
    });

    return axisInfo;
  }

  bool match(Operation *op) override {
    return isa<triton::MakeTensorDescOp>(op);
  }
};

class DescriptorLoadOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto loadOp = cast<triton::DescriptorLoadOp>(op);
    LDBG("DescriptorLoadOpAxisInfoVisitor: " << *op);

    auto resultType = cast<RankedTensorType>(loadOp.getResult().getType());
    unsigned rank = resultType.getRank();

    AxisInfo::DimVectorT contiguity, divisibility, constancy;

    // For descriptor loads, return conservative axis info.
    for (unsigned d = 0; d < rank; ++d) {
      contiguity.push_back(1);
      divisibility.push_back(1);
      constancy.push_back(1);
    }

    auto axisInfo = AxisInfo(std::move(contiguity), std::move(divisibility),
                             std::move(constancy));

    LLVM_DEBUG({
      std::string axisStr;
      llvm::raw_string_ostream os(axisStr);
      axisInfo.print(os);
      LDBG("-- " << axisStr);
    });

    return axisInfo;
  }

  bool match(Operation *op) override {
    return isa<triton::DescriptorLoadOp>(op);
  }
};

template <typename OpTy>
class CastOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }

  bool match(Operation *op) override { return isa<OpTy>(op); }
};

class LLVMConstantOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto constOp = cast<LLVM::ConstantOp>(op);
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (intAttr) {
      int64_t value = intAttr.getValue().getZExtValue();
      return AxisInfo(/*contiguity=*/{1},
                      /*divisibility=*/{highestPowOf2Divisor(value)},
                      /*constancy=*/{1},
                      /*knownConstantValue=*/{value});
    }
    auto splatAttr = dyn_cast<SplatElementsAttr>(constOp.getValue());
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      int64_t value = splatAttr.getSplatValue<APInt>().getZExtValue();
      TensorType ty = cast<TensorType>(splatAttr.getType());
      return AxisInfo(
          /*contiguity=*/AxisInfo::DimVectorT(ty.getRank(), 1),
          /*divisibility=*/
          AxisInfo::DimVectorT(ty.getRank(), highestPowOf2Divisor(value)),
          /*constancy=*/
          AxisInfo::DimVectorT(ty.getShape().begin(), ty.getShape().end()),
          /*knownConstantValue=*/{value});
    }
    return AxisInfo();
  }

  bool match(Operation *op) override { return isa<LLVM::ConstantOp>(op); }
};

class LLVMAddOpAxisInfoVisitor final : public AxisInfoVisitor {
public:
  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    const auto &lhsInfo = operands[0]->getValue();
    const auto &rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();

    std::optional<int64_t> constantValue;
    if (lhsInfo.getConstantValue().has_value() &&
        rhsInfo.getConstantValue().has_value())
      constantValue = {lhsInfo.getConstantValue().value() +
                       rhsInfo.getConstantValue().value()};

    AxisInfo::DimVectorT contiguity, divisibility, constancy;
    for (auto d = 0; d < rank; ++d) {
      if (constantValue.has_value()) {
        contiguity.push_back(1);
        auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
        constancy.push_back(resTy ? resTy.getShape()[d] : 1);
        divisibility.push_back(
            highestPowOf2Divisor<int64_t>(constantValue.value()));
      } else {
        contiguity.push_back(std::max(
            std::gcd(lhsInfo.getConstancy(d), rhsInfo.getContiguity(d)),
            std::gcd(lhsInfo.getContiguity(d), rhsInfo.getConstancy(d))));
        constancy.push_back(
            std::gcd(lhsInfo.getConstancy(d), rhsInfo.getConstancy(d)));
        divisibility.push_back(
            std::gcd(lhsInfo.getDivisibility(d), rhsInfo.getDivisibility(d)));
      }
    }
    return AxisInfo(std::move(contiguity), std::move(divisibility),
                    std::move(constancy), constantValue);
  }

  bool match(Operation *op) override { return isa<LLVM::AddOp>(op); }
};

} // anonymous namespace

void AxisInfoExt::addVisitors(mlir::triton::AxisInfoVisitorList &visitors) {
  visitors.append<MakeTensorPtrOpAxisInfoVisitor>();
  visitors.append<AdvanceOpAxisInfoVisitor>();
  visitors.append<MakeTensorDescOpAxisInfoVisitor>();
  visitors.append<DescriptorLoadOpAxisInfoVisitor>();
  visitors.append<CastOpAxisInfoVisitor<arith::IndexCastOp>>();
  visitors.append<LLVMConstantOpAxisInfoVisitor>();
  visitors.append<LLVMAddOpAxisInfoVisitor>();
}

ModuleAxisInfoAnalysis::ModuleAxisInfoAnalysis(ModuleOp moduleOp)
    : triton::ModuleAxisInfoAnalysis(moduleOp, AxisInfoExt::addVisitors) {}

AxisInfo *ModuleAxisInfoAnalysis::getAxisInfo(Value value) {
  auto funcOp = value.getParentRegion()->getParentOfType<FunctionOpInterface>();
  auto *axisInfoMap = getFuncData(funcOp);
  if (!axisInfoMap)
    return nullptr;
  auto it = axisInfoMap->find(value);
  if (it == axisInfoMap->end())
    return nullptr;
  return &(it->second);
}

unsigned ModuleAxisInfoAnalysis::getContiguity(Value value) {
  auto tensorTy = ttgi::getRankedTensorType(value.getType());
  if (!tensorTy)
    return 1;
  auto linAttr = gpu::toLinearEncoding(tensorTy);
  auto order = linAttr.getOrder();
  unsigned align = getAlignment(value);

  auto uniqueContigPerThread = linAttr.getContigPerThread();
  assert(order[0] < uniqueContigPerThread.size() &&
         "Unexpected uniqueContigPerThread size");
  unsigned contiguity = uniqueContigPerThread[order[0]];
  LDBG("getContiguity uniqueContigPerThread = " << contiguity);
  contiguity = std::min(align, contiguity);

  return contiguity;
}

unsigned ModuleAxisInfoAnalysis::getAlignment(Value value) {
  auto tensorTy = ttgi::getRankedTensorType(value.getType());
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfo(value);
  if (!axisInfo)
    return 1;
  auto linAttr = gpu::toLinearEncoding(tensorTy);
  auto order = linAttr.getOrder();

  if (order[0] >= axisInfo->getRank())
    return 1;

  auto maxMultipleBytes = axisInfo->getDivisibility(order[0]);
  auto maxContig = axisInfo->getContiguity(order[0]);

  auto elemTy = tensorTy.getElementType();
  if (auto ptrTy = dyn_cast<PointerType>(elemTy))
    elemTy = ptrTy.getPointeeType();
  auto elemNumBits = elemTy.getIntOrFloatBitWidth();
  auto elemNumBytes = std::max<unsigned>(elemNumBits / 8, 1);
  auto maxMultiple = std::max<int64_t>(maxMultipleBytes / elemNumBytes, 1);
  unsigned alignment = std::min(maxMultiple, maxContig);
  LDBG("getAlignment order[0] "
       << order[0] << " maxMultipleBytes = " << maxMultipleBytes
       << " maxContig = " << maxContig << " elemNumBits = " << elemNumBits
       << " maxMultiple = " << maxMultiple << " alignment " << alignment);
  LLVM_DEBUG({
    std::string axisStr;
    llvm::raw_string_ostream os(axisStr);
    axisInfo->print(os);
    LDBG("-- " << axisStr);
  });
  return alignment;
}

unsigned ModuleAxisInfoAnalysis::getMaskAlignment(Value mask) {
  auto tensorTy = ttgi::getRankedTensorType(mask.getType());
  if (!tensorTy)
    return 1;
  auto *axisInfo = getAxisInfo(mask);
  if (!axisInfo)
    return 1;
  auto linAttr = gpu::toLinearEncoding(tensorTy);

  auto maskOrder = linAttr.getOrder();
  if (maskOrder[0] >= axisInfo->getRank())
    return 1;
  auto alignment = std::max<unsigned>(axisInfo->getConstancy(maskOrder[0]), 1);
  LDBG("getMaskAlignment maskOrder[0] " << maskOrder[0] << " alignment "
                                        << alignment);
  LLVM_DEBUG({
    std::string axisStr;
    llvm::raw_string_ostream os(axisStr);
    axisInfo->print(os);
    LDBG("-- " << axisStr);
  });
  return alignment;
}

} // namespace mlir::triton::intel
