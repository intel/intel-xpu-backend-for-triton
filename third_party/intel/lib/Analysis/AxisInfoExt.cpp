#include "intel/include/Analysis/AxisInfoExt.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "intel-axis-info-ext"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::intel {

namespace ttgi = mlir::triton::gpu::intel;

namespace {

// Upstream AxisInfoVisitorImpl / BinaryOpVisitorImpl / AddSubOpAxisInfoVisitor
// are defined in an anonymous namespace in lib/Analysis/AxisInfo.cpp, so they
// cannot be instantiated or subclassed from outside that translation unit.  We
// replicate the base templates here for Intel-specific op visitors.
template <typename OpTy> class AxisInfoVisitorImpl : public AxisInfoVisitor {
public:
  using AxisInfoVisitor::AxisInfoVisitor;

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) final {
    return getAxisInfo(cast<OpTy>(op), operands);
  }

  bool match(Operation *op) final { return isa<OpTy>(op); }

  virtual AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) = 0;
};

template <typename OpTy>
class BinaryOpVisitorImpl : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto lhsInfo = operands[0]->getValue();
    auto rhsInfo = operands[1]->getValue();
    auto rank = lhsInfo.getRank();
    auto constantValue = getConstantValue(op, lhsInfo, rhsInfo);
    if (constantValue.has_value()) {
      auto resTy = dyn_cast<RankedTensorType>(op.getType());
      AxisInfo::DimVectorT constancy =
          resTy ? to_vector(resTy.getShape()) : AxisInfo::DimVectorT(rank, 1);
      AxisInfo::DimVectorT contiguity(rank, 1);
      AxisInfo::DimVectorT divisibility(
          rank, highestPowOf2Divisor<int64_t>(constantValue.value()));
      return AxisInfo(contiguity, divisibility, constancy, constantValue);
    }
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (auto d = 0; d < rank; ++d) {
      contiguity.push_back(getContiguity(op, lhsInfo, rhsInfo, d));
      constancy.push_back(getConstancy(op, lhsInfo, rhsInfo, d));
      divisibility.push_back(getDivisibility(op, lhsInfo, rhsInfo, d));
    }
    return AxisInfo(contiguity, divisibility, constancy, constantValue);
  }

protected:
  virtual int64_t getContiguity(OpTy op, const AxisInfo &lhs,
                                const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual int64_t getDivisibility(OpTy op, const AxisInfo &lhs,
                                  const AxisInfo &rhs, int dim) {
    return 1;
  }

  virtual int64_t getConstancy(OpTy op, const AxisInfo &lhs,
                               const AxisInfo &rhs, int dim) {
    return std::gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  virtual std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                                  const AxisInfo &rhs) {
    return {};
  }
};

//===----------------------------------------------------------------------===//
// Intel-specific visitors
//===----------------------------------------------------------------------===//

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
    // 2D block I/O operates on the inner 2 dimensions (rank-2 and rank-1).
    // For those dims the divisibility is bounded by the stride of the *other*
    // inner dimension; for batch dims (dim < rank-2) use conservative
    // divisibility of 1; for 1-D tensors the single stride suffices.
    unsigned relevantIdx = (rank >= 2 && dim == rank - 1)   ? rank - 2
                           : (rank >= 2 && dim == rank - 2) ? rank - 1
                                                            : dim;
    const AxisInfo &relevantStride = strideInfo[relevantIdx];
    divisibility.push_back(
        contiguity[dim] > 1
            ? std::min(ptrDivisibility, relevantStride.getDivisibility()[0])
            : 1);
    constancy.push_back(1);
  }

  return AxisInfo(std::move(contiguity), std::move(divisibility),
                  std::move(constancy), std::nullopt);
}

class MakeTensorPtrOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::MakeTensorPtrOp> {
public:
  using AxisInfoVisitorImpl<triton::MakeTensorPtrOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::MakeTensorPtrOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    LDBG("MakeTensorPtrOpAxisInfoVisitor: " << *op);

    auto tensorType = cast<RankedTensorType>(
        cast<PointerType>(op.getResult().getType()).getPointeeType());
    unsigned rank = op.getShape().size();

    // TODO: Support higher rank tensors.  The makeTensorPtrAxisInfo helper
    // now handles rank > 2 (for MakeTensorDescOp), so this guard can be
    // removed once MakeTensorPtrOp callers are validated with rank > 2.
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
};

class AdvanceOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::AdvanceOp> {
public:
  using AxisInfoVisitorImpl<triton::AdvanceOp>::AxisInfoVisitorImpl;
  AxisInfo
  getAxisInfo(triton::AdvanceOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

class MakeTensorDescOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::MakeTensorDescOp> {
public:
  using AxisInfoVisitorImpl<triton::MakeTensorDescOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::MakeTensorDescOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    LDBG("MakeTensorDescOpAxisInfoVisitor: " << *op);

    RankedTensorType tensorType =
        cast<TensorDescType>(op.getResult().getType()).getBlockType();
    unsigned rank = op.getShape().size();

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
};

class DescriptorLoadOpAxisInfoVisitor final
    : public AxisInfoVisitorImpl<triton::DescriptorLoadOp> {
public:
  using AxisInfoVisitorImpl<triton::DescriptorLoadOp>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(triton::DescriptorLoadOp op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    LDBG("DescriptorLoadOpAxisInfoVisitor: " << *op);

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
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
};

//===----------------------------------------------------------------------===//
// LLVM dialect and IndexCast visitors
//===----------------------------------------------------------------------===//

template <typename OpTy>
class CastOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

template <typename OpTy>
class ConstantOpAxisInfoVisitor final : public AxisInfoVisitorImpl<OpTy> {
public:
  using AxisInfoVisitorImpl<OpTy>::AxisInfoVisitorImpl;

  AxisInfo
  getAxisInfo(OpTy op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    auto boolAttr = dyn_cast<BoolAttr>(op.getValue());
    if (intAttr || boolAttr) {
      int64_t value{};
      if (intAttr)
        value = intAttr.getValue().getSExtValue();
      else
        value = boolAttr.getValue() ? 1 : 0;
      return AxisInfo(/*contiguity=*/{1},
                      /*divisibility=*/{highestPowOf2Divisor(value)},
                      /*constancy=*/{1},
                      /*knownConstantValue=*/{value});
    }
    // TODO: generalize to dense attr
    auto splatAttr = dyn_cast<SplatElementsAttr>(op.getValue());
    if (splatAttr && splatAttr.getElementType().isIntOrIndex()) {
      APInt apValue = splatAttr.template getSplatValue<APInt>();
      // Use getZExtValue for 1-bit integers (booleans) to avoid sign extension
      // turning true into -1, use getSExtValue for wider integers to properly
      // handle negative values.
      int64_t value = apValue.getBitWidth() == 1 ? apValue.getZExtValue()
                                                 : apValue.getSExtValue();
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
};

// AddSubOpAxisInfoVisitor for LLVM::AddOp.
//
// We cannot reuse the upstream AddSubOpAxisInfoVisitor<> template because:
//  1. It lives in an anonymous namespace (lib/Analysis/AxisInfo.cpp) and cannot
//     be instantiated from here.
//  2. Upstream's getConstantValue() has if-constexpr branches for
//  arith::AddIOp,
//     arith::SubIOp, and triton::AddPtrOp, but no branch for LLVM::AddOp — so
//     constant folding would silently return {} (no value).
//  3. Upstream's getDivisibility() both-contiguous path calls multiplyDivisor()
//     and gcd(), which are also in the upstream anonymous namespace.
//
// This simplified version is equivalent to the upstream template instantiated
// for a plain integer addition (no SubIOp sign-flip, no AddPtrOp pointer
// arithmetic) and correctly folds constants for LLVM::AddOp.
template <typename OpTy>
class AddSubOpAxisInfoVisitor final : public BinaryOpVisitorImpl<OpTy> {
public:
  using BinaryOpVisitorImpl<OpTy>::BinaryOpVisitorImpl;

private:
  int64_t getContiguity(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                        int dim) override {
    return std::max(std::gcd(lhs.getConstancy(dim), rhs.getContiguity(dim)),
                    std::gcd(lhs.getContiguity(dim), rhs.getConstancy(dim)));
  }

  int64_t getDivisibility(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                          int dim) override {
    return std::gcd(lhs.getDivisibility(dim), rhs.getDivisibility(dim));
  }

  int64_t getConstancy(OpTy op, const AxisInfo &lhs, const AxisInfo &rhs,
                       int dim) override {
    return std::gcd(lhs.getConstancy(dim), rhs.getConstancy(dim));
  }

  std::optional<int64_t> getConstantValue(OpTy op, const AxisInfo &lhs,
                                          const AxisInfo &rhs) override {
    if (lhs.getConstantValue().has_value() &&
        rhs.getConstantValue().has_value()) {
      return {lhs.getConstantValue().value() + rhs.getConstantValue().value()};
    }
    return {};
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// AxisInfoExt
//===----------------------------------------------------------------------===//

void AxisInfoExt::addVisitors(AxisInfoVisitorList &visitors) {
  visitors.append<MakeTensorPtrOpAxisInfoVisitor>();
  visitors.append<AdvanceOpAxisInfoVisitor>();
  visitors.append<MakeTensorDescOpAxisInfoVisitor>();
  visitors.append<DescriptorLoadOpAxisInfoVisitor>();
  visitors.append<CastOpAxisInfoVisitor<arith::IndexCastOp>>();
  // TODO: Remove rules for LLVM::ConstantOp, LLVM::AddOp
  // when scf.for supports integer induction variables
  visitors.append<ConstantOpAxisInfoVisitor<LLVM::ConstantOp>>();
  visitors.append<AddSubOpAxisInfoVisitor<LLVM::AddOp>>();
}

//===----------------------------------------------------------------------===//
// ModuleAxisInfoAnalysis
//===----------------------------------------------------------------------===//

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
  // FIXME: This is not as good as it could be, as we don't need to restrict
  // the analysis to one dimension. We should determine contiguity on the
  // flattenOuts() layout
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

  // FIXME: should this be an assertion instead?
  // Temporarily added to avoid crashing on some tests.
  // See issue #3842.
  if (order[0] >= axisInfo->getRank())
    return 1;

  auto maxMultipleBytes = axisInfo->getDivisibility(order[0]);
  auto maxContig = axisInfo->getContiguity(order[0]);

  auto elemTy = tensorTy.getElementType();
  // Get the pointee type if we have a tensor of ptrs to compute contiguity for
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

  // FIXME: should this be an assertion instead?
  // Temporarily added to avoid crashing on some tests.
  // See issue #3842.
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
