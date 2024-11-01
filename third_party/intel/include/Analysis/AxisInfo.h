#ifndef TRITON_INTEL_ANALYSIS_AXISINFO_H
#define TRITON_INTEL_ANALYSIS_AXISINFO_H

#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::intel {

// Module level axis info analysis based on the call graph, assuming that we do
// not have recursive functions.
//
// Since each function will be called multiple times, we need to calculate the
// axis info based on the axis info of all the callers.  In the future, we can
// perform optimization using function cloning so that each call site will have
// unique axis info.

class ModuleAxisInfoAnalysis : public triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : triton::ModuleAxisInfoAnalysis(moduleOp) {
    funcMap.clear();

    SmallVector<FunctionOpInterface> funcs;
    for (auto root : getRoots()) {
      walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
          // Pre-order edge walk callback
          [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
          // Post-order node walk callback
          [&](FunctionOpInterface funcOp) {
            funcs.push_back(funcOp);
            funcMap.try_emplace(funcOp, AxisInfoMapT{});
          });
    }
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

  AxisInfo *getAxisInfo(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *axisInfoMap = getFuncData(funcOp);
    if (!axisInfoMap) {
      return nullptr;
    }
    auto it = axisInfoMap->find(value);
    if (it == axisInfoMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }

  unsigned getPtrContiguity(Value ptr);
  unsigned getPtrAlignment(Value ptr);
  unsigned getMaskAlignment(Value mask);

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};

} // namespace mlir::triton::intel

#endif
