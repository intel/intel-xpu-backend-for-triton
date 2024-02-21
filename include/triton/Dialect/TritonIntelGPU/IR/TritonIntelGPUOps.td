#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TTIG_PrefetchCacheOp : TTIG_Op<"prefetch_cache",
                                  [
                                  //AttrSizedOperandSegments,
                                   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                                  // MemoryEffects<[MemRead, MemWrite]>,
                                   //TypesMatchWith<"infer mask type from src type",
                                   //                "src", "mask", "getI1SameShape($_self)",
                                   //                "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
                                   //TypesMatchWith<"infer other type from src type",
                                   //                "src", "other", "getPointeeType($_self)",
                                   //                "($_op.getOperands().size() <= 4) || std::equal_to<>()">
                                                   ]> {
  let summary = "prefetch data to L3 cache";

  let description = [{
    This is operation to prefetch cache
  }];

  let arguments = (ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr);

  let builders = [
      // A tensor of pointers
      // OpBuilder<(ins "Value":$ptr_johnlu)>,
  ];

    // Format: `tt.load operands attrs : optional(type(ptr)) -> type(result)`
  let assemblyFormat = "$ptr attr-dict `:` type($ptr)";

  let extraClassDeclaration = [{
  }];

  //let hasCustomAssemblyFormat = 1;
}

#endif
