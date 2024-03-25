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

def TT_TensorRelated : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

def TTIG_GlueOp : TTIG_Op<"glue", [// operands have same type: SameOperandsType,
                               Pure]> {
    let summary = "glue/concat opearands to a larger size";

    let arguments = (ins Variadic<TT_TensorRelated>:$operands);

    let results = (outs TT_TensorRelated:$result);

    let assemblyFormat = "$operands attr-dict `:` type($operands) `->` type($result)";
}

def TTIG_ExtractOp : TTIG_Op<"extract", [Pure]> {
    let summary = "extract a sub value from base at idx";

    let arguments = (ins TT_TensorRelated:$base, I32Attr:$idx);

    let results = (outs TT_TensorRelated:$result);

    let assemblyFormat = "$base `,` $idx attr-dict `:` type($base) `->` type($result)";
}

def TTIG_CastOp : TTIG_Op<"cast", [Pure]> {
    let summary = "bitcast a tensor to another type with the same overall bits";

    let arguments = (ins TT_Tensor:$src);

    let results = (outs TT_Tensor:$dst);

    let assemblyFormat = "$src attr-dict `:` type($src) `to` type($dst)";
}

#endif
