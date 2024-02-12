#ifndef TRITON_INTEL_GPU_ATTRDEFS
#define TRITON_INTEL_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

//===----------------------------------------------------------------------===//
// Intel DPAS Layout Encoding
//===----------------------------------------------------------------------===//

def DpasEncodingAttr : DistributedEncoding<"DpasEncoding", "intel_mma_encoding", [MmaEncodingTrait], TritonIntelGPU_Dialect> {
  let mnemonic = "dpas";

  let description = [{
An encoding for the tensors distributed across the threads for the C and D operands of XMX tensor core operation.
The XMX tensor core operation is defined for matrix matmul as: D=A*B+C
The shape of the of XMX tensor core operation is defined by systolic depth, repeat count, execution size and operations per channel.

The encoding is characterized by parameters:
        - `repeatCount` which shall be in the range [1, 8]
        - `systolicDepth` For PVC/ATSM, the size is 8.
        - `executionSize` For PVC, the size is 16. For ATSM, the size is 8.
        - `opsPerChannel` 4 for 8 bit scalar type, 2 for 16 bit scalar type, 1 for 32 bit scalar type.
        - `warpsPerCTA`
        - `sugGroupSize` valid sub group size is 8/16/32


The layout example repeat_count=8, systolic_depth=8, execution_size=16 and operands_per_chan=2 for warp size 32.
For A operand:
                                   systolic depth = 8
<------------------------------------------------------------------------------------------------->
opsPerChan=2
<--------->
t0  ...  t0   t1  ... t1   t2  ... t2  t3  ... t3  t4  ... t4   t5  ... t5  t6  ... t6  t7  ... t7    ^
t8  ...  t8   t9  ... t9   t10 ... t10 t11 ... t11 t12 ... t12  t13 ... t13 t14 ... t14 t15 ... t15   |
t16 ...  t16  t17 ... t17  t18 ... t18 t19 ... t19 t20 ... t20  t21 ... t21 t22 ... t22 t23 ... t23   |
t24 ...  t24  t25 ... t25  t26 ... t26 t27 ... t27 t28 ... t28  t29 ... t29 t30 ... t30 t31 ... t31   | repeat count <= 8
t0  ...  t0   t1  ... t1   t2  ... t2  t3  ... t3  t4  ... t4   t5  ... t5  t6  ... t6  t7  ... t7    |
t8  ...  t8   t9  ... t9   t10 ... t10 t11 ... t11 t12 ... t12  t13 ... t13 t14 ... t14 t15 ... t15   |
t16 ...  t16  t17 ... t17  t18 ... t18 t19 ... t19 t20 ... t20  t21 ... t21 t22 ... t22 t23 ... t23   |
t24 ...  t24  t25 ... t25  t26 ... t26 t27 ... t27 t28 ... t28  t29 ... t29 t30 ... t30 t31 ... t31   v

For B operand:
               execution size = 16
<------------------------------------------------------------->
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15     ^             ^
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .       | opsPerChan=2|
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15     v             |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |  systolic depth = 8
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15                   |
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
t0  t1  t2  t3  t4  t5  t6  t7  t8  t9  t10 t11 t12 t13 t14 t15                   |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   |
.   .   .   .   .   .   .   .   .   .   .   .   .   .   .   .                     |
t16 t17 t18 t19 t20 t21 t22 t23 t24 t25 t26 t27 t28 t29 t30 t31                   v

This pattern repeats every warpsPerTile[0] (resp. warpsPerTile[1]) blocks
along the row (resp. col) dimension.
}];

  let parameters = (
    ins
    "unsigned":$repeatCount,
    "unsigned":$systolicDepth,
    "unsigned":$executionSize,
    "unsigned":$opsPerChannel,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    "unsigned":$subGroupSize
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{

    SmallVector<unsigned> getShapeA() const;
    SmallVector<unsigned> getShapeB() const;
    SmallVector<unsigned> getShapeC() const;
    SmallVector<int64_t> getDPASRepetitions(ArrayRef<int64_t> shape, int opIdx) const;
    SmallVector<unsigned> getSizePerThreadForOperands(unsigned opIdx) const;
    SmallVector<unsigned> getShapePerCTATileForDotOperands(ArrayRef<int64_t> shape, int opIdx) const;
    unsigned getTotalElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const;

    bool supportReduction() const {
      return true;
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}


#endif
