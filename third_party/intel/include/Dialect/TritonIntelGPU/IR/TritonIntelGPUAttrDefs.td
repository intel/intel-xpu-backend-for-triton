#ifndef TRITON_INTEL_GPU_ATTRDEFS
#define TRITON_INTEL_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

//===----------------------------------------------------------------------===//
// Intel DPAS Layout Encoding
//===----------------------------------------------------------------------===//

def DpasEncodingAttr : DistributedEncoding<"DpasEncoding", "intel_dpas_encoding",
                                          [MmaEncodingTrait], TritonIntelGPU_Dialect> {
  let mnemonic = "dpas";

  let description = [{
An encoding for the tensors distributed across the threads for the C and D operands of XMX tensor core operation
and its corresponding A and B operands layout with the DPAS encoding as parent.
The XMX tensor core operation is defined for matrix matmul as: D=A*B+C
The shape of the of XMX tensor core operation is defined by systolic depth, repeat count, execution size and operations per channel.

The encoding is characterized by parameters:
        - `repeatCount` which shall be in the range [1, 8]
        - `systolicDepth` For PVC/ATSM, the size is 8.
        - `executionSize` For PVC, the size is 16. For ATSM, the size is 8.
        - `opsPerChannel` 4 for 8 bit scalar type, 2 for 16 bit scalar type, 1 for 32 bit scalar type.
        - `warpsPerCTA` indicates the distribution of the warps in the block. The order is [1, 0] for rank 2.
        - `repCluster` indicates the cluster size of the repetitions of the DPAS tile.
        - `sugGroupSize` Currently only sub group size 16 is supported.

The values of the matrix is distributed across the threads in the subgroup as row-major order.
  - If the column size of the matrix is equal to the number of threads in the subgroup, a single value name represents a single rows of the matrix.
  - If the column size of the matrix is less than the number of threads in the subgroup, a single value name represents multiple rows of the matrix.
  - If the column size of the matrix is larger than the number of the threads in the subgroup, a single row of the matrix requires multiple value name.

Example 1, the column size of the matrix is 16 and the number of threads in the subgroup is 16.
The DPAS encoding of repeatCount=8, systolicDepth=8, executionSize=16, opsPerChannel=2 and sugGroupSize=16.

The layout for A operand:
                       K = 16 (K = systolic depth * opsPerChan)
<---------------------------------------------------------------------------->

t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   ^
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   | M = 8 (repeat count)
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   v

The layout for B operand:
                        N = 16 (N = execution size)
<---------------------------------------------------------------------------->

t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    ^
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |  K = 16 (K = systolic depth * opsPerChan)
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15    v

The layout for C operand and result D:
                    N = 16 (N = execution size)
<---------------------------------------------------------------------------->
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   ^
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   | M = 8 (M=repeat count)
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   v

Example 2, the column size of the matrix is 8 and the number of threads in the subgroup is 16.
The DPAS encoding of repeatCount=8, systolicDepth=8, executionSize=16, opsPerChannel=1 and sugGroupSize=16.

The layout for A operand:
  K = 8 (K = systolic depth * opsPerChan)
<---------------------------------------->

t0   t1   t2   t3   t4   t5   t6   t7    ^
t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7    |
t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7    | M = 8 (repeat count)
t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7    |
t8   t9   t10  t11  t12  t13  t14  t15   v

The layouts for B operand is like the one of opsPerChan=2 but the K size is 8.
The layouts for C and D operands are same as the one of opsPerChan=2.

Example 3, the column size of the matrix is 32 and the number of threads in the subgroup is 16.
The DPAS encoding of repeatCount=8, systolicDepth=8, executionSize=16, opsPerChannel=4 and sugGroupSize=16.

The layout for A operand:
                       K = 32 (K = systolic depth * opsPerChan)
<----------------------------------------------------------------------------------------------------------------------------------->

t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   ^
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   | M = 8 (repeat count)
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   v

The layouts for B operand is like the one of opsPerChan=2 but the K size is 32.
The layouts for C and D operands are same as the one of opsPerChan=2.

The patterns (illustrated above) repeats every warpsPerTile[0] (resp. warpsPerTile[1]) blocks
along the row (resp. col) dimension.  And the repetitions are clustered of the size of repCluster to optimize the memory accessing.

Suppose we have a `tt.dot` operation of the block size [64, 128] += [64, 32] * [32, 128] of hf16/bf16.
The `warpsPerCTA` set to [2, 2]. The number of repetitions of the DPAS tile per warp is: A=8, B=8, C,D=16.
The DPAS repetitions are distributed as follows:

                                                warp[:0]  warp[:1]  warp[:0]  warp[:1]
                                              |----^----|----^----|----^----|----^----|
                                              repCluster[1]
                                              <--------->
                                              ┌────┬────┬────┬────┬────┬────┬────┬────┐
                                              │R0  │R1  │    │    │R4  │R5  │    │    │
                                              │    │    │    │    │    │    │    │    │
                                              ├────┼────┼────┼────┼────┼────┼────┼────┤
                                              │R2  │R3  │    │    │R6  │R7  │    │    │
                                              │    │    │    │    │    │    │    │    │
                                              └────┴────┴────┴────┴────┴────┴────┴────┘

            -                ^ ┌────┬────┐    ┌────┬────┬────┬────┬────┬────┬────┬────┐
            |                | │R0  │R2  │    │R0  │R1  │    │    │R4  │R5  │    │    │
            |                | │    │    │    │    │    │    │    │    │    │    │    │
   warp[0:] < repCluster[0]  | ]────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                | │R1  │R3  │    │R2  │R3  │    │    │R6  │R7  │    │    │
            |                | │    │    │    │    │    │    │    │    │    │    │    │
            -                v ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            |                  │    │    │    │    │    │    │    │    │    │    │    │
   warp[1:] <                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            -                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                  │R4  │R6  │    │R8  │R9  │    │    │R12 │R13 │    │    │
            |                  │    │    │    │    │    │    │    │    │    │    │    │
   warp[0:] <                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                  │R5  │R7  │    │R10 │R11 │    │    │R14 │R15 │    │    │
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            -                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            |                  │    │    │    │    │    │    │    │    │    │    │    │
   warp[1:] <                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            |                  │    │    │    │    │    │    │    │    │    │    │    │
            -                  └────┴────┘    └────┴────┴────┴────┴────┴────┴────┴────┘

}];

  let parameters = (
    ins
    "unsigned":$repeatCount,
    "unsigned":$systolicDepth,
    "unsigned":$executionSize,
    "unsigned":$opsPerChannel,
    ArrayRefParameter<"unsigned">:$warpsPerCTA__,
    ArrayRefParameter<"unsigned">:$repCluster,
    "unsigned":$subGroupSize
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{

    SmallVector<unsigned> getDPASInstShapeA() const;
    SmallVector<unsigned> getDPASInstShapeB() const;
    SmallVector<unsigned> getDPASInstShapeC() const;
    SmallVector<unsigned> getShapeA() const;
    SmallVector<unsigned> getShapeB() const;
    SmallVector<unsigned> getShapeC() const;
    SmallVector<int64_t> getDPASRepetitions(ArrayRef<int64_t> shape, int opIdx) const;
    SmallVector<unsigned> getSizePerThreadForOperand(int kWidth,unsigned opIdx) const;
    SmallVector<unsigned> getElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, unsigned opIdx) const;
    SmallVector<unsigned> getShapePerCTATileForOperand(ArrayRef<int64_t> shape, int kWidth, int opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
    unsigned getTotalElemsPerThreadForOperand(ArrayRef<int64_t> shape, Type eltTy, int kWidth, int opIdx) const;

    bool supportReduction() const {
      return true;
    }

    SmallVector<unsigned> getContigPerThread();
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Intel Warp Encoding
//===----------------------------------------------------------------------===//

def WarpEncodingAttr : TritonGPU_Attr<"WarpEncoding", "intel_warp_encoding",
                                     [], TritonIntelGPU_Dialect> {
  let mnemonic = "warp";

  let description = [{
   An encoding characterized by two tuples -- thread tile size and warp tile size
   which specify the amount of elements owned by each thread and warp respectively.
   currently all their meaning remain the same as above blocked encoding.
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread,
    ArrayRefParameter<"unsigned">:$threadsPerWarp,
    ArrayRefParameter<"unsigned">:$order // the fastest-changing axis first
  );

  let extraClassDeclaration = [{
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
  }];

  let hasCustomAssemblyFormat = 1;
}

#endif
