#ifndef TRITON_INTEL_GPU_ATTRDEFS
#define TRITON_INTEL_GPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"

//===----------------------------------------------------------------------===//
// Intel DPAS Layout Encoding
//===----------------------------------------------------------------------===//

def DpasEncodingAttr : DistributedEncoding<"DpasEncoding", "intel_dpas_encoding",
                                          [MmaEncodingTrait, DeclareLayoutEncodingMethods]> {
  let dialect = TritonIntelGPU_Dialect;
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
        - `opsPerChannel` 4 for 8 bit scalar type of A/B operands of DPAS instruction,
                          2 for 16 bit scalar type of A/B operands of DPAS instruction,
                          1 for 32 bit scalar type of A/B operands of DPAS instruction.
        - `warpsPerCTA` indicates the distribution of the warps in the block. The order is [1, 0] for rank 2.
        - `repCluster` indicates the cluster size of the repetitions of the DPAS tile.
        - `threadsPerWarp__` AKA threadsPerWarp, use the name threadsPerWarp__ to avoid conflicting
                            with the `getThreadsPerWarp` in interface DistributedLayout. Currently only 16 is supported.

The values of the matrix is distributed across the threads in the subgroup as row-major order.
  - If the column size of the matrix is equal to the number of threads in the subgroup, one scalar represents one row of the matrix in register.
  - If the column size of the matrix is less than the number of threads in the subgroup, one scalar represents multiple rows of the matrix in register.
  - If the column size of the matrix is larger than the number of the threads in the subgroup, one scalar represents partial row of the matrix in register.

Example 1, the column size of the matrix is 16 and the number of threads in the subgroup is 16.
The DPAS encoding of repeatCount=8, systolicDepth=8, executionSize=16, opsPerChannel=2 and threadsPerWarp=16.

The layout for A operand:
                       K = 16 (K = systolic depth * opsPerChan)
<---------------------------------------------------------------------------->

t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   ^
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   | M = 8 (M = repeat count)
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
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   | M = 8 (M = repeat count)
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12  t13  t14  t15   v

Example 2, the column size of the matrix is 8 and the number of threads in the subgroup is 16.
The DPAS encoding of repeatCount=8, systolicDepth=8, executionSize=16, opsPerChannel=1 and threadsPerWarp=16.

The layout for A operand:
  K = 8 (K = systolic depth * opsPerChan)
<---------------------------------------->

t0   t1   t2   t3   t4   t5   t6   t7    ^
t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7    |
t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7    | M = 8 (M = repeat count)
t8   t9   t10  t11  t12  t13  t14  t15   |
t0   t1   t2   t3   t4   t5   t6   t7    |
t8   t9   t10  t11  t12  t13  t14  t15   v

The layouts for B operand is like the one of opsPerChan=2 but the K size is 8.
The layouts for C and D operands are same as the one of opsPerChan=2.

Example 3, the column size of the matrix is 32 and the number of threads in the subgroup is 16.
The DPAS encoding of repeatCount=8, systolicDepth=8, executionSize=16, opsPerChannel=4 and threadsPerWarp=16.

The layout for A operand:
                       K = 32 (K = systolic depth * opsPerChan)
<----------------------------------------------------------------------------------------------------------------------------------->

t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   ^
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   | M = 8 (M = repeat count)
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   |
t0 t0   t1 t1   t2 t2   t3 t3   t4 t4   t5 t5   t6 t6   t7 t7   t8 t8   t9 t9   t10 t10  t11 t11  t12 t12  t13 t13  t14 t14  t15 t15   v

The layouts for B operand is like the one of opsPerChan=2 but the K size is 32.
The layouts for C and D operands are same as the one of opsPerChan=2.

The patterns (illustrated above) repeats every warpsPerTile[0] (resp. warpsPerTile[1]) blocks
along the row (resp. col) dimension. And the repetitions are clustered of the size of repCluster to optimize the memory accessing.

Suppose we have a `tt.dot` operation of the block size [64, 128] = [64, 32] * [32, 128] of f16/bf16. And its input tensor layout is defined as follows:
```
#dpas = #ttig.dpas<{repeatCount = 8, systolicDepth = 8, executionSize = 16, opsPerChan = 2, threadsPerWarp = 16, warpsPerCTA = [2, 2], repCluster = [2, 2]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#dpas, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#dpas, kWidth=2}>

%d = tt.dot %a, %b, %c : tensor<64x32xf16, #dot_operand_a> * tensor<32x128xf16, #dot_operand_b> -> tensor<64x128xf32, #dpas>
```
The semantic of this `tt.dot` includes GEMM tiling configuration as:

                                               warp[:0]  warp[:1]  warp[:0]  warp[:1]
                                             |----^----|----^----|----^----|----^----|
                                             repCluster[1]
                                             <--------->
                                             ┌────┬────┬────┬────┬────┬────┬────┬────┐
                                             │W0R0│W0R1│W1R0│W1R1│W0R4│W0R5│W1R4│W1R5│
                                             │W2R0│W2R1│W3R0│W3R1│W2R4│W2R5│W3R4│W3R5│
      warpPerCTA = [[W0, W1],                ├────┼────┼────┼────┼────┼────┼────┼────┤
                    [W2, W3]]                │W0R2│W0R3│W1R2│W1R3│W0R6│W0R7│W1R6│W1R7│
                                             │W2R2│W2R3│W3R2│W3R3│W2R6│W2R7│W3R6│W3R7│
                                             └────┴────┴────┴────┴────┴────┴────┴────┘


           -                ^ ┌────┬────┐    ┌────┬────┬────┬────┬────┬────┬────┬────┐
           |                | │W0R0│W0R2│    │W0R0│W0R1│W1R0│W1R1│W0R4│W0R5│W1R4│W1R5│
           |                | │W1R0│W1R2│    │    │    │    │    │    │    │    │    │
  warp[0:] < repCluster[0]  | ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                | │W0R1│W0R3│    │W0R2│W0R3│W1R2│W1R3│W0R6│W0R7│W1R6│W1R7│
           |                | │W1R1│W1R3│    │    │    │    │    │    │    │    │    │
           -                v ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                  │W2R0│W2R2│    │W2R0│W2R1│W3R0│W3R1│W2R4│W2R5│W3R4│W3R5│
           |                  │W3R0│W3R2│    │    │    │    │    │    │    │    │    │
  warp[1:] <                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                  │W2R1│W2R1│    │W2R2│W2R3│W3R2│W3R3│W2R6│W2R7│W3R6│W3R7│
           |                  │W3R1│W3R1│    │    │    │    │    │    │    │    │    │
           -                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                  │W0R4│W0R6│    │W0R8│W0R9│W1R8│W1R9│W0  │W0  │W1  │W1  │
           |                  │W1R4│W1R6│    │    │    │    │    │R12 │R13 │R12 │R13 │
  warp[0:] <                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                  │W0R5│W0R7│    │W0  │W0  │W1  │W1  │W0  │W0  │W1  │W1  │
           |                  │W1R5│W1R7│    │R10 │R11 │R10 │R11 │R14 │R15 │R14 │R15 │
           -                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                  │W2R4│W2R6│    │W2R8│W2R9│W3R8│W3R8│W2  │W2  │W3  │W3  │
           |                  │W3R4│W3R6│    │    │    │    │    │R12 │R13 │R12 │R13 │
  warp[1:] <                  ├────┼────┤    ├────┼────┼────┼────┼────┼────┼────┼────┤
           |                  │W2R5│W2R7│    │W2  │W2  │W3  │W3  │W2  │W2  │W3  │W3  │
           |                  │W3R5│W3R7│    │R10 │R11 │R10 │R10 │R14 │R15 │R14 │R15 │
           -                  └────┴────┘    └────┴────┴────┴────┴────┴────┴────┴────┘


}];

  let parameters = (
    ins
    "unsigned":$repeatCount,
    "unsigned":$systolicDepth,
    "unsigned":$executionSize,
    "unsigned":$opsPerChannel,
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    ArrayRefParameter<"unsigned">:$repCluster,
    "unsigned":$threadsPerWarp
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    enum class OpIdx : unsigned {
      OperandA = 0u,
      OperandB = 1u,
      OperandC = 2u
    };

    SmallVector<unsigned> getDPASInstShapeA() const;
    SmallVector<unsigned> getDPASInstShapeB() const;
    SmallVector<unsigned> getDPASInstShapeC() const;
    SmallVector<unsigned> getShapeA() const;
    SmallVector<unsigned> getShapeB() const;
    SmallVector<unsigned> getShapeC() const;

    SmallVector<int64_t> getDPASRepetitions(ArrayRef<int64_t> shape, OpIdx opIdx) const;
    SmallVector<unsigned> getElemsPerThreadForOperands(ArrayRef<int64_t> shape, Type eltTy, OpIdx opIdx) const;
    SmallVector<unsigned> getRepOrderForOperand(OpIdx opIdx) const;
    unsigned getTotalElemsPerThreadForOperand(ArrayRef<int64_t> shape, Type eltTy, int kWidth, OpIdx opIdx) const;

    // Forwarder functions for casting unsigned to OpIdx.
    SmallVector<int64_t> getDPASRepetitions(ArrayRef<int64_t> shape, unsigned opIdx) const {
      return getDPASRepetitions(shape, static_cast<OpIdx>(opIdx));
    }
    SmallVector<unsigned> getRepOrderForOperand(unsigned opIdx) const {
      return getRepOrderForOperand(static_cast<OpIdx>(opIdx));
    }
    unsigned getTotalElemsPerThreadForOperand(ArrayRef<int64_t> shape, Type eltTy, int kWidth, unsigned opIdx) const {
      return getTotalElemsPerThreadForOperand(shape, eltTy, kWidth, static_cast<OpIdx>(opIdx));
    }

    SmallVector<unsigned> getContigPerThread() const;

    struct DPASCapability {
      explicit DPASCapability(unsigned minSGSize) : executionSize(minSGSize) {}
      DPASCapability() = default;

      bool isPVC() const {
        return executionSize == 16;
      }
      bool isFalconShore() const {
        return executionSize == 16;
      }
      bool isATSM() const {
        return executionSize == 8;
      }

      static constexpr unsigned systolicDepth = 8u;
      static constexpr unsigned repeatCount = 8u;
      static constexpr unsigned opsChanBitWidths = 32u;
      unsigned executionSize = 0u;
    };

    static DPASCapability getDPASCapability(mlir::ModuleOp mod);
    static unsigned getOpsPerChannel(Type elemType);
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// Intel Warp Encoding
//===----------------------------------------------------------------------===//

def WarpEncodingAttr : DistributedEncoding<"WarpEncoding", "intel_warp_encoding",
                                     [DeclareLayoutEncodingMethods]> {
  let dialect = TritonIntelGPU_Dialect;
  let mnemonic = "warp";

  let description = [{
   An encoding characterized by two tuples -- thread tile size and warp tile size
   which specify the amount of elements owned by each thread and warp respectively.
   currently all their meaning remain the same as above blocked encoding.
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$sizePerThread_,
    ArrayRefParameter<"unsigned">:$threadsPerWarp_,
    ArrayRefParameter<"unsigned">:$order_ // the fastest-changing axis first
  );

  let extraClassDeclaration = extraDistributedDeclaration #  [{
    unsigned getTotalElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getElemsPerThread(ArrayRef<int64_t> shape, Type eltTy) const;
    SmallVector<unsigned> getSizePerThread() const;
    SmallVector<unsigned> getThreadsPerWarp() const;
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Intel Subgroup2DBlock Encoding
//===----------------------------------------------------------------------===//

def Subgroup2DBlockEncodingAttr : DistributedEncoding<"Subgroup2DBlockEncoding", "subgroup_2d_block_encoding", [MmaEncodingTrait]> {
  let dialect = TritonIntelGPU_Dialect;
  let mnemonic = "subgroup_2d_block";

  let description = [{
    An encoding for tensors produced via Intel Subgroup 2D Block IO operations.

    The subgroup 2D block IO operations read or write two-dimensional blocks of data from a two-dimensional region of memory. The Subgroup 2D Block Encoding layout is parameterized by the block width, block height, and block count for the individual load instructions and the distribution and replication of loads across warps.

    The SPV_INTEL_2d_block_io extension documentation provides more information on the subgroup 2D block IO operations and parameters: https://github.khronos.org/SPIRV-Registry/extensions/INTEL/SPV_INTEL_2d_block_io.html

    For the layout, the following parameters are required:
    - `instrShape` : contains the (height, width) block parameters for the block io operation
    - `numBlocks` : the block count parameter allows a single load to load multiple blocks in row-major order (useful for increasing cache line utilization)
    - `threadsPerWarp` : currently a scalar, this parameter allows us to support different subgroup / warp configurations. Because the 2d block io operation is a subgroup operation, the size of the subgroup is important in determining the ordering of the loaded tensor.
    - `warpsPerCTA` : the number of warps per block / subgroups per workgroup and their distribution
    - `order` : The order within the block, used to determine along which dimension to broadcast.
    - `kWidth` : Currently unused, but keeping because we will likely need it for layout conversions.
    - `CTALayout` : Describes how blocks are distributed among work-groups/thread blocks.
  }];

  let parameters = (
    ins
    ArrayRefParameter<"unsigned">:$warpsPerCTA,
    "CTALayoutAttr":$CTALayout,
    ArrayRefParameter<"unsigned">:$instrShape,
    "unsigned":$numBlocks,
    ArrayRefParameter<"unsigned">:$order,
    "unsigned":$kWidth,
    "unsigned":$threadsPerWarp
  );

  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<unsigned> getRepOrderForOperand(int opIdx) const;
    static SmallVector<unsigned, 3> getInstrShapeForLayout(DistributedEncodingTrait layout, ArrayRef<int64_t> shape, bool memoryRowMajor, unsigned kWidth, MLIRContext* context);
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif
