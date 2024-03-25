###################
Triton architecture
###################

The goal of this document is to describe the high-level architecture and roadmap of the solution outlining components and interactions. The intended audience is triton & triton backend developers. No a priori knowledge is assumed, however, the introduction is concise (use links to familiarize yourself with the basics as necessary).

************
Introduction
************

Triton [c1]_ [c10]_ has two main use cases: triton language JIT compiler and pytorch 2.0 [c2]_ low-level backend compiler. Here we focus on the latter but keep the former in mind (as well as supporting other frameworks).

.. image :: ../pics/pt2.png

From PyTorch point of view, a model FX graph [c3]_ is captured by Dynamo [c4]_ (and AOT Autograd if necessary) which is passed to Inductor [c5]_. The Inductor converts the broad set of PyTorch’s ATen operators into a narrower set of stable primitive operators (PrimTorch) [c6]_ via “decompositions” (a customizable mechanism for complex operator expansion). The resulting graph is translated into Triton kernels which are decorated python functions produced by the Inductor [c7]_. Triton frontend consumes function sources, the internals follow traditional compiler architecture: an AST is built and traversed to produce an MLIR module. The module is progressively lowered into the LLVM dialect. A regular LLVM module is then produced and converted to an appropriate representation depending on the target (e.g., PTX for Nvidia). A compiled kernel is then returned with a launcher stub. Kernels are then executed by PyTorch through Inductor’s scheduling & wrapper interfaces.


****************************
Triton MLIR compiler details
****************************

Code generation overview
========================


The Triton structure can be broken down into three major parts, following the classical compiler structure:

1. The frontend consumes a decorated Python kernel and transforms it into Triton IR (MLIR Triton dialect).
2. The middle-end consumes the IR and progressively lowers it to the TritonGPU dialect applying optimizations.
3. The backend lowers TritonGPU dialect to LLVM dialect, then converts it to LLVM IR and an appropriate format depending on the target (e.g., ptx/cubin in Nvidia case).

There are three main IR stages on which most of the optimizations are done: Triton dialect, TritonGPU dialect [c8]_, and LLVM IR. Note: Vendor-specific backends introduce additional dialects to help with TritonGPU lowering.

.. image :: ../pics/triton.png

In addition to the custom dialects Triton reuses upstream ones like arith & math for math, scf/cf for control flow, index, func, nvvm for low-level GPU operations. Triton also introduces some Nvidia-specific dialects for lowering to Hopper target.

Torch Inductor IR (input language)
----------------------------------
Torch Inductor is a pure Python compiler of PyTorch models. The Inductor uses “define-by-run” loop level IR. It supports dynamic shapes/strides. The Inductor primarily targets GPUs with Triton and CPUs by generating C++ sources with OpenMP. At a high-level Inductor  performs:

* **Graph operators’ decomposition** into a smaller set of operators (~250 PrimTorch ops instead of ~2k PyTorch ops [c11]_; Note: the sets are not stabilized!).
* **Loop-level IR graph lowering**: remove views, broadcasting, indexing simplification, materialization vs reuse decisions, layout optimization, loop reordering.
* **Scheduling**: horizontal/vertical/reduction fusion , tiling, memory planning, buffer reuse, autotuning.
* **Backend code generation** (depending on the target).
* **Wrapper code generation**: basically, an unrolled interpreter loop to run the generated kernel (gather the inputs, symbolic shape processing, allocate the tensors, and invoke the kernel).

The sequence of the above steps produces a ``call`` function that controls the guards [c13]_ and then calls a sequence of Aten/Extern kernels intermixed with generated triton functions. Former are resolved through an algorithm selection mechanism [c12]_. The latter is a Python function with ``@triton.jit`` decorator, e.g. (fused bias from a linear layer and a Relu):

.. code-block:: python
  import torch
  import triton
  import triton.language as tl

  @triton.jit
  def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
      xnumel = 16
      xoffset = tl.program_id(0) * XBLOCK
      xindex = xoffset + tl.arange(0, XBLOCK)[:]
      xmask = xindex < xnumel
      x0 = xindex % 8
      x2 = xindex
      tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
      tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
      tmp2 = tmp0 + tmp1
      tmp3 = triton_helpers.maximum(0, tmp2)
      tl.store(in_out_ptr0 + (x2), tmp3, xmask)


Each kernel will additionally be annotated with meta information such as size hints, signature, device type, mutated args, and others.
By default, Inductor reuses efficient implementations of compute-heavy operations such as matrix multiplication or flash attention via dispatching a call to the operator's native implementation in Aten. Consider a model:

.. code-block:: python

  class CausalSelfAttention(nn.Module):

      def __init__(self, num_heads: int, embed_dimension: int, bias: bool=False, is_causal: bool=False, dropout:float=0.0):
          super().__init__()
          assert embed_dimension % num_heads == 0
          # key, query, value projections for all heads, but in a batch
          self.c_attn = nn.Linear(embed_dimension, 3 * embed_dimension, bias=bias)
          # output projection
          self.c_proj = nn.Linear(embed_dimension, embed_dimension, bias=bias)
          # regularization
          self.dropout = dropout
          self.resid_dropout = nn.Dropout(dropout)
          self.num_heads = num_heads
          self.embed_dimension = embed_dimension
          # Perform causal masking
          self.is_causal = is_causal

      def forward(self, x):
          # calculate query, key, values for all heads in batch and move head forward to be the batch dim
          query_projected = self.c_attn(x)

          batch_size = query_projected.size(0)
          embed_dim = query_projected.size(2)
          head_dim = embed_dim // (self.num_heads * 3)

          query, key, value = query_projected.chunk(3, -1)
          query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
          key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
          value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

          if self.training:
              dropout = self.dropout
              is_causal = self.is_causal
          else:
              dropout = 0.0
              is_causal = False

          y = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout, is_causal=is_causal)
          y = y.transpose(1, 2).view(batch_size, -1, self.num_heads * head_dim)

          y = self.resid_dropout(self.c_proj(y))
          return y


  num_heads = 8
  heads_per_dim = 64
  batch_size = 1
  max_seq_len = 128
  embed_dimension = num_heads * heads_per_dim
  dtype = torch.float16
  p = torch.randn(batch_size, max_seq_len, embed_dimension, dtype=dtype).cuda()
  model = CausalSelfAttention(num_heads=num_heads, embed_dimension=embed_dimension, bias=True, is_causal=True, dropout=0.1).to("cuda").to(dtype)

The generated code for a single causal attention block would look like the following. Note the linear layers and the attention blocks produce a standalone kernel call (only the fused native dropout makes it to a Triton kernel in this case).

.. code-block:: python

  def call(args):
      primals_1, primals_2, primals_3 = args
      args.clear()
      assert_size_stride(primals_1, (1536, 512), (512, 1))
      assert_size_stride(primals_2, (512, 512), (512, 1))
      assert_size_stride(primals_3, (1, 128, 512), (65536, 512, 1))
      with torch.cuda._DeviceGuard(0):
          torch.cuda.set_device(0) # no-op to ensure context
          buf0 = empty_strided((128, 1536), (1536, 1), device='cuda', dtype=torch.float16)
          # Source Nodes: [l__self___c_attn], Original ATen: [aten.mm]
          extern_kernels.mm(reinterpret_tensor(primals_3, (128, 512), (512, 1), 0), reinterpret_tensor(primals_1, (512, 1536), (1, 512), 0), out=buf0)
          del primals_1
          # Source Nodes: [scaled_dot_product_attention], Original ATen: [aten._scaled_dot_product_flash_attention]
          buf1 = aten._scaled_dot_product_flash_attention(reinterpret_tensor(buf0, (1, 8, 128, 64), (0, 64, 1536, 1), 0), reinterpret_tensor(buf0, (1, 8, 128, 64), (0, 64, 1536, 1), 512), reinterpret_tensor(buf0, (1, 8, 128, 64), (0, 64, 1536, 1), 1024), 0.1, True)
          buf2 = buf1[0]
          assert_size_stride(buf2, (1, 8, 128, 64), (65536, 64, 512, 1))
          buf3 = buf1[1]
          assert_size_stride(buf3, (1, 8, 128), (1024, 128, 1))
          buf4 = buf1[2]
          assert_size_stride(buf4, (2, ), (1, ))
          buf5 = buf1[3]
          assert_size_stride(buf5, (2, ), (1, ))
          buf6 = buf1[6]
          assert_size_stride(buf6, (), ())
          buf7 = buf1[7]
          assert_size_stride(buf7, (), ())
          del buf1
          buf9 = empty_strided((128, 512), (512, 1), device='cuda', dtype=torch.float16)
          # Source Nodes: [l__self___c_proj], Original ATen: [aten.mm]
          extern_kernels.mm(reinterpret_tensor(buf2, (128, 512), (512, 1), 0), reinterpret_tensor(primals_2, (512, 512), (1, 512), 0), out=buf9)
          buf10 = empty_strided((1, ), (1, ), device='cuda', dtype=torch.int64)
          # Source Nodes: [], Original ATen: []
          aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf10)
          buf12 = empty_strided((1, 128, 512), (65536, 512, 1), device='cuda', dtype=torch.bool)
          buf13 = reinterpret_tensor(buf9, (1, 128, 512), (65536, 512, 1)); del buf9  # reuse
          # Source Nodes: [l__self___resid_dropout], Original ATen: [aten.native_dropout]
          stream0 = get_cuda_stream(0)
          triton_poi_fused_native_dropout_0.run(buf13, buf10, buf12, 0, 65536, grid=grid(65536), stream=stream0)
          return (buf13, reinterpret_tensor(primals_3, (128, 512), (512, 1), 0), reinterpret_tensor(buf0, (1, 8, 128, 64), (0, 64, 1536, 1), 0), reinterpret_tensor(buf0, (1, 8, 128, 64), (0, 64, 1536, 1), 512), reinterpret_tensor(buf0, (1, 8, 128, 64), (0, 64, 1536, 1), 1024), buf2, buf3, buf4, buf5, buf6, buf7, reinterpret_tensor(buf2, (128, 512), (512, 1), 0), buf12, reinterpret_tensor(primals_2, (512, 512), (512, 1), 0), )

It is worth noting that Inductor can replace these native aten implementations with triton templated kernels.
Inductor then passes the generated kernels (Triton source code) to the Triton compiler.

Triton frontend
---------------
Triton frontend is responsible for converting the input python-like language to the intermediate representation (Triton MLIR dialect). Consider an example kernel for softmax calculation.

.. code-block:: python

  @triton.jit
  def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
      row_idx = tl.program_id(0)
      row_start_ptr = input_ptr + row_idx * input_row_stride
      col_offsets = tl.arange(0, BLOCK_SIZE)
      input_ptrs = row_start_ptr + col_offsets
      row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
      row_minus_max = row - tl.max(row, axis=0)
      numerator = tl.exp(row_minus_max)
      denominator = tl.sum(numerator, axis=0)
      softmax_output = numerator / denominator
      output_row_start_ptr = output_ptr + row_idx * output_row_stride
      output_ptrs = output_row_start_ptr + col_offsets
      tl.store(output_ptrs, softmax_output, mask=col_offsets< n_cols)


The resulting IR follows the input language almost 1 to 1:

.. code-block:: none

  tt.func public @softmax_kernel_0d1d234(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %4 = tt.splat %2 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.splat %arg4 : (i32) -> tensor<1024xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<1024xi32>
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant dense<0xFF800000> : tensor<1024xf32>
    %8 = tt.load %5, %7, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %9 = tt.call @max__fp32S1024S__1cconstexpr_0__2cconstexpr_False__3cconstexpr_True_(%8) : (tensor<1024xf32>) -> f32
    %10 = tt.splat %9 : (f32) -> tensor<1024xf32>
    %11 = arith.subf %8, %10 : tensor<1024xf32>
    %12 = math.exp %11 : tensor<1024xf32>
    %13 = tt.call @sum__fp32S1024S__1cconstexpr_0_(%12) : (tensor<1024xf32>) -> f32
    %14 = tt.splat %13 : (f32) -> tensor<1024xf32>
    %15 = arith.divf %12, %14 : tensor<1024xf32>
    %16 = arith.muli %0, %arg3 : i32
    %17 = tt.addptr %arg0, %16 : !tt.ptr<f32>, i32
    %18 = tt.splat %17 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %20 = tt.splat %arg4 : (i32) -> tensor<1024xi32>
    %21 = arith.cmpi slt, %3, %20 : tensor<1024xi32>
    tt.store %19, %15, %21 {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.return
  }

.. image :: ../pics/prog-model.png

As seen in the example above, Triton relies on pointer arithmetic mixed with a wide set of ‘built  -ins’ (e.g., ``tl.program_id()``) calls to produce the IR. There is tensor creation, shape manipulation, math, memory, and some other built-ins available (see [c16]_ for the complete set). The program model (SPMD) assumes that an executor runs a number of ‘programs’ that process different data. The kernel can accept torch tensors and treat them as a tensor of pointers. Each kernel is assumed to be single-threaded, each working on a ‘block’ of data (e.g., ``BLOCK_SIZE: tl.constexpr`` in the kernel example above; in this case happens to equal 1024). Triton “automatically” parallelizes the execution across the range of data. Since the block size affects hardware mapping (e.g., shared memory access) the value is a compile-time constant. Automatic parallelization basically means that users do not need to explicitly control and synchronize (e.g., for shared memory access). Calls to math functions are emitted as additional functions usually containing libdevice calls (or similar).
Additionally, Triton provides a runtime and a JIT, and caches previously compiled kernels for reuse. Python binding is done through pybind11 [c24]_.
The resulting IR is passed to the optimizer (middle-end).

Triton optimizations
====================
Triton’s optimizer uses custom MLIR and default LLVM optimization passes to improve kernel performance. Passes are primarily run over Triton dialect, TritonGPU dialect, and LLVM IR. There’re some common passes like inline, LICM, CSE, DSE that are run at each stage as well as dialect specific optimizations that are described below.

Triton dialect
--------------
Triton dialect [c17]_ closely mimics the language built-ins exposed to the user. Its input types are basic types like floating point of different formats, pointers, and tensors of basic types. The operations are: tensor creation and shape manipulation, tensor pointer arithmetic, SPMD primitives, loads/stores, reductions, scans, atomics, debug ops, and some others (e.g., some weird like a modified func.call op).
At this level the optimizer runs:

* Combine pass – applying rewrite rules for IR simplification.
* Broadcast reordering.
* Tensor pointer rewriting.

TritonGPU dialect
-----------------
TritonGPU dialect [c18]_ exposes GPU-specific operators. After converting Trition dialect to TritonGPU the following set of optimizations are run:

* Coalescing – make sure the dimension with greatest contiguity is first.
* Layout conversion removal.
* Thread locality optimization.
* Matmul acceleration pipeline.
* Dot operands optimization.
* Software loop Pipelining.
* Prefetching – add hoisted multi-buffering in the shared memory for the dot operator inside a loop.
* Data duplication reduction.
* Instruction reordering.

The most important thing about the dialect is that it changes how tensors are represented by adding a layout. The layout attribute determines how the data should be partitioned across GPU threads. There are two classes of layouts: shared and distributed.

Shared layout class
^^^^^^^^^^^^^^^^^^^
This layout is used for tensors that can be accessed within shared memory by different GPU threads. The layout describes elements swizzling to avoid shared memory access bank conflicts. The main purpose of the layout is to, as the name suggests, shared memory mapping.
Example:

.. code-block:: none

  A_{0, 0}  A_{0, 1}  A_{0, 2}  A_{0, 3} ...   [phase 0] \ per_phase = 2
  A_{1, 0}  A_{1, 1}  A_{1, 2}  A_{1, 3} ...   [phase 0] /
  groups of vec=2 elements
  are stored contiguously
  _ _ _ _ /\_ _ _ _
  A_{2, 2}  A_{2, 3}  A_{2, 0}  A_{2, 1} ...   [phase 1] \ per phase = 2
  A_{3, 2}  A_{3, 3}  A_{3, 0}  A_{3, 1} ...   [phase 1] /


An actual shared layout is described by the following parameters:

* Swizzling parameters. These control swizzling patterns (phase)

    * **Vec** – represents the number of elements in a “package” to be swizzled.
    * Multiple consecutive rows can have the same swizzling pattern. The number of rows that have the same swizzling pattern is **perPhase**. Calculated based on the parent MMFA/MMA encoding.
    * **maxPhase** – represents the total number of patterns. This is usually set according to how shared memory is accessed to minimize bank conflicts.

* **Order** – an array, fastest changing axis first
* **CTA Layout** – containing CTAs (groups) per CGA (grid), CTASplitNum, and CTAOrder.
* **hasLeadingOffset** – Boolean value when set to true means when matrix is stored shared memory, there will be an offset not only in the stride dimension, but also in the leading dimension. For example, a matrix of size 16x128 and data type I8 is stored in the shared memory with 64B-swizzle mode. The offset of the element with index (0, 64) will be 16*64, compared to 1*64 when the hasLeadingOffset is false.

Example [c20]_. Assume 16 (M) by 16 (N) tensor A and each element is a f32. And we want to do swizzling along the N dim (row).

.. image :: ../pics/shared1.png

The swizzling is done for volta, so perPhase = 128 / (elementsPerRow * elementTypeInBytes) = 128 / (16*4) = 2. In this toy example, without assuming any access pattern, we can set maxPhase to 8, so that we have enough swizzling patterns to cover all the 16 rows. Let's assume vec = 2 as the value is decided by the user of the shared memory. Swizzling function is the xor function: col_swizzled = (col / vec) ^ phase * vec.
The data layout in shared memory becomes:

.. image :: ../pics/shared2.png

The solid line unites tensor elements that are processed by a single thread.

Distributed layout class
^^^^^^^^^^^^^^^^^^^^^^^^
The Distributed encoding describes the layout tensor:math:`L` with the 4-level hierarchy of multiple threads on GPU. It is abstracted from the top to the bottom as Groups Per Grid->Subgroups per Group->Threads Per Subgroup->Values Per Thread. For Groups (CTA) Per Grid (CGA) and Subgroups (Warps) Per Group (CTA) level, the linear id is distributed contiguously with the shape and order.
For example, a shape/order pair defines a distribution layout:

.. code-block:: none

  shape = [4, 4]
  order = [0, 1] // The fastest-changing axis first
  ->
  layout = [0  4  8  12]
          [1  5  9  13]
          [2  6  10 14]
          [3  7  11 15]

For the Threads Per Subgroup (Warp) and Values Per Thread level, the linear id distribution is variant for each sub-class encoding.
The layout function :math:`L` of this layout is then defined, for an index :math:`i \in R^D`, as follows: TODO  :math:`d \in D` ????, Is it dimesion?

.. math::

  L(A)[i_d] =& L[i_d + k_d * A_{shape}[d]] \bmod L_{shape}[d]; \\
             & \forall k_d : i_d + k_d * A_{shape}[d] < L_{shape}[d]\\

The two presented classes form additional layout encodings.

Blocked layout
^^^^^^^^^^^^^^
The blocked layout is a distributed layout where each subgroup (warp) owns a contiguous portion of the target tensor. This is typically the kind of data layout used to promote memory coalescing in LoadInst and StoreInst. It is characterized by three tuples – thread tile size, subgroup (warp) tile size, and block tile size – which specify the number of elements owned by each GPU thread, subgroup, and group respectively. The purpose of the blocked layout is to describe the register file mapping.
The actual parameter set is the following:

* **sizePerThread** – defines the thread tile size, e.g., a tuple {2, 2} would mean each thread owns a 2 by 2 square matrix of elements.
* **threadsPerWarp** – defines the subgroup or warp tile size. Since a subgroup size has very limited options this would look like e.g. {8, 4} for SIMD32. The example would mean that each subgroup will process a set of 8 elements in 4 rows (and the assignment to the thread is determined by sizePerThread).
* **warpsPerCTA** – defines how a tensor is split between the subgroups that build up a group. E.g., a {2, 1} would mean a “horizontal” tensor partitioning and {1, 2} – “vertical”.
* **Order** – an array, fastest changing axis first
* **CTA Layout** – containing CTAs (groups) per CGA (grid), CTASplitNum, and CTAOrder.

Todo: example of non-contiguous access.
Following are a couple of examples from Triton’s inline doc (numbers mean thread ID, positions mean elements in a tensor):
Example 1, a row-major coalesced layout may partition a 16x16 tensor over 2 warps (i.e. 64 threads) as follows:

.. code-block:: none

  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  ...
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
for

.. code-block:: none

  #triton_gpu.blocked_layout<{
    sizePerThread = {2, 2}
    threadsPerWarp = {8, 4}
    warpsPerCTA = {1, 2}
    CTAsPerCGA = {1, 1}
  }>

Example 2, a row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) as follows:

.. code-block:: none

  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  ...                                                 ...
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35  0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39  4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  ...                                                 ...
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63  28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

.. code-block:: none

  #triton_gpu.blocked_layout<{
    sizePerThread = {2, 2}
    threadsPerWarp = {8, 4}
    warpsPerCTA = {1, 2}
    CTAsPerCGA = {1, 1}
  }>

Example 3, A row-major coalesced layout may partition a 32x32 tensor over 2 warps (i.e. 64 threads) and
4 CTAs (taking 2x2 for example) as follows:


.. code-block:: none

  CTA [0,0]                                              CTA [0,1]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  ...                                                    ...
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

  CTA [1,0]                                              CTA [1,1]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]  [ 0  0  1  1  2  2  3  3  ; 32 32 33 33 34 34 35 35 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]  [ 4  4  5  5  6  6  7  7  ; 36 36 37 37 38 38 39 39 ]
  ...                                                    ...
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]
  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]  [ 28 28 29 29 30 30 31 31 ; 60 60 61 61 62 62 63 63 ]

for

.. code-block:: none

  #triton_gpu.blocked_layout<{
    sizePerThread = {2, 2}
    threadsPerWarp = {8, 4}
    warpsPerCTA = {1, 2}
    CTAsPerCGA = {2, 2}
  }>


The last piece of the puzzle for dot operator lowering (see Dot product optimization & layout lowering) is the matrix multiplication input (dot) and output (mma) operands layouts.

Dot operand layout
^^^^^^^^^^^^^^^^^^
In the TritonGPU dialect, considering ``d = tt.dot a, b, c``. ``tt.dot``'s operands ``a`` and ``b`` must be of DotOperandEncodingAttr distributed layout.

MMA layout
^^^^^^^^^^
MMA layouts provide the register file mapping for the result of a matrix multiplication instruction. There are different layouts for different hardware (e.g., MFMA for AMD, NvidiaMma for Nvidia, DPAS for Intel). See Nvidia’s examples at [c26]_.

Dot product optimization & layout lowering
==========================================
GPUs provide specific instructions for efficient matrix multiplication (Nvidia’s MMA [c21]_, Intel’s DPAS [c22]_, and AMD’s MFMA [c23]_). These are usually implemented as systolic arrays and produce/consume a tile of input and output values (as opposed to regular instructions consuming 1 operand at a time). The performance of workloads using these instructions is highly dependent on data throughput, thus the overall flow looks like the following:

* Load input operand tiles from the global device memory into the shared memory. These tiles will have a *shared* layout.
* Load a small portion of the data to the register file. These will have a *dot* layout.
* Execute the MM instruction. The result of the instruction is written back to the register file and will have a *mma* (or similar) layout.

Layouts dependency example (an arrow from Dot layout to MMA layout means MMA is a parent of Dot layout):

.. image :: ../pics/encoding.png

A single dot operator is likely to be mapped to multiple MMA instructions. For Nvidia flow, these will be emitted as inline assembly into LLVM (e.g., ``llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 …``).

Layout conversion
=================
To produce the desired memory behavior described in the previous section, triton GPU introduces layouts conversion (by means of ConvertLayoutOp). An input tensor represented in a blocked layout is sliced and inserted into a shared layout, e.g.:

.. code-block:: none

  %61 = triton_gpu.insert_slice_async %39, %58, %c0_i32, %60, %cst_1 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32x!tt.ptr<f16>, #blocked> -> tensor<4x64x32xf16, #shared>
  triton_gpu.async_commit_group


The main loop of the GEMM would then extract a slice (a reimplementation of tensor.extract_slice [c25]_) from the shared memory, converting arguments to the dot layout and producing mma layout with the dot operator, e.g.:

.. raw:: html

  <div class="highlight-none notranslate"><div class="highlight"><pre><span></span>
  %107:14 = <b>scf.for</b> %arg9 = %c0_i32 to %51 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %39, %arg12 = %49, %arg13 = %94, %arg14 = %100, %arg15 = %101, %arg16 = %102, %arg17 = %85, %arg18 = %86, %arg19 = %c2_i32, %arg20 = %c3_i32, %arg21 = %c1_i32, %arg22 = %104, %arg23 = %106) -> (tensor<64x128xf32, #mma>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, tensor<4x64x32xf16, #shared>, tensor<4x32x128xf16, #shared1>, tensor<64x32xf16, #shared>, tensor<32x128xf16, #shared1>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x128x!tt.ptr<f16>, #blocked1>, i32, i32, i32, tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>)  : i32 {
      %126 = <b>triton_gpu.extract_slice</b> %arg15[0, 16] [64, 16] [1, 1] : tensor<64x32xf16, #shared> to tensor<64x16xf16, #shared>
      %127 = <b>triton_gpu.convert_layout</b> %126 : (tensor<64x16xf16, <b>#shared</b>>) -> tensor<64x16xf16, <b>#triton_gpu.dot_op</b><{opIdx = 0, parent = #mma, kWidth = 2}>>
      %128 = <b>triton_gpu.extract_slice</b> %arg16[16, 0] [16, 128] [1, 1] : tensor<32x128xf16, <b>#shared1</b>> to tensor<16x128xf16, <b>#shared1</b>>
      %129 = <b>triton_gpu.convert_layout</b> %128 : (tensor<16x128xf16, #shared1>) -> tensor<16x128xf16, <b>#triton_gpu.dot_op</b><{opIdx = 1, parent = #mma, kWidth = 2}>>
      %130 = tt.dot %arg22, %arg23, %arg10 {allowTF32 = true} : tensor<64x16xf16, <b>#triton_gpu.dot_op</b><{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x128xf16, <b>#triton_gpu.dot_op</b><{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x128xf32, <b>#mma</b>>
      %131 = <b>tt.dot</b> %127, %129, %130 {allowTF32 = true} : tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x128xf16, <b>#triton_gpu.dot_op</b><{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<64x128xf32, <b>#mma</b>>
      ...
  </pre></div></div>

The result of the processing is then converted back to blocked layout to be stored to the main GPU memory, e.g.:

.. code-block:: none

  %125 = triton_gpu.convert_layout %108 : (tensor<64x128xf16, #mma>) -> tensor<64x128xf16, #blocked1>
  tt.store %117, %125, %124 {cache = 1 : i32, evict = 1 : i32} : tensor<64x128xf16, #blocked1>


See TritonDotPattern.

Pipelining optimization
=======================
The pipelining pass is split in two parts. The first one creates a modulo schedule. The second – emits prologue and epilogue and rewrites the inner loop. There is currently a single ad hoc scheduling for the matmul. It creates the schedule and inserts async loads as well as wait ops. An example of the expansion in case we break the loop into three stages (S0, S1, S2) is as follows:

.. code-block:: none

  S0(0)                        // Prologue
  S0(1) S1(0)                  // Prologue
  scf.for %I = %C0 to %N - 2 {
    S0(I+2) S1(I+1) S2(I)      // Pipelined kernel
  }
  S1(N) S2(N-1)                // Epilogue
  S2(N)                        // Epilogue

Prefetches insertion
Prefetch pass attempts to prefetch the operands of a tt.dot op. It adds slice extraction from an input tensor and inserts layout conversion ops. The latter ones will then be lowered to shared memory loads.
Here’s an example of the transformation:

.. code-block:: none

  %a: tensor<128x32xf16, #enc>
  scf.for %iv = ... iter_args(%a_arg = %a, ...) {
    %d = tt.dot %a_arg, %b, %c
    ...
    scf.yield %a_next, ...
  }

Is translated to:

.. code-block:: none

  %a: tensor<128x32xf16, #enc>
  %a_tmp = tensor.extract_slice %a[0, 0] [128, 16]
  %a_prefetch = triton_gpu.convert_layout %a_tmp
  scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
  {
    %x = tt.dot %a_arg, %b, %c
    %a_tmp_rem = tensor.extract_slice %a_buf[0, 16] [128, 16]
    %a_prefetch_next = triton_gpu.convert_layout %a_tmp_rem
    ...
    scf.yield %next_a, ..., %a_prefetch_next
  }

*****************
Intel GPU backend
*****************
Intel GPU backend [c27]_ for Triton reuses most of the Triton upstream infrastructure and optimizations arriving at a similar representation for device specific lowering (TritonGPU -> LLVMIR). At this point the backend provides custom passes, layouts, and dialects to adjust the emitted LLVM IR. The IR is then translated to the Standard Portable Intermediate Representation (SPIR-V) [c28]_ to be consumed by Intel Graphics Compiler (IGC) [c29]_.

Components
==========
The Intel GPU backend consists of three major components:

* Triton fork for upstream work
* Intel GPU backend (plugin)

SIMD vs SIMT code generation
============================
IGC provides two distinct ways of compiling a compute kernel:

* **Scalar path** – OpenCL-like kernels, SIMT programming model, when a value in the IR represents an OpenCL’s Work Item [c30]_ (or a logical thread). The logical thread is usually mapped to a SIMD lane (e.g., there usually will be 32 of logical threads in a warp; so, APIs provide synchronization primitives for scalar values that communicate to the whole warp by the compiler inserting the right asm instructions).
* **Vector path** – SIMD-kernels, in this programming model the IR operates on vectors that are mapped to a physical thread. The compiler (originates from the C-for-Metal [c31]_) operates with explicit vectors and vector sizes.
The modes are mostly separate within IGC and make different assumptions about the input IR. Each path exposes a set of intrinsics: GenISA intrinsics [c32]_ for the scalar path with scalar arguments and GenX intrinsics [c33]_ (or vc-instrinsics – the open source name) for the vector path with explicitly vector arguments.

From the execution point of view the two modes are incompatible (in the driver), however, there’s a feature to allow for kernels to do cross-context calls (in dpc++ these are invoke_simd and invoke_spmd, e.g., [c34]_). Those have an overhead and are tricky to use.
Intel GPU backed has thus two paths for Triton kernels compilation:

* SIMT – the default approach (same as AMD/Nvidia) that lowers TritonGPU IR using the layouts described above.
* SIMD – an approach suitable for dense operations that transforms TritonGPU to “warp-level” IR (similar to auto-vectorization), adjusts operator argument sizes and maps the result to XeGPU dialect [c35]_.

At a higher level, the two approaches represent only the way the IR is looked at (e.g., Triton IR can be thought of "SIMD" in a way that it operates on tensors; and autovectorization converts initial sizes to appropriate hardware-defined vector widths for actual instructions).

Runtime
=======
Current state & Motivation
--------------------------
Triton backend for Intel GPU uses an API to interact with the GPU driver. `Upstream work <https://github.com/pytorch/pytorch/issues/114723>`_ on eager mode operator implementation relies on SYCL runtime to allocate and move device memory, invoke kernels, and synchronize those GPU queues. Current packaging assumes users install oneAPI base toolkit to have access to the runtime library. Triton’s initial runtime implementation is Level Zero based. It faced synchronization problems when interacting with IPEX and the currently proposed solution is to introduce SYCL runtime dependency for Triton./

Thoroughly designing components interaction provides an opportunity for user experience and performance improvements.

Analysis results
----------------
Triton needs a runtime to bundle a kernel invocation as well as memory movement to and from GPUs. The are three main options to consider: `SYCL runtime <https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html>`_, `Level Zero runtime <https://github.com/oneapi-src/level-zero>`_, and `Unified runtime <https://github.com/oneapi-src/unified-runtime>`_.

Triton is usually used together with PyTorch in a sense that Triton kernels consume PyTorch tensors and often represent some custom operation. Hence, Triton runtime needs to interact with PyTorch components to guarantee synchronization. These components include:

* The basic device memory allocation & movement (e.g., ``torch.randn(1823, 781, device='xpu')``)
* Aten operators’ implementations via oneDNN
* Pytorch distributed modes via oneCCL
* Habana’s Synapse backend (oneDNN + custom compiler)

Fundamentally, a PyTorch+triton package needs to be able to allocate and move device memory, run kernels, and synchronize on events. All of these can be done at the lowest level (L0).

Using the lowest level possible has following benefits:

* Minimizing the number of dependencies of the package. E.g., Level Zero is a self-contained small (~200KB) loader library that is easily packaged.
* Minimizing the overhead of operation invocations. I.e., SYCL introduces additional layers of abstraction atop Leve Zero API.
* Stabilizing the interface. I.e., having lower-level stable API at the bottom frees the package from updates and compatibility issues as well as allows easier higher level abstractions evolution.
* Narrowing the surface of potential package conflicts.

Using SYCL runtime benefits are:

* Improving the quality of SYCL components by exposing them to more usage scenarios.
* Tactical short-term development speed-up as some of the functionality can be reused (e.g., IPEX memory management).

The key to providing seamless access to Intel’s hardware is frictionless user experience. From user’s perspective, a library is expected to have the minimal possible set of dependencies, not have conflicts with other installations that they might have and be “debugable” and well documented. Triton as a component has no meaningful dependency on SYCL runtime, so applying the above principles it should use the lowest layer possible.

The main technical obstacle in using L0 or Unified runtime for all the components is the presence of the host part of SYCL kernels developed by oneDNN. Internal wrapper structures and integration headers make it complex to execute SYCL fat binaries to be executed by a lower-level runtime. Although possible via dumping the IR/assembly generated by the SYCL compiler and feeding them to lower-level runtime directly for device code and generating the integration headers (e.g., similar technique is used by Unified runtime for CTS tests), the mechanism is ad-hoc.

Components interaction surface is limited to synchronization, that is, passing and consuming events and waiting on them. Having one component use low-level runtime and the other one a high-level creates interoperability issue. This is partially covered by existing functionality such as `Native Driver Access <https://oneapi-src.github.io/unified-runtime/core/PROG.html#native-driver-access>`_ but there are scenarios in which the support is not enough. An example of this would be a component using L0 runtime submitting a kernel and passing control flow to a component that uses SYCL runtime. The latter does not wait on the native L0 queue to drain before reading the data.

Ultimately this means that Triton is forced to have a dependency on SYCL runtime only to wrap the native queue handles for other components to respect the synchronization point.

Directions
----------
Going forward, removing SYCL runtime dependency from Triton can be achieved with some joint effort (Triton, upstream work for memory allocation, oneDNN, oneCCL, and runtimes). Without significant changes to runtimes there are two areas of improvement: data allocation and movement API and synchronization with SYCL-based components.

Memory allocation and movement does not depend on SYCL and can be consumed by SYCL-based components with relatively minor changes. Using Unified runtime is an appealing option as it has the necessary interop capabilities with L0 and can potentially become the standard mechanism for accelerator interaction in PyTorch. It also does not restrict the language choice for kernels implementation and may avoid the need for additional runtimes interop features.

Pytorch has a `mechanism of streams <https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream>`_ (a linear sequence of execution that belongs to a specific device) to orchestrate kernel invocation and data movement. Using PyTorch’s synchronization abstractions can help decouple Triton kernels and Aten operator implementations by communicating via a stream (handled at the Inductor level). The stream can use a low-level runtime and have wrappers for SYCL-based consumers. Its implementation will live inside PyTorch as a separate component.

There is also an option to introduce SYCL kernel invocation support into Unified runtime to make it indifferent to the input. This path allows for putting Unified runtime as the first-class citizen tool for all the components.

Triton will have an opportunity to be used without PyTorch while not having a redundant dependency on SYCL runtime and display all the beneficial qualities of relying on a low-level runtime.

*******************
Links and materials
*******************

.. [c1] Triton repo: https://github.com/openai/triton
.. [c2] PyTorch 2.0 release notes: https://pytorch.org/get-started/pytorch-2.0/#developervendor-experience
.. [c3] FX graph documentation: https://pytorch.org/docs/stable/fx.html
.. [c4] Torch Dynamo deep dive: https://pytorch.org/docs/stable/torch.compiler_deepdive.html
.. [c5] Torch Inductor introduction & design: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
.. [c6] PrimTorch: https://pytorch.org/get-started/pytorch-2.0/#primtorch-stable-primitive-operators
.. [c7] Torch Inductor Triton codegen sources: https://github.com/pytorch/pytorch/blob/95a86ed9ca107329151e0dc172386d50dd3471c6/torch/_inductor/codegen/triton.py
.. [c8] Triton dialects: https://triton-lang.org/main/dialects/dialects.html
.. [c9] Torch Inductor details from hot chips 2023: https://youtu.be/i-dOWSHk3Wk?si=EmnM3pnOglh13j8s&t=828
.. [c10] Triton paper (Triton: an intermediate language and compiler for tiled neural network computations) https://dl.acm.org/doi/abs/10.1145/3315508.3329973
.. [c11] Pytorch IRs: https://pytorch.org/docs/master/ir.html#irs
.. [c12] Extern operators selection mechanism: https://github.com/pytorch/pytorch/blob/94db6578ccee2551c986d92c245e0a0729b99449/torch/_inductor/select_algorithm.py
.. [c13] Guards overview: https://pytorch.org/docs/stable/torch.compiler_guards_overview.html
.. [c14] Triton heuristics: https://github.com/pytorch/pytorch/blob/6ebb26d572d5fcdc6ac0d1297bdf8d1eb5d20722/torch/_inductor/triton_heuristics.py
.. [c15] Softmax implementation example: https://github.com/openai/triton/blob/ded624282e67e5f58db332380e6ff088f276d534/python/tutorials/02-fused-softmax.py
.. [c16] Triton language: https://triton-lang.org/main/python-api/triton.language.html
.. [c17] Triton dialect ops: https://github.com/openai/triton/blob/ded624282e67e5f58db332380e6ff088f276d534/include/triton/Dialect/Triton/IR/TritonOps.td
.. [c18] TritonGPU dialect ops: https://github.com/openai/triton/blob/ded624282e67e5f58db332380e6ff088f276d534/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td
.. [c19] Triton layouts definition: https://github.com/openai/triton/blob/ded624282e67e5f58db332380e6ff088f276d534/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td
.. [c20] Swizzling examples for shared layout: https://github.com/openai/triton/discussions/2026#discussioncomment-6746579
.. [c21] Nvidia’s Matrix Multiply-Accumulate Instructions: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions
.. [c22] Intel’s Xe-HPG overview & white paper: https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-the-xe-hpg-architecture.html
.. [c23] AMD’s Matrix cores: https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/
.. [c24] pybind11: https://github.com/pybind/pybind11
.. [c25] Tensor extract slice: https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract_slice-tensorextractsliceop
.. [c26] Matrix fragments for mma.m16n8k16: https://docs-nvidia-com.translate.goog/cuda/parallel-thread-execution/index.html?_x_tr_sl=auto&_x_tr_tl=en&_x_tr_hl=en&_x_tr_pto=wapp#warp-level-matrix-fragment-mma-16816-float
.. [c27] Intel XPU backend for Triton repo: https://github.com/intel/intel-xpu-backend-for-triton
.. [c28] SPIR-V: https://www.khronos.org/spir/
.. [c29] Intel Graphics Compiler: https://github.com/intel/intel-graphics-compiler
.. [c30] OpenCL 3.0 API specification: https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_API.html#_execution_model
.. [c31] C-for-Metal: High Performance SIMD Programming on Intel GPUs: https://arxiv.org/abs/2101.11049
.. [c32] GenISA intrinsics: https://github.com/intel/intel-graphics-compiler/blob/4a1798982e29564baba0265b19a4752f8f458219/IGC/GenISAIntrinsics/Intrinsic_definitions.py
.. [c33] GenX intrinsics: https://github.com/intel/vc-intrinsics
.. [c34] Sycl ext invoke_simd: https://github.com/intel/llvm/blob/d3c8a7e621ba41be5c11ebad1bce8cd1af216117/sycl/doc/extensions/experimental/sycl_ext_oneapi_invoke_simd.asciidoc
.. [c35] XeGPU dialect: https://github.com/intel/mlir-extensions
