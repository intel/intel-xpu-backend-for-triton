# Intel XPU Backend for Triton: What's New and What's Next

**Triton Developer Summit — October 19, 2026**

---

## Proposal Abstract (150 words)

Two years ago, we introduced the Intel XPU Backend for Triton, demonstrating XMX engine
support, DPAS-based GEMM performance, and the critical role of block pointers for efficient
memory access on Intel GPUs. Since then, the landscape has evolved dramatically — both in
hardware and in Triton's programming model.

This talk covers three major developments: (1) New Intel GPU architectures — Battlemage (BMG)
for client and Crescent Island (CRI/Xe3P) for data center — and what new capabilities they
unlock for Triton kernel developers. (2) Following the community's deprecation of block
pointers, how Intel Triton natively handles tensor descriptors using hardware 2D block I/O,
the performance benefits this brings, and an honest look at when tensor descriptors don't
win — and what we're doing about it. (3) Real-world ecosystem adoption: Intel Triton powering
vLLM, SGLang, and PyTorch workloads with MXFP4 quantization and flash attention.

---

## Section 1: Where We Were (3 min)

### Slide: "Two Years Ago at This Summit"

- Introduced Intel XPU Backend for Triton as an out-of-tree backend
- Demonstrated:
  - XMX (Xe Matrix eXtension) engine and DPAS instruction support
  - GEMM performance on PVC (Ponte Vecchio / Data Center GPU Max)
  - Block pointer → 2D block I/O pipeline for efficient memory access
- Key message then: **block pointers are essential for Intel GPU performance**
- Runtime stack: SYCL + Level Zero + IGC (Intel Graphics Compiler)

### Slide: "What's Changed Since Then"

- Triton community deprecated block pointers, moved to tensor descriptors
- Two new GPU architecture generations shipped (Xe2, Xe3P)
- Backend matured from proof-of-concept to production ecosystem integration
- Upstream collaboration: regular merges with OpenAI Triton, layout conversion
  improvements contributed back

---

## Section 2: New Hardware Landscape (5 min)

### Slide: "Intel GPU Architecture Evolution"

| Generation | Products | Key Triton-Relevant Features |
|------------|----------|------------------------------|
| Xe-HPC (2022) | PVC (Data Center GPU Max) | DPAS exec=16, 2D block I/O, 512KB L1/SLM, HBM |
| Xe2 (2024) | **BMG** (Arc B-series) | DPAS exec=16, 2D block I/O, MXFP4 scaled dot |
| Xe3P (2025+) | **CRI** (Crescent Island) | BF8 DPAS, elementwise prefetch, 256B prefetch |

### Slide: "BMG — Bringing Datacenter Triton to Client"

**What is BMG?**
- Xe2 architecture, Arc B-series discrete client GPU
- Same DPAS execution size (16) as PVC — kernels port directly
- Full 2D block I/O support
- Scaled matrix multiply for MXFP4/FP8 quantized inference

**What it means for Triton developers:**
- Write once, run on both datacenter (PVC) and client (BMG) Intel GPUs
- DPAS and 2D block I/O support means same optimization strategies apply
- Enables local development and testing with consumer hardware

**Speaker notes:** Emphasize that BMG lets developers buy an Arc B580/B770 and develop
Triton kernels locally that will run at scale on datacenter GPUs. Lower barrier to entry.

### Slide: "CRI (Xe3P) — New Capabilities for Datacenter"

**What is CRI?**
- Xe3P architecture, next-gen datacenter GPU (Crescent Island)
- Builds on PVC/BMG foundation with new capabilities

**New Triton-relevant features:**
- **BF8 DPAS**: Native BF16 accumulation with FP8 inputs
  (`BF16_BF16_FP8_FP8` engine type) — better precision for inference
- **Elementwise prefetch**: Software pipelining for non-dot loads
  - Previously only dot operands were prefetched in the pipeline
  - CRI enables prefetching arbitrary loads (elementwise ops, reductions)
  - Overlaps memory latency with compute for non-matmul kernels
- **256-byte prefetch**: Wider prefetch instructions for higher bandwidth
- **Split barriers**: Asynchronous producer-consumer synchronization

**Speaker notes:** The elementwise prefetch story is compelling — it shows that Intel
hardware capabilities are driving new compiler optimizations, not just matching NVIDIA
features. This is a differentiator.

---

## Section 3: Tensor Descriptors — The Deep Dive (10 min)

### Slide: "The Block Pointer → Tensor Descriptor Migration"

**Timeline:**
- 2024: Intel backend relied heavily on block pointers for 2D block I/O
- 2025: Triton community deprecated block pointers
- 2025-2026: Intel backend migrated to native tensor descriptor support

**What changed in the codebase (concrete commits):**
- Removed `RewriteTensorPointer` pass entirely
- Removed Intel block pointer handling code
- Added native `MakeTensorDescOp` → `DescriptorLoadOp/StoreOp` flow
- `RewriteTensorDescriptorToPointer` pass: new fallback for when descriptors
  can't be used directly

### Slide: "How Intel Handles Tensor Descriptors Natively"

**The pipeline:**
```
tt.make_tensor_descriptor (Python frontend)
    ↓
[TTIR passes: optimize contiguous gathers, scatter rewrites]
    ↓
RewriteTensorDescriptorToPointer (fallback for non-2D-block cases)
    ↓
MaterializeBlockPointer (annotate 2D block I/O eligibility)
    ↓
[LLVM lowering: emit 2D block load/store/prefetch intrinsics]
```

**Key insight:** Intel hardware has dedicated 2D block I/O units in the LSC
(Load/Store Controller). Tensor descriptors map *directly* to hardware surface
descriptors — base pointer, width, height, pitch. No software emulation needed.

**Constraints enforced:**
- 16-byte aligned base pointer
- 16-byte aligned pitch (stride between rows)
- Stride-one in the last (contiguous) dimension
- Maximum rank: 5

### Slide: "Performance: Tensor Descriptor vs Tensor of Pointers"

**Show benchmark data for:**
- GEMM (various sizes): TD vs ToP on PVC, BMG
- Flash attention forward/backward: before/after TD migration
- vLLM unified attention kernel: with and without 2D block loads

**Key talking points:**
- 2D block loads bypass the gather/scatter path entirely
- Hardware handles pitch compensation and address calculation
- Fewer instructions = less register pressure = better occupancy

**TODO: Collect actual benchmark numbers for these comparisons.**

### Slide: "When Tensor Descriptors Don't Win"

**Case 1: Scattered access patterns**
- When offsets are data-dependent and non-contiguous
- Falls back to scalar gather/scatter (N independent loads)
- Example: attention with dynamic sparse masks, indirect indexing

**Case 2: Alignment violations**
- Base pointer not 16-byte aligned
- Pitch not a multiple of 16 bytes
- Common with odd tensor shapes or sliced tensors

**Case 3: Tile shapes below 2D block minimums**
- Hardware has minimum tile dimensions for 2D block I/O
- Very small tiles (e.g., 1 row) can't use block loads
- Fallback: 1D block I/O or scalar loads

**Case 4: 1D strided access patterns**
- Access pattern is strided but 1-dimensional
- Doesn't naturally map to 2D surface descriptor

### Slide: "Closing the Gap — What We're Doing About It"

**Optimization 1: Contiguous Gather Rewrite**
```
Before: N scalar gather operations (one per element)
After:  Single 2D block load (when offsets form contiguous range)
```
The `RewriteContiguousGather` pattern detects when gather offsets form a
contiguous range and replaces N scattered loads with one 2D block load.
"Orders of magnitude" improvement per code comments.

**Optimization 2: Multi-Range Gather**
When offsets have compile-time-constant sub-ranges:
- Emit one 2D block load per contiguous sub-range
- Concatenate results
- Still much better than N scalar loads

**Optimization 3: 1D Strided → 2D Block Reshape**
```
Before: 1D strided load (can't use 2D block I/O)
After:  Reshape to 2D tile → 2D block load → reshape back
```
The `MaterializeBlockPointer` pass now detects 1D strided patterns and
reshapes them into 2D tiles that qualify for block I/O.

**Optimization 4: Base Address Alignment Compensation**
Hardware requires 64-byte aligned base addresses. The backend compensates:
```
alignedPtr = ptr & ~0x3f        // Align down to 64 bytes
offset = ptr & 0x3f             // Track the misalignment
adjustedWidth = width + offset   // Widen the surface to include padding
adjustedX = x + offset/elemSize  // Shift the start coordinate
```
This is transparent to the user — non-aligned tensors "just work."

### Slide: "Call to Action: Use Tensor Descriptors"

**For kernel authors:**
- Use `triton.make_tensor_descriptor()` instead of `make_block_ptr` or manual pointers
- Ensure base pointer and strides are 16-byte aligned when possible
- Prefer 2D tile shapes that fit hardware minimums
- The compiler does the rest — 2D block I/O is selected automatically

**For framework developers (vLLM, SGLang, PyTorch):**
- Allocate tensors with alignment-friendly shapes
- Pass stride information through to Triton kernels
- Test with Intel GPUs in CI (we can help!)

---

## Section 4: Ecosystem Adoption (7 min)

### Slide: "Intel Triton in the Wild"

**vLLM:**
- 9 dedicated test suites covering the full inference stack
- 40+ validated models (LLaMA, Qwen, DeepSeek, multimodal)
- Key kernels: unified attention, paged attention, MOE, speculative decoding
- FP16, FP8, MXFP4 quantization support
- Tensor descriptor-based unified attention kernel for 2D block loads on Q matrix

**SGLang:**
- Scaled-MM benchmark integration
- Kernel validation on Intel hardware

**PyTorch (Inductor):**
- Regular pin updates tracking upstream PyTorch
- Inductor test suite validation
- Intel Extension for PyTorch (IPEX) integration

**Liger Kernels:**
- Recently integrated (pin added)
- Community kernel library running on Intel GPUs

### Slide: "Flash Attention on Intel GPUs"

**Forward pass:**
- Tensor descriptor-based implementation
- Unconditional TWISST_GRID layout optimization
- FlexAttention support with causal and custom masks

**Backward pass:**
- Tensor descriptor migration complete (replacing regular pointers)
- Performance tuning for register pressure

**Benchmarks:**
- Causal FlexAttention benchmarks on PVC and BMG
- Paged attention for vLLM inference
- Custom mask support for diverse model architectures

**TODO: Include specific speedup numbers.**

### Slide: "MXFP4 Quantization"

- Block Scale DPAS (BDPAS) instruction: `D = C + (A * B) * scaleA * scaleB`
- Hardware support on Xe2 (BMG) and Xe3P (CRI)
- Production kernels for MXFP4 x MXFP4 matrix multiply
- Enables efficient 4-bit quantized inference for LLMs
- Tuned for small-batch inference (the common vLLM use case)

### Slide: "Compiler Optimizations That Matter for Real Workloads"

| Optimization | What It Does | Impact |
|-------------|-------------|--------|
| Software pipelining | Overlap memory loads with compute | Hide memory latency |
| Auto-GRF mode | Detect spills, recompile with 256 registers | Reduce spill/fill overhead |
| HoistLayoutConversions | Move layout changes out of loops with GRF budget | Reduce register pressure |
| ReduceVariableLiveness | Shorten live ranges of large tensors | Reduce register pressure |
| Contiguous gather rewrite | Replace N scalar loads with 1 block load | Massive bandwidth improvement |

---

## Section 5: What's Next (3 min)

### Slide: "Looking Forward"

**Hardware:**
- CRI (Xe3P) datacenter GPU availability
- Continued BMG ecosystem growth on client

**Compiler:**
- Deeper tensor descriptor optimization (more cases covered)
- Gluon dialect integration (warp specialization, tensor memory)
- Improved cross-backend portability

**Ecosystem:**
- Broader vLLM model coverage
- SGLang production deployment support
- Community kernel libraries (Liger, Triton Kernels)

### Slide: "How to Get Involved"

- GitHub: `intel/intel-xpu-backend-for-triton`
- Works on consumer Intel Arc GPUs (BMG) — try it today
- CI runs on PVC and BMG — PRs welcome
- vLLM XPU support: production-ready for inference
- Documentation: Kobuk guide (link in README)

---

## Appendix: Backup Slides

### Backup: "Intel GPU Memory Hierarchy"

| Architecture | L1 Cache / SLM | L2 Cache |
|-------------|----------------|----------|
| Xe-HPG (DG2/Arc A) | 128 KB SLM, 256 KB L1 per Xe-core | 16 MB |
| Xe-HPC (PVC) | 512 KB L1/SLM per Xe-core | 408 MB (HBM + cache) |
| Xe2 (BMG) | TBD | TBD |

Key difference from NVIDIA: Intel loads dot operands directly to registers from memory.
No SLM staging needed for A/B matrices. The hardware I/O buffer and cache handle
redundant accesses.

### Backup: "DPAS Instruction Details"

```
D[M,N] = C[M,N] + A[M,K] x B[K,N]

M = repeatCount (up to 8)
N = executionSize (16 for PVC/BMG, 8 for DG2)
K = systolicDepth(8) x opsPerChannel(varies by dtype)
```

opsPerChannel: FP16/BF16 → 2, INT8/FP8 → 4, TF32 → 1, FP4 → 8

### Backup: "Compilation Pipeline Overview"

```
Python kernel → TTIR → TTGIR → LLVM IR → SPIR-V → ZEBIN (native binary)
                 ↑        ↑         ↑          ↑
              Frontend  Layout    Lowering   IGC JIT
                       + DPAS   to GenISA/
                       + Pipeline  SPIR-V
```

5 stages, with 2D block I/O decisions made in TTGIR (MaterializeBlockPointer)
and final intrinsic selection in LLVM lowering.

### Backup: "Tensor Descriptor Constraints Cheat Sheet"

| Constraint | Value | Why |
|-----------|-------|-----|
| Base pointer alignment | 16 bytes | Hardware surface descriptor requirement |
| Pitch alignment | 16 bytes (128 bits) | LSC alignment for 2D addressing |
| Stride-one dimension | Last dim must be 1 | Contiguous memory access requirement |
| Max rank | 5 | Hardware descriptor limit |
| Min tile width | Varies by dtype | 2D block I/O minimum |
| Max tile height | 32 rows | 2D block load limit |
| Max tile bytes/row | 64 bytes | 2D block load limit |
| Base address alignment | 64 bytes (compensated) | Hardware requires, compiler handles |

---

## TODOs Before Submission

- [ ] Collect GEMM benchmark numbers: TD vs ToP on PVC and BMG
- [ ] Collect flash attention speedup numbers
- [ ] Get CRI availability timeline for public disclosure
- [ ] Decide on talk length (30 min assumed above, adjust if 20 or 45)
- [ ] Add speaker bio and affiliation
- [ ] Review with team for any NDA-restricted content (especially CRI details)
- [ ] Create visual diagrams for the tensor descriptor pipeline
- [ ] Prepare live demo if possible (GEMM on BMG showing TD vs ToP?)
