# Worked Examples — Block Pointer → Tensor Descriptor

Verbatim before→after pairs to pattern-match against. Each pair corresponds to a
kernel in `scripts/validate_rules.py`.

> **Validation status: HARDWARE-VALIDATED** on Intel Data Center GPU Max 1100
> (PVC) via `scripts/validate_rules.py --check-ir` (result: `ALL VALIDATED`,
> exit 0). Every pair below passed both (a) numeric equality before≡after≡
> reference (`torch.testing.assert_close`) and (b) the IR check confirming the
> AFTER descriptor kernel lowers to `Subgroup2DBlock` 2D-block-I/O — except the
> two rank-1 pairs (`copy1d`, unit-stride 1D descriptor; and `copy1d_strided`,
> the non-unit-stride masked-pointer fallback), which are correctly
> correctness-only with no 2D block I/O today (by design). Re-run any time on an
> XPU host (after initializing the oneAPI runtime):
> `python scripts/validate_rules.py --check-ir`.

The canonical real-world reference is the repo's own GEMM benchmark:
`benchmarks/triton_kernels_benchmark/gemm_benchmark.py` (tensor descriptor version).

---

## 1. Simple 2D GEMM (Rules 1, 2, 3, 4, 7)

**Before (block pointer):**
```python
a_bp = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                         offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
b_bp = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                         offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for _ in range(0, K, BLOCK_K):
    a = tl.load(a_bp, boundary_check=(0, 1))
    b = tl.load(b_bp, boundary_check=(0, 1))
    acc += tl.dot(a, b)
    a_bp = tl.advance(a_bp, (0, BLOCK_K))
    b_bp = tl.advance(b_bp, (BLOCK_K, 0))
c_bp = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                         offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
tl.store(c_bp, acc, boundary_check=(0, 1))
```

**After (tensor descriptor):**
```python
a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   block_shape=(BLOCK_M, BLOCK_K))
b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                   block_shape=(BLOCK_K, BLOCK_N))
c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                   block_shape=(BLOCK_M, BLOCK_N))
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
off_k = 0
for _ in range(0, K, BLOCK_K):
    a = a_desc.load([pid_m * BLOCK_M, off_k])
    b = b_desc.load([off_k, pid_n * BLOCK_N])
    acc += tl.dot(a, b)
    off_k += BLOCK_K
c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)
```
The advancing K dim becomes `off_k`; the static M/N offsets are passed inline.
All three descriptors are 2D, last-stride-1, local, and dot-feeding → fast path.

---

## 2. Transposed B operand (Rule 6)

B is stored `(N, K)` row-major; the dot wants `(K, N)`. The block pointer
expressed this with `order=(0, 1)`. The descriptor keeps **last stride == 1** by
describing B in its native `(N, K)` layout and transposing the loaded block.

**Before:**
```python
b_bp = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                         offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
...
b = tl.load(b_bp, boundary_check=(0, 1))
```

**After:**
```python
b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(N, K), strides=(stride_bn, stride_bk),
                                   block_shape=(BLOCK_N, BLOCK_K))
...
b = b_desc.load([pid_n * BLOCK_N, off_k]).T
```
Do **not** instead write a `(K, N)` descriptor with a non-1 last stride — that
falls off the 2D-block-I/O path.

---

## 3. Batched GEMM (Rules 8 / 11 rank-3 fold)

Fold the batch index into `base`; keep the descriptor **2D**.

**Before:**
```python
bid = tl.program_id(2)
off_a = bid.to(tl.int64) * stride_az
a_bp = tl.make_block_ptr(base=a_ptr + off_a, shape=(M, K), strides=(stride_am, stride_ak),
                         offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
```

**After:**
```python
bid = tl.program_id(2)
off_a = bid.to(tl.int64) * stride_az
a_desc = tl.make_tensor_descriptor(base=a_ptr + off_a, shape=(M, K), strides=(stride_am, stride_ak),
                                   block_shape=(BLOCK_M, BLOCK_K))
```
Note `.to(tl.int64)` on the batch index to avoid int32 overflow on large strides.

---

## 4. Boundary-masked tensor-of-pointer load (Rule 9, boundary case)

A `K` not divisible by `BLOCK_K` produces a masked edge tile. The boundary mask
`offs_k < K - k*BLOCK_K` folds into the descriptor `shape=(M, K)` / `(K, N)`.

**Before:**
```python
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
for k in range(0, tl.cdiv(K, BLOCK_K)):
    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
    ...
    a_ptrs += BLOCK_K * stride_ak
```

**After:**
```python
a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                   block_shape=(BLOCK_M, BLOCK_K))
off_k = 0
for _ in range(0, tl.cdiv(K, BLOCK_K)):
    a = a_desc.load([pid_m * BLOCK_M, off_k])   # shape=(M,K) zero-pads the edge tile
    ...
    off_k += BLOCK_K
```

For a **data-dependent** mask (causal, value predicate) instead:
```python
v = desc.load([row, col])
v = tl.where(mask, v, other)   # load full block, re-apply predicate in registers
```

---

## 5. Rank-1 copy, unit stride (Rule 11, rank-1)

Translate to a **1D descriptor**. XPU does not give 1D descriptors 2D block I/O
today (the 1D fast path is pending), so flag that this lowers via the pointer
path — but the form is correct and forward-compatible. (Validated by the
`copy1d` pair.)

**Before:**
```python
x_bp = tl.make_block_ptr(base=x_ptr, shape=(N,), strides=(1,),
                         offsets=(pid * BLOCK,), block_shape=(BLOCK,), order=(0,))
v = tl.load(x_bp, boundary_check=(0,))
```

**After:**
```python
x_desc = tl.make_tensor_descriptor(base=x_ptr, shape=(N,), strides=(1,), block_shape=(BLOCK,))
v = x_desc.load([pid * BLOCK])
```

---

## 5b. Rank-1, non-unit stride → masked pointer load (Rule 11, rank-1 strided)

A strided 1D view (last/only stride `!= 1`) **cannot** be a descriptor — its last
stride must be 1, so `strides=(H,)` fails to compile (`Tensor descriptor last dim
must be 1 but got H`). This is the one `boundary_check` that cannot be folded into
a descriptor `shape`; reproduce it as an explicit `mask=` on a tensor-of-pointer
load/store (pointer path, no 2D block I/O). This is SGLang FLA's `p_beta`/`p_g`
pattern. (Validated by the `copy1d_strided` pair.)

**Before:**
```python
p_beta = tl.make_block_ptr(beta + bos * H + i_h, (T,), (H,),
                           (i_t * BT,), (BT,), (0,))   # stride H != 1
b_beta = tl.load(p_beta, boundary_check=(0,))
```

**After:**
```python
o = i_t * BT + tl.arange(0, BT)
b_beta = tl.load(beta + bos * H + i_h + o * H, mask=o < T, other=0.0)
# strided rank-1 store: tl.store(g + bos * H + i_h + o * H, b_val, mask=o < T)
```
Do **not** write `tl.make_tensor_descriptor(beta + ..., (T,), (H,), (BT,))` — a
non-unit last stride does not compile. Unlike Rule 9 (which *removes* a boundary
mask by folding it into a descriptor `shape`), here the mask is *retained* on the
pointer path because no legal descriptor exists.

---

## 6. Block pointer received as a helper argument (Rule 12)

A block pointer can only be passed `@triton.jit`→`@triton.jit`, and doing so
defeats the fast path. The fix changes the **interface**, not just the body.

**Before:**
```python
@triton.jit
def load_tile(bp):                         # receives a block pointer
    return tl.load(bp, boundary_check=(0, 1))

@triton.jit
def kernel(a_ptr, M, K, ...):
    bp = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                           offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    a = load_tile(bp)
```

**After (helper builds the descriptor locally; call sites pass raw ingredients):**
```python
@triton.jit
def load_tile(a_ptr, M, K, stride_am, stride_ak,
              row, col, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                     block_shape=(BLOCK_M, BLOCK_K))
    return desc.load([row, col])

@triton.jit
def kernel(a_ptr, M, K, ...):
    a = load_tile(a_ptr, M, K, stride_am, stride_ak, pid_m * BLOCK_M, 0, BLOCK_M, BLOCK_K)
```
**Caller-affecting:** every call site of `load_tile` must change. Report them.
