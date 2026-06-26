# Annotated PTX Walkthrough

For readers who know general assembler but not PTX. Covers the parts that matter
for the TMA descriptor-update optimization; skips DWARF/debug boilerplate.

## PTX in 10 lines

- Registers are **virtual, typed, unlimited**. `%r` = b32, `%rd` = b64,
  `%rs` = b16, `%p` = predicate (bool). `.reg .b32 %r<47>;` just declares "up to
  46 b32 regs". ptxas assigns real registers later.
- **Predication:** `@%p2 instr` runs `instr` only if `%p2` is true — branchless
  per-thread conditional (like ARM predication).
- **Memory space is in the opcode:** `ld.global`/`st.global` (DRAM),
  `ld.shared`/`st.shared::cta` (SMEM), `ld.param` (kernel args).
- **`bar.sync 0`** = `__syncthreads()` (block barrier). `bar.warp.sync` = warp.
- **Special regs:** `%tid.x` (thread in block), `%ctaid.x` (block id),
  `%nctaid.x` (grid dim).
- **`// begin/end inline asm`** = instructions Triton emitted by hand (the TMA
  intrinsics). Everything else is LLVM-generated.
- `cvt.u64.u32 d, s` = zero-extend convert. `cvta.global` = convert a generic
  pointer to a `.global` address. `shl.b64 d,s,N` = `d = s << N`.
- `mad.wide.u32 d,a,b,c` = `d = a*b + c` (wide). `mad.lo` = low bits.
- `setp.lt.u32 %p, a, b` = `%p = (a < b)` — sets a predicate.
- `elect.sync` picks one leader lane in a warp (used to do "one thread does X").

## Shared prologue (both kernels identical here)

```
mov.u32   %r11, %ctaid.x;        // r11 = blockIdx.x  (= seq_idx)
cvt.u64.u32 %rd6, %r11;          // rd6 = (u64) seq_idx
ld.param.b64 %rd7, [..._param_2];// rd7 = block_tables_ptr
ld.param.b64 %rd8, [..._param_6];// rd8 = stride_bt_0
mul.lo.s64  %rd9, %rd8, %rd6;    // rd9 = seq_idx * stride_bt_0
shl.b64     %rd10, %rd9, 2;      // rd10 = rd9 * 4   (int32 elem size; the *4 = paged_kv_load.py:71)
add.s64     %rd1, %rd7, %rd10;   // rd1 = &block_tables[seq_idx][0]   <-- per-seq base, loop-invariant
...
mov.u32   %r12, %tid.x;
and.b32   %r1, %r12, 127;        // r1 = tid % 128  (lane within the 128-thread block)
shr.u32   %r2, %r12, 5;          // r2 = tid / 32   (warp index)
...
mov.b32   %r18, global_smem;     // r18 = base of the block's shared-memory arena
...
shl.b64   %rd27, %rd11, 1;       // rd27 = stride_kv_1 * 2  (bytes; bf16=2)  -- the global_stride field value
```

Key invariant values established before the loop: `%rd1` (block-table base),
`%r18` (SMEM base), `%rd27` (the descriptor's `global_stride`, in bytes),
`%rd28` (the global descriptor scratch addr, built earlier). None of these
change across iterations.

The predicates that gate the TMA intrinsics:
```
setp.eq.b32 %p2, %r1, 0;         // p2 = (lane == 0)        -> "thread 0 only"
setp.lt.u32 %p1, %r1, 32;        // p1 = (lane < 32)        -> "first warp only"
```
So the 16 `tensormap.replace.tile.*` run on **thread 0 alone** (`@%p2`), and the
zero-fill / cp_fenceproxy / acquire run on the **first warp** (`@%p1`). That
"thread 0 only" is why the 16 replaces are *serialized* — one lane does them all.

---

## BASELINE loop body — the problem

`$L__BB0_1:` … `@%p11 bra $L__BB0_1;`  (paged_kv_load_kernel.ptx lines 105-361)

```
$L__BB0_1:
  // --- compute this iteration's KV base address (genuinely loop-varying) ---
  add.s64 %rd24, %rd1, %rd49;            // rd24 = &block_tables[seq][j]   (rd49 = j*4 counter)
  ld.global.b32 { %r31 }, [ %rd24 + 0 ]; // r31 = physical_block_idx = block_tables[seq][j]   <-- RUNTIME LOAD
  cvt.s64.s32 %rd45, %r31;               // sign-extend to 64b
  mul.lo.s64  %rd46, %rd4, %rd45;        // rd46 = physical_block_idx * stride_kv_0
  shl.b64     %rd47, %rd46, 1;           // * 2 (bf16 bytes)
  add.s64     %rd26, %rd3, %rd47;        // rd26 = kv_cache_ptr + ...  <-- THE NEW global_address

  // --- (re)build the WHOLE descriptor in SMEM, every iteration ---
  bar.sync 0;
  @%p1 st.shared::cta.b32 [ %r32 + 0 ], %r40;   // ZERO-FILL 128B of SMEM (first warp). r40=0.
  bar.warp.sync -1;
  cvt.u64.u32 %rd25, %r18;               // rd25 = SMEM descriptor staging addr (= global_smem+0)

  @%p2 tensormap.replace.tile.global_address... [%rd25], %rd26;  // field: base       (VARIES)
  @%p2 tensormap.replace.tile.rank...          [%rd25], 0x1;     // field: rank=1     (invariant)
  @%p2 tensormap.replace.tile.box_dim...       [%rd25],0x0,%r33; // box_dim[0]=64     (invariant)
  @%p2 tensormap.replace.tile.box_dim...       [%rd25],0x1,%r34; // box_dim[1]=16     (invariant)
  @%p2 tensormap.replace.tile.global_dim...    [%rd25],0x0,%r35; // global_dim[0]=128 (invariant)
  @%p2 tensormap.replace.tile.global_dim...    [%rd25],0x1,%r34; // global_dim[1]=16  (invariant)
  @%p2 tensormap.replace.tile.global_stride... [%rd25],0x0,%rd27;// global_stride     (invariant)
  @%p2 tensormap.replace.tile.element_stride...[%rd25],0x0,%r36; // =1                (invariant)
  @%p2 tensormap.replace.tile.element_stride...[%rd25],0x1,%r36; // =1                (invariant)
  @%p2 tensormap.replace.tile.elemtype...      [%rd25],0xa;      // =10 (bf16)        (invariant)
  @%p2 tensormap.replace.tile.interleave...    [%rd25],0x0;      // =0                (invariant)
  @%p2 tensormap.replace.tile.swizzle_mode...  [%rd25],0x3;      // =3 (128B swizzle) (invariant)
  @%p2 tensormap.replace.tile.fill_mode...     [%rd25],0x0;      // =0                (invariant)

  // --- publish SMEM descriptor -> global, then make TMA proxy observe it ---
  @%p1 tensormap.cp_fenceproxy.global.shared::cta... [%rd28], [%rd25], 0x80;  // copy 128B SMEM->global + release
  @%p1 fence.proxy.tensormap::generic.acquire.gpu [%rd28], 0x80;             // proxy re-acquire
  @%p1 cp.async.bulk.commit_group ;
  @%p1 cp.async.bulk.wait_group.read 0 ;
  bar.sync 0;

  // --- set up mbarrier + issue the TMA load of the actual KV tile ---
  ... mbarrier.init / arrive.expect_tx ...
  @%p4 cp.async.bulk.tensor.2d... [%r38], [%rd29, {...}], [%r37];  // TMA: global KV -> SMEM tile at %r38
  ... mbarrier.try_wait (spin until tile arrived) ...

  // --- read tile out of SMEM, store to output ---
  ld.shared.b16 %rs1, [%r3]; ... (16 loads)
  st.global.b16 [%rd30], {%rs1}; ... (16 stores)

  // --- loop update ---
  add.s64 %rd49, %rd49, 4;    // j counter += 4 bytes (i32 stride)
  add.s64 %rd48, %rd48, %rd2; // output ptr advance
  setp.ne.b64 %p11, %rd49, 2048;
  @%p11 bra $L__BB0_1;        // loop while j != 512 (2048/4)   NUM_KV_BLOCKS=512
```

**Everything between the zero-fill and the cp_fenceproxy is rebuilt every
iteration**, but only the *first* line (`global_address` ← `%rd26`) actually
changes. 12 invariant replaces + a 128-byte zero-fill are pure waste, serialized
on thread 0, sitting on the critical path before the publish.

---

## OPTIMIZED loop body — the fix

`paged_kv_load_kernel.optimized.ptx`. Two changes: a one-time HOISTED block
before the loop, and a stripped loop body.

### Hoisted block (runs ONCE, just before `$L__BB0_1`)

```
// relocate descriptor to a PERSISTENT, non-overlapping SMEM home (see below)
add.s32 %r48, %r18, 4224;              // r48 = global_smem + 4224   (past tile+mbarrier)
add.s32 %r32, %r48, %r17;              // zero-fill dest now relative to r48
...
cvt.u64.u32 %rd25, %r48;               // rd25 = relocated descriptor staging addr
mov.b32 %r40, 0;
bar.sync 0;
@%p1 st.shared::cta.b32 [ %r32 + 0 ], %r40;   // zero-fill ONCE
bar.warp.sync -1;
@%p2 tensormap.replace.tile.rank...          // all 12 INVARIANT fields, ONCE
@%p2 tensormap.replace.tile.box_dim... (x2)
@%p2 tensormap.replace.tile.global_dim... (x2)
@%p2 tensormap.replace.tile.global_stride...
@%p2 tensormap.replace.tile.element_stride... (x2)
@%p2 tensormap.replace.tile.elemtype...
@%p2 tensormap.replace.tile.interleave_layout...
@%p2 tensormap.replace.tile.swizzle_mode...
@%p2 tensormap.replace.tile.fill_mode...
// NOTE: global_address is NOT built here — done per-iteration below.
```

### Loop body (stripped)

```
$L__BB0_1:
  add.s64 %rd24, %rd1, %rd49;
  ld.global.b32 { %r31 }, [ %rd24 + 0 ]; // physical_block_idx (still per-iter)
  cvt.s64.s32 %rd45, %r31;
  mul.lo.s64  %rd46, %rd4, %rd45;
  shl.b64     %rd47, %rd46, 1;
  add.s64     %rd26, %rd3, %rd47;        // rd26 = new global_address

  // --- the ONLY per-iteration descriptor work ---
  @%p2 tensormap.replace.tile.global_address... [%rd25], %rd26;   // 1 field
  @%p1 tensormap.cp_fenceproxy.global.shared::cta... [%rd28],[%rd25],0x80; // republish
  @%p1 fence.proxy.tensormap::generic.acquire.gpu [%rd28], 0x80;          // re-acquire
  @%p1 cp.async.bulk.commit_group ;
  @%p1 cp.async.bulk.wait_group.read 0 ;
  bar.sync 0;

  ... (mbarrier + TMA tile load + SMEM read + global store, UNCHANGED) ...
  @%p11 bra $L__BB0_1;
```

Loop body goes from **13 replaces + zero-fill** → **1 replace**. The
`cp_fenceproxy` + `acquire` stay (the irreducible floor — the global descriptor
must be re-published and the proxy re-acquired after the address changes).

---

## The subtle correctness fix (why `+4224`)

In the baseline the descriptor staging is at `global_smem + 0` — which
**overlaps** the data tile region `[0, 4096)` (the TMA copy writes the KV tile
to `global_smem + 0` or `+2048`). That overlap is fine in the baseline *because*
the descriptor is rebuilt from scratch every iteration after the tile clobbers
it.

For the hoisted version the descriptor must **survive** across iterations, so it
needs its own home that the data tile never touches. `+4224` puts it past the
tile (`[0,4096)`) and the mbarrier (`+4096`). Cost: the kernel's shared-memory
allocation grows by 128 bytes — the concrete form of the "persistent SMEM costs
occupancy" tradeoff.

See [NOTES.md](NOTES.md) for the full finding and the SMEM-layout evidence.
