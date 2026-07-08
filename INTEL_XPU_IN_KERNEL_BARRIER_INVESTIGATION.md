# Intel XPU In-Kernel Barrier Investigation

Date: 2026-06-22

This note records the debugging trail for calling Intel SHMEM synchronization primitives from inside Triton kernels, especially `dl.barrier_all()` and `dl.sync_all()`.

> **UPDATE 2026-06-22 (backend-fork continuation): ROOT CAUSE FOUND AND FIXED.**
> The original "Short Conclusion" below (finding #1) was a **misdiagnosis** and is
> kept only as a debugging trail. The real root cause is not that "Triton-side
> guards do not mask atomics"; it is that **`xpu.tid(0)` did not return a per-lane
> work-item id at all.** It lowered to `llvm.genx.local.id.x`, an intrinsic the
> Intel SPIR-V backend does not recognize (only `llvm.genx.GenISA.*` is
> whitelisted), so it survived translation as an unresolved import that resolves
> to a uniform `0` on every lane. A guard like `if xpu.tid(0) == 0` was therefore
> **not divergent**: it was uniformly true (all 128 lanes enter) or uniformly
> false. Every lane then ran the guarded collective, driving the 1-PE psync word
> to 128 and hanging. See the new section
> [**Resolution: `xpu.tid` Was Not Per-Lane**](#resolution-xputid-was-not-per-lane).

## Short Conclusion (ORIGINAL — finding #1 superseded; see Resolution)

There are three separate findings:

1. **Root cause for Triton 1-PE `dl.sync_all()`: Triton-side guards do not mask atomics inside linked SYCL bitcode.** A one-shot psync probe showed the ISHMEM psync word is incremented 128 times even when the Triton call is guarded with `if xpu.tid(0) == 0`. A minimal non-ISHMEM external atomic probe reproduces the same behavior: a guarded external atomic increments by 128. Since ISHMEM expects the 1-PE psync value to become exactly `1`, it spins forever after the psync value overshoots to `128`. **[SUPERSEDED — the guard was non-divergent because `xpu.tid(0)` was uniformly 0; the atomic ran on all 128 lanes because all 128 lanes entered the "guard", not because the guard failed to mask a single divergent lane.]**

2. **Triton 1-PE in-kernel sync/barrier hangs when GPU IPC is enabled.** Native SYCL/ISHMEM 1-PE sync/barrier passes with GPU IPC enabled, so the 1-PE hang points at the Triton-linked ISHMEM integration rather than a baseline ISHMEM/Level Zero/SYCL failure. **[Still true, and now explained: an unguarded in-kernel collective runs on all 128 lanes. The fix is to leader-guard it with a real per-lane id.]**

3. **2-PE in-kernel sync/barrier hangs on this machine even in native SYCL.** This machine exposes only one Level Zero XPU, so two MPI ranks share one physical visible device. A device-side cross-process barrier requires both ranks' kernels to make concurrent progress; on one visible XPU that is not a reliable assumption. **[Unchanged — separate one-XPU scheduling limitation.]**

A two-GPU node is needed for a fair 2-PE in-kernel barrier test, but the Triton 1-PE GPU-IPC hang should be fixed or understood first.

## Relevant Files

- Triton experiment: `python/triton_dist/test/intel/test_kernel_barrier_all.py`
- Triton ISHMEM bindings: `python/triton_dist/language/extra/xpu/libishmem_device.py`
- Python host ISHMEM/module init: `python/triton_dist/pyishmem.py`
- Triton module global init helper: `python/triton_dist/ishmem_module_init.cpp`
- ISHMEM barrier implementation: `/home/jovyan/ishmem/src/collectives/barrier.cpp`
- ISHMEM team sync implementation: `/home/jovyan/ishmem/src/collectives/sync_impl.h`

## Hardware/Runtime Context

Current visible accelerator inventory:

```text
$ ONEAPI_DEVICE_SELECTOR=level_zero:* sycl-ls --ignore-device-selectors
[level_zero:gpu][level_zero:0] Intel(R) Data Center GPU Max 1100
```

`clinfo -l` also reports one Intel GPU device. `xpu-smi` was not available in the tested shell.

The Intel test launcher currently maps MPI ranks to the Python test process and the test itself calls:

```python
torch.xpu.set_device(0)
```

So with `--nproc_per_node 2`, both MPI ranks submit kernels to the same visible XPU.

## Terminology

### GPU IPC

GPU IPC means GPU inter-process communication. In ISHMEM, when multiple PEs are local to the same node and GPU IPC is enabled, device code can directly address peer GPU symmetric memory instead of routing every operation through a host/runtime proxy.

### `psync` Path

The `psync` path is ISHMEM's intra-node device-side team synchronization path. Each team has symmetric scratch words, roughly:

```cpp
long psync[2];
int psync_idx;
```

For device `ishmemi_team_sync`, the fast intra-node path does roughly:

```text
pick current psync slot
advance psync_idx for the next sync
for each PE in the team:
    atomic_add(that PE's psync slot, 1)
while atomic_load(my psync slot) != team_size:
    spin
atomic_store(my psync slot, 0)
```

For a one-PE team, this should effectively increment the local `psync` to `1`, observe that `team_size == 1`, reset it, and return.

### Proxy Fallback

When the fast GPU-IPC/intra-node path is not used, device code writes a request into an ISHMEM ring buffer and waits for the host-side ISHMEM proxy/runtime to complete it. In `sync_impl.h`, this is the `sync_team_fallback()` path:

```cpp
ishmemi_proxy_blocking_request(req);
```

Setting `ISHMEM_ENABLE_GPU_IPC=0` forces this proxy/runtime path instead of the intra-node `psync` path.

## Triton Experiment Cases

The opt-in test file registers these cases:

| Case | Kernel behavior |
| --- | --- |
| `no_barrier` | Remote store through `dl.remote_ptr()`, no in-kernel barrier |
| `barrier_only` | `progress=1`, `dl.barrier_all()`, `progress=2` |
| `sync_all_only` | `progress=1`, `dl.sync_all()`, `progress=2` |
| `sync_all_leader_only` | Triton-side `if xpu.tid(0) == 0: dl.sync_all()` with work-group barriers around the call |
| `check` | Remote store, `dl.barrier_all()`, then read local value |

The `sync_all_only` case is useful because `dl.sync_all()` maps directly to `ishmem_sync_all()`, which calls `ishmemi_team_sync(ISHMEM_TEAM_WORLD)`. That isolates the team-sync/`psync` behavior from the extra barrier request in `ishmem_barrier_all()`.

## Observed Results

### Triton Results

| Probe | Result | Interpretation |
| --- | --- | --- |
| 2 PEs, `no_barrier` | Passed | Triton launch, `my_pe`, `n_pes`, `remote_ptr`, symmetric memory, and module init are broadly working. |
| 2 PEs, `barrier_only` | Timed out | `dl.barrier_all()` alone is enough to hang. |
| 2 PEs, `sync_all_only` | Timed out | The lower-level `ishmemi_team_sync` path also hangs. |
| 1 PE, `barrier_only`, GPU IPC enabled | Timed out | Not only a two-rank scheduling issue. |
| 1 PE, `sync_all_only`, GPU IPC enabled | Timed out | The Triton-linked `ishmemi_team_sync`/`psync` path is suspect. |
| 1 PE, `sync_all_leader_only`, GPU IPC enabled | Timed out | Triton-side lane guarding is not sufficient. |
| 1 PE, `barrier_only`, `ISHMEM_ENABLE_GPU_IPC=0` | Passed | The proxy/runtime path can complete for one PE from Triton. |
| 1 PE, `sync_all_only`, `ISHMEM_ENABLE_GPU_IPC=0` | Passed | Confirms the one-PE failure is specific to the GPU-IPC/`psync` branch. |

### Triton-Specific Atomic Masking Probe

The decisive temporary probes were:

```text
/tmp/triton_ishmem_debug_as1.cpp
/tmp/triton_ishmem_debug_as1_probe.py
/tmp/triton_external_atomic_as1.cpp
/tmp/triton_external_atomic_probe.py
/tmp/triton_ishmem_sync_wrapper.cpp
/tmp/triton_sync_wrapper_probe.py
```

A permanent repo-local reproducer for the non-ISHMEM external atomic masking issue now lives at:

```text
reproducers/intel_external_atomic_masking/
```

Run it with `--case all --expect bug` to confirm the current behavior, or `--case triton_guard --expect fixed` after a backend fix.

The cleaned ISHMEM psync one-shot probe performs the same psync setup as `ishmemi_team_sync`, but stops after one atomic add/load/reset attempt instead of spinning. With one PE and GPU IPC enabled, it printed:

```text
psync.index = 0
psync.next_index = 1
psync.my_psync_ptr = <same as remote_psync_ptr>
psync.remote_psync_ptr = <same as my_psync_ptr>
psync.team_size = 1
psync.start = 0
psync.last_pe = 0
psync.stride = 1
psync.local_index = 1
psync.delta = 0
psync.value_before = 0
psync.atomic_fetch_add_previous = 63
psync.value_after_add = 128
psync.value_after_optional_reset = 128
kernel.progress_after = 2
```

This proves the psync address and team metadata are sane enough to access, and the no-spin one-shot returns. The bad behavior is the count: a single guarded Triton call causes the linked SYCL atomic to contribute 128 increments. `team_size` is `1`, so the real `ishmemi_team_sync` loop waits for `psync == 1` and never exits once the value becomes `128`.

A minimal non-ISHMEM reproducer confirmed this is not ISHMEM-specific. The Triton kernel contained:

```python
if xpu.tid(0) == 0:
    external_atomic_inc(out)
```

The generated LLIR really does branch on `llvm.genx.local.id.x() == 0`, but the linked SYCL function body:

```llvm
define internal spir_func void @external_atomic_inc_as1(ptr addrspace(1) %0) {
  %2 = call i64 @_Z18__spirv_AtomicIAddPU3AS1liil(..., i64 1)
  ...
}
```

still produced:

```text
[128, 127, 128, 0, 1, 2, 0, 0]
```

So a Triton-side guard around an external SYCL atomic call does not restrict the atomic to one lane. Plain external stores behind the same guard worked, which is why simpler metadata reads and `dl.my_pe()` can look healthy.

Moving the local-id guard inside the linked SYCL bitcode changes the result. A temporary external function that does its own `get_local_id(0)` check before the atomic produced:

```text
[1, 0, 1, 0, 1, 2, 0, 0]
```

A temporary `ishmem_sync_all` wrapper with the same internal local-id guard also passed for one PE with GPU IPC enabled once the wrapper was forced to link against Triton's `ishmem_sync_all` alias:

```text
[0, 1, 2, -1, -1, -1, -1, -1]
```

Representative Triton commands:

```bash
# 2-PE control: passed
timeout 45 conda run --no-capture-output -n dist-triton-build \
  bash scripts/launch_intel.sh --nproc_per_node 2 \
  python/triton_dist/test/intel/test_kernel_barrier_all.py --case no_barrier

# 1-PE Triton sync with GPU IPC enabled: timed out
timeout 30 conda run --no-capture-output -n dist-triton-build \
  /opt/intel/oneapi/mpi/2021.17/bin/mpirun -np 1 \
  -genv I_MPI_FABRICS=shm \
  -genv ISHMEM_RUNTIME=MPI \
  -genv ISHMEM_ENABLE_GPU_IPC=1 \
  -genv LD_LIBRARY_PATH=/home/jovyan/ishmem/install/lib:/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH \
  -genv FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.17/libfabric/lib/prov \
  -genv TRITON_BACKENDS_IN_TREE=1 \
  -genv ISHMEM_HOME=/home/jovyan/ishmem/install \
  python python/triton_dist/test/intel/test_kernel_barrier_all.py --case sync_all_only

# 1-PE Triton sync with GPU IPC disabled: passed
timeout 20 conda run --no-capture-output -n dist-triton-build \
  /opt/intel/oneapi/mpi/2021.17/bin/mpirun -np 1 \
  -genv I_MPI_FABRICS=shm \
  -genv ISHMEM_RUNTIME=MPI \
  -genv ISHMEM_ENABLE_GPU_IPC=0 \
  -genv LD_LIBRARY_PATH=/home/jovyan/ishmem/install/lib:/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH \
  -genv FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.17/libfabric/lib/prov \
  -genv TRITON_BACKENDS_IN_TREE=1 \
  -genv ISHMEM_HOME=/home/jovyan/ishmem/install \
  python python/triton_dist/test/intel/test_kernel_barrier_all.py --case sync_all_only
```

### Native SYCL/ISHMEM Control

A temporary native probe was built at:

```text
/tmp/ishmem_native_probe.cpp
/tmp/ishmem_native_probe_cmake/build/ishmem_native_probe
```

The probe launches a SYCL `single_task` that does:

```cpp
*progress = 1;
ishmem_sync_all();      // or ishmem_barrier_all()
*progress = 2;
```

Build method:

```text
find_package(ISHMEM REQUIRED PATHS /home/jovyan/ishmem/install/lib/cmake/ishmem)
find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(ishmem_native_probe PRIVATE ISHMEM::ISHMEM MPI::MPI_CXX)
```

Native results:

| Probe | Result | Interpretation |
| --- | --- | --- |
| 1 PE, native `ishmem_sync_all()`, GPU IPC enabled | Passed | Native one-PE `psync` path works. |
| 1 PE, native `ishmem_barrier_all()`, GPU IPC enabled | Passed | Native one-PE barrier path works. |
| 2 PEs, native `ishmem_sync_all()`, GPU IPC enabled, one visible XPU | Timed out | Cross-process in-kernel sync still problematic on one XPU. |
| 2 PEs, native `ishmem_barrier_all()`, GPU IPC enabled, one visible XPU | Timed out | Same concurrent-progress/scheduling concern. |

Representative native commands:

```bash
# Native 1-PE sync: passed
timeout 45 /opt/intel/oneapi/mpi/2021.17/bin/mpirun -np 1 \
  -genv I_MPI_FABRICS=shm \
  -genv ISHMEM_RUNTIME=MPI \
  -genv ISHMEM_ENABLE_GPU_IPC=1 \
  -genv LD_LIBRARY_PATH=/home/jovyan/ishmem/install/lib:/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH \
  -genv FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.17/libfabric/lib/prov \
  /home/jovyan/ishmem/install/bin/ishmrun --disable-bind \
  /tmp/ishmem_native_probe_cmake/build/ishmem_native_probe sync

# Native 1-PE barrier: passed
timeout 45 /opt/intel/oneapi/mpi/2021.17/bin/mpirun -np 1 \
  -genv I_MPI_FABRICS=shm \
  -genv ISHMEM_RUNTIME=MPI \
  -genv ISHMEM_ENABLE_GPU_IPC=1 \
  -genv LD_LIBRARY_PATH=/home/jovyan/ishmem/install/lib:/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH \
  -genv FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.17/libfabric/lib/prov \
  /home/jovyan/ishmem/install/bin/ishmrun --disable-bind \
  /tmp/ishmem_native_probe_cmake/build/ishmem_native_probe barrier

# Native 2-PE sync on one visible XPU: timed out
timeout 60 /opt/intel/oneapi/mpi/2021.17/bin/mpirun -np 2 \
  -genv I_MPI_FABRICS=shm \
  -genv ISHMEM_RUNTIME=MPI \
  -genv ISHMEM_ENABLE_GPU_IPC=1 \
  -genv LD_LIBRARY_PATH=/home/jovyan/ishmem/install/lib:/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH \
  -genv FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.17/libfabric/lib/prov \
  /home/jovyan/ishmem/install/bin/ishmrun --disable-bind \
  /tmp/ishmem_native_probe_cmake/build/ishmem_native_probe sync
```

## Interpretation

### What Is Not Broken Globally

Native SYCL/ISHMEM can execute a one-PE device `ishmem_sync_all()` and `ishmem_barrier_all()` with GPU IPC enabled. Therefore, the underlying one-PE ISHMEM/Level Zero/SYCL `psync` path is not globally broken.

### What Looks Triton-Specific

Triton hangs on the same one-PE GPU-IPC sync path where native SYCL passes. The latest evidence narrows this to mixed Triton/SYCL codegen around divergent calls into linked SYCL bitcode:

- `global_info` is initialized well enough for `dl.my_pe()` and `dl.n_pes()`.
- Triton-visible team metadata is correct for one PE: `team.size == 1`, `team.only_intra == 1`, `local_pes[0] == 1`, `ipc_buffer_delta[1] == 0`, and `psync` starts at `0`.
- A no-spin psync atomic add/load returns, so the psync address is accessible.
- The atomic contribution is wrong when the call is guarded from Triton: linked SYCL atomics execute across the 128-lane work-group/SIMD execution context instead of once.

That explains why `dl.my_pe()` and metadata reads can work while `dl.sync_all()` hangs. The basic module/global-info path is fine; the `psync` loop's exact-count protocol is broken by over-incrementing.

## Backend Fork Investigation Plan

The likely root fix belongs in the forked Triton Intel backend rather than in ISHMEM. The working backend repo for this environment is:

```text
/home/jovyan/dist-triton/intel-xpu-backend-for-triton
```

Current branch observed during this investigation:

```text
triton-distributed-intel-xpu
```

The permanent reproducer in this repo should be the starting point:

```text
reproducers/intel_external_atomic_masking/
```

### 1. Reproduce Against the Backend Fork

Run the reproducer from this distributed repo while using the editable Triton/backend install built from the fork:

```bash
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu

timeout 60 conda run --no-capture-output -n dist-triton-build \
  /opt/intel/oneapi/mpi/2021.17/bin/mpirun -np 1 \
  -genv I_MPI_FABRICS=shm \
  -genv ISHMEM_RUNTIME=MPI \
  -genv ISHMEM_ENABLE_GPU_IPC=1 \
  -genv LD_LIBRARY_PATH=/home/jovyan/ishmem/install/lib:/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH \
  -genv FI_PROVIDER_PATH=/opt/intel/oneapi/mpi/2021.17/libfabric/lib/prov \
  -genv TRITON_BACKENDS_IN_TREE=1 \
  -genv ISHMEM_HOME=/home/jovyan/ishmem/install \
  python reproducers/intel_external_atomic_masking/repro_external_atomic_masking.py \
    --case all --expect bug
```

Current buggy signature:

```text
plain: [0, 0, 0, 1234, 1, 2, 0, 0]
triton_guard: [128, 127, 128, 0, 1, 2, 0, 0]
internal_guard: [1, 0, 1, 0, 1, 2, 0, 0]
```

After a backend fix, this should pass:

```bash
python reproducers/intel_external_atomic_masking/repro_external_atomic_masking.py \
  --case triton_guard --expect fixed
```

### 2. Capture and Compare Compiler Artifacts

Capture TTIR, TTGIR, LLIR, and SPIR-V artifacts for at least these reproducer cases:

- `plain`: external function call behind a Triton guard, but only a normal store inside the external function.
- `triton_guard`: external atomic call behind a Triton guard; currently increments 128 times.
- `internal_guard`: the local-id guard is inside the SYCL bitcode; currently increments once.

The first comparison target is LLIR. In the buggy `triton_guard` case, the LLIR already shows a branch around the external call:

```llvm
%tid_is_zero = icmp eq i32 %tid, 0
br i1 %tid_is_zero, label %call_external, label %done
```

but the inlined/linked external function body contains an unpredicated SPIR-V atomic:

```llvm
call i64 @_Z18__spirv_AtomicIAddPU3AS1liil(..., i64 1)
```

That is the point where active-lane information appears to stop applying to side effects inside linked external bitcode.

### 3. Inspect Candidate Backend Areas

Start with these confirmed files/areas in the fork:

```text
third_party/intel/lib/TritonIntelGPUToLLVM/ControlFlowOpToLLVM.cpp
third_party/intel/lib/TritonIntelGPUToLLVM/SPMDOpToLLVM.cpp
third_party/intel/lib/TritonIntelGPUToLLVM/LoadStoreOpToLLVM.cpp
third_party/intel/lib/TritonIntelGPUToLLVM/SPIRVTargetInfo.cpp
third_party/intel/lib/TritonIntelGPUToLLVM/TritonGPUToLLVM.cpp
third_party/intel/lib/Target/SPIRV/SPIRVTranslation.cpp
third_party/intel/lib/Target/LLVMIR/LLVMPasses.h
third_party/intel/lib/Target/LLVMIR/LLVMIRFreezeMaskedDivRem.cpp
```

Questions to answer in those areas:

- How does the Intel backend represent Triton's active-lane predicate for divergent control flow?
- Why do Triton-native stores lower to predicated operations while side effects inside linked external functions do not?
- Are `extern_call` operations treated as ordinary scalar function calls even when they occur in divergent control flow?
- Does the external bitcode linking/inlining step erase or bypass any predicate/mask metadata that the Intel lowering expects?
- Does SPIR-V translation have a way to preserve an active-lane mask around function calls or atomics, or must this be handled before SPIR-V translation?

### 4. Try the Smallest Backend Fix First

The most direct fix would be to preserve the Triton active-lane predicate across side-effecting external calls. Possible implementation shapes, from most desirable to most conservative:

1. Predicate or mask the linked external function's side-effecting operations when the call occurs under divergent Triton control flow.
2. Lower divergent `extern_call` operations through a backend helper that executes the linked call only for active lanes.
3. Reject or diagnose side-effecting `extern_call` inside divergent control flow until a correct masked lowering exists.

A backend fix should be validated by the reproducer, not by the ISHMEM barrier test first. The minimal acceptance criterion is:

```text
triton_guard: [1, 0, 1, 0, 1, 2, 0, 0]
```

### 5. Add an Upstreamable Regression Test

Once the backend fix is understood, port the reproducer into the backend fork's test structure as a non-ISHMEM Intel backend/codegen test. The test should assert:

- A plain external store behind a Triton guard still runs correctly.
- A guarded external atomic increments once, not by the work-group size.
- An internally guarded external atomic remains a passing control.

This framing is upstreamable because it is a generic Intel Triton backend issue: divergent Triton control flow around linked external device bitcode must preserve side-effect masking. It does not require ISHMEM to demonstrate the bug.

### What Looks Like a One-XPU Scheduling Limitation

Native SYCL also hangs for 2 PEs on the current one-visible-XPU node. That supports the concurrent-progress concern: two separate MPI processes launch kernels that spin waiting for each other, but the device/runtime is not guaranteed to keep both cross-process kernels resident and forward-progressing on one XPU.

This is separate from the Triton 1-PE issue.

## Recommended Next Steps

> Items 1–3 below predate the [Resolution](#resolution-xputid-was-not-per-lane)
> and are kept for context. Item 1's premise is now false: a leader guard with the
> fixed `xpu.tid` does **not** over-contribute to `psync`.

1. **~~Do not rely on Triton-side guards for linked SYCL atomic collectives.~~ [OBSOLETE]** This was only true while `xpu.tid(0)` returned a uniform `0`. With the fix, `if xpu.tid(0) == 0: dl.sync_all()` correctly runs on one lane and passes.

2. **Prototype permanent SYCL-compiled wrapper functions for collectives.** The passing workaround moved the local-id guard into SYCL bitcode and called that wrapper unconditionally from Triton. A real version should live in the repo rather than `/tmp`, avoid the forced-link hack, and cover `sync_all`/`barrier_all` semantics deliberately.

3. **Fix or document external bitcode call masking in the compiler/backend.** The minimal external atomic reproducer is small enough to use as a backend regression test: a guarded external atomic should increment once, not 128 times.

4. **Decide whether to keep the `global_info` 8-byte module-init patch.** The LLIR reports `"sycl-device-global-size"="8"`, and rank queries still pass with the 8-byte write, but this patch did not fix sync by itself.

5. **Use a two-GPU node only after the 1-PE Triton case is understood.** Two GPUs are needed for a fair multi-PE in-kernel barrier test, but they will not explain why Triton 1-PE `sync_all` hangs while native 1-PE `sync_all` passes.

---

## Resolution: `xpu.tid` Was Not Per-Lane

This section supersedes original finding #1. It was produced by continuing the
investigation from inside the backend fork
(`/home/jovyan/dist-triton/intel-xpu-backend-for-triton`), but the actual fix
landed in the distributed repo's XPU language extension because that is where
the `xpu.tid` binding lives.

### The Decisive Experiment

The earlier conclusion assumed `xpu.tid(0)` was per-lane and that a guard around
the call simply failed to mask the linked atomic. To test that assumption, the
proven external-atomic helper was guarded with **different** constants and ranges
(work-group size is 128, so `tid` should range over `0..127`):

```text
guard tid==0    : atomic_count = 128   (markers set)
guard tid==5    : atomic_count = 0     (nothing ran at all)
guard tid==200  : atomic_count = 0
range 0<=tid<1  : atomic_count = 128
range 0<=tid<128: atomic_count = 128
range 3<=tid<7  : atomic_count = 0
range 64<=tid<128: atomic_count = 0
```

If `xpu.tid(0)` were genuinely per-lane, `tid==5` would activate lane 5 and the
unmasked external atomic would fire **once**. It fired **zero** times. Every
guard that includes `0` fires on all 128 lanes; every guard that excludes `0`
fires on none. The only consistent explanation is that **`xpu.tid(0)` evaluates
to a uniform `0` on every work-item**, so `if tid == 0` is uniformly true and
`if tid == k (k!=0)` is uniformly false.

### Why

`xpu.tid(0)` lowered to the LLVM intrinsic `llvm.genx.local.id.x`
(`triton_dist/language/extra/xpu/language_extra.py`). The Intel SPIR-V backend
only whitelists `llvm.genx.GenISA.*` as allowed unknown intrinsics
(`third_party/intel/lib/Target/SPIRV/SPIRVTranslation.cpp`,
`setSPIRVAllowUnknownIntrinsics({"llvm.genx.GenISA."})`), and nothing in the
backend defines or maps `llvm.genx.local.id.*`, `llvm.genx.local.size.*`, or
`llvm.genx.sub.group.*`. In the generated SPIR-V the symbol survives as a plain
`LinkageAttributes ... Import` external function call (not a `LocalInvocationId`
BuiltIn), and the runtime resolves it to a uniform `0`.

The LLIR comparison that originally looked like "masking lost across the call"
was actually showing two **different** thread-id sources:

- the `scf.if` branch was built from `llvm.genx.local.id.x()` (the uniform-0
  intrinsic), and
- the Triton stores were lowered to `llvm.genx.GenISA.PredicatedStore` whose
  predicate the Intel store-lowering re-derives independently from the **real**
  per-lane OpenCL builtin `_Z12get_local_idj` (`getThreadId` →
  `gpu::ThreadIdOp`).

So the stores were correct (their own real per-lane predicate), the branch was
uniformly-true, and the unpredicated external atomic ran on all 128 lanes. That
is the entire "128×" effect.

NVIDIA and AMD do not have this problem because their `tid` bindings use real,
backend-recognized intrinsics: `llvm.nvvm.read.ptx.sreg.tid.*` and
`llvm.amdgcn.workitem.id.*`.

### The Fix

In `python/triton_dist/language/extra/xpu/language_extra.py`, the work-item
queries now use the real SPIR-V/OpenCL builtins (resolved per-lane by IGC)
instead of the unrecognized `llvm.genx.*` names:

| Helper | Old (uniform 0) | New (per-lane builtin) |
| --- | --- | --- |
| `tid(axis)` | `llvm.genx.local.id.{x,y,z}` | `_Z12get_local_idj` (`get_local_id`) |
| `ntid(axis)` | `llvm.genx.local.size.{x,y,z}` | `_Z14get_local_sizej` (`get_local_size`) |
| `sub_group_id` | `llvm.genx.sub.group.id` | `_Z16get_sub_group_idv` |
| `sub_group_local_id` | `llvm.genx.sub.group.local.id` | `_Z22get_sub_group_local_idv` |
| `sub_group_size` | `llvm.genx.sub.group.size` | `_Z18get_sub_group_sizev` |

`get_local_id`/`get_local_size` return `size_t` (i64) and take a single
`unsigned int` axis argument; results are truncated back to `int32` to preserve
the existing CUDA-style API. All five mangled names were confirmed to compile
against the SYCL device builtins with `icpx -fsycl -fsycl-device-only`.

### Verification (all 1-PE, GPU IPC enabled, fresh Triton cache)

| Check | Before fix | After fix |
| --- | --- | --- |
| Reproducer `triton_guard` (`--expect fixed`) | `[128, 127, 128, ...]` | `[1, 0, 1, 0, 1, 2, 0, 0]` ✅ |
| Reproducer `plain` / `internal_guard` | passed | still pass |
| Guard discriminator `tid==5` | 0 (uniform) | atomic fires exactly once on lane 5 |
| `test_kernel_barrier_all.py --case sync_all_leader_only` | flaky hang | **PASSED 4/4** (and the standalone `fixed_leader` probe 5/5) |
| Leader-guarded `dl.sync_all()` probe | hang | **PASSED** (progress=2) |
| Unguarded `dl.sync_all()` / `barrier_only` / `sync_all_only` | hang | **still hang** (expected — see below) |

### Important Caveat: Stale Triton Cache

The fix lives in a Python language binding, so the compiled-kernel cache must be
invalidated for it to take effect. The shared default cache
(`~/.triton/cache`, ~11 GB here) held pre-fix kernels and made
`sync_all_leader_only` hang even after the edit. Running with a fresh
`TRITON_CACHE_DIR` and `TRITON_ALWAYS_COMPILE=1` made it pass 4/4 deterministically.
When validating language-binding changes, always force a clean recompile.

### Remaining: Unguarded Collectives Still Hang (By Design)

`barrier_only` and `sync_all_only` call `dl.barrier_all()` / `dl.sync_all()`
**unconditionally**, i.e. on all 128 lanes. At 1 PE this still overshoots the
psync word and hangs — this is a *usage* problem, not a compiler bug, and the
`xpu.tid` fix does not change it. A leader-guarded call
(`if xpu.tid(0) == 0: dl.sync_all()`) now passes. Recommended follow-up: make
the `dl.barrier_all` / `dl.sync_all` bindings restrict themselves to a single
leader lane internally (as the `internal_guard` control does) so they are safe
to call unconditionally from Triton, matching the CUDA/ROCm `libshmem_device`
expectations.

### Upstreamable Regression Test

The existing reproducer (`reproducers/intel_external_atomic_masking/`, in the
distributed repo) already encodes the acceptance criterion: with the fix,
`triton_guard` must be `[1, 0, 1, 0, 1, 2, 0, 0]`. A more direct backend-level
regression test is "a `tid(0)`-guarded side effect runs on exactly one lane",
which does not require ISHMEM at all and could live in the Intel backend test
suite.

---

## Resolution: Signal / Notify / Wait Path (2026-06-23)

A follow-up pass audited the device-side signalling intrinsics
(`signal_set`, `signal_add`, `signal_fetch`, `signal_wait_until`, the `notify`
op, and the `common_ops.py` kernels that use them). All findings below are from
the distributed repo's `python/triton_dist/language/extra/xpu/libishmem_device.py`
and `python/triton_dist/kernels/intel/common_ops.py`, verified empirically at
1 PE with GPU IPC enabled and a fresh Triton cache.

### Bugs Found

1. **Every `tl.cast` in the Intel ISHMEM bindings used a stale kwarg.** All 27
   cast sites in `libishmem_device.py` called
   `tl.cast(x, ty, _builder=_semantic.builder)`. In this Triton version `cast`
   (and all `@builtin`s) take **`_semantic=`**; any other keyword makes the
   `@builtin` wrapper raise *"`_semantic` argument must be provided…"*. So
   `signal_set`, `signal_add`, `signal_wait_until`, `pe_accessible`, every
   `putmem`/`put_nbi`, and the team ops would all fail at compile time the moment
   they were actually called. NVSHMEM's bindings use `_semantic=` in all 127 of
   their cast sites; the Intel file was simply never updated after the
   `_builder`→`_semantic` rename. **Fix:** `sed 's/_builder=_semantic.builder/_semantic=_semantic/g'`.

2. **`signal_set` / `signal_add` referenced device symbols that do not exist.**
   The bindings asked for `ishmem_signal_set` / `ishmem_signal_add`, but
   `libishmem_device.bc` defines neither (in any mangling). The real device
   symbols are **`ishmemx_signal_set`** / **`ishmemx_signal_add`** (note the
   `ishmemx_` extension prefix; mangled `_Z18ishmemx_signal_setPmmi` /
   `_Z18ishmemx_signal_addPmmi`). With the wrong name the kernel compiles but
   fails at load with `ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED`.
   (`ishmem_signal_fetch` and `ishmem_signal_wait_until` *do* exist unmangled, so
   those bindings had only bug #1.) **Fix:** rename to the `ishmemx_` symbols.

3. **`common_ops.py` called functions that don't exist on `dl`.**
   `signal_wait_kernel` called `dl.notify_wait_until(...)` and
   `broadcast_signal_kernel` / `signal_reduce_to_root_kernel` called
   `dl.notify(...)`. `dl` is `libishmem_device`, which has neither — `notify` is
   a `distributed`-dialect builtin in `triton_dist.language`, and there is no
   `notify_wait_until` anywhere. All three kernels raised `AttributeError` on
   first compile; nothing imported or tested them, so it went unnoticed.
   **Fix:** rewrote them to use `dl.signal_wait_until` (with an
   `ISHMEM_CMP_*` constant) and `dl.signal_set` / `dl.signal_add` (with the
   target `pe`), each guarded to a single leader work-item.

### `signal_wait_until` itself is fine — the producer side was the problem

`signal_wait_until` is a pure poll (`while (compare(*sig, cmp, val)) spin`). It
reads only, so unlike the counting collectives it does not overshoot when run on
many lanes, and once the cast bug is fixed it works. The hazardous side is the
**producer**: `signal_add` is a counting atomic, so the same all-lane
multiplicity bug as `sync_all` applies — see below.

### Lane-multiplicity interaction (depends on the `xpu.tid` fix)

Measured at 1 PE with the corrected symbols:

| Kernel | Result | Meaning |
| --- | --- | --- |
| `signal_add(+1)` ×3, guarded `if xpu.tid(0)==0` | signal = **3** | leader guard works (needs the per-lane `tid` fix) |
| `signal_add(+1)` **unguarded** (all lanes) | signal = **128** | counting signal op multiplied by work-group size |
| `signal_set(42)` → `signal_wait_until(EQ,42)` → `signal_fetch` | progress=2, value=42 | full producer→consumer round trip works |

So `signal_add` has exactly the same "guard it or it counts 128×" property as
`sync_all`/`barrier_all`, and the leader guard only became trustworthy after the
`xpu.tid` per-lane fix. `signal_set` is idempotent across lanes (writes the same
value), so unguarded `signal_set` is merely wasteful, not incorrect.

### `distributed.wait` lowering gaps (not yet fixed)

The Intel `WaitOpConversion` (`lib/Conversion/TritonDistributedToLLVM/INTEL/
DistributedOpToLLVM.cpp`) lowers `distributed.wait` to a single
`ishmem_signal_wait_until` and **drops most of the op's semantics**:

- **`numBarriers` is ignored.** The op is documented as waiting on a *list* of
  `num_barriers` signals; NVIDIA loops over them (one per lane up to warp size).
  Intel waits on only the first address.
- **`scope` and `semantic` attributes are ignored.** NVIDIA emits
  `ld.global.<sem>.<scope>` and a closing `bar.warp.sync`; Intel emits neither.
- **The comparison is hardcoded to `ISHMEM_CMP_GE` (4).** The op carries no cmp
  today, but a multi-value or exact-match wait cannot be expressed.

No Intel kernel currently uses `distributed.wait` (they use `dl.signal_wait_until`
directly), so this is latent — but any ported kernel relying on multi-signal
`wait` semantics would silently wait on one signal.

### Missing fused `putmem_signal` on Intel

NVSHMEM exposes `putmem_signal` / `putmem_signal_nbi` (write data **and** set the
flag in one ordered op). The Intel `libishmem_device` has no such binding, even
though the device library defines the symbols (`ishmem_putmem_signal`,
`ishmem_put<N>_signal`, typed `ishmem_<T>_put_signal`, and `_nbi` variants). A
ported producer/consumer kernel that relies on the fused op must instead do a
separate `putmem` + `quiet`/fence + `signal_set` on Intel. Worth adding for
parity.

### Verification commands

All four signal checks above were run via
`mpirun -np 1 … python <probe>` with `ISHMEM_ENABLE_GPU_IPC=1`, a fresh
`TRITON_CACHE_DIR`, and `TRITON_ALWAYS_COMPILE=1` (the same harness as the
`xpu.tid` work — see the stale-cache caveat). After the binding fixes, the real
`dl.signal_set` / `dl.signal_add` / `dl.signal_wait_until` / `dl.signal_fetch`
round trip and all three repaired `common_ops.py` kernels compile and run.

---

## Resolution: `NotifyOpConversion` Backend Fix + LLVM Cache Restore (2026-06-23)

### Backend bug: `distributed.notify` lowered to nonexistent symbols

The platform-independent `notify` op (`triton_dist.language.notify`, used by the
NVIDIA kernels and the intended cross-platform producer-side primitive) lowered,
on Intel, through `NotifyOpConversion` in
`lib/Conversion/TritonDistributedToLLVM/INTEL/DistributedOpToLLVM.cpp`. That
conversion hardcoded the device symbols **`ishmem_signal_set`** /
**`ishmem_signal_add`** — which, as the signal/notify audit found, **do not
exist** in `libishmem_device.bc` (only the `ishmemx_`-prefixed forms do). So any
kernel using `notify` would compile but fail to load with
`ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED`. This is the same root issue as the
Python-binding symbol bug, one layer down in the C++ lowering, and it is why the
`common_ops.py` kernels had been (incorrectly) reaching for other names.

**Fix:** point the lowering at the real extension symbols:

```cpp
funcName = (SET) ? "ishmemx_signal_set" : "ishmemx_signal_add";
```

`common_ops.py` was then rewritten to use the proper `notify` abstraction
(`dist.notify(ptr, peer, signal=v, sig_op="set"/"add", comm_scope="gpu")`,
leader-guarded), matching how the NVIDIA kernels are written, instead of calling
`dl.signal_*` directly.

### Verified (1 PE, GPU IPC, rebuilt `libtriton.so`)

| Check | Result |
| --- | --- |
| `notify(sig_op="set", 42)` self → read back | **42** ✅ |
| `notify(sig_op="add", +7)` ×2 self → read back | **14** ✅ (was a load-time unlink failure) |
| `common_ops.broadcast_signal_kernel` / `signal_reduce_to_root_kernel` | compile + lower ✅ |
| Regression: external-atomic reproducer `triton_guard` | `[1,0,1,0,1,2,0,0]` ✅ |
| Regression: leader-guarded `sync_all` (1 PE) | PASSED, progress=2 ✅ |

### Build environment note: the LLVM cache had to be restored first

Rebuilding `libtriton.so` (needed for any C++ backend change) was initially
blocked: this Triton pins LLVM **`62b7cf96`** (v23, in
`3rdparty/triton/cmake/llvm-hash.txt`), but the shared cache
`~/.triton/llvm/llvm-62b7cf96-ubuntu-x64` had been **emptied** (0 files) — only
an unrelated **`0729a74e`** (v22, pinned by a *different* checkout,
`~/triton-flex-compare/...`) remained complete. `~/.triton/llvm` is shared across
every Triton checkout on the host, keyed by hash; a neighbor project's setup plus
a cache eviction left this repo's pinned LLVM as an empty husk. Linking against
the v22 tree failed with `undefined symbol: …IntegerRangeAnalysis…` (a v23 MLIR
API absent in v22).

**Fix:** re-download the pinned tarball and extract into the cache:

```bash
curl -L -o /tmp/llvm.tar.gz \
  https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-62b7cf96-ubuntu-x64.tar.gz
rm -rf ~/.triton/llvm/llvm-62b7cf96-ubuntu-x64
tar xzf /tmp/llvm.tar.gz -C ~/.triton/llvm/
```

This restored a complete LLVM 23 (559 static libs, MLIR cmake + headers). The
CMake cache also had stale `0729a74e` paths (from debugging) that had to be
reset to `62b7cf96` before `cmake -G Ninja … && ninja libtriton.so` succeeded.
The blob store is online and serves the exact pinned hash, so this is the
canonical recovery whenever a rebuild reports `Unsupported LLVM version 22` or an
`IntegerRangeAnalysis` undefined symbol.