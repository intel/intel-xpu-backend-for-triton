# PVC Setup Guide — Getting `run_tests.sh` to Pass

This is a step-by-step guide for setting up `triton-distributed-intel-xpu` from a fresh
clone on an Intel **Data Center GPU Max (PVC)** machine, up to the point where the Intel
test runner `python/triton_dist/test/intel/run_tests.sh` works.

It consolidates `BUILD_SETUP.md`, `CLAUDE.md`, and the launch/test scripts into a single
linear checklist. If you hit an error, jump to [Troubleshooting](#troubleshooting) at the
bottom — most failures map to a single missed step here.

> **Conventions used below**
> - Repo root: `/home/jovyan/dist-triton/triton-distributed-intel-xpu`
> - ishmem install prefix: `/home/jovyan/ishmem/install`
> - Conda env: `dist-triton-build` (Python 3.10)
> - oneAPI: `/opt/intel/oneapi` (2025.3)
>
> Adjust paths if your machine differs, but keep them consistent across every step.

---

## 0. Prerequisites (verify before you start)

These must already exist on the PVC host. None are installed by this guide.

| Component | Version | How to check |
|-----------|---------|--------------|
| Intel Data Center GPU Max (PVC) | 1+ device | `xpu-smi discovery` |
| Intel oneAPI (DPC++/icpx, MPI, Level Zero) | 2025.3 | `ls /opt/intel/oneapi/compiler/2025.3/bin/icpx` |
| Intel MPI | 2021.17 | `ls /opt/intel/oneapi/mpi/latest/bin/mpirun` |
| Level Zero headers | 1.26.2 | `ls /usr/include/level_zero/ze_api.h` |
| CMake | ≥ 3.28 | `cmake --version` |
| Ninja | system | `ninja --version` |
| Conda / Miniconda | any | `conda --version` |

> ⚠️ **Do NOT `source /opt/intel/oneapi/setvars.sh` (or `oneapi-vars.sh`).** It injects a
> `libur_loader.so` that conflicts with PyTorch's bundled SYCL runtime and breaks XPU
> detection. Throughout this guide we set the few oneAPI paths we need **manually**.

> 🛠️ **Prefer to run it?** [scripts/setup_pvc.sh](scripts/setup_pvc.sh) automates every
> step below. It is idempotent (skips finished work) and stops with clear instructions if a
> manual prerequisite is missing.
>
> ```bash
> bash scripts/setup_pvc.sh --scan         # inventory: required components vs. what you have
> bash scripts/setup_pvc.sh --check-only   # scan + verify prerequisites, no build
> bash scripts/setup_pvc.sh                # full setup (Phases A, B, C)
> bash scripts/setup_pvc.sh --run-tests    # full setup, then run run_tests.sh
> ```
>
> The rest of this document explains what the script does and why, step by step.

The setup has three phases:

1. **Phase A** — Build triton-distributed (the compiler + distributed passes).
2. **Phase B** — Build Intel SHMEM (ishmem) and its three artifacts.
3. **Phase C** — Rebuild the Intel distributed pass against ishmem, wire up the runtime
   environment, and run `run_tests.sh`.

---

## Phase A — Build triton-distributed

### A.1 Create the conda environment

```bash
conda create -n dist-triton-build python=3.10 -y
conda activate dist-triton-build

# PyTorch with native XPU support (no IPEX needed) — from the cached wheel
pip install /home/jovyan/triton/intel-xpu-backend-for-triton/.scripts_cache/pytorch/dist/torch-2.13.0a0+git393c805-cp310-cp310-linux_x86_64.whl

# Build dependencies
pip install cmake ninja pybind11 wheel numpy pyelftools packaging pytest
```

> **Use `dist-triton-build`, never `python-3.10`.** The `python-3.10` env is reserved for
> other work. IPEX (Intel Extension for PyTorch) is **not** required — PyTorch 2.13.0a0 has
> native `torch.xpu`. Ignore any "IPEX not available" warnings.

Sanity check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

### A.2 Initialize the Triton submodule

```bash
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu
git submodule update --init 3rdparty/triton
```

This pulls commit `3ab7ba7e6` from the fork named in `.gitmodules`
(`sarah12121212/intel-xpu-backend-for-triton`, branch `triton-distributed-intel-xpu`).

### A.3 Materialize the backend symlinks ⭐ (easy to miss)

The submodule stores backend files (`third_party/{intel,nvidia,amd}/backend/...`) as git
**symlinks**. After checkout these become regular files containing a *path string* instead
of real source — so the backend won't import. Replace them with the actual source from the
parent commit `9dc5ce0c7`:

```bash
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu/3rdparty/triton

for tp in intel nvidia amd; do
    for f in $(git ls-tree -r HEAD third_party/$tp/ | grep "^120000" | awk '{print $4}'); do
        git show 9dc5ce0c7:"$f" > "$f" 2>/dev/null
    done
done
```

`120000` is the git mode for a symlink. We use parent commit `9dc5ce0c7` deliberately —
**not** the original Intel fork's `main`, whose newer backend files import symbols (e.g.
`decompose_descriptor`) that don't exist in this pinned Triton base.

**Special case** — one distributed-only file isn't in the parent commit. Copy it from this
repo's own source:

```bash
cp /home/jovyan/dist-triton/triton-distributed-intel-xpu/python/triton_dist/language/extra/xpu/libishmem_device.py \
   third_party/intel/language/intel/libishmem_device.py
```

> The repo's `scripts/materialize_triton_backend_symlinks.py` does **not** work here — it
> checks `os.path.isdir(".git")`, which is False inside a submodule (`.git` is a file). Use
> the loop above.

### A.4 Build (first pass — ishmem not yet available)

```bash
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu
conda activate dist-triton-build

export TRITON_BUILD_DISTRIBUTED=ON
export TRITON_BUILD_INTEL_DISTRIBUTED=OFF   # turned ON later, once ISHMEM_HOME exists
export TRITON_BUILD_PROTON=OFF
export TRITON_ENABLE_PROTON=OFF
export TRITON_BUILD_TOOLS=OFF
export USE_TRITON_DISTRIBUTED_AOT=0
export MAX_JOBS=8

pip install -e python --no-build-isolation --use-pep517
```

The first build downloads a prebuilt LLVM (hash `62b7cf96`, ~1.5 GB) to `~/.triton/llvm/`.
If the network is unavailable, set `LLVM_SYSPATH` to an existing matching LLVM.

If `libtriton.so` doesn't appear after the pip install (pip sometimes finishes the Python
packaging without finishing the C++ link), build explicitly:

```bash
cd python/build/cmake.linux-x86_64-cpython-3.10
ninja -j8
```

Expected artifacts:
- `python/triton/_C/libtriton.so` (~921 MB) — the Triton compiler with the Intel backend
- `python/triton/_C/libtriton_distributed.so` (~5 MB) — distributed MLIR passes + bindings

### A.5 Verify the core build

```bash
export TRITON_BACKENDS_IN_TREE=1
python -c "
import triton
from triton.backends import backends
print(f'Triton {triton.__version__}, backends: {list(backends.keys())}')
import triton_dist
from triton._C.libtriton.distributed import passes
print('Distributed passes: OK')
"
```

Expected:
```
Triton 3.7.0, backends: ['intel', 'nvidia']
Distributed passes: OK
```

---

## Phase B — Build Intel SHMEM (ishmem) + artifacts

ishmem is **not** part of oneAPI. It's a separate open-source project, and Triton needs
**three** derived artifacts from it that don't ship prebuilt:

| Artifact | Used by | Why it's needed |
|----------|---------|-----------------|
| `libishmem.so` | runtime linking | shared form of the static lib |
| `libishmem_cwrap.so` | Python (`pyishmem.py`, ctypes) | re-exports hidden symbols with default visibility |
| `libishmem_device.bc` | Triton kernel compile | device-side ishmem functions linked into kernels |

### B.1 Build and install ishmem 1.5.1

```bash
cd /home/jovyan
git clone https://github.com/oneapi-src/ishmem.git
cd ishmem
git checkout v1.5.1

mkdir build && cd build
cmake .. \
    -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icx \
    -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icpx \
    -DENABLE_MPI=ON \
    -DENABLE_OPENSHMEM=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DCMAKE_INSTALL_PREFIX=/home/jovyan/ishmem/install

make -j8
make install
```

Produces `…/install/lib/libishmem.a`, `…/install/include/ishmem.h`, `…/install/bin/ishmrun`.

### B.2 Build `libishmem.so` (shared library)

```bash
/opt/intel/oneapi/compiler/2025.3/bin/icpx -fsycl -shared -fPIC \
    -Wl,--whole-archive /home/jovyan/ishmem/install/lib/libishmem.a -Wl,--no-whole-archive \
    -lze_loader -L/opt/intel/oneapi/mpi/latest/lib -lmpi \
    -o /home/jovyan/ishmem/install/lib/libishmem.so
```

### B.3 Build `libishmem_cwrap.so` (C-linkage wrapper for Python ctypes)

`libishmem.so` marks its symbols `LOCAL` (hidden), so `ctypes.CDLL()` can't call them.
A thin wrapper re-exports the needed functions with `default` visibility.

Create `/tmp/ishmem_cwrap.cpp`:

```cpp
#include <ishmem.h>
#include <ishmemx.h>
#include <cstddef>
#include <cstdint>

extern "C" {

__attribute__((visibility("default"))) void c_ishmem_init() { ishmem_init(); }
__attribute__((visibility("default"))) void c_ishmem_finalize() { ishmem_finalize(); }
__attribute__((visibility("default"))) int  c_ishmem_my_pe() { return ishmem_my_pe(); }
__attribute__((visibility("default"))) int  c_ishmem_n_pes() { return ishmem_n_pes(); }
__attribute__((visibility("default"))) void* c_ishmem_malloc(size_t size) { return ishmem_malloc(size); }
__attribute__((visibility("default"))) void c_ishmem_free(void* ptr) { ishmem_free(ptr); }
__attribute__((visibility("default"))) void c_ishmem_barrier_all() { ishmem_barrier_all(); }
__attribute__((visibility("default"))) void c_ishmem_fence() { ishmem_fence(); }
__attribute__((visibility("default"))) void c_ishmem_quiet() { ishmem_quiet(); }

// IMPORTANT: return the VALUE stored in ishmemi_gpu_info (the USM device pointer),
// NOT &ishmemi_gpu_info (the host address of the variable). Returning the address-of
// makes module_init write a host pointer into the device global → GPU segfault at 0x0.
__attribute__((visibility("default"))) uint64_t c_ishmem_get_gpu_info_ptr() {
    extern void* ishmemi_gpu_info;
    return reinterpret_cast<uint64_t>(ishmemi_gpu_info);
}

__attribute__((visibility("default"))) void* c_ishmem_ptr(const void* dest, int pe) { return ishmem_ptr(dest, pe); }
__attribute__((visibility("default"))) void c_ishmem_int_p(int* dest, int value, int pe) { ishmem_int_p(dest, value, pe); }
__attribute__((visibility("default"))) int  c_ishmem_int_g(const int* src, int pe) { return ishmem_int_g(src, pe); }
__attribute__((visibility("default"))) void c_ishmem_putmem(void* dest, const void* source, size_t nelems, int pe) { ishmem_putmem(dest, source, nelems, pe); }
__attribute__((visibility("default"))) void c_ishmem_getmem(void* dest, const void* source, size_t nelems, int pe) { ishmem_getmem(dest, source, nelems, pe); }

} // extern "C"
```

Compile (the `--whole-archive` is essential — without it the linker drops the LOCAL
symbols the wrapper calls):

```bash
/opt/intel/oneapi/compiler/2025.3/bin/icpx -fsycl -shared -fPIC \
    -I/home/jovyan/ishmem/install/include \
    -o /home/jovyan/ishmem/install/lib/libishmem_cwrap.so \
    /tmp/ishmem_cwrap.cpp \
    -Wl,--whole-archive /home/jovyan/ishmem/install/lib/libishmem.a -Wl,--no-whole-archive \
    -lze_loader -L/opt/intel/oneapi/mpi/latest/lib -lmpi
```

### B.4 Build `libishmem_device.bc` (device bitcode)

This bitcode is linked into every Triton kernel so device-side `ishmem_my_pe()`,
`ishmem_ptr()`, etc. resolve to real implementations. It takes four sub-steps.

Tool locations (note: `llvm-as`/`llvm-dis` come from Triton's downloaded LLVM, not oneAPI):

```bash
BUNDLER=/opt/intel/oneapi/compiler/2025.3/bin/compiler/clang-offload-bundler
LLVM_LINK=/opt/intel/oneapi/compiler/2025.3/bin/compiler/llvm-link
NM=/opt/intel/oneapi/compiler/2025.3/bin/compiler/llvm-nm
LLVM_DIS=/home/jovyan/.triton/llvm/llvm-62b7cf96-ubuntu-x64/bin/llvm-dis
LLVM_AS=/home/jovyan/.triton/llvm/llvm-62b7cf96-ubuntu-x64/bin/llvm-as
```

**Step 1 — unbundle device bitcode from the SYCL fat objects:**

```bash
mkdir -p /home/jovyan/ishmem/device_bc_build && cd /home/jovyan/ishmem/device_bc_build
ar x /home/jovyan/ishmem/install/lib/libishmem.a
mkdir -p device_bcs
for obj in *.o; do
    $BUNDLER --unbundle --type=o \
        --targets=sycl-spir64_gen-unknown-unknown \
        --input="$obj" --output="device_bcs/${obj%.o}.bc" 2>/dev/null
done
```

**Step 2 — link all device bitcode:**

```bash
$LLVM_LINK device_bcs/*.bc -o libishmem_device_raw.bc
```

**Step 3 — add C-name aliases.** ishmem is C++ (mangled names like `_Z12ishmem_my_pev`),
but Triton's lowering emits unmangled C calls. Generate aliases mapping C names → mangled
names, appended to the **same** `.ll` (LLVM aliases must live in the module that defines
their target). Save as `generate_aliases.py` in the `device_bc_build` dir:

```python
import subprocess, re
NM = "/opt/intel/oneapi/compiler/2025.3/bin/compiler/llvm-nm"
result = subprocess.run([NM, "libishmem_device_raw.bc"], capture_output=True, text=True)
mangled_names = [parts[2] for line in result.stdout.splitlines()
                 if len(parts := line.strip().split()) >= 3 and parts[1] == 'T']
demangle = subprocess.run(["c++filt"], input="\n".join(mangled_names),
                          capture_output=True, text=True)
demangled = demangle.stdout.strip().split("\n")
seen, aliases = set(), []
for mangled, full in zip(mangled_names, demangled):
    m = re.match(r'(ishmem\w*|ishmemx\w*)\(', full)
    if m:
        c = m.group(1)
        if c != mangled and c not in seen:
            seen.add(c)
            aliases.append(f"@{c} = alias void (...), ptr @{mangled}")
with open("libishmem_device_raw.ll", "a") as f:
    f.write("\n; === C-name aliases ===\n")
    for a in aliases:
        f.write(a + "\n")
```

```bash
$LLVM_DIS libishmem_device_raw.bc -o libishmem_device_raw.ll
python3 generate_aliases.py
$LLVM_AS libishmem_device_raw.ll -o libishmem_device_with_aliases.bc
```

**Step 4 — mark `global_info` as `externally_initialized`.** ishmem's `device_global`
`global_info` holds a pointer to the runtime's GPU state. Without this attribute, LLVM sees
a zero-initialized global nothing writes to and constant-folds every load to `null` →
segfault. The attribute tells LLVM the value is written externally (by `module_init` at
runtime):

```bash
$LLVM_DIS libishmem_device_with_aliases.bc -o /tmp/patched.ll
sed -i 's/@global_info = \(.*\) global/@global_info = \1 externally_initialized global/g' /tmp/patched.ll
$LLVM_AS /tmp/patched.ll -o libishmem_device.bc
cp libishmem_device.bc /home/jovyan/ishmem/install/lib/libishmem_device.bc
```

**Verify** the C-named functions are defined (`T`):

```bash
$NM /home/jovyan/ishmem/install/lib/libishmem_device.bc \
  | grep -E " T (ishmem_my_pe|ishmem_n_pes|ishmem_ptr|ishmem_barrier_all)$"
```

### B.5 Confirm all three artifacts are present

```bash
ls -la /home/jovyan/ishmem/install/lib/{libishmem.so,libishmem_cwrap.so,libishmem_device.bc}
```

---

## Phase C — Wire up the Intel pass + runtime, then run tests

### C.1 Rebuild with the Intel distributed pass ⭐ (the step the original build skipped)

Phase A built with `TRITON_BUILD_INTEL_DISTRIBUTED=OFF` because `ISHMEM_HOME` didn't exist
yet. Now that ishmem is installed, reconfigure with it **ON** and relink. Without this you
get:

```
AttributeError: module 'triton._C.libtriton.distributed.passes.ttgpuir.intel'
                has no attribute 'add_distributed_to_llvm'
```

```bash
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu/python/build/cmake.linux-x86_64-cpython-3.10
ISHMEM_HOME=/home/jovyan/ishmem/install cmake -DTRITON_BUILD_INTEL_DISTRIBUTED=ON .
ninja -j8
```

This recompiles the Intel lowering pass (`INTEL/ConvertIntelDistributedToLLVM.cpp`,
`INTEL/DistributedOpToLLVM.cpp`) and relinks `libtriton.so` + `libtriton_distributed.so`.

### C.2 Set the runtime environment

Use this exact block in any shell where you run tests. (`run_tests.sh` re-exports most of
these itself, but set them so single-command runs and verification snippets work too.)

```bash
conda activate dist-triton-build
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu

export TRITON_BACKENDS_IN_TREE=1
export ISHMEM_HOME=/home/jovyan/ishmem/install
export MPI_DIR=/opt/intel/oneapi/mpi/latest
export LD_LIBRARY_PATH=$ISHMEM_HOME/lib:$MPI_DIR/lib:$LD_LIBRARY_PATH
export PATH=$MPI_DIR/bin:$PATH
export FI_PROVIDER_PATH=$MPI_DIR/libfabric/lib/prov
```

> Again: **do not** source `setvars.sh`. The `ISHMEM_HOME is not a supported variable`
> warning from the ishmem runtime is harmless — it's our convention, not theirs.

### C.3 Smoke test (single process, no MPI)

```bash
python -c "
import os
os.environ['TRITON_BACKENDS_IN_TREE'] = '1'
os.environ['ISHMEM_HOME'] = '/home/jovyan/ishmem/install'
import triton, triton_dist, torch
from triton.backends import backends
from triton._C.libtriton.distributed import passes
from triton_dist import pyishmem
from triton_dist.language.extra.xpu import libishmem_device
print(f'Triton {triton.__version__}, backends: {list(backends.keys())}')
print(f'PyTorch {torch.__version__}, XPU: {torch.xpu.get_device_name(0)}')
print(f'ISHMEM loaded: {pyishmem._ISHMEM_AVAILABLE}')
print('OK')
"
```

### C.4 Single MPI test (verifies the full pipeline)

```bash
bash scripts/launch_intel.sh --nproc_per_node 2 \
    python/triton_dist/test/intel/test_ishmem_api.py --case check_rank_query
```

Expected:
```
[PE 0] PASSED: check_rank_query (pe=0, npes=2)
[PE 1] PASSED: check_rank_query (pe=1, npes=2)
```

If you see this, the whole chain works: Triton DSL → TTIR → TTGIR → LLVM IR (distributed
lowering) → SPIR-V (with ishmem device bitcode) → ZEBIN → Level Zero module (`global_info`
initialized by `module_init`) → correct device-side ishmem results.

### C.5 Run the full suite

```bash
bash python/triton_dist/test/intel/run_tests.sh
```

Useful variants:

```bash
bash python/triton_dist/test/intel/run_tests.sh --single-only   # no-MPI tests only
bash python/triton_dist/test/intel/run_tests.sh --mpi-only      # 2-PE tests only
bash python/triton_dist/test/intel/run_tests.sh test_all_gather # one file, all its cases
```

The runner sets `ISHMEM_HOME`, `TRITON_BACKENDS_IN_TREE`, `LD_LIBRARY_PATH`, and
`FI_PROVIDER_PATH` internally and launches MPI tests via `mpirun -np 2 -genv …`. Most cases
run under 2 PEs; a few (`test_kernel_barrier_all::check`) run single-PE.

A clean run ends with:

```
========================================================================
TEST SUMMARY
========================================================================
  Passed: N
  Skipped: M
  Failed: 0
```

(`run_tests.sh` exits non-zero if any test fails; tests that exit code 77 are reported as
SKIPPED, not failures.)

---

## Quick reference — minimal end-to-end

For a machine where Phases A & B were already done once and you just need a working shell:

```bash
conda activate dist-triton-build
cd /home/jovyan/dist-triton/triton-distributed-intel-xpu
export TRITON_BACKENDS_IN_TREE=1
export ISHMEM_HOME=/home/jovyan/ishmem/install
export MPI_DIR=/opt/intel/oneapi/mpi/latest
export LD_LIBRARY_PATH=$ISHMEM_HOME/lib:$MPI_DIR/lib:$LD_LIBRARY_PATH
export PATH=$MPI_DIR/bin:$PATH
export FI_PROVIDER_PATH=$MPI_DIR/libfabric/lib/prov
bash python/triton_dist/test/intel/run_tests.sh
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `0 active drivers` / backend not found | `TRITON_BACKENDS_IN_TREE` unset, or symlinks not materialized | Set the env var (C.2); re-run the symlink loop (A.3) |
| `cannot import name 'decompose_descriptor'` | Backend files came from a newer Intel fork than this submodule | Re-materialize from parent commit `9dc5ce0c7`, not `main` (A.3) |
| `… intel has no attribute 'add_distributed_to_llvm'` | Built with `TRITON_BUILD_INTEL_DISTRIBUTED=OFF` | Reconfigure ON + `ninja` (C.1) |
| `libishmem_cwrap.so` exports nothing | Linked without `--whole-archive` | Re-link with `--whole-archive` (B.3) |
| Kernel segfaults at `0x0` / `Expected pe=1, got 0` | `global_info` not patched, or `c_ishmem_get_gpu_info_ptr` returned `&var` | Apply `externally_initialized` (B.4 step 4); ensure cwrap returns the value, not address-of (B.3) |
| `Pointer argument doesn't reference accessible memory` | CPU tensor sent to XPU (old IPEX-gated test) | Tests should use `torch.xpu.is_available()` directly; IPEX is not required |
| `Alias must point to a definition` | Aliases generated in a separate module | Append aliases to the `.ll` that holds the function bodies (B.4 step 3) |
| `libur_loader.so` conflict / XPU vanishes | Sourced `setvars.sh`/`oneapi-vars.sh` | Don't source it; set MPI/ishmem paths manually (C.2) |
| MPI children can't find libs | Intel MPI doesn't inherit parent env | Pass `-genv LD_LIBRARY_PATH/ISHMEM_HOME/TRITON_BACKENDS_IN_TREE/PATH` (handled by `launch_intel.sh` / `run_tests.sh`) |
| Compilation behaves unexpectedly after changes | Stale Triton cache | Clear the `triton_cache/` directory |
| `terminate called without an active exception` at exit (single-PE) | ishmem finalization-order issue during teardown | Known/benign — does not affect test correctness |

---

## What each phase gave you

- **Phase A** — `libtriton.so` + `libtriton_distributed.so`, the Intel backend discoverable
  in-tree.
- **Phase B** — the three ishmem artifacts the runtime and the kernel compiler need.
- **Phase C** — the Intel distributed LLVM lowering pass compiled in, environment wired up,
  and `run_tests.sh` green.

For deeper background on *why* each ishmem artifact is built the way it is, see
[BUILD_SETUP.md](BUILD_SETUP.md). For the history of bugs found and fixed while getting here,
see [HISTORY.md](HISTORY.md).
