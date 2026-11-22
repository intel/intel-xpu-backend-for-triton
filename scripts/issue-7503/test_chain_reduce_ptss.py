"""
Diagnostic sweep for issue #7503: `test_chain_reduce[1-max-src_layout2-256-*]` exceeds the
per-thread scratch space (PTSS) limit on A770 and B580.

Compiles the kernel of `python/test/unit/intel/test_core.py::test_chain_reduce` for every shape,
layout, op and axis, once per GRF mode and compile path, and records what IGC made of it: the GRF
count and spill it settled on, or the scratch space it asked for when it gave up.

A PTSS overflow is the measurement here, not a failure, and is reported as XFAIL with the numbers
in the reason. A case fails only on a wrong result or on an error that is not a PTSS overflow.

GRF modes: `default` takes the backend's automatic large-GRF retry. On the JIT path, when the
retry fails too, `load_binary` re-raises the *first* build's error, so the scratch figure it
reports is the default-GRF one. `128` and `256` are explicit, which disables the retry, so each
reports its own figure.

Compile paths: `jit` hands SPIR-V to Level Zero at load time (`driver.c`), which is what the
Windows CI does. `aot` compiles with `ocloc` first (`compiler.py`), which is what Linux CI does
through `TRITON_XPU_GEN_NATIVE_CODE=1`; it is skipped when `ocloc` is not on PATH.

Set `ISSUE_7503_RESULTS_DIR` to write one JSON record per case, plus `env.json`, for
`summarize.py`.
"""
# pytest idioms: `M`/`N` are test_core.py's parameter names, fixtures arrive as parameters, and the JIT
# path builds the device binary only in `CompiledKernel._init_handles`.
# pylint: disable=invalid-name,too-many-arguments,too-many-positional-arguments,too-many-locals
# pylint: disable=redefined-outer-name,protected-access

import json
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
from triton import knobs
from triton.runtime.errors import OutOfResources

DEVICE = "xpu"
THREADS_PER_WARP = 32

# Same order as `layouts` in test_core.py, so `src_layout<i>` names the same layout there and here.
LAYOUTS = [
    ([1, 4], [1, THREADS_PER_WARP], [4, 1], [1, 0]),
    ([1, 4], [1, THREADS_PER_WARP], [2, 2], [1, 0]),
    ([1, 4], [THREADS_PER_WARP // 32, 32], [1, 4], [1, 0]),
    ([1, 4], [8, THREADS_PER_WARP // 8], [2, 2], [0, 1]),
]
SHAPES = [[128, 128], [256, 128], [256, 256], [128, 256]]

# IGC's wording: "total scratch space exceeds HW supported limit: 428288 bytes (max permitted PTSS
# 262144 bytes)". The JIT path passes it through in the exception text with required/limit 0/0.
PTSS_NUMBERS_RE = re.compile(r"(\d+)\s*bytes\s*\(max permitted PTSS\s*(\d+)\s*bytes\)", re.IGNORECASE)
PTSS_MARKERS = ("total scratch space", "scratch space exceeds", "max permitted PTSS", "per-thread scratch space")

RESULTS_DIR = os.environ.get("ISSUE_7503_RESULTS_DIR")


def blocked_layout(size_per_thread, threads_per_warp, warps_per_cta, order) -> str:
    # Formatted like `BlockedLayout.__str__` in test_core.py.
    return (f"#ttg.blocked<{{sizePerThread={size_per_thread}, threadsPerWarp={threads_per_warp}, "
            f"warpsPerCTA={warps_per_cta}, order={order}}}>")


def chain_reduce_ir(M: int, N: int, src_layout: str, op: str, first_axis: int) -> str:
    # The IR of test_core.py::test_chain_reduce, verbatim apart from the f-string plumbing.
    if op == "sum":
        op_str = """
        %13 = arith.addi %arg2, %arg3 : i32
        tt.reduce.return %13 : i32"""
    else:
        op_str = """
        %13 = arith.cmpi "sgt", %arg2, %arg3 : i32
        %14 = arith.select %13, %arg2, %arg3 : i32
        tt.reduce.return %14 : i32"""
    reduced = M if first_axis == 1 else N
    return f"""
    #src = {src_layout}
    module attributes {{"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = {THREADS_PER_WARP} : i32}} {{
    tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<i32> {{tt.divisibility = 16 : i32}}) {{
        %cst = arith.constant dense<{N}> : tensor<{M}x1xi32, #src>
        %0 = tt.make_range {{end = {M} : i32, start = 0 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>>
        %1 = tt.expand_dims %0 {{axis = 1 : i32}} : tensor<{M}xi32, #ttg.slice<{{dim = 1, parent = #src}}>> -> tensor<{M}x1xi32, #src>
        %2 = arith.muli %1, %cst : tensor<{M}x1xi32, #src>
        %3 = tt.make_range {{end = {N} : i32, start = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>>
        %4 = tt.expand_dims %3 {{axis = 0 : i32}} : tensor<{N}xi32, #ttg.slice<{{dim = 0, parent = #src}}>> -> tensor<1x{N}xi32, #src>
        %5 = tt.broadcast %2 : tensor<{M}x1xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %6 = tt.broadcast %4 : tensor<1x{N}xi32, #src> -> tensor<{M}x{N}xi32, #src>
        %7 = arith.addi %5, %6 : tensor<{M}x{N}xi32, #src>
        %8 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %9 = tt.addptr %8, %7 : tensor<{M}x{N}x!tt.ptr<i32>, #src>, tensor<{M}x{N}xi32, #src>
        %10 = tt.load %9 : tensor<{M}x{N}x!tt.ptr<i32>, #src>
        %11 = "tt.reduce"(%10) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = {first_axis} : i32}} : (tensor<{M}x{N}xi32, #src>) -> tensor<{reduced}xi32, #ttg.slice<{{dim = {first_axis}, parent = #src}}>>
        %12 = "tt.reduce"(%11) ({{
        ^bb0(%arg2: i32, %arg3: i32):
        {op_str}
        }}) {{axis = 0 : i32}} : (tensor<{reduced}xi32, #ttg.slice<{{dim = {first_axis}, parent = #src}}>>) -> i32
        tt.store %arg1, %12 : !tt.ptr<i32>
        tt.return
    }}
    }}
    """


def build_log(err: OutOfResources) -> str:
    # The AOT path chains the `CalledProcessError` of the failed `ocloc` run.
    return getattr(err.__cause__, "output", "") or ""


def is_ptss_overflow(err: OutOfResources) -> bool:
    text = f"{err.name} {build_log(err)}".lower()
    return any(marker.lower() in text for marker in PTSS_MARKERS)


def ptss_numbers(err: OutOfResources) -> list[list[int]]:
    """
    Returns every (required, limit) pair IGC reported, in log order.

    The AOT path parses one pair into `required`/`limit`, and the chained `CalledProcessError`
    holds the full `ocloc` log of the last attempt. The JIT path only has the first 1 KiB of the
    IGC log, in `name`.
    """
    text = f"{err.name}\n{build_log(err)}"
    pairs = [[int(required), int(limit)] for required, limit in PTSS_NUMBERS_RE.findall(text)]
    if not pairs and err.required:
        pairs = [[int(err.required), int(err.limit)]]
    return pairs


def write_record(name: str, record: dict) -> None:
    if not RESULTS_DIR:
        return
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    (out_dir / f"{safe_name}.json").write_text(json.dumps(record, indent=1))


@pytest.fixture(scope="module", autouse=True)
def environment_record():
    props = torch.xpu.get_device_properties(0)
    target = triton.runtime.driver.active.get_current_target()
    record = {
        "platform": platform.platform(),
        "device": props.name,
        "driver_version": getattr(props, "driver_version", ""),
        "arch": target.arch.get("arch", "") if isinstance(target.arch, dict) else str(target.arch),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "ocloc": shutil.which("ocloc") or "",
        "forced_native_code": bool(knobs.intel.gen_native_code or knobs.intel.dump_shader_info),
    }
    write_record("env", record)
    print(f"\nissue-7503 environment: {record}")
    yield record


@pytest.mark.parametrize("path", ["jit", "aot"])
@pytest.mark.parametrize("grf_mode", ["default", "128", "256"], ids=lambda g: f"grf-{g}")
@pytest.mark.parametrize("M, N", SHAPES)
@pytest.mark.parametrize("layout", range(len(LAYOUTS)), ids=lambda i: f"src_layout{i}")
@pytest.mark.parametrize("op", ["sum", "max"])
@pytest.mark.parametrize("first_axis", [0, 1])
def test_chain_reduce_ptss(first_axis, op, layout, M, N, grf_mode, path, environment_record, tmp_path, request):
    # Parameter order gives ids like `1-max-src_layout2-256-128-grf-default-jit`: the
    # test_core.py::test_chain_reduce id, then the GRF mode and the compile path.
    if path == "aot" and not environment_record["ocloc"]:
        pytest.skip("ocloc is not on PATH")
    if path == "jit" and environment_record["forced_native_code"]:
        pytest.skip("TRITON_XPU_GEN_NATIVE_CODE or TRITON_INTEL_ENABLE_IGC_SHADER_DUMP forces the AOT path")

    record = {
        "id": request.node.callspec.id, "first_axis": first_axis, "op": op, "layout": layout, "M": M, "N": N,
        "grf_mode": grf_mode, "path": path
    }
    src = tmp_path / "test_chain_reduce.ttgir"
    src.write_text(chain_reduce_ir(M, N, blocked_layout(*LAYOUTS[layout]), op, first_axis))
    options = {"generate_native_code": path == "aot"}
    if grf_mode != "default":
        options["grf_mode"] = grf_mode

    start = time.perf_counter()
    try:
        kernel = triton.compile(str(src), options=options)
        # The JIT path builds the device binary here, not in `triton.compile`.
        kernel._init_handles()
    except OutOfResources as err:
        record["build_seconds"] = round(time.perf_counter() - start, 2)
        if not is_ptss_overflow(err):
            record.update(status="error", message=str(err)[:4000])
            write_record(record["id"], record)
            raise
        pairs = ptss_numbers(err)
        record.update(status="ptss", ptss=pairs, message=str(err)[:4000])
        write_record(record["id"], record)
        pytest.xfail("PTSS overflow: " + (", ".join(f"{req} > {lim} bytes" for req, lim in pairs) or "no numbers"))
    except Exception as err:
        record.update(status="error", build_seconds=round(time.perf_counter() - start, 2),
                      message=f"{type(err).__name__}: {err}"[:4000])
        write_record(record["id"], record)
        raise

    record.update(
        build_seconds=round(time.perf_counter() - start, 2),
        n_regs=kernel.n_regs,
        n_spills=kernel.n_spills,
        warp_size=kernel.metadata.warp_size,
        build_flags=kernel.metadata.build_flags,
        native=kernel.metadata.generate_native_code,
    )

    x = RandomState(17).randint(0, 4, (M, N)).astype("int32")
    x_tri = torch.tensor(x, device=DEVICE)
    z_tri = torch.zeros((1, ), dtype=torch.int32, device=DEVICE)
    kernel[(1, 1, 1)](x_tri, z_tri)
    expected = int(np.sum(x) if op == "sum" else np.max(x))
    actual = int(z_tri.cpu().item())
    record.update(status="ok" if actual == expected else "wrong", expected=expected, actual=actual)
    write_record(record["id"], record)
    np.testing.assert_equal(actual, expected)
