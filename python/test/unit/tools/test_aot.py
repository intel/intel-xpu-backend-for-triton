import glob
import os
import pytest
import re
import subprocess
import sys
import tempfile
import shutil
import sysconfig

import numpy as np

import triton
from triton._internal_testing import is_cuda, is_xpu, is_hip
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.driver import include_dirs, library_dirs
from triton.backends.intel.driver import COMPILATION_HELPER

kernel_utils_src = """
import triton

@triton.jit
def mul(x, y):
    return x * y
"""

kernel_src = """
import triton
import triton.language as tl
import kernel_utils

@triton.jit
def kernel(C, A, B, M, N, K,
          stride_cm, stride_cn,
          stride_am, stride_ak,
          stride_bk, stride_bn,
          BLOCK_M: tl.constexpr,
          BLOCK_N: tl.constexpr,
          BLOCK_K: tl.constexpr):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
  offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
  offs_k = tl.arange(0, BLOCK_K)
  a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
  b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

  accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(0, tl.cdiv(K, BLOCK_K)):
      # Load the next block of A and B, generate a mask by checking the K dimension.
      # If it is out of bounds, set it to 0.
      a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
      b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
      # We accumulate along the K dimension.
      accumulator += tl.dot(a, b)
      # Advance the ptrs to the next K block.
      a_ptrs += BLOCK_K * stride_ak
      b_ptrs += BLOCK_K * stride_bk

  c = kernel_utils.mul(accumulator, accumulator)
  # Write back the block of the output matrix C with masks.
  offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
  tl.store(c_ptrs, c)
"""

test_utils_src = """
#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "kernel.h"

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    for (int i = 0; i < size; i++) {
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {
            fprintf(file, ",");
        }
    }
    fclose(file);
}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file %s\\n", filename);
        return;
    }
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {
        index++;
    }
    fclose(file);
}"""


def select_compiler():
    gxx = shutil.which("g++")
    icpx = shutil.which("icpx")
    cl = shutil.which("cl")
    cxx = (icpx or cl) if os.name == "nt" else (icpx or gxx)
    if cxx is None:
        raise RuntimeError("Failed to find C++ compiler. Please specify via CXX environment variable.")
    return cxx


def _cxx_compile_cmd(cxx: str, src: list, include_dirs: list, only_compile: bool = True) -> list:
    if "cl.EXE" in cxx or "clang-cl" in cxx:
        command = [cxx] + src + ["/I" + include_dir for include_dir in include_dirs
                                 ] + ["/Zc:__cplusplus", "/std:c++17", "/MD", "/nologo", "/O2", "/EHsc", "/wd4996"]
        if only_compile:
            command += ["/c"]
    else:
        command = [cxx] + src + ["-I" + include_dir for include_dir in include_dirs
                                 ] + ["-fPIC" if os.name != "nt" else "-Wno-deprecated-declarations"]
        if only_compile:
            command += ["-c"]
    return command


def _cxx_link_cmd(cxx: str, o_files: list, out: str, extra_library_dirs: list = [], extra_libraries: list = [],
                  shared_lib: bool = True) -> list:
    extra_link_args = []
    if os.name == "nt":
        libname_without_ext = out.split(".")[0]
        extra_link_args = [f"/IMPLIB:{libname_without_ext}.lib"]

    library_dirs = COMPILATION_HELPER.library_dir + COMPILATION_HELPER.libsycl_dir + extra_library_dirs
    if "cl.EXE" in cxx or "clang-cl" in cxx:
        command = [cxx] + [*o_files, *(["/LD"] if shared_lib else []), "/link", f"/OUT:{out}"] + [
            "/LIBPATH:" + library_dir for library_dir in library_dirs
        ] + ["sycl8.lib", "ze_loader.lib"] + [f"{lib}.lib" for lib in extra_libraries] + extra_link_args
    else:
        command = [cxx] + [*o_files, *(["-shared"] if shared_lib else []), "-o", out] + [
            "-L" + library_dir for library_dir in library_dirs
        ] + ["-lsycl8" if os.name == "nt" else "-lsycl", "-lze_loader"] + [f"-l{lib}"
                                                                           for lib in extra_libraries] + extra_link_args

    return command


def _cxx_cmd(cxx: str, src: list, out: str, include_dirs: list, extra_library_dirs: list,
             extra_libraries: list) -> list:
    compile_command = _cxx_compile_cmd(cxx, src, include_dirs, only_compile=False)
    link_command = _cxx_link_cmd(cxx, [], out, extra_library_dirs, extra_libraries, shared_lib=False)
    return compile_command + link_command[1:]


def gen_kernel_library_xpu(dir, libname):
    cpp_files = glob.glob(os.path.join(dir, "*.cpp"))
    cxx = select_compiler()
    command = _cxx_compile_cmd(cxx, cpp_files, COMPILATION_HELPER.include_dir)
    subprocess.run(command, check=True, cwd=dir)

    if "cl.EXE" in cxx or "clang-cl" in cxx:
        o_files = glob.glob(os.path.join(dir, "*.obj"))
    else:
        o_files = glob.glob(os.path.join(dir, "*.o"))
    command = _cxx_link_cmd(cxx, o_files, libname)
    subprocess.run(command, check=True, cwd=dir)


def gen_kernel_library(dir, libname):
    if is_xpu():
        gen_kernel_library_xpu(dir, libname)
    else:
        c_files = glob.glob(os.path.join(dir, "*.c"))
        subprocess.run(
            ["gcc"] + c_files + ["-I", include_dirs[0], "-c", "-fPIC"],
            check=True,
            cwd=dir,
        )
        o_files = glob.glob(os.path.join(dir, "*.o"))

        command = ["gcc", *o_files, "-shared", "-o", libname]
        for lib_dir in library_dirs():
            command.extend(["-L", lib_dir])
        subprocess.run(command, check=True, cwd=dir)


def gen_test_bin(dir, M, N, K, exe="test", algo_id=0):
    exe_extension = sysconfig.get_config_var("EXE")
    exe = exe + exe_extension
    test_src = f"""
int main(int argc, char **argv) {{
  int M = {M}, N = {N}, K = {K};

  // initialize CUDA handles
  CUdevice dev;
  CUcontext ctx;
  CUstream stream;
  CUdeviceptr A, B, C;
  CUresult err = 0;
  cuInit(0);
  cuDeviceGet(&dev, 0);
  cuCtxCreate(&ctx, 0, dev);
  cuMemAlloc(&A, M * K * 2);
  cuMemAlloc(&B, K * N * 2);
  cuMemAlloc(&C, M * N * 4);
  cuStreamCreate(&stream, 0);
  load_matmul_fp16();

  // initialize input data
  int16_t hA[M*K];
  int16_t hB[K*N];
  memset(hA, 0, M*K*2);
  memset(hB, 0, K*N*2);
  read_csv_to_buffer(argv[1], hA, M*K);
  read_csv_to_buffer(argv[2], hB, K*N);
  cuMemcpyHtoD(A, hA, M*K*2);
  cuMemcpyHtoD(B, hB, K*N*2);

  // launch kernel
  CUresult ret;
  int algo_id = {algo_id};
  if (algo_id == 0) {{
    ret = matmul_fp16_default(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1);
  }} else {{
    ret = matmul_fp16(stream, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
  }}
  if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
  assert(ret == 0);

  // read data
  int32_t hC[M*N];
  memset(hC, 0, M*N*4);
  cuMemcpyDtoH(hC, C, M*N*4);
  write_buffer_to_csv(argv[3], hC, M*N);

  // free cuda handles
  unload_matmul_fp16();
  cuMemFree(A);
  cuMemFree(B);
  cuMemFree(C);
  cuCtxDestroy(ctx);
}}
"""
    src = test_utils_src + test_src
    if is_xpu():
        src = f"""
#include "kernel.h"
#include <assert.h>
#include <cmath>
#include <cstddef>
#include <level_zero/ze_api.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sycl/sycl.hpp>

static void write_buffer_to_csv(char *filename, int32_t *buffer, int size) {{
    FILE *file = fopen(filename, "w");
    if (file == NULL) {{
        printf("Could not open file %s\\n", filename);
        return;
    }}
    for (int i = 0; i < size; i++) {{
        fprintf(file, "%d", buffer[i]);
        if (i < size - 1) {{
            fprintf(file, ",");
        }}
    }}
    fclose(file);
}}

static void read_csv_to_buffer(char *filename, int16_t *buffer, int size) {{
    FILE *file = fopen(filename, "r");
    if (file == NULL) {{
        printf("Could not open file %s\\n", filename);
        return;
    }}
    int index = 0;
    while (fscanf(file, "%hd,", &buffer[index]) != EOF && index < size) {{
        index++;
    }}
    fclose(file);
}}
int main(int argc, char ** argv) {{
    constexpr int M = {M}, N = {N}, K = {K};

    // initialize sycl handles
    sycl::queue q{{sycl::gpu_selector_v}};
    sycl::ext::intel::device_ptr<sycl::float16> A =
        sycl::malloc_device<sycl::float16>(M * K * 2, q);
    sycl::ext::intel::device_ptr<sycl::float16> B =
        sycl::malloc_device<sycl::float16>(K * N * 2, q);
    sycl::ext::intel::device_ptr<sycl::float16> C =
        sycl::malloc_device<sycl::float16>(M * N * 4, q);

    // initialize input data
    int16_t hA[M * K];
    int16_t hB[K * N];
    memset(hA, 0, M * K * 2);
    memset(hB, 0, K * N * 2);
    read_csv_to_buffer(argv[1], hA, M * K);
    read_csv_to_buffer(argv[2], hB, K * N);
    q.memcpy(A, hA, M * K * 2).wait();
    q.memcpy(B, hB, K * N * 2).wait();

    // launch kernel
    load_matmul_fp16();
    int32_t ret;
    int algo_id = {algo_id};
    if (algo_id == 0) {{
        ret = matmul_fp16_default(q, C, A, B, M, N, K, N, 1, K, 1, N, 1);
    }} else {{
        ret = matmul_fp16(q, C, A, B, M, N, K, N, 1, K, 1, N, 1, {algo_id});
    }}
    if (ret != 0) fprintf(stderr, "kernel launch failed\\n");
    assert(ret == 0);

    q.wait();

    // read data
    int32_t hC[M * N];
    memset(hC, 0, M * N * 4);
    q.memcpy(hC, C, M * N * 4).wait();
    write_buffer_to_csv(argv[3], hC, M * N);

    // free sycl resources
    unload_matmul_fp16();
    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);
}}
"""
    src_name = "test.c"
    if is_xpu():
        src_name = "test.cpp"
    with open(os.path.join(dir, src_name), "w") as file:
        file.write(src)

    if is_cuda():
        command = ["gcc", "test.c"]
        for inc_dir in include_dirs:
            command.extend(["-I", inc_dir])
        for lib_dir in library_dirs():
            command.extend(["-L", lib_dir])
        command.extend(["-l", "cuda", "-L", dir, "-l", "kernel", "-o", exe])

    if is_xpu():
        cxx = select_compiler()
        command = _cxx_cmd(cxx, ["test.cpp"], exe, COMPILATION_HELPER.include_dir, [dir], ["kernel"])
    subprocess.run(command, check=True, cwd=dir)


def write_triton_kernels(dir, src, util_src):
    kernel_path = os.path.join(dir, "kernel.py")
    with open(kernel_path, "w") as file:
        file.write(src)

    kernel_utils_path = os.path.join(dir, "kernel_utils.py")
    with open(kernel_utils_path, "w") as file:
        file.write(util_src)

    return kernel_path


def _compile_kernel(dir, signature, kernel_name, out_name, out_path, num_warps, grid, generate_native_code,
                    kernel_path):
    compiler_path = os.path.join(triton.tools.__path__[0], "compile.py")

    subprocess.run(
        [
            sys.executable,
            compiler_path,
            "-n",
            kernel_name,
            "--signature",
            signature,
            "--out-name",
            out_name,
            "-o",
            out_path,
            "-w",
            str(num_warps),
            "-g",
            grid,
            *(["-gnc"] if generate_native_code else []),
            kernel_path,
        ],
        check=True,
        cwd=dir,
    )


# Edge case kernel with no specialization
def compile_aot_kernel_no_specialization(dir, kernel_path, dtype, BM, BN, BK, generate_native_code):
    # compile all desired configs
    sig = f"*fp32, *{dtype}, *{dtype}, i32, i32, i32, i32, i32, i32, i32, i32, i32, {BM}, {BN}, {BK}"
    name = f"matmul_{dtype}"
    grid = f"M/{BM}, N/{BN}, 1"
    _compile_kernel(
        dir=dir,
        signature=sig,
        kernel_name="kernel",
        out_name=name,
        out_path=name,
        num_warps=1,
        grid=grid,
        generate_native_code=generate_native_code,
        kernel_path=kernel_path,
    )


def compile_aot_kernels(dir, kernel_path, dtype, BM, BN, BK, generate_native_code, ha_hb_hints):
    # compile all desired configs
    for ha, hb in ha_hb_hints:
        sig = f"*fp32:16, *{dtype}:16, *{dtype}:16, i32, i32, i32, i32{ha}, i32:1, i32{hb}, i32:1, i32:16, i32:1, {BM}, {BN}, {BK}"
        name = f"matmul_{dtype}"
        grid = f"M/{BM}, N/{BN}, 1"
        _compile_kernel(
            dir=dir,
            signature=sig,
            kernel_name="kernel",
            out_name=name,
            out_path=name,
            num_warps=1,
            grid=grid,
            generate_native_code=generate_native_code,
            kernel_path=kernel_path,
        )


def link_aot_kernels(dir):
    linker_path = os.path.join(triton.tools.__path__[0], "link.py")

    # link all desired configs
    h_files = glob.glob(os.path.join(dir, "*.h"))
    subprocess.run([sys.executable, linker_path] + h_files + ["-o", "kernel"], check=True, cwd=dir)


def generate_matmul_test_data(dir, M, N, K):
    a = np.random.randn(M * K).astype(np.float16).reshape((M, K))
    b = np.random.randn(M * K).astype(np.float16).reshape((K, N))
    a_path = os.path.join(dir, "a.csv")
    b_path = os.path.join(dir, "b.csv")
    c_path = os.path.join(dir, "c.csv")
    for x, path in [(a, a_path), (b, b_path)]:
        x.view(np.int16).ravel().tofile(path, sep=",")
    return a, b, a_path, b_path, c_path


def check_hasco_binary_str(tmp_dir: str, dtype: str):
    # Linking is not yet enabled on HIP backend so just check compilation for now.
    h_files = glob.glob(f"matmul_{dtype}.*.h", root_dir=tmp_dir)
    cpp_files = glob.glob(f"matmul_{dtype}.*.cpp", root_dir=tmp_dir)
    assert len(h_files) == 1, "Expected one .h file"
    assert len(cpp_files) == 1, "Expected one .cpp file"
    pattern = re.compile(r'HSACO_NAME\[(\d+)\]')
    with open(os.path.join(tmp_dir, cpp_files[0]), "r") as cpp_file:
        content = cpp_file.read()
        matches = pattern.findall(content)
        assert len(matches) == 1, "Expected one HSACO_NAME definition"
        assert int(matches[0]) > 16, "Expected valid HSACO object binary string"


# Test edge case where the provided kernel signature has no specializations
@pytest.mark.parametrize("generate_native_code", [True, False])
def test_compile_link_matmul_no_specialization(generate_native_code):
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"

        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernel_no_specialization(tmp_dir, kernel_path, dtype, BM, BN, BK, generate_native_code)
        if is_hip():
            check_hasco_binary_str(tmp_dir, dtype)
            return

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so" if os.name != "nt" else "kernel.dll")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        subprocess.run([os.path.join(tmp_dir, "test"), a_path, b_path, c_path], env=env, check=True, cwd=tmp_dir)
        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("generate_native_code", [True, False])
def test_compile_link_matmul(generate_native_code):
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, generate_native_code, ha_hb_hints=[(":16", ":16")])
        if is_hip():
            check_hasco_binary_str(tmp_dir, dtype)
            return
        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so" if os.name != "nt" else "kernel.dll")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        subprocess.run([os.path.join(tmp_dir, "test"), a_path, b_path, c_path], env=env, check=True, cwd=tmp_dir)

        # read data and compare against reference
        c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
        c_tri = c.reshape((M, N)).view(np.float32)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))
        np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("generate_native_code", [True, False])
def test_launcher_has_no_available_kernel(generate_native_code):
    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"
        BM, BN, BK = 16, 16, 16

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)
        compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, generate_native_code, ha_hb_hints=[(":1", ":1")])
        if is_hip():
            check_hasco_binary_str(tmp_dir, dtype)
            return

        link_aot_kernels(tmp_dir)

        # compile test case
        M, N, K = 16, 16, 16
        gen_kernel_library(tmp_dir, "libkernel.so" if os.name != "nt" else "kernel.dll")
        gen_test_bin(tmp_dir, M, N, K)

        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)

        # run test case
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = tmp_dir + ":" + env.get("LD_LIBRARY_PATH", "")
        result = subprocess.run(
            [os.path.join(tmp_dir, "test"), a_path, b_path, c_path],
            env=env,
            cwd=tmp_dir,
            capture_output=True,
            text=True,
        )

        # It should fail since the launcher requires all the strides be 1 while they are not.
        # On windows: 3221226505 == 0xc0000409: STATUS_STACK_BUFFER_OVERRUN
        assert result.returncode == -6 if os.name != "nt" else 0xc0000409
        assert "kernel launch failed" in result.stderr


@pytest.mark.skipif(not is_cuda() and not is_xpu(), reason="Requires CUDA or XPU")
def test_compile_link_autotune_matmul():
    # this test is pretty slow, so we only run with the native binary
    generate_native_code = True

    np.random.seed(3)

    with tempfile.TemporaryDirectory() as tmp_dir:
        dtype = "fp16"

        kernel_path = write_triton_kernels(tmp_dir, kernel_src, kernel_utils_src)

        tile_sizes = [
            [16, 16, 16],
            [64, 64, 32],
        ]

        for ts in tile_sizes:
            BM, BN, BK = ts[0], ts[1], ts[2]
            compile_aot_kernels(tmp_dir, kernel_path, dtype, BM, BN, BK, generate_native_code,
                                ha_hb_hints=[(":16", ":16"), (":16", ""), ("", ":16")])

        link_aot_kernels(tmp_dir)

        gen_kernel_library(tmp_dir, "libkernel.so" if os.name != "nt" else "kernel.dll")

        # compile test case
        M, N, K = 64, 64, 64
        # initialize test data
        a, b, a_path, b_path, c_path = generate_matmul_test_data(tmp_dir, M, N, K)
        c_ref = np.matmul(a.astype(np.float32), b.astype(np.float32))

        for algo_id in range(len(tile_sizes)):
            # generate and run test case
            test_name = f"test_{algo_id}"
            gen_test_bin(tmp_dir, M, N, K, exe=test_name, algo_id=algo_id)

            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = tmp_dir + ":" + env.get("LD_LIBRARY_PATH", "")
            subprocess.run(
                [os.path.join(tmp_dir, test_name), a_path, b_path, c_path],
                check=True,
                cwd=tmp_dir,
                env=env,
            )

            # read data and compare against reference
            c = np.genfromtxt(c_path, delimiter=",", dtype=np.int32)
            c_tri = c.reshape((M, N)).view(np.float32)
            np.testing.assert_allclose(c_tri, c_ref * c_ref, atol=1e-4, rtol=1e-4)


def test_ttgir_to_asm():
    src = """
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = {warp_size} : i32, "ttg.num-ctas" = 1 : i32}} {{
  tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {{
    tt.return
  }}
}}
"""
    target = GPUTarget("hip", "gfx942", 64) if is_hip() else GPUTarget("cuda", 80, 32)
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "empty_kernel.ttgir")
        with open(kernel_path, "w") as fp:
            fp.write(src.format(warp_size=target.warp_size))
        k = triton.compile(kernel_path, target=target)
        if is_cuda():
            ptx = k.asm["ptx"]
            assert ".target sm_80" in ptx
            assert ".address_size 64" in ptx
        elif is_hip():
            amdgcn = k.asm["amdgcn"]
            assert '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"' in amdgcn
            assert '.wavefront_size: 64' in amdgcn


def test_ttgir_to_spv():
    src = """
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func public @sum_kernel_0d1d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
    tt.return
  }
}
"""
    # ensure spv output so we can grep the spirv file
    os.environ["TRITON_XPU_GEN_NATIVE_CODE"] = "0"
    with tempfile.TemporaryDirectory() as tmp_dir:
        kernel_path = os.path.join(tmp_dir, "empty_kernel.ttgir")
        with open(kernel_path, "w") as fp:
            fp.write(src)
        k = triton.compile(kernel_path, target=triton.runtime.driver.active.get_current_target())
        spv = k.asm['spvdis']
        assert "OpCapability Kernel" in spv
        assert "LocalSize 128 1 1" in spv
        assert "SubgroupSize 32" in spv
