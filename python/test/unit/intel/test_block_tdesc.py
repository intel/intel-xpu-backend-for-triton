import pytest
import torch
import pathlib

import triton
from triton._internal_testing import is_xpu


@pytest.fixture(autouse=True, params=[False, True], ids=["branch-io", "predicated-io"])
def predicated_io(request, monkeypatch):
    if request.param:
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_LOAD", "1")
        monkeypatch.setenv("TRITON_INTEL_PREDICATED_STORE", "1")
    yield


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [64, 64], [64, 32], [32, 32], [16, 64], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_load_store(M, N, dtype_str, device, tmp_path: pathlib.Path):
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_load_store(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %stride_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c0_i32 = arith.constant 0 : i32

            %src_desc = tt.make_tensor_descriptor %arg0, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>>

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32]
                    : !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>> -> tensor<{M}x{N}x{ty}, #blocked>

            %dst_desc = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>>

            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32], %data
                                : !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>>, tensor<{M}x{N}x{ty}, #blocked>

            tt.return
        }}
    }}
    """
    torch.manual_seed(42)

    torch_dtype = getattr(torch, dtype_str)
    if torch_dtype.is_floating_point:
        a = torch.randn((M, N), dtype=torch_dtype, device=device)
    else:
        a = torch.randint(low=-127, high=128, size=(M, N), dtype=torch_dtype, device=device)

    x = torch.empty_like(a)

    temp_file = tmp_path / "test_tdesc_load_store.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)
    assert torch.equal(a, x)


@pytest.mark.parametrize("M, N",
                         [[256, 64], [256, 32], [128, 32], [128, 16], [64, 64], [64, 32], [32, 32], [16, 64], [16, 16]])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int8"])
@pytest.mark.skipif(not is_xpu(), reason="Tensor descriptor tests are specific to the XPU backend")
def test_tdesc_load_zero_padding(M, N, dtype_str, device, tmp_path: pathlib.Path):
    """Load a MxN block through a descriptor whose shape is (M-1)x(N-1).

    The last row and last column are out of bounds and must be zero-padded.
    Input is filled with ones so any zero in the output indicates padding.
    """
    num_warps = 4
    threads_per_warp = 32

    ty = {"float32": "f32", "float16": "f16", "int8": "i8"}[dtype_str]

    ir = f"""
    #blocked = #ttg.blocked<{{sizePerThread = [1, 1], threadsPerWarp = [1, {threads_per_warp}], warpsPerCTA = [1, {num_warps}], order = [1, 0]}}>
    module attributes {{ttg.target = "xpu", "ttg.num-warps" = {num_warps} : i32, "ttg.threads-per-warp" = {threads_per_warp} : i32}} {{
        tt.func public @descriptor_load_store_pad(%arg0: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}, %arg1: !tt.ptr<{ty}> {{tt.divisibility = 16 : i32}}) {{
            %stride_i64 = arith.constant {N} : i64
            %c1_i64 = arith.constant 1 : i64
            %cM_minus1 = arith.constant {M - 1} : i32
            %cN_minus1 = arith.constant {N - 1} : i32
            %cM_i32 = arith.constant {M} : i32
            %cN_i32 = arith.constant {N} : i32
            %c0_i32 = arith.constant 0 : i32

            // Source descriptor with shape (M-1)x(N-1) — last row/col out of bounds
            %src_desc = tt.make_tensor_descriptor %arg0, [%cM_minus1, %cN_minus1], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>>

            %data = tt.descriptor_load %src_desc [%c0_i32, %c0_i32]
                    : !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>> -> tensor<{M}x{N}x{ty}, #blocked>

            // Destination descriptor with full shape so we can store everything
            %dst_desc = tt.make_tensor_descriptor %arg1, [%cM_i32, %cN_i32], [%stride_i64, %c1_i64]
                        : !tt.ptr<{ty}>, !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>>

            tt.descriptor_store %dst_desc [%c0_i32, %c0_i32], %data
                                : !tt.tensordesc<tensor<{M}x{N}x{ty}, #blocked>>, tensor<{M}x{N}x{ty}, #blocked>

            tt.return
        }}
    }}
    """

    torch_dtype = getattr(torch, dtype_str)
    a = torch.ones((M, N), dtype=torch_dtype, device=device)
    x = torch.empty_like(a)

    temp_file = tmp_path / "test_tdesc_load_zero_padding.ttgir"
    temp_file.write_text(ir)
    kernel = triton.compile(str(temp_file))

    kernel[(1, 1, 1)](a, x)

    # Build expected: ones everywhere except last row and last column are zero-padded
    expected = torch.ones((M, N), dtype=torch_dtype, device=device)
    expected[M - 1, :] = 0
    expected[:, N - 1] = 0

    assert torch.equal(x, expected)
