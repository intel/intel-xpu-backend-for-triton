import triton
import triton.language as tl
import os
import pytest
import torch


def is_perf_warning_enabled():
    return os.environ.get('MLIR_ENABLE_REMARK', '0') == '1'


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def test_mma_remark(capfd):
    if is_cuda():
        capability = torch.cuda.get_device_capability()
        if capability[0] < 9:
            pytest.skip("Requires sm >= 90 to run")

    os.environ['MLIR_ENABLE_REMARK'] = '1'

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn):
        a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                        block_shape=(32, 128), order=(1, 0))
        b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                        block_shape=(128, 32), order=(0, 1))
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), offsets=(0, 0),
                                        block_shape=(32, 32), order=(1, 0))
        a = tl.load(a_block_ptr)
        b = tl.load(b_block_ptr)
        c = tl.dot(a, b)
        tl.store(c_block_ptr, c)

    triton.compile(
        triton.compiler.ASTSource(
            fn=matmul_kernel, signature={
                0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9:
                'i32', 10: 'i32', 11: 'i32'
            }, constants={}))
    captured = capfd.readouterr()

    assert "remark: Warning: can't use MMA V3 for the dot op" in captured.err, "expect MMA V3 remark"
    assert "note: see current operation:" in captured.err
    os.environ['MLIR_ENABLE_REMARK'] = '0'
