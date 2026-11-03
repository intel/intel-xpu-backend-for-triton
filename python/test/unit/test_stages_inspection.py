import triton

import os
import pathlib
import hashlib
import pytest
from triton._internal_testing import is_cuda
from triton._instrumentation import instrument, is_enabled, register_instrumentation, unregister_instrumentation


def test_instrumentation_mode():
    options = {"instrumentation_mode": "gsan,consan"}

    assert is_enabled(options, "consan")
    assert not is_enabled(options, "proton")


def test_dynamic_instrumentation_and_dialect_loading():
    calls = []

    assert not instrument("context", point="load-dialects", backend="test")
    register_instrumentation(point="load-dialects", backend="test", callback=lambda context: calls.append(
        ("dialects", context)))
    register_instrumentation(point="ttgpuir-to-llvmir", backend="test", callback=lambda _pm: calls.append("passes"))
    try:
        assert instrument("context", point="load-dialects", backend="test")
        assert instrument("pm", point="ttgpuir-to-llvmir", backend="test", context="module-context")
        assert calls == [("dialects", "context"), ("dialects", "module-context"), "passes"]
    finally:
        unregister_instrumentation(point="load-dialects", backend="test")
        unregister_instrumentation(point="ttgpuir-to-llvmir", backend="test")

    assert not instrument("context", point="load-dialects", backend="test")


@pytest.mark.xfail(not is_cuda(), reason="only currently tested on CUDA", run=False)
def test_inspection(monkeypatch, fresh_knobs, tmp_path: pathlib.Path):
    stage_name = 'make_ttgir'
    curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
    repro_path = tmp_path / "repro_prefix"

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    inspect_stages_hook_called = False
    make_ttgir_wrapper_called = False

    def get_key():
        return pathlib.Path(__file__).read_text()

    def get_hash():
        return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

    def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(arg is None for arg in (stages, options, language, capability)):
            return get_key(), get_hash()
        nonlocal inspect_stages_hook_called
        inspect_stages_hook_called = True

        def make_ttgir_wrapper(src, metadata, options, capability):
            nonlocal make_ttgir_wrapper_called
            make_ttgir_wrapper_called = True
            return self.make_ttgir(src, metadata, options, capability)

        stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

    @triton.jit
    def k1():
        return

    @triton.jit
    def k2():
        return

    # Run once to get the clean/golden repro dump
    k1[(1, )]()
    assert not inspect_stages_hook_called and not make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    golden_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # Setup hook and call again, check if hooks got called
    fresh_knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    k2[(1, )]()
    assert inspect_stages_hook_called and make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    hook_repro = curr_repro_path.read_text()

    # Check that repros match
    assert golden_repro.replace('k1', 'dummy') == hook_repro.replace('k2', 'dummy')


@pytest.mark.parametrize("stages,frontend_unroll,expected", [(1, 1, True), (2, 1, False), (1, 4, False)])
def test_intel_one_stage_dot_loop_hint(monkeypatch, stages, frontend_unroll, expected):
    from triton._C import libtriton
    if not hasattr(libtriton, "intel"):
        pytest.skip("Intel backend is required")
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", "bmg")
    target = GPUTarget("xpu", {
        "architecture": 21479031808, "arch": "bmg", "name": "Intel Arc B580",
        "driver_version": "1.17.39758+11", "gpu_eu_count": 160, "gpu_subslice_count": 20,
        "max_work_group_size": 1024, "max_num_sub_groups": 64, "sub_group_sizes": [16, 32],
        "has_2d_block_io": True, "has_bfloat16_conversion": True,
        "has_subgroup_matrix_multiply_accumulate": True, "local_mem_size": 131072,
        "max_grf_mode": "256", "__intel_already_queried_extensions__": True,
    }, 32)

    @triton.jit
    def dot_loop(A, B, C, UNROLL: tl.constexpr):
        a = tl.make_tensor_descriptor(A, (256, 1024), (1024, 1), (256, 32))
        b = tl.make_tensor_descriptor(B, (1024, 256), (256, 1), (32, 256))
        acc = tl.full((256, 256), 0, tl.float32)
        for k in tl.range(0, 1024, 32, loop_unroll_factor=UNROLL):
            acc = tl.dot(a.load([0, k]), b.load([k, 0]), acc)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 256)
        tl.store(C + rows[:, None] * 256 + cols[None, :], acc)

    source = ASTSource(dot_loop, {"A": "*bf16", "B": "*bf16", "C": "*fp32", "UNROLL": "constexpr"},
                       {"UNROLL": frontend_unroll},
                       {(i,): [("tt.divisibility", 16)] for i in range(3)})
    compiled = triton.compile(source, target=target, options={"num_warps": 32, "num_stages": stages,
                                                           "grf_mode": "256", "generate_native_code": False, "gemm_policy": "unconditional"})
    assert ("llvm.loop.unroll.count" in compiled.asm["llir"]) == expected


@pytest.mark.parametrize("policy,stages,kind,expected_cap,expected_unroll", [
    (None, 2, "plain", True, False),
    (None, 3, "exp", True, False),
    ("learned", 2, "mixed", False, False),
    ("reference", 2, "mixed", False, False),
    ("learned", 1, "mixed", False, False),
    ("baseline", 2, "plain", False, False),
    ("learned", 2, "plain", True, False),
    ("reference", 2, "plain", True, False),
    ("learned", 3, "plain", True, False),
    ("reference", 3, "plain", False, False),
    ("learned", 3, "exp", True, False),
    ("reference", 3, "exp", False, False),
    ("learned", 2, "transpose", False, False),
    ("learned", 2, "unknown_math", False, False),
    ("learned", 2, "b_math", False, False),
    ("learned", 2, "multiple", False, False),
    ("learned", 1, "plain", False, True),
    ("reference", 1, "plain", False, True),
    ("baseline", 1, "plain", False, False),
    ("learned", 1, "multiple", False, False),
    ("learned", 1, "pointer", False, False),
    ("learned", 2, "pointer", False, False),
    ("reference", 2, "pointer", False, False),
    ("learned", 2, "fp16", False, False),
    ("intervention", 1, "pointer", False, False),
    ("intervention", 1, "mixed", False, False),
    ("intervention", 1, "unknown_math", False, False),
    ("intervention", 1, "multiple", False, False),
    ("intervention", 2, "pointer", False, False),
    ("intervention", 2, "mixed", False, False),
    ("intervention", 2, "unknown_math", False, False),
    ("intervention", 2, "multiple", False, False),
    ("unconditional", 1, "pointer", False, False),
    ("unconditional", 1, "mixed", False, False),
    ("unconditional", 1, "unknown_math", False, False),
    ("unconditional", 1, "multiple", False, False),
    ("unconditional", 2, "pointer", False, False),
    ("unconditional", 2, "mixed", False, False),
    ("unconditional", 2, "unknown_math", False, False),
    ("unconditional", 2, "multiple", False, False),
])
def test_intel_gemm_policy_decisions(monkeypatch, policy, stages, kind,
                                    expected_cap, expected_unroll):
    import json
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource

    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", "bmg")
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "0")
    target = GPUTarget("xpu", {
        "architecture": 21479031808, "arch": "bmg", "name": "Intel Arc B580",
        "driver_version": "1.17.39758+11", "gpu_eu_count": 160, "gpu_subslice_count": 20,
        "max_work_group_size": 1024, "max_num_sub_groups": 64, "sub_group_sizes": [16, 32],
        "has_2d_block_io": True, "has_bfloat16_conversion": True,
        "has_subgroup_matrix_multiply_accumulate": True, "local_mem_size": 131072,
        "max_grf_mode": "256", "__intel_already_queried_extensions__": True,
    }, 32)

    @triton.jit
    def kernel(A, B, C, KIND: tl.constexpr):
        if KIND == "transpose":
            ad = tl.make_tensor_descriptor(A, (16384, 256), (256, 1), (32, 256))
        else:
            ad = tl.make_tensor_descriptor(A, (256, 16384), (16384, 1), (256, 32))
        bd = tl.make_tensor_descriptor(B, (16384, 256), (256, 1), (32, 256))
        acc = tl.full((256, 256), 0, tl.float32)
        for k in range(0, 16384, 32):
            if KIND == "pointer":
                rows = tl.arange(0, 256)
                ks = k + tl.arange(0, 32)
                cols = tl.arange(0, 256)
                a = tl.load(A + rows[:, None] * 16384 + ks[None, :])
                b = tl.load(B + ks[:, None] * 256 + cols[None, :])
            elif KIND == "transpose":
                a = ad.load([k, 0]).T
                b = bd.load([k, 0])
            else:
                a = ad.load([0, k])
                b = bd.load([k, 0])
            if KIND == "mixed":
                ks = k + tl.arange(0, 32)
                cols = tl.arange(0, 256)
                b = tl.load(B + ks[:, None] * 256 + cols[None, :])
            if KIND == "exp":
                a = tl.exp(a.to(tl.float32)).to(tl.bfloat16)
            if KIND == "unknown_math":
                a = tl.abs(a)
            if KIND == "b_math":
                b = tl.exp(b.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(a, b, acc)
            if KIND == "multiple":
                acc = tl.dot(a, b, acc)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 256)
        tl.store(C + rows[:, None] * 256 + cols[None, :], acc)

    dtype = "*fp16" if kind == "fp16" else "*bf16"
    src = ASTSource(kernel, {"A": dtype, "B": dtype, "C": "*fp32", "KIND": "constexpr"},
                    {"KIND": kind}, {(i,): [("tt.divisibility", 16)] for i in range(3)})
    options = dict(num_warps=32, num_stages=stages, grf_mode="256")
    if policy is not None:
        options["gemm_policy"] = policy
    compiled = triton.compile(src, target=target, options=options)
    assert ("repCluster = [2, 2]" in compiled.asm["ttgir"]) == expected_cap
    assert ('!"llvm.loop.unroll.count", i32 2' in compiled.asm["llir"]) == expected_unroll
    if policy != "baseline" and kind in {"plain", "exp", "transpose"}:
        assert "ttig.gemm_feature_vectors = [array<f64:" in compiled.asm["ttgir"]
    if kind in {"pointer", "mixed"} and policy != "baseline":
        assert "ttig.gemm_feature_vectors = []" in compiled.asm["ttgir"]


def test_intel_gemm_policy_cache_key_and_invalid_mode():
    from triton.backends.intel.compiler import XPUOptions
    options = [XPUOptions(gemm_policy=mode) for mode in
               ["baseline", "intervention", "reference", "learned"]]
    assert len({opt.hash() for opt in options}) == 4
    with pytest.raises(ValueError, match="Unknown experimental GEMM policy"):
        XPUOptions(gemm_policy="misspelled")


@pytest.mark.parametrize("kind,pad_a,pad_b,reduction,stages,unroll,policy,columns,cap", [('descriptor', 0, 0, 16384, 2, False, 'learned', 256, True), ('descriptor', 32, 0, 16384, 2, False, 'learned', 256, True), ('descriptor', 0, 32, 16384, 2, False, 'learned', 256, True), ('pointer', 0, 0, 16384, 2, False, 'learned', 256, False), ('pointer', 32, 32, 16384, 2, False, 'learned', 256, False), ('transpose', 32, 0, 16384, 2, False, 'learned', 256, False), ('descriptor', 0, 0, 8320, 1, False, 'learned', 256, False), ('descriptor', 32, 0, 8320, 1, False, 'learned', 256, False), ('descriptor', 64, 0, 8320, 1, False, 'learned', 256, False), ('pointer', 0, 32, 12352, 1, False, 'learned', 256, False), ('transpose', 32, 0, 16384, 1, True, 'learned', 256, False), ('descriptor', 0, 32, 11136, 2, False, 'reference', 256, False), ('descriptor', 0, 64, 11136, 2, False, 'reference', 256, True), ('descriptor', 0, 32, 12288, 2, False, 'reference', 256, True), ('descriptor', 32, 0, 8320, 1, False, 'learned', 8192, False), ('descriptor', 32, 0, 18560, 1, False, 'learned', 8192, False), ('descriptor', 32, 0, 8320, 1, False, 'reference', 8192, False), ('exp', 0, 32, 11136, 2, False, 'reference', 256, False)])
def test_intel_gemm_alignment_features(monkeypatch, kind, pad_a, pad_b, reduction, stages, unroll, policy, columns, cap):
    import re
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", "bmg")
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "0")
    target = GPUTarget("xpu", {
        "architecture": 21479031808, "arch": "bmg", "name": "Intel Arc B580",
        "driver_version": "1.17.39758+11", "gpu_eu_count": 160, "gpu_subslice_count": 20,
        "max_work_group_size": 1024, "max_num_sub_groups": 64, "sub_group_sizes": [16, 32],
        "has_2d_block_io": True, "has_bfloat16_conversion": True,
        "has_subgroup_matrix_multiply_accumulate": True, "local_mem_size": 131072,
        "max_grf_mode": "256", "__intel_already_queried_extensions__": True,
    }, 32)

    @triton.jit
    def kernel(A, B, C, KIND: tl.constexpr, PA: tl.constexpr, PB: tl.constexpr, KK: tl.constexpr, NN: tl.constexpr):
        if KIND == "transpose":
            ad = tl.make_tensor_descriptor(A, (KK, 256), (256 + PA, 1), (32, 256))
        else:
            ad = tl.make_tensor_descriptor(A, (256, KK), (KK + PA, 1), (256, 32))
        bd = tl.make_tensor_descriptor(B, (KK, NN), (NN + PB, 1), (32, 256))
        acc = tl.full((256, 256), 0, tl.float32)
        for k in range(0, KK, 32):
            if KIND == "pointer":
                rows = tl.arange(0, 256)
                ks = k + tl.arange(0, 32)
                cols = tl.arange(0, 256)
                a = tl.load(A + rows[:, None] * (KK + PA) + ks[None, :])
                b = tl.load(B + ks[:, None] * (NN + PB) + cols[None, :])
            elif KIND == "transpose":
                a = ad.load([k, 0]).T
                b = bd.load([k, 0])
            else:
                a = ad.load([0, k])
                b = bd.load([k, 0])
            if KIND == "exp":
                a = tl.exp(a.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(a, b, acc)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 256)
        tl.store(C + rows[:, None] * 256 + cols[None, :], acc)

    src = ASTSource(kernel, {"A": "*bf16", "B": "*bf16", "C": "*fp32", "KIND": "constexpr", "PA": "constexpr", "PB": "constexpr", "KK": "constexpr", "NN": "constexpr"},
                    {"KIND": kind, "PA": pad_a, "PB": pad_b, "KK": reduction, "NN": columns}, {(i,): [("tt.divisibility", 16)] for i in range(3)})
    compiled = triton.compile(src, target=target,
                              options=dict(num_warps=32, num_stages=stages, grf_mode="256", gemm_policy=policy))
    text = compiled.asm["ttgir"]
    if kind == "pointer":
        assert "ttig.gemm_feature_vectors = []" in text
        assert "repCluster = [2, 2]" not in text
        assert 'llvm.loop.unroll.count' not in compiled.asm["llir"]
        return
    feature_text = re.search(r'ttig\.gemm_feature_vectors = \[([^\]]*)\]', text)[1]
    features = list(map(float, re.search(r'array<f64: ([^>]+)>', feature_text)[1].split(',')))
    pitch_text = re.search(r'ttig\.gemm_pitch_bytes = \[([^\]]*)\]', text)[1]
    pitches = list(map(int, re.search(r'array<i64: ([^>]+)>', pitch_text)[1].split(',')))
    expected_a = 2 * ((256 if kind == "transpose" else reduction) + pad_a)
    expected_b = 2 * (columns + pad_b)
    assert pitches == [expected_a, expected_b]
    assert len(features) == 13
    assert features[-2:] == [expected_a & -expected_a, expected_b & -expected_b]
    assert features[6] == (kind != "pointer")
    assert features[7] == (kind != "transpose")

    assert ('!"llvm.loop.unroll.count", i32 2' in compiled.asm["llir"]) == unroll

    if cap is not None:
        assert ("repCluster = [2, 2]" in text) == cap

    assert features[8] == (kind == "exp")


@pytest.mark.parametrize("policy,stages", [("learned", 1), ("learned", 2),
                                          ("reference", 1), ("reference", 2), ("intervention", 1), ("intervention", 2),
                                          ("unconditional", 1), ("unconditional", 2)])
def test_intel_gemm_unknown_pitch_falls_back(monkeypatch, policy, stages):
    import re
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource
    monkeypatch.setenv("TRITON_INTEL_DEVICE_ARCH", "bmg")
    monkeypatch.setenv("TRITON_XPU_GEN_NATIVE_CODE", "0")
    target = GPUTarget("xpu", {
        "architecture": 21479031808, "arch": "bmg", "name": "Intel Arc B580",
        "driver_version": "1.17.39758+11", "gpu_eu_count": 160, "gpu_subslice_count": 20,
        "max_work_group_size": 1024, "max_num_sub_groups": 64, "sub_group_sizes": [16, 32],
        "has_2d_block_io": True, "has_bfloat16_conversion": True,
        "has_subgroup_matrix_multiply_accumulate": True, "local_mem_size": 131072,
        "max_grf_mode": "256", "__intel_already_queried_extensions__": True,
    }, 32)

    @triton.jit
    def kernel(A, B, C, S):
        ad = tl.make_tensor_descriptor(A, (256, 16384), (S, 1), (256, 32))
        bd = tl.make_tensor_descriptor(B, (16384, 256), (256, 1), (32, 256))
        acc = tl.full((256, 256), 0, tl.float32)
        for k in range(0, 16384, 32):
            acc = tl.dot(ad.load([0, k]), bd.load([k, 0]), acc)
        rows = tl.arange(0, 256)
        cols = tl.arange(0, 256)
        tl.store(C + rows[:, None] * 256 + cols[None, :], acc)

    src = ASTSource(kernel, {"A": "*bf16", "B": "*bf16", "C": "*fp32", "S": "i64"}, {},
                    {(i,): [("tt.divisibility", 16)] for i in range(3)})
    compiled = triton.compile(src, target=target,
                              options=dict(num_warps=32, num_stages=stages,
                                           grf_mode="256", gemm_policy=policy))
    snapshot = re.search(r'ttig\.gemm_pitch_bytes = \[([^\]]*)\]', compiled.asm["ttgir"])[1]
    pitches = list(map(int, re.search(r'array<i64: ([^>]+)>', snapshot)[1].split(',')))
    assert pitches == [0, 512]
    assert "repCluster = [2, 2]" not in compiled.asm["ttgir"]
    assert 'llvm.loop.unroll.count' not in compiled.asm["llir"]
