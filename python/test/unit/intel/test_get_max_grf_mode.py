"""Unit test for `get_max_grf_mode()`'s "cri" vs. everything-else GRF policy.

https://github.com/intel/intel-xpu-backend-for-triton/issues/8074

`get_max_grf_mode()` is the single source of truth `parse_target` calls to
populate `dev_prop['max_grf_mode']`, from which every consumer (the ocloc
auto-large-GRF retry, `driver.c`'s JIT retry, and the `ttig.max_grf_mode`
module attribute read by `RegisterPressureAnalysis`) reaches it as
`opt.max_grf_mode` rather than re-deriving it. A driver- or out-of-tree
arch-module-supplied override participates via `parse_target`'s own
`tgt_prop.get('max_grf_mode', get_max_grf_mode(tgt_prop))` call -- the same
`tgt_prop.get(key, default)` idiom every other per-target capability in that
function uses, so it is not this function's own concern and has no dedicated
test here, matching its siblings. Before this test existed, nothing pinned
the actual policy ("512" on "cri", "256" everywhere else): the one hardcoded
assertion of "256" elsewhere in the suite (`test_native_code_generation.py`)
is `@pytest.mark.xfail(is_xpu_cri())`, i.e. inert on exactly the target this
policy treats differently.

Pure `dict -> str`, no device access, so this needs no XPU and no `device`
fixture.
"""
from triton.backends.intel.compiler import get_max_grf_mode


def test_get_max_grf_mode_cri():
    assert get_max_grf_mode({"arch": "cri"}) == "512"


def test_get_max_grf_mode_non_cri():
    assert get_max_grf_mode({"arch": "bmg"}) == "256"
    assert get_max_grf_mode({"arch": "pvc"}) == "256"
