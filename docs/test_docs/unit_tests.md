
# Dependencies
1. All components listed in [SetUp](../setup.md) wiki page.

2. Tests related

* `pytest`
* `pytest-xdist` : (Optional Plugin) If you wish to run in parallel. A known limitation is that this won't capture the `-s` flag of `pytest`, so if you wish to debug and capture the output, it is not recommended to use this.


```Bash
pip install pytest
# Optional
pip install pytest-xdist
```


# Triton Unit Tests
This test case lies under `triton/python/tests/unit`.
[Note: ] The `triton/python/tests/regression` is not supported yet.


**Important**
1. If you are running tests including math lib, you should set the following flag to the triton lib. If you are running CI/Nightly tests, it is recommended to set this before a full coverage test.
```
export TRITON_LIBDEVICE_PATH={abs-path-to-triton}/python/triton/third_party/xpu/
```

2. To run the tests, one need to manually add the `import intel_extension_for_pytorch` before using xpu.



**Run single test**
```
# Please make sure added `import intel_extension_for_pytorch` to the test_file.py first.
(triton-tests)$
pytest -sv -k [your pattern(eg. test_dot)] test_file.py --device=xpu
```

**Run all tests**
```
# Please make sure added `import intel_extension_for_pytorch` to the test_file.py first.
(triton-tests)$
pytest -v test_file.py --device=xpu
```

Note that the `test_file.py` could be `.`, meaning running all tests on the current folder.

## Optional
**Run tests in parallel**
You need install `pytest-xdist` first.
```
pytest -n 8 ... (The same with before)
```

# Torch Inductor Tests

This test case lies under `pytorch/tests/inductor`. We are on an active developing process, some changes are internally are going to be upstreamed to PyTorch shortly. We will update the docs then.


# Appendix
## Skipping tests

There are two way to skip:

1. Skip the whole test function by using a decorator:

```Python
@pytest.mark.skip("skip Reason")
# Or use @pytest.mark.skipIf("skip Reason")
def test_my_func():
```
2. Skip inside function

```Python
def test_my_func():
    pytest.skip("Skip Reason:")
    # OR use skipIf
```
