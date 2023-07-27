- [Dependencies](#dependencies)
- [Triton Unit Tests](#triton-unit-tests)
  - [Optional](#optional)
- [Torch Inductor Tests](#torch-inductor-tests)
- [Appendix](#appendix)
  - [Skipping tests](#skipping-tests)


# Dependencies

1. Tests related

* `pytest`
* `pytest-xdist` : (Optional Plugin) If you wish to run in parallel. A known limitation is that this won't capture the `-s` flag of `pytest`, so if you wish to debug and capture the output, it is not recommended to use this.


```Bash
pip install pytest
# Optional
pip install pytest-xdist
```


# Triton Unit Tests
This test case lies under `triton/python/tests/unit/language`.

[Note: ] The other tests `triton/python/tests/` are not ported support yet.


**Important**
1. If you are running tests including math lib, you should set the following flag to the triton lib. It is recommended to set this before a full coverage test.
    ```
    export TRITON_LIBDEVICE_PATH={abs-path-to-triton}/python/triton/third_party/xpu/
    ```

1. To run the tests, one needs to manually add the `import intel_extension_for_pytorch` before using xpu.
2. If there are cases using `triton.compile`, one needs to use `kernel = triton.compile(name, device_type='xpu')` instead of just using `triton.compile(name)`.


**Run single test**
```
# Please make sure added `import intel_extension_for_pytorch` to the test_file.py first.
(tests/unit/language)$
pytest -sv -k [your pattern(eg. test_bin_op)] test_file.py --device=xpu
```

The `-s` flag will capture all the outputs. If you don't use it, then just run without this flag.


**Run all tests**
```
# Please make sure added `import intel_extension_for_pytorch` to the test_file.py first.
(tests/unit/language)$
pytest -v test_file.py --device=xpu
```

Note that the `test_file.py` could be `.`, meaning running all tests on the current folder.

## Optional
**Run tests in parallel**

You need to install `pytest-xdist` first.
```
pytest -n 8 ... (The same as before)
```

# Torch Inductor Tests

This test case lies under `pytorch/tests/inductor`. We are in an active development process, some changes are internally going to be upstreamed to PyTorch shortly. We will update the docs then.


# Appendix
## Skipping tests

There are two ways to skip:

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
