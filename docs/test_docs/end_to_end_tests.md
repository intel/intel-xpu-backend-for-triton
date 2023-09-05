- [Overview](#overview)
- [Pre-Request](#pre-request)
- [Use the Hugging Face model](#use-the-hugging-face-model)
  - [TL;DR](#tldr)
- [Detail for commands](#detail-for-commands)
    - [Debugging Tips](#debugging-tips)
  - [Profiling](#profiling)
    - [End-to-end Tests Setting:](#end-to-end-tests-setting)
      - [Profiling Settings](#profiling-settings)
      - [Profiling Tips](#profiling-tips)


# Overview
This doc contains [Torchdynamo Benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo) setup for XPU Backend for Triton\*.

The Benchmark contains different suites and shares as a common frontend usage. This doc below is an example showing [Hugging Face\*](https://huggingface.co/) End-to-End models for triton.

# Pre-Request
The PyTorch version should be the same as the one in [installation guide for intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#installation-guide).


# Use the Hugging Face model

## TL;DR
PyTorch benchmark will automatically download necessary dependencies.

Simply run the model using the following sh file. Note that there are some tricks for debugging. It is recommended to refer to [Debugging Tips](#debugging-tips).



First, copy the sh file  [intel_xpu_backend/.github/scripts/inductor_xpu_test.sh](../../.github/scripts/inductor_xpu_test.sh) to the PyTorch source folder, then run the `sh` file with the command:

```Bash
# Run all models
bash xpu_run_batch.sh huggingface amp_bf16 training performance  xpu 0 static 2 1

# Run single model `T5Small`
bash xpu_run_batch.sh huggingface amp_bf16 training performance  xpu 0 static 2 1 T5Small
```

There are also useful env flag, for example:
- `TORCHINDUCTOR_CACHE_DIR={some_DIR}`: Where the cache files are put. It is useful when debugging.
- `TORCH_COMPILE_DEBUG=1`: Whether print debug info.
- `TRITON_XPU_PROFILE=ON`: Show XPU triton kernels for debug.

By default, the cache dir is under `/tmp/torchinductor_{user}/`, it is recommended to change the cache dir to a new place when you are debugging. For example,

```Bash
LOG_DIR=${WORKSPACE}/inductor_log/${SUITE}/${MODEL}/${DT}
mkdir -p ${LOG_DIR}
export TORCHINDUCTOR_CACHE_DIR=${LOG_DIR}

```


# Detail for commands

Below is the detail for those who are interested in more fine-grained control.

Normally, the command will be like the following:

```Bash
python benchmarks/dynamo/${SUITE}.py --only ${MODEL} --accuracy --amp -dxpu -n50 --no-skip --dashboard ${Mode_extra}  --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv
```
The full arg lists could be found with the following command:

```Bash
python benchmarks/dynamo/huggingface.py --help
```

In addition to the argument, there are configs in Python code to control the behavior:


Please go to [torch._dynamo.config](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py) and [torch._inductor.config](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) to find all configs.

One example of using the config is in [Debugging Tips](#debugging-tips). Please set the config according to your need.

### Debugging Tips

It is recommended to set the following environment variables for debugging:

- `TORCHINDUCTOR_CACHE_DIR={some-dir}`: Set this for where torchinductor cache is put.
- `TRITON_CACHE_DIR={some-dir}`: Where the triton cache is. By default, it is under the `TORCHINDUCTOR_CACHE_DIR/triton` folder.
- `TORCH_COMPILE_DEBUG_DIR={some-dir}`: Where the compile debug files be put. You could see folders like `aot_torchinductor` containing the torchinductor logs, and `torchdynamo` folder containing the dynamo log.
- `TORCH_COMPILE_DEBUG=1`: Detailed for TorchInductor Tracing. It will print a lot of messages. Thus it is recommended to redirect the output to the file. By setting this flag, the re-producible Python file could be easily found.


Alternatively, the above env flag could also be set in a Python file like below, these three configurations could help to generate more readable kernel names.

```Python
# helps to generate descriptive kernel names
torch._inductor.config.triton.ordered_kernel_names = True
torch._inductor.config.triton.descriptive_kernel_names = True
torch._inductor.config.kernel_name_max_ops = 8
```

**Reproducing Errors with Smaller Python File**

Re-running from the overall model is quite a burden, you could try to reproduce the error using a smaller Python file.
To reproduce the result, one could set the flag `TORCH_COMPILE_DEBUG=1`. Then the graph will be printed. Note that there are a lot of outputs, one could direct the output to a file.

```Bash
TORCH_COMPILE_DEBUG=1 python ... &> test.log
```

For now, we need to go into the output log to find where the reproduced code is. By looking at the above output, there are some lines like below:

```Bash
torch._inductor.debug: [WARNING] GoogleFnet__3_inference_3 debug trace: /tmp/torchinductor_username/rc/dlkmcaknezrsmfxw5emr4pdy5qtny47pozz5wihpvwhsi7x3elg.debug
```
In this folder, you could find the file structure like below:

```
.
├── cdlkmcaknezrsmfxw5emr4pdy5qtny47pozz5wihpvwhsi7x3elg.debug
│   ├── debug.log
│   ├── fx_graph_readable.py
│   ├── fx_graph_runnable.py
│   ├── fx_graph_transformed.py
│   ├── ir_post_fusion.txt
│   ├── ir_pre_fusion.txt
│   └── output_code.py
└── cdlkmcaknezrsmfxw5emr4pdy5qtny47pozz5wihpvwhsi7x3elg.py
```

The `cdlkmcaknezrsmfxw5emr4pdy5qtny47pozz5wihpvwhsi7x3elg.py` contains the runnable file that we need.

You could open that Python file, import `intel_extension_for_pytorch`, and then run the Python file as normal.


In the future, you could use minifer to produce the above, by enabling the following flags:

```Python
torch._dynamo.config.repro_after="dynamo"
```


## Profiling

To profile the result, one should use the `performance` mode instead of `accuracy`. i.e, One should use

```Bash
python benchmarks/dynamo/${SUITE}.py  ... --performance ...
```

For now, we use the [profiler_legacy](https://github.com/intel/intel-extension-for-pytorch/blob/xpu-master/docs/tutorials/features/profiler_legacy.md) to catch the profiling result.

A typical profiling code would look like below:

```Python
# import all necessary libraries
import torch
import intel_extension_for_pytorch

# these lines won't be profiled before enabling profiler tool
input_tensor = torch.randn(1024, dtype=torch.float32, device='xpu:0')

# enable legacy profiler tool with a `with` statement
with torch.autograd.profiler_legacy.profile(use_xpu=True) as prof:
    # do what you want to profile here after the `with` statement with proper indent
    output_tensor_1 = torch.nonzero(input_tensor)
    output_tensor_2 = torch.unique(input_tensor)

# print the result table formatted by the legacy profiler tool as your wish
print(prof.key_averages().table(sort_by="self_xpu_time_total"))
```
### End-to-end Tests Setting:

#### Profiling Settings

For E2E tests, there are several places to change. You should cd to `pytorch/benchmarks/dynamo` and change the `common.py` as below. Note that the line number may not be the same, but the change places are unique.

```diff
@@ -530,7 +536,7 @@ def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
     @contextlib.contextmanager
     def maybe_profile(*args, **kwargs):
         if kwargs.pop("enabled", True):
-            with torch.profiler.profile(*args, **kwargs) as p:
+            with torch.autograd.profiler_legacy.profile(enabled=True, use_xpu=True, *args, **kwargs) as p:
                 yield p
         else:
             yield
@@ -540,7 +546,7 @@ def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwa
rgs):
         prof: torch.profiler.profile = kwargs.pop("p", None)
         mark = kwargs.pop("mark", None)
         if prof:
-            with torch.profiler.record_function(mark):
+            with torch.autograd.profiler.record_function(mark):
                 yield
         else:
             yield
```
#### Profiling Tips

To run the model, you should add the `--export-profiler-trace` flag when running. Because use the profiling process will link libtorch, this will greatly reduce the kernel compiling time. It is highly recommended to **run twice** for quicker result:

1. On the first run, run the model **without** `--export-profiler-trace` flag. This will generate necessary caches.
2. On the second run, run with `--export-profiler-trace` flag. This will actually do the profiling result.


If you wish to make kernel name more readable, you could enable with the following config:

```Python
# common.py
torch._inductor.config.triton.ordered_kernel_names = True
torch._inductor.config.triton.descriptive_kernel_names = True
torch._inductor.config.kernel_name_max_ops = 8
```

The chrome trace file by default will export to `torch._dynamo.config.base_dir`, you could control this process by setting `torch._dynamo.config.base_dir` to the folder you want.


One example of the result shown as below:

```Log
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg      Self XPU    Self XPU %     XPU total  XPU time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                               aten::mm        17.10%      xx.yyy ms        17.35%    xx.yyy ms    xx.yyy us    xx.yyy ms        25.33%     xx.yyy ms     xx.yyy  us          1940
XPU Triton kernel:triton_fused__unsafe_view__unsafe_...         0.35%      xx.yyy ms         0.35%    xx.yyy ms    xx.yyy us    xx.yyy ms        18.07%     xx.yyy ms     xx.yyy  us           120
XPU Triton kernel:triton_fused__unsafe_view_18__unsa...         0.35%      xx.yyy ms         0.35%    xx.yyy ms    xx.yyy us    xx.yyy ms        17.31%     xx.yyy ms     xx.yyy  us           120
                                              aten::bmm         5.99%      xx.yyy ms         6.06%    xx.yyy ms    xx.yyy us    xx.yyy ms        10.99%     xx.yyy ms     xx.yyy  us           720
XPU Triton kernel:triton_fused__unsafe_view_18__unsa...         0.40%      xx.yyy ms         0.40%    xx.yyy ms    xx.yyy us    xx.yyy ms         7.89%     xx.yyy ms     xx.yyy  us           120
XPU Triton kernel:triton_fused__unsafe_view__unsafe_...         0.46%      xx.yyy ms         0.46%    xx.yyy ms    xx.yyy us    xx.yyy ms         3.07%     xx.yyy ms     xx.yyy  us           240
XPU Triton kernel:triton_fused__unsafe_view_18__unsa...         0.07%      xx.yyy us         0.07%    xx.yyy us    xx.yyy us    xx.yyy ms         2.71%     xx.yyy ms     xx.yyy  us            20
XPU Triton kernel:triton_fused_convert_element_type_...         3.39%      xx.yyy ms         3.39%    xx.yyy ms    xx.yyy us    xx.yyy ms         2.70%     xx.yyy ms     xx.yyy  us          1440
XPU Triton kernel:triton_fused_add_clone_convert_ele...         1.40%      xx.yyy ms         1.40%    xx.yyy ms    xx.yyy us    xx.yyy ms         2.48%     xx.yyy ms     xx.yyy  us           720
...
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: xxx.yyyms
Self XPU time total: xxx.yyyms

```
