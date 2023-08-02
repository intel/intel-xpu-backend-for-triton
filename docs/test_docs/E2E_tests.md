- [Overview](#overview)
- [Pre-Request](#pre-request)
- [Use the Hugging Face model](#use-the-hugging-face-model)
  - [TL;DR](#tldr)
- [Detail for commands](#detail-for-commands)
    - [Debugging Tips](#debugging-tips)
  - [Profiling](#profiling)
    - [E2E Tests Setting:](#e2e-tests-setting)
      - [Profiling Settings](#profiling-settings)
      - [Profiling Tips](#profiling-tips)


# Overview
This doc contains [Torchdynamo Benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo) setup for Intel® XPU Backend for Triton\*.

The Benchmark contains different suites and shares as a common frontend usage. This doc below is an example showing [Hugging Face](https://huggingface.co/) End-to-End models for triton.

# Pre-Request
The PyTorch version should be the same as the one in [installation guide for intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#installation-guide).


# Use the Hugging Face model

## TL;DR
PyTorch benchmark will automatically download necessary dependencies.

Simply run the model using the following sh file. Note that there are some tricks for debugging. It is recommended to refer to [Debugging Tips](#debugging-tips).



First, create a sh file called `single_run.sh` under the PyTorch source folder, then run using the following `sh` file with the command:

```
./single_run.sh huggingface AlbertForMaskedLM float32 training accuracy xpu 0
```

```Bash
#! /bin/bash
# This script works for xpu / cuda device inductor tests

SUITE=${1:-huggingface}     # huggingface / torchbench / timm_models
MODEL=${2:-AlbertForMaskedLM}
DT=${3:-float32}            # float32 / float16 / amp
MODE=${4:-inference}        # inference / training
SCENARIO=${5:-accuracy}     # accuracy / performance
DEVICE=${6:-xpu}            # xpu / cuda
CARD=${7:-0}                # 0 / 1 / 2 / 3 ...

WORKSPACE=`pwd`
LOG_DIR=${WORKSPACE}/inductor_log/${SUITE}/${MODEL}/${DT}
mkdir -p ${LOG_DIR}
LOG_NAME=inductor_${SUITE}_${MODEL}_${DT}_${MODE}_${DEVICE}_${SCENARIO}
export TORCHINDUCTOR_CACHE_DIR=${LOG_DIR}

Mode_extra=""
if [[ $MODE == "training" ]]; then
    echo "Testing with training mode."
    Mode_extra="--training "
fi

ulimit -n 1048576

ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --only ${MODEL} --${SCENARIO} --${DT} -d${DEVICE} -n50 --no-skip --dashboard ${Mode_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv &> ${LOG_DIR}/${LOG_NAME}.log
cat ${LOG_DIR}/${LOG_NAME}.csv

```

The log file is under `inductor_log/${SUITE}/${MODEL}/${DT}/${LOG_NAME}.log`.

If you need to do a batch run for the whole suite, delete the `--only ${MODEL}` from the above script and rerun.

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
orch._inductor.debug: [WARNING] GoogleFnet__3_inference_3 debug trace: /tmp/torchinductor_username/rc/dlkmcaknezrsmfxw5emr4pdy5qtny47pozz5wihpvwhsi7x3elg.debug
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
+torch._dynamo.config.repro_after="dynamo"

```


## Profiling

To profile the result, one should use the `performance` mode instead of `accuracy`. i.e, One should use

```
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
### E2E Tests Setting:

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
                                               aten::mm        17.10%      74.087ms        17.35%      75.140ms      38.732us      98.681ms        25.33%      98.681ms      50.867us          1940
XPU Triton kernel:triton_fused__unsafe_view__unsafe_...         0.35%       1.499ms         0.35%       1.499ms      12.490us      70.400ms        18.07%      70.400ms     586.668us           120
XPU Triton kernel:triton_fused__unsafe_view_18__unsa...         0.35%       1.520ms         0.35%       1.520ms      12.670us      67.464ms        17.31%      67.464ms     562.201us           120
                                              aten::bmm         5.99%      25.969ms         6.06%      26.266ms      36.481us      42.810ms        10.99%      42.810ms      59.459us           720
XPU Triton kernel:triton_fused__unsafe_view_18__unsa...         0.40%       1.720ms         0.40%       1.720ms      14.336us      30.744ms         7.89%      30.744ms     256.201us           120
XPU Triton kernel:triton_fused__unsafe_view__unsafe_...         0.46%       2.008ms         0.46%       2.008ms       8.367us      11.961ms         3.07%      11.961ms      49.836us           240
XPU Triton kernel:triton_fused__unsafe_view_18__unsa...         0.07%     319.170us         0.07%     319.170us      15.959us      10.550ms         2.71%      10.550ms     527.504us            20
XPU Triton kernel:triton_fused_convert_element_type_...         3.39%      14.700ms         3.39%      14.700ms      10.208us      10.537ms         2.70%      10.537ms       7.317us          1440
XPU Triton kernel:triton_fused_add_clone_convert_ele...         1.40%       6.072ms         1.40%       6.072ms       8.433us       9.650ms         2.48%       9.650ms      13.402us           720
...
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 433.194ms
Self XPU time total: 389.655ms

```
