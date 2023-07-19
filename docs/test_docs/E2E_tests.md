- [Overview](#overview)
- [Pre-Request](#pre-request)
- [Use Hugging Face model](#use-hugging-face-model)
  - [TL;DR](#tldr)
- [Detail for commands](#detail-for-commands)
    - [Debugging Tips](#debugging-tips)
  - [Profiling](#profiling)


# Overview
This doc contains [Torchdynamo Benchmarks](https://github.com/pytorch/pytorch/tree/main/benchmarks/dynamo) setup for Intel® XPU Backend for Triton\*.

The Benchmark contains different suites and shares as a common frontend usage. This doc below is an example showing [Hugging Face](https://huggingface.co/) End-to-End models for triton.

# Pre-Request
The PyTorch version should be the same as the one in [installation guide for intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#installation-guide) .


# Use Hugging Face model

## TL;DR
PyTorch benchmark will automatically download necessary dependencies.

Simply run the model using the following sh file. Note that there are some tricks for debugging. It is recommended to refer to [Debugging Tips](#debugging-tips).



First create a sh file called `single_run.sh` under pytorch source folder, then run using the following `sh` file with the command:

```
./single_run.sh huggingface AlbertForMaskedLM float32 training accuracy xpu 0
```

```Bash
#! /bin/bash
# This script work for xpu / cuda device inductor tests

SUITE=${1:-huggingface}     # huggingface / torchbench / timm_models
MODEL=${2:-AlbertForMaskedLM}
DT=${3:-float32}            # float32 / float16 / amp
MODE=${4:-inference}        # inference / training
SCENARIO=${5:-accuracy}     # accuracy / performance
DEVICE=${6:-xpu}            # xpu / cuda
CARD=${7:-0}                # 0 / 1 / 2 / 3 ...
SHAPE=${8:-static}          # static / dynamic

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

Shape_extra=""
if [[ $SHAPE == "dynamic" ]]; then
    echo "Testing with dynamic shapes."
    Shape_extra="--dynamic-shapes --dynamic-batch-only "
fi

ulimit -n 1048576

ZE_AFFINITY_MASK=${CARD} python benchmarks/dynamo/${SUITE}.py --only ${MODEL} --${SCENARIO} --${DT} -d${DEVICE} -n50 --no-skip --dashboard ${Mode_extra} ${Shape_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv &> ${LOG_DIR}/${LOG_NAME}.log
cat ${LOG_DIR}/${LOG_NAME}.csv

```

The log file is under `inductor_log/${SUITE}/${MODEL}/${DT}/${LOG_NAME}.log`.

If you need to do a batch run for the whole suite, delete the `--only ${MODEL}` from the above script and rerun.

# Detail for commands

Below are the detail for those who interested for more fine-grained control.

Normally, the command will be like the following:

```Python
# The following code could not run, need modification
python benchmarks/dynamo/${SUITE}.py --only ${MODEL} --accuracy --amp -dxpu -n50 --no-skip --dashboard ${Mode_extra} ${Shape_extra} --backend=inductor --timeout=4800 --output=${LOG_DIR}/${LOG_NAME}.csv
```
The full arg lists could be found with the following command:

```Python
python benchmarks/dynamo/huggingface.py --help
```

In addition to the argument, there are config in Python code to control the behavior:


Please go to [torch._dynamo.config](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py) and [torch._inductor.config](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) to find all configs.

One example of using the config is in [Debugging Tips](#debugging-tips). Please set the config according to your need.

### Debugging Tips

It is recommended to set the following environment variables for debugging:

- `TORCHINDUCTOR_CACHE_DIR={some-dir}`: Set this for where torchinductor cache is put.
- `TRITON_CACHE_DIR={some-dir}`: Where the triton cache is. By default it is under `TORCHINDUCTOR_CACHE_DIR/triton` folder.
- `TORCH_COMPILE_DEBUG_DIR={some-dir}`: Where the compile debug files be put. You could see folders like `aot_torchinductor` containing the torchinductor logs, and `torchdynamo` folder containing dynamo log.
- `TORCH_COMPILE_DEBUG=1`: Detailed for TorchInductor Tracing. It will print a lot of messages. Thus it is recommended to re-direct output to file. By setting this flag, the re-producible python file could be easily find.


Alternatively, the above env flag could also be set in Python file like below, these three configurations could help to generate more readable kernel names.

```Python
# helps to generate descriptive kernel names
torch._inductor.config.triton.ordered_kernel_names = True
torch._inductor.config.triton.descriptive_kernel_names = True
torch._inductor.config.kernel_name_max_ops = 8
```

**Reproducing Errors with Smaller Python File**

Re-running from the overall model is quit a burden, you could try to re-produce the error using smaller python file.
To reproduce the result, one could set the flag `TORCH_COMPILE_DEBUG=1`. Then the graph will be printed. Note that there are a lot of outputs, one could direct the output to file.

```Bash
TORCH_COMPILE_DEBUG=1 python ... &> test.log
```

For now, we need to go into the output log to find where the reproduce code is. By looking at the above output, there are some lines like below:

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

You could open that python file, import `intel_extension_for_pytorch`, and then run the python file as normal.


In the future, you could use minifer to produce the above, by enabling the following flags:

```Python
+torch._dynamo.config.repro_after="dynamo"

```


## Profiling

To profile the result, one should use the `performance` mode instead of `accuracy`. i.e, One should use

```
python benchmarks/dynamo/${SUITE}.py  ... --performance ...
```
