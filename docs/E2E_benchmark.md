- [Pre-Request](#pre-request)
  - [Workaround for PyTorch](#workaround-for-pytorch)
- [Use Hugging Face model](#use-hugging-face-model)
  - [TL;DR](#tldr)
  - [Detail for commands](#detail-for-commands)
    - [Profiling](#profiling)
    - [Debugging Tips](#debugging-tips)
    - [Full config lists](#full-config-lists)


# Pre-Request
In addition to packages used for triton, `pandas` are the only one needed:

```Bash
pip install pandas
```
## Workaround for PyTorch

You need to add the following changes for PyTorch:
https://github.com/intel-innersource/frameworks.ai.pytorch.private-gpu/pull/136

We will update the BKC when the PR is merged (or upstreamed).

Then build pytorch as former BKC.



# Use Hugging Face model

## TL;DR
PyTorch benchmark will automatically download necessary dependencies.

Apply following changes to open debug flags and use the sh file to run. Note you need to change the `torch._dynamo.config.base_dir` to anywhere you want.

First Enable the inductor config. The following make the kernel name more readable.

```diff
diff --git a/benchmarks/dynamo/common.py b/benchmarks/dynamo/common.py
index 1fbd012d82..63eb1f9ba6 100644
--- a/benchmarks/dynamo/common.py
+++ b/benchmarks/dynamo/common.py
@@ -50,6 +50,11 @@ except ImportError:
     pass
 
 log = logging.getLogger(__name__)
+torch._dynamo.config.base_dir = os.path.join("/home/user/pytorch/model_run", os.environ["BASE_DIR"] )
+# torch._dynamo.config.repro_after="dynamo"
+torch._inductor.config.triton.ordered_kernel_names = True
+torch._inductor.config.triton.descriptive_kernel_names = True
+torch._inductor.config.kernel_name_max_ops = 8
```

Then run using the following `sh` file.


```Bash
declare -a test_cases=("BlenderbotSmallForCausalLM"
                        "YituTechConvBert" 
                        "MobileBertForMaskedLM" 
                        "MobileBertForQuestionAnswering" 
                        "T5Small" 
                        "PegasusForConditionalGeneration")


mv model_run model_run_old
mkdir model_run

for model in "${test_cases[@]}"
do


    SUITE=${1:-huggingface}
    MODEL=${2:-$model}
    # MODEL=${2:-GoogleFnet}
    DT=${3:-float32}
    CHANNELS=${4:-first}
    SHAPE=${5:-static}
    BS=${6:-0}

    Shape_extra=""
    if [[ $SHAPE == "dynamic" ]]; then
        echo "Testing with dynamic shapes."
        Shape_extra="--dynamic-shapes "
    fi

    Channels_extra=""
    if [[ ${CHANNELS} == "last" ]]; then
        Channels_extra="--channels-last "
    fi

    BS_extra=""
    if [[ ${BS} -gt 0 ]]; then
        BS_extra="--batch_size=${BS} "
    fi

    echo "Running -----" ${MODEL}
    ulimit -n 1048576

    rm -rf /tmp/torchinductor_$USER

    mkdir model_run/${MODEL}

    BASE_DIR=${MODEL} TORCH_COMPILE_DEBUG=1 TORCH_COMPILE_DEBUG_DIR=${MODEL} python benchmarks/dynamo/${SUITE}.py -dxpu -n10 --no-skip --dashboard ${Channels_extra} ${BS_extra} ${Shape_extra} --inductor --amp --performance  --only ${MODEL} --export-profiler-trace --output-directory=model_run/${MODEL} &> model_run/${MODEL}/run.log
    mv /tmp/torchinductor_$USER/ model_run/${MODEL}/
    mv torch_compile_debug /model_run/${MODEL}
    rm -rf /model_run/${MODEL}/torchinductor_$USER/triton
    zip -r ${MODEL}.zip ${MODEL}
    echo "Finished ----- " ${MODEL}
done

echo "ALL Finished ----- "
```


## Detail for commands





Normally, the command will be like the following:

```Python
# The following code could not run, need modification
python benchmarks/dynamo/huggingface.py --float32 --backend=inductor -dxpu --accuracy{/performance} --inference -n5 --no-skip --dashboard --only GoogleFnet
```
The full arg lists could be found with the following command:

```Python
python benchmarks/dynamo/huggingface.py --help
```


### Profiling

To profile the result, one should use the `performance` mode instead of `accuracy`. i.e, One should use 

```
python benchmarks/dynamo/${SUITE}.py  ... --performance --output-directory=hf_result_folder --
```

Note that for the `--output-director`, the folder must be created before running the script.

For more detailed tutorial, please refer to [dynamo troubleshooting](https://github.com/pytorch/pytorch/blob/main/docs/source/compile/troubleshooting.rst).


Note that there are cases when the output folder is not as expected. It is recommended to set the `torch._dynamo.config.base_dir`:

```Python
torch._dynamo.config.base_dir = your_abs_path
```


### Debugging Tips

**Graph output**
To reproduce the result, one could set the flag `TORCH_COMPILE_DEBUG=1`. Then the graph will be printed. Note that there are a lot of outputs, one could direct the output to file.

```Bash
TORCH_COMPILE_DEBUG=1 python ... &> test.log
```

** Minifier Reproducing Code**

For now, we need to go into the output log to find where the reproduce code is. By looking at the above output, there are some lines like below:

```Bash
[2023-06-28 00:56:30,793] torch._inductor.debug: [WARNING] GoogleFnet__3_inference_3 debug trace: /tmp/torchinductor_username/rc/crcu4u3v2uxa6oml5vn7bpwpo3oiqran4j3xl5og6dixkcv72d3v.debug
```
In this folder, you could find files like below:

```
debug.log  
fx_graph_readable.py 
fx_graph_runnable.py  
fx_graph_transformed.py  
ir_post_fusion.txt  
ir_pre_fusion.txt  
output_code.py
```

The `output_code.py` contains the triton graph that we need.

If you need minifer to produce the above, you could enable the following flags:

```Python
+torch._dynamo.config.repro_after="dynamo"

```
Then you could find `minifier_launcher.py` under the default folder `torch_compile_debug`.

### Full config lists

Please go to [torch._dynamo.config](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/config.py) and [torch._inductor.config](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/config.py) to find all configs.