# Triton kernel extraction

* Follow [End-to-End-Test Doc](https://github.com/intel/intel-xpu-backend-for-triton/wiki/End-to-End-Test) to setup environment for pytorch model suites.
  * Feel free to ask if any issues on the environment
* Steps to extract kernel in short:
  * `git clone https://github.com/Stonepia/pytorch.git -b liyang/kernel_extraction`
  * donwload script [kernel_collection.py](./kernel_collection.py)
  * `python kernel_collection.py huggingface amp_bf16 inference performance /path/to/pytorch`
  * kernels from huggingface models are extracted to `extracted_kernels` folder in the same folder as `kernel_collection.py`
* Explanation
  * With Env Flag `TORCH_COMPILE_DEBUG=1`, Pytorch Inductor generated triton kernels will be saved to `TORCHINDUCTOR_CACHE_DIR`
    Details are under [xpu_run_batch.sh](https://github.com/Stonepia/pytorch/blob/f01e701952f632dfcc24d8c77916dbc8f1cf28af/xpu_run_batch.sh#L20-L35)
    There will be too many files with filename in hash code, we should find them with kernel name.
  * With `--export-profiler-trace`, we can save kernel performance summary table, from which we can get target kernel name.
    Modified code here: [torch/benchmarks/dynamo/common.py](https://github.com/Stonepia/pytorch/blob/f01e701952f632dfcc24d8c77916dbc8f1cf28af/benchmarks/dynamo/common.py#L743-L756)
  * The remaining task is find the target kernel file in cached files, extract target triton kernel code to separate file.
  * Current strategy is pick the top 1 XPU time cost kernel for each model, extract them out as microbenchmark cases.
  * `huggingface amp_bf16 inference performance` are options, refer comments in that script to choose other mode.
  * `/path/to/pytorch` must point to the repo-branch in step 1, there are some necessary changes to dump cached info.
