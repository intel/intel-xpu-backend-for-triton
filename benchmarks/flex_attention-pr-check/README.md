## For profiling llama3.1

The basic command that run llama3.1-8b is (assume llama3.1-8b models is downloaded)

```bash
python run_llm_inductor_greedy.py -m meta-llama/Meta-Llama-3.1-8B --max-new-tokens 128 \
  --input-tokens 1024 --num-warmup 2 --num-iter 7 --compile --profile
```

## Full steps for profiling llama3.1

1. Install pytorch & triton:

```bash
## setup pytorch and triton environment
git clone https://github.com/intel/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton/scripts
bash ./install-pytorch.sh --force-reinstall
bash ./compile-triton.sh
cd ..

## install deps
pip uninstall torchvision torchaudio -y
pip install --pre torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu --no-deps
```

2. install transformers

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout $(<../transformers-commit.txt)
git apply ../transformers-patch-for-timing.diff
git submodule sync
git submodule update --init --recursive
python setup.py develop
cd ..
```

3. run llama3.1 profiling

```bash
bash triton-llama3-profiling.sh
```
