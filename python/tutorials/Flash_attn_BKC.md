## Collect Flash Attenion Perf: Triton & XeTLA

### XeTLA

```bash
# prepare XeTLA library
git clone https://github.com/intel-innersource/libraries.gpu.xetla.git
cd libraries.gpu.xetla
git checkout def1c682ecbd6a8f3b21042ee0f2e5040fae9871
# fix `is_device_copyable` incompatible error
wget -O demo_xetla_flash_attn_patch.diff https://raw.githubusercontent.com/Dewei-Wang-sh/intel-xpu-backend-for-triton/perf_attn/python/tutorials/demo_xetla_flash_attn_patch.diff
git apply demo_xetla_flash_attn_patch.diff
cd ..
# scp to IDC server like:
scp -r -J guest@146.152.232.8 /path/to/libraries.gpu.xetla YOUR_IDSID@100.80.168.33:/home/YOUR_IDSID

# login IDC server and run
cd $HOME
git clone https://github.com/intel-sandbox/DPCPP-Kernel.git
cd DPCPP-Kernel
git checkout f3e0f725d6f1c16dff820df5a0a83f4cd8e01a35
# modify `xetla_base` in flash_attention/mha/src/run.sh 
cd flash_attention/mha/src
bash run.sh
```


### Triton

```bash
git clone -b perf_attn https://github.com/Dewei-Wang-sh/intel-xpu-backend-for-triton.git
cd intel-xpu-backend-for-triton
scripts/compile-triton.sh --triton --venv
source .venv/bin/activate
scripts/compile-pytorch-ipex.sh --pinned
bash collect.sh
```