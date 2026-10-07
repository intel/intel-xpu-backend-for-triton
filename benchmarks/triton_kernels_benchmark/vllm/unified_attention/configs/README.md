# Unified-attention configurations

This directory contains offline-tuned configurations for the unified-attention kernel.
Each JSON file covers one device name and static attention key:

- Q/K/V and output dtypes, head size, and query heads per KV head.
- KV block size, tensor-descriptor usage, and KV quantization mode.
- Sliding-window size, softcap usage, and Q/KV memory layouts.

Filenames use upstream fused-MoE's comma-separated `key=value` convention:

```text
H=128,GQA=16,B=16,device_name=Intel(R)_Arc(TM)_B580_Graphics,dtype=bfloat16-bfloat16-bfloat16-bfloat16,td=1,kv_quant=0,softcap=0,window=0,q_layout=contiguous,kv_layout=contiguous.json
```

`H` is head size, `GQA` is query heads per KV head, and `B` is KV block size.
`device_name` replaces spaces with underscores. `dtype` lists Q, K, V, and output
dtypes in that order. `td` and `softcap` are Boolean flags encoded as 0 or 1;
`kv_quant` is the KV quantization mode and `window` is the sliding-window size
(0 disables it). Layout fields describe Q and KV storage. Every static-key field
is included; no hash is used. The provided configurations target Intel Arc B580.

Within each file, entries match the 2D/3D path, segment count, maximum Q/KV-length
buckets, and sequence count. Runtime selects the nearest recorded query-work point
(total query tokens × query heads), choosing the smaller point on a tie.
Each entry selects `block_m`, `tile_size`, `num_warps`, `num_stages`, `grf_mode`,
and `heads_per_program`. A `null` config selects the heuristic fallback; missing
profiles or unmatched dynamic regimes also use the fallback.

See [tune_unified_attention.py](../tune_unified_attention.py) to generate profiles.
With the unified-attention patch applied, run from the parent directory:

```bash
python tune_unified_attention.py --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --tp-size 2 --batch-size 1 8 32 --query-len 1 --kv-len 1024 \
  --tune --save-dir configs
```

The script reads model configuration without loading weights and generates synthetic
attention tensors. Model inputs default to QKV-sliced queries and interleaved KV;
use `--q-layout contiguous --kv-layout contiguous` for separate contiguous tensors.
Omit `--tune` to benchmark existing profiles. Set `VLLM_TUNED_CONFIG_FOLDER` to use
an external config directory at runtime.

Upstream documents its analogous format in the
[fused-MoE configs README](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/configs/README).
