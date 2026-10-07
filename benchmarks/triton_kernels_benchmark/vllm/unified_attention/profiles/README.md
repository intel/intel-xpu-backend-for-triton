# Unified-attention configurations

This directory contains offline-tuned configurations for the unified-attention kernel.
Each JSON file covers one device name and static attention key:

- Q/K/V and output dtypes, head size, and query heads per KV head.
- KV block size, tensor-descriptor usage, and KV quantization mode.
- Sliding-window size, softcap usage, and Q/KV memory layouts.

Filenames have the form
`unified_attention_<dtype>-<device>_h<head_size>_gqa<query_heads_per_kv>_b<block_size>_<hash>.json`.
The hash identifies the full device/static-key combination, including fields omitted
from the readable prefix. The provided configurations target Intel Arc B580.

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
  --tune --save-dir profiles
```

The script reads model configuration without loading weights and generates synthetic
attention tensors. Model inputs default to QKV-sliced queries and interleaved KV;
use `--q-layout contiguous --kv-layout contiguous` for separate contiguous tensors.
Omit `--tune` to benchmark existing profiles. Set `VLLM_TUNED_CONFIG_FOLDER` to use
an external profile directory at runtime.

Intel CI-specific adapters and input manifests are documented in
[scripts/vllm/ua_tuning](../../../../../scripts/vllm/ua_tuning/README.md).
