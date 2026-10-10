This directory contains offline-tuned configurations for the unified attention kernel.
Each JSON file covers one device name and static attention key.

Static keys are: Q/K/V and output dtypes, head size, query heads per KV head, KV block size, tensor-descriptor usage, KV quantization mode, Sliding window size, softcap usage, and Q/KV memory layouts.

Example filename:
```text
H=128,GQA=16,B=16,device_name=Intel(R)_Arc(TM)_B580_Graphics,dtype=bfloat16-bfloat16-bfloat16-bfloat16,td=1,kv_quant=0,softcap=0,window=0,q_layout=contiguous,kv_layout=contiguous.json
```

`H` is head size, `GQA` is query heads per KV head, and `B` is KV block size.
`dtype` lists Q, K, V, and output
dtypes in that order. `td` and `softcap` are flags encoded as 0 or 1,
`kv_quant` is the KV quantization mode and `window` is the sliding-window size
(0 disables it). Layout fields describe Q and KV storage.

Within each file, entries match the 2D/3D path, segment count, maximum Q/KV-length
buckets, and sequence count. Runtime selects the nearest recorded query-work point
(total query tokens × query heads), choosing the smaller point on a tie.
Each entry selects `block_m`, `tile_size`, `num_warps`, `num_stages`, `grf_mode`,
and `heads_per_program`. A `null` config selects the heuristic fallback; missing
profiles or unmatched dynamic regimes also use the fallback.
