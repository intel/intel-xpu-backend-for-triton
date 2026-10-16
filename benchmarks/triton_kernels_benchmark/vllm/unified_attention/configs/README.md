This directory contains offline-tuned configurations for the unified attention kernel.
Each JSON file covers one device name and static attention key.

Static keys are: Q/K/V and output dtypes, head size, query heads per KV head, KV block size, tensor-descriptor usage, KV quantization mode, sliding window size, and softcap usage.

Example filename:
```text
H=128,G=16,B=16,dev=Intel(R)_Arc(TM)_B580_Graphics,dt=401736f1,td=1,kq=0,sc=0,w=0.json
```

`H` is head size, `G` is query heads per KV head, `B` is KV block size, and
`dev` is the device name with spaces replaced by underscores.
`dt` is the first eight hexadecimal characters of SHA-256 over the full Q, K, V,
and output dtype names. Full dtype names and static keys remain in the JSON and
are checked when loading.

`td` and `sc` (softcap) are flags encoded as 0 or 1, `kq` is the KV quantization
mode, and `w` is the sliding-window size (0 disables it).

Within each file, entries match the 2D/3D path, segment count, and maximum
Q/KV-length buckets. Runtime selects the nearest recorded sequence count, then the
nearest query-work point (total query tokens × query heads) at that sequence count.
Both selections use absolute distance and choose the smaller value on a tie.
Each entry selects `block_m`, `tile_size`, `num_warps`, `num_stages`, `grf_mode`,
and `heads_per_program`. A `null` config selects the heuristic fallback; missing
profiles or unmatched dynamic regimes also use the fallback.
