#!/usr/bin/env python3
"""Turn a capture.py output (capture.json + .spv + index tensors) into a case directory for repro.cpp.
Usage: gen_case.py <capture-dir> <case-dir> [launch-index]"""
import json, os, shutil, sys
cap, out = sys.argv[1], sys.argv[2]; idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
d = json.load(open(os.path.join(cap, "capture.json"))); l = d["launches"][idx]
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(cap, l["spv"]), os.path.join(out, "kernel.spv"))
ELEM = {"torch.float8_e4m3fn": 1, "torch.bfloat16": 2, "torch.float32": 4, "torch.int32": 4}
INIT = {"q": "rand_fp8", "k": "rand_fp8", "v": "rand_fp8", "out": "zero"}
tens, lines, seen = {}, [], set()
for a in l["args"]:
    if a["kind"] != "tensor" or a["tensor"] in seen: continue
    seen.add(a["tensor"]); n = 1
    for s in a["shape"]: n *= s
    t = a["tensor"]; src = os.path.join(cap, t + ".bin")
    if t in INIT: init = INIT[t]
    elif os.path.exists(src): shutil.copy(src, os.path.join(out, t + ".bin")); init = f"file {t}.bin"
    else: init = "zero"
    lines.append(f"tensor {t} {a['dtype'].split('.')[-1]} {n * ELEM[a['dtype']]} {init}")
flops = 0
for q, kv in d["seq_lens"]:
    if d["sliding_window"] is not None: kv = min(kv, d["sliding_window"])
    flops += 2 * d["head_size"] * q * kv * 2 * d["q_heads"]
hdr = [f"label {d['case']}", f"kernel {l['kernel']}", "spv kernel.spv", f"grid {l['grid'][0]} {l['grid'][1]} {l['grid'][2] if len(l['grid']) > 2 else 1}",
       f"local {l['num_warps'] * l['threads_per_warp']} 1 1", f"flops {flops}", "output out"]
body = []
for a in l["args"]:
    if a["kind"] == "tensor": body.append(f"arg tensor {a['tensor']}")
    elif a["kind"] == "null": body.append("arg null -")
    else:
        t = {"i32": "i32", "i64": "i64", "fp32": "f32"}[a["type"]]; body.append(f"arg {t} {a['value']!r}")
body += ["arg null -", "arg null -"]   # trailing global_scratch / profile_scratch pointers (global_scratch_size == 0)
open(os.path.join(out, "case.cfg"), "w").write("\n".join(hdr + lines + body) + "\n")
print("build_flags recorded by Triton for this module:", repr(l["build_flags"]))
