# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import pathlib
import time
from itertools import chain

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def trace_handler(profile_obj):
    print(profile_obj.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))


parser = argparse.ArgumentParser("LLM generation (greedy search) script for inductor torch.compile path",
                                 add_help=False)
parser.add_argument(
    "-m",
    "--model-name-or-path",
    default="meta-llama/Llama-2-7b-hf",
    type=str,
    help="path to model or model name in HF hub",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["fp32", "bf16", "fp16"],
    help="bf16 or fp32",
    default="bf16",
)
parser.add_argument("--max-new-tokens", default=32, type=int, help="output max new tokens")
parser.add_argument("--input-tokens", default="32", type=str)
parser.add_argument("--page-size", default=32, type=int)
parser.add_argument("--prompt", default=None, type=str)
parser.add_argument("--num-iter", default=1, type=int, help="num iter")
parser.add_argument("--num-warmup", default=0, type=int, help="num warmup")
parser.add_argument("--batch-size", default=1, type=int, help="batch size")
parser.add_argument("--device", default="xpu", type=str)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

if args.dtype == "bf16":
    amp_enabled = True
    load_dtype = torch.bfloat16
elif args.dtype == "fp32":
    amp_enabled = False
    load_dtype = torch.float
elif args.dtype == "fp16":
    amp_enabled = True
    load_dtype = torch.float16
else:
    assert False, "This script only support bf16 and fp32 as dtype"

attn_type = "flex_attention"
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_dtype,
                                             attn_implementation=attn_type).to(args.device)
if attn_type == "paged_attention":
    model.generation_config.cache_implementation = "paged"
    model.config.page_size = args.page_size

if args.compile:
    with torch.no_grad(), torch.autocast(enabled=amp_enabled, device_type=args.device, dtype=load_dtype):
        print("compile Enabled")
        model.forward = torch.compile(model.forward, dynamic=True)

# greedy search
generate_kwargs = {
    "do_sample": False,
    "temperature": 0.9,
    "num_beams": 1,
    "token_latency": True,
}
current_path = pathlib.Path(__file__).parent.resolve()
if args.prompt is not None:
    prompt = args.prompt
else:
    with open(str(current_path) + "/prompt.json", encoding="utf-8") as f:
        prompt_pool = json.load(f)
    if "llama" in prompt_pool and args.input_tokens in prompt_pool["llama"]:
        prompt = prompt_pool["llama"]["2048"]
    else:
        raise SystemExit(
            "[ERROR] No such input_tokens prompt in prompt.json, Please use --prompt if want to use custom input.")

prompt = [prompt] * args.batch_size
inputs = tokenizer(prompt, return_tensors="pt", max_length=int(args.input_tokens))
input_ids = inputs.input_ids.to(args.device)
attention_mask = inputs.attention_mask.to(args.device)

input_size = input_ids.size(dim=1)
print(f"---- Prompt size: {input_size}")

# warmup
with torch.no_grad(), torch.autocast(enabled=amp_enabled, device_type=args.device, dtype=load_dtype):
    for _ in range(args.num_warmup):
        model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens, **generate_kwargs)

if args.profile:
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
    ], schedule=torch.profiler.schedule(wait=0, warmup=2, active=5), on_trace_ready=trace_handler,
                                record_shapes=True) as prof:
        with torch.no_grad(), torch.autocast(enabled=amp_enabled, device_type=args.device, dtype=load_dtype):
            for i in range(7):

                model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens,
                               **generate_kwargs)
                prof.step()
# benchmark
num_iter = args.num_iter - args.num_warmup
total_time = 0.0
total_list = []
gen_text = None
with torch.no_grad(), torch.autocast(enabled=amp_enabled, device_type=args.device, dtype=load_dtype):
    for _ in range(num_iter):
        torch.xpu.synchronize()
        tic = time.time()
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens,
                                **generate_kwargs)
        gen_ids = output[0]
        gen_text = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        torch.xpu.synchronize()
        toc = time.time()
        total_time += toc - tic
        total_list.append(output[1])

print(gen_text, flush=True)
print("\n", "-" * 10, "Summary:", "-" * 10)
latency = total_time / num_iter
print(f"inference-latency: {latency:.3f} sec.")
first_latency = np.mean([x[0] for x in total_list])
if args.max_new_tokens > 1:
    next_latency_list = list(chain(*[x[1:] for x in total_list]))
    next_latency_list.sort()
    average_next_latency = np.mean(next_latency_list)
    p90_latency = np.percentile(next_latency_list, 90)
print(f"first-token-latency: {first_latency:.3f} sec.")
if args.max_new_tokens > 1:
    print(f"rest-token-latency: {average_next_latency:.3f} sec.")
    print(f"P90-rest-token-latency: {p90_latency:.3f} sec.")
