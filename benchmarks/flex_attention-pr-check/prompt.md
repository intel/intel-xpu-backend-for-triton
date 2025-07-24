帮我处理两个文件，这两个文件分别是xpu和cuda的UT logs，格式基本一致，里面都包含了类似以下格式的语句：
```
--- Running: python ../hoshibara-pytorch/test/inductor/test_flex_decoding.py TestFlexDecodingCUDA.test_builtin_score_mods_bfloat16_score_mod8_head_dims2_cuda_bfloat16 ---
/home/sparse/miniforge3/envs/xingyuan-flex-attention-enable-20250303/lib/python3.10/site-packages/torch/__init__.py:1624: UserWarning: This API is going to be deprecated, please see https://pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:78.)
  _C._set_float32_matmul_precision(precision)
.
----------------------------------------------------------------------
Ran 1 test in 9.207s

OK
very small ref error of  tensor(2.0730, device='cuda:0', dtype=torch.float64)
very small ref error of  tensor(2.0852, device='cuda:0', dtype=torch.float64)
frames [('total', 6), ('ok', 6)]
inline_call []
stats [('calls_captured', 51), ('unique_graphs', 6)]
inductor [('triton_bundler_save_kernel', 84), ('async_compile_cache_miss', 20), ('async_compile_cache_hit', 8), ('pattern_matcher_count', 4), ('pattern_matcher_nodes', 4), ('benchmarking.InductorBenchmarker.benchmark_gpu', 4), ('fxgraph_cache_miss', 2), ('triton_bundler_save_static_autotuner', 2)]
aot_autograd [('total', 2), ('autograd_cache_miss', 2), ('ok', 2), ('autograd_cache_saved', 1)]
graph_break []
--- Finished: python ../hoshibara-pytorch/test/inductor/test_flex_decoding.py TestFlexDecodingCUDA.test_builtin_score_mods_bfloat16_score_mod8_head_dims2_cuda_bfloat16 ---
--- Exit Status: 0 (Took 13 seconds) ---
```

在单个文件中，我需要其中的几个信息：
1. UT名字：TestFlexDecodingCUDA.test_builtin_score_mods_bfloat16_score_mod8_head_dims2_cuda_bfloat1
2. UT实际耗时：Ran 1 test in 9.207s
3. 退出代码：Exit Status: 0
4. 总体耗时：(Took 13 seconds)

XPU对应的文件也与CUDA一致，其中类名、函数名中的CUDA和cuda会变成XPU和xpu。
我现在同时提供给你xpu和cuda生成的结果，需要你帮我生成一个csv，它共有7行，分别是：
1. CUDA UT名字
2. CUDA UT实际耗时
3. CUDA UT总体耗时
4. CUDA UT退出代码
5. XPU UT名字
6. XPU UT实际耗时
7. XPU UT总体耗时
8. XPU UT退出代码
9. CUDA UT实际耗时/XPU UT实际耗时

因为可能存在有些UT没法匹配上的问题，这些匹配不上的UT就输出到后面就行，如果XPU没有对应的，XPU那边的信息就留空，CUDA也是一样

不需要在给我的代码中提供测试用的格式，这个我能自己来。