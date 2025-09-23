# FIXME: make these configurations part of `BenchmarkConfig` when it will be used for all benchmarks
# format: bench_name: (n_warmup, n_repeat)
DO_BENCH_SETUP = {
    # Maximum across torch-native=10, triton=800, torch-jit=10, xetla=100, onednn=800
    # For onednn more warmup very slowly makes performance worse
    "fused_softmax": (800, 10),

    # This warmup logic improves performance on BMG significantly
    # For FWD mode in triton & cutlass: Some configs increase performance with warmup as a step function, but some slowly decrease with saturation
    # Performance is best at 250-400ms range, but we want stable, not just best at ~600ms (triton/cutlass providers)
    # n_warmup_fwd = 600
    # For BWD mode: Performance doesn't really improve much with warmup for triton, but xetla benefit from more warmup
    # n_warmup_bwd = 400  # Maximum across xetla=400, triton=10, onednn=10
    # We keep old warmup value, because new warmup makes perfomance on PVC slightly worse
    "flash_attention": (10, 10),

    # Maximum across torch=200, triton=600
    "flex_attention_causal_mask": (600, 10),

    # There is still performance variance for triton, probably caused by random choice of autotune config
    "flex_attention_custom_masks": (200, 10),

    # Maximum across onednn=600, triton=800, xetla=10, cutlass=600
    "gemm": (800, 10),

    # Maximum across onednn=600, triton=1000
    # For onednn and triton: Some configs increase performance with warmup as a step function, but some
    # slowly decrease with saturation. Performance is best at 150-200ms range, but we want stable, not just best
    "gemm_postop_addmatrix": (1000, 10),

    # Some configs increase performance with warmup as a step function, but some slowly decrease with saturation.
    # Performance is best at 200-400ms range, but we want stable, not just best
    "gemm_postop_gelu": (1000, 10),

    # Some configs increase performance with warmup as a step function, but some slowly decrease with saturation. Performance is best at 200-400ms range, but we want stable, not just best
    # This warmup improves performance on BMG
    # n_warmup = 800
    # We keep old warmup for now because longer warmup make perfomance on PVC worse
    "gemm_preop_exp": (10, 10),

    # Maximum across onednn=10, triton=100, xetla=300
    # custom format: bench_name: (n_warmup, n_repeat, n_repeat_xtl)
    "gemm_splitk": (300, 10, 100),

    # Maximum across onednn=10, triton=1000, xetla=100
    "gemm_streamk": (1000, 10),
    "prefix_sums": (1000, 100),
}

try:
    # The easiest way to overwrite a config that eliminates merge conflicts
    from .do_bench_configs_rewrite import DO_BENCH_SETUP_REWRITE
    DO_BENCH_SETUP = DO_BENCH_SETUP_REWRITE
except ImportError:
    pass


def get_benchmark_setup(bench_name: str) -> tuple:
    return DO_BENCH_SETUP[bench_name]
