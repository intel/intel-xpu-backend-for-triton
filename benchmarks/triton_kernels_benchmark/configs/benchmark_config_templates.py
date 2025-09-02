from triton_kernels_benchmark.benchmark_testing import BenchmarkCategory, BenchmarkConfig

from triton_kernels_benchmark import (
    fused_softmax,
    gemm_benchmark,
    gemm_tensor_desc_benchmark,
    gemm_tensor_of_ptr_benchmark,
    flash_attention_benchmark,
    flash_attention_tensor_desc_benchmark,
    prefix_sums,
)

CONFIGS = [
    BenchmarkConfig(
        key="softmax",
        get_benchmark=fused_softmax.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.SOFTMAX},
        description="Triton Softmax kernel benchmark",
    ),
    BenchmarkConfig(
        key="gemm",
        get_benchmark=gemm_benchmark.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.GEMM},
        description="Triton GEMM kernel benchmark",
    ),
    BenchmarkConfig(
        key="gemm-tensor-of-ptr",
        get_benchmark=gemm_tensor_of_ptr_benchmark.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.EXPERIMENTAL, BenchmarkCategory.GEMM},
        description="GEMM kernel benchmark - with tensor of pointer",
    ),
    BenchmarkConfig(
        key="gemm-tensor-desc",
        get_benchmark=gemm_tensor_desc_benchmark.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.EXPERIMENTAL, BenchmarkCategory.GEMM},
        description="GEMM kernel benchmark - with tensor descriptor",
    ),
    BenchmarkConfig(
        key="gemm_bt",
        get_benchmark=gemm_benchmark.get_benchmark,
        run_opts={"transpose_b": True},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.GEMM},
        description="Triton GEMM (A@B^t) kernel benchmark",
    ),
    BenchmarkConfig(
        key="gemm_at",
        get_benchmark=gemm_benchmark.get_benchmark,
        run_opts={"transpose_a": True},
        categories={BenchmarkCategory.OPTIONAL, BenchmarkCategory.GEMM},
        description="Triton GEMM (A^t@B) kernel benchmark",
    ),
    BenchmarkConfig(
        key="flash_attention",
        get_benchmark=flash_attention_benchmark.get_benchmark,
        run_opts={"fa_kernel_mode": "fwd"},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.FLASH_ATTENTION},
        description="FlashAttention forward kernel benchmark",
    ),
    BenchmarkConfig(
        key="flash_attention_tensor_desc",
        get_benchmark=flash_attention_tensor_desc_benchmark.get_benchmark,
        run_opts={"fa_kernel_mode": "fwd"},
        categories={BenchmarkCategory.EXPERIMENTAL, BenchmarkCategory.FLASH_ATTENTION},
    ),
    BenchmarkConfig(
        key="flash_attention_bwd",
        get_benchmark=flash_attention_benchmark.get_benchmark,
        run_opts={"fa_kernel_mode": "bwd"},
        categories={BenchmarkCategory.OPTIONAL, BenchmarkCategory.FLASH_ATTENTION},
        description="FlashAttention backward kernel benchmark",
    ),
    BenchmarkConfig(
        key="prefix-sums",
        get_benchmark=prefix_sums.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.OPTIONAL, BenchmarkCategory.PREFIX_SUMS},
        description="Prefix Sums kernel benchmark",
    ),
    # FIXME: add optional - splitK, streamk, gemm with pre-op or postops, microbenchmarks
    # FIXME: Experimental - FlexAttention
]
