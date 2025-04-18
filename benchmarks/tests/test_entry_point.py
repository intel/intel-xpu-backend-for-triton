from typing import Callable, List, Optional, Type

import pytest

from triton_kernels_benchmark.benchmark_testing import MarkArgs
from triton_kernels_benchmark.benchmark_utils import BenchmarkCategory, BenchmarkConfigs

ALL_CATEGORIES = [cat.value for cat in BenchmarkCategory]


@pytest.mark.parametrize(
    (
        "collect_only",
        "configs_filter",
        "categories_filter",
        "providers_filter",
        "expected_exception",
        "configs_count",
        "providers_count",
    ),
    (
        [True, "all", ALL_CATEGORIES, [], None, lambda x: x > 1, lambda x: x > 1],
        [False, "fused_softmax", ["optional"], ["triton"], AssertionError, None, None],
        [False, "fused_softmax", ALL_CATEGORIES, ["triton"], None, lambda x: x == 1, lambda x: x == 1],
        [False, "fused_softmax", ALL_CATEGORIES, ["onednn"], AssertionError, None, None],
    ),
)
def test_collect_only(
    collect_only: bool,
    configs_filter: str,
    categories_filter: List[str],
    providers_filter: List[str],
    expected_exception: Optional[Type[BaseException]],
    configs_count: Callable[[int], bool],
    providers_count: Callable[[int], bool],
):

    def benchmark_configs():
        return BenchmarkConfigs._from_args(  # pylint: disable=W0212
            args=MarkArgs(),
            collect_only=collect_only,
            configs_filter=configs_filter,
            categories_filter=categories_filter,
            providers_filter=providers_filter,
        )

    if expected_exception:
        with pytest.raises(expected_exception):
            benchmark_configs().run()
    else:
        configs = benchmark_configs().configs
        benchmark_configs().run()
        assert configs_count(len(configs))
        providers_counts = [len(config.config_summary.selected_providers) for config in configs]
        assert providers_count(max(providers_counts))
        assert providers_count(min(providers_counts))
