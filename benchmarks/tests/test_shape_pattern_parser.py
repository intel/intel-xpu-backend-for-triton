import pytest

from triton_kernels_benchmark.benchmark_utils import BenchmarkCategory, BenchmarkConfigs
from triton_kernels_benchmark.benchmark_shapes_parser import ShapePatternParser

CONFIG_SHAPES = [
    shape for config in BenchmarkConfigs._get_configs(  # pylint: disable=W0212
        configs_filter=BenchmarkConfigs._get_all_configs().keys(),  # pylint: disable=W0212
        categories_filter=[cat.value for cat in BenchmarkCategory],
        providers_filter=[],
        shape_pattern=None,
    ) for shape in config.shapes
]


@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("[1-2-3]", [1, 2, 3]),
        ("[16-32-1024-64]", [16, 32, 1024, 64]),
        ("[True-False-123]", ["True", "False", 123]),
    ],
)
def test_parse_valid_shapes(input_string, expected):
    assert ShapePatternParser.parse(input_string) == expected


@pytest.mark.parametrize("known_shape", CONFIG_SHAPES)
def test_parse_all_known_shapes(known_shape):
    ShapePatternParser.parse(known_shape)


@pytest.mark.parametrize(
    "pattern, tokens",
    [
        ("[16-*-1024-*]", [16, "*", 1024, "*"]),
    ],
)
def test_parse_valid_pattern_shapes(pattern, tokens):
    parsed_tokens = ShapePatternParser.parse(pattern, pattern_shape=True)
    assert tokens == parsed_tokens


def test_parse_bracketed_star_without_pattern_shape_fails():
    with pytest.raises(ValueError):
        ShapePatternParser.parse("[*]")


@pytest.mark.parametrize(
    "invalid_string",
    [
        "1-2-3",
        "[]",
        "[ ]",
        "[ - ]",
        "[-]",
        "[--]",
        "[- -]",
        "[1-2-@-4]",
    ],
)
def test_parse_bracketed_invalid(invalid_string):
    with pytest.raises(ValueError):
        ShapePatternParser.parse(invalid_string, pattern_shape=True)
    with pytest.raises(ValueError):
        ShapePatternParser(invalid_string)


@pytest.mark.parametrize(
    "pattern",
    [
        "[1-2-3]",
        "[a-b-c]",
        "[*-b-3]",
    ],
)
def test_init_valid(pattern):
    parser = ShapePatternParser(pattern)
    assert parser.pattern == pattern
    assert len(parser.pattern_tokens) == parser.pattern_dims


@pytest.fixture
def sample_shapes():
    return [
        "[16-32-1024-64-False-bwd]",
        "[16-64-1024-32-True-fwd]",
        "[32-32-512-64-False-bwd]",
        "[16-32-1024-64-False-fwd]",
    ]


def test_filter_exact_match(sample_shapes):  # pylint: disable=W0621
    parser = ShapePatternParser("[16-32-1024-64-False-bwd]")
    assert parser.filter_by_pattern(sample_shapes) == ["[16-32-1024-64-False-bwd]"]


def test_filter_wildcard_match(sample_shapes):  # pylint: disable=W0621
    parser = ShapePatternParser("[16-*-1024-64-False-*]")
    expected = [
        "[16-32-1024-64-False-bwd]",
        "[16-32-1024-64-False-fwd]",
    ]
    assert parser.filter_by_pattern(sample_shapes) == expected


@pytest.mark.parametrize(
    "pattern, shapes",
    [
        ("[*]", ["bad-shape", "[foo]", "[123]"]),
    ],
)
def test_filter_invalid_shape_strings(pattern, shapes):
    parser = ShapePatternParser(pattern)
    with pytest.raises(ValueError):
        parser.filter_by_pattern(shapes)


@pytest.mark.parametrize(
    "pattern, shapes",
    [
        ("[1-2-3]", ["[1-2]"]),
    ],
)
def test_filter_dims_mismatch(pattern, shapes):
    parser = ShapePatternParser(pattern)
    with pytest.raises(ValueError) as exc_info:
        parser.filter_by_pattern(shapes)
    assert "mismatch" in str(exc_info.value)


@pytest.mark.parametrize(
    "pattern, shapes",
    [
        ("[*]", ["[foo]", "[bar]", "[123]"]),
    ],
)
def test_filter_single_wildcard_matches_all(pattern, shapes):
    parser = ShapePatternParser(pattern)
    assert parser.filter_by_pattern(shapes) == shapes
