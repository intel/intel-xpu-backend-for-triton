from typing import List, Union
from dataclasses import dataclass, field

import re


@dataclass
class ShapePatternParser:
    pattern: str
    pattern_tokens: List[str] = field(init=False)
    pattern_dims: int = field(init=False)

    def __post_init__(self):
        self.pattern_tokens = self.parse(self.pattern, pattern_shape=True)
        self.pattern_dims = len(self.pattern_tokens)

    def __str__(self):
        return self.pattern

    @staticmethod
    def parse(shape_string: str, pattern_shape: bool = False) -> List[Union[int, str]]:
        pattern_match = re.fullmatch(r"\[(.*)\]", shape_string)
        if not pattern_match:
            raise ValueError(
                f"Invalid format: {shape_string!r}, only patterns similar to [16-*-1024-*-bwd] are supported", )
        inner_string = pattern_match.group(1)
        if not inner_string:
            raise ValueError(f"Empty shape - {inner_string}")
        tokens = inner_string.split("-")
        if not any(tokens):
            raise ValueError(f"Empty shape - {inner_string}")
        result: List[Union[int, str]] = []
        for token in tokens:
            try:
                result.append(int(token))
            except ValueError:
                if token.isalnum() or token == "*" and pattern_shape:
                    result.append(token)
                else:
                    raise ValueError(  # pylint: disable=W0707
                        f"Unsupported shape or shape pattern {shape_string}"
                        "Each shape element could be either int, alphanumeric string or '*' in shape pattern")
        return result

    def matches_pattern(self, shape_string: str) -> bool:
        tokens = self.parse(shape_string)
        shape_dims = len(tokens)
        if shape_dims != self.pattern_dims:
            raise ValueError(f"Input shape dims {shape_dims} and pattern shape dims {self.pattern_dims} mismatch")
        for pattern_token, token in zip(self.pattern_tokens, tokens):
            if pattern_token == "*":
                continue
            if pattern_token != token:
                break
        else:
            return True
        return False

    def __call__(self, shape_string: str) -> bool:
        return self.matches_pattern(shape_string)

    def filter_by_pattern(self, shape_strings: List[str]) -> List[str]:
        return [shape_string for shape_string in shape_strings if self.matches_pattern(shape_string)]
