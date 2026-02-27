# Python Coding Guidelines

## Naming Conventions

```python
# Variables and functions: snake_case
num_warps = 4
def parse_target(arch_str):
    ...

# Classes: CamelCase
class XPUBackend(BaseBackend):
    ...

# Constants: UPPER_SNAKE_CASE
THREADS_PER_WARP = 16
GPU_DIALECT = "ttg"
```

## Formatting

- **Line length**: 120 characters max
- **Formatter**: Ruff (primary), YAPF (secondary) — configured in `pyproject.toml`
- **Linting**: Ruff with `E501`, `E701`, `E731`, `E741` ignored

## Language Features

- **Minimum Python version**: 3.10
- Use modern type annotations and language features (e.g., `str | None` instead of `Optional[str]`, match statements)
- Prefer explicit over implicit behavior

## Type Hints

Use type hints on function signatures. Prefer modern Python 3.10+ syntax:

```python
# Preferred: modern syntax
def find_sycl(include_dir: list[str]) -> tuple[list[str], list[str]]:
    ...

# Preferred: union syntax (3.10+)
def min_dot_size(device_props: dict | GPUTarget) -> int:
    ...

# Acceptable: typing module for complex types
from typing import Optional, Union
def process(value: Union[Dict, GPUTarget]) -> int:
    ...
```

Use `TYPE_CHECKING` to avoid circular imports:
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .runtime.cache import CacheManager
```

## Import Order

Standard Python ordering:
1. `__future__` annotations
2. Standard library (`os`, `re`, `functools`, `dataclasses`, etc.)
3. Third-party packages (`torch`, `numpy`, `pytest`)
4. Local/project imports (`triton`, `triton._C.libtriton`, backend modules)

## Docstrings

Use triple-quoted docstrings for public functions with Arguments/Returns/Raises sections:

```python
def find_sycl(include_dir: list[str]) -> tuple[list[str], list[str]]:
    """
    Looks for the sycl library in known places.

    Arguments:
      include_dir: list of include directories to pass to compiler.

    Returns:
      enriched include_dir and libsycl.so location.

    Raises:
      AssertionError: if library was not found.
    """
```

## Architecture

- Use dataclasses for data containers and configuration
- Pass configuration as dataclass instances; avoid global state
- Prefer composition over inheritance; use mixins sparingly
- Use `defusedxml` for XML parsing (security)
- Keep CLI argument parsing separate from business logic

### Dataclasses

Use `@dataclass` for configuration and option structs. Validate in `__post_init__`:

```python
@dataclass
class XPUOptions:
    num_warps: int = 4
    num_stages: int = 2
    grf_mode: str = 'default'

    def __post_init__(self):
        if self.num_warps <= 0 or (self.num_warps & (self.num_warps - 1)) != 0:
            raise AssertionError("num_warps must be a power of 2")
```

## Error Handling

- Use `assert` for internal invariants
- Use `raise AssertionError(msg)` or domain-specific exceptions for validation
- No bare `except:` — always catch specific exceptions

## Triton Kernel Conventions

```python
@triton.jit
def kernel_name(
    a_ptr, b_ptr, c_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    ...
```

- Pointer arguments first, then dimensions, then strides, then `tl.constexpr` block sizes
- Use `tl.constexpr` for compile-time constants
- Use `UPPER_SNAKE_CASE` for constexpr block size parameters
