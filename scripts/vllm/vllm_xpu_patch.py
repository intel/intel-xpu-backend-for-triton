#!/usr/bin/env python
"""AST-guided XPU patcher for vLLM test files.

Scans Python test files for hardcoded CUDA references and applies
source-level replacements to make them XPU-compatible. Uses the ast
module to locate patterns precisely, then performs text replacements
that preserve formatting, comments, and whitespace.

Usage:
    python scripts/vllm-xpu-patch.py <vllm_root>
"""

import ast
import re
import sys
from pathlib import Path


def _find_cuda_patterns(source: str) -> list[dict]:
    """Use AST to find hardcoded CUDA patterns and their locations."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    patterns: list[dict] = []
    for node in ast.walk(tree):
        # device="cuda" in function default parameters
        if isinstance(node, ast.FunctionDef):
            defaults = node.args.defaults
            arg_names = [arg.arg for arg in node.args.args]
            # Match defaults with their argument names (defaults align to rightmost args)
            num_defaults = len(defaults)
            if num_defaults > 0:
                for i, default in enumerate(defaults):
                    arg_idx = len(arg_names) - num_defaults + i
                    arg_name = arg_names[arg_idx]
                    if arg_name == "device" and isinstance(default, ast.Constant) and default.value == "cuda":
                        patterns.append({
                            "type": "device_default_param",
                            "line": default.lineno,
                            "col": default.col_offset,
                        })

        # device="cuda" keyword arguments
        if isinstance(node, ast.keyword):
            if (node.arg == "device" and isinstance(node.value, ast.Constant) and node.value.value == "cuda"):
                patterns.append({
                    "type": "device_kwarg",
                    "line": node.value.lineno,
                    "col": node.value.col_offset,
                })

        # torch.device("cuda") calls
        if isinstance(node, ast.Call):
            func = node.func
            is_torch_device = (isinstance(func, ast.Attribute) and func.attr == "device"
                               and isinstance(func.value, ast.Name) and func.value.id == "torch")
            if is_torch_device and node.args and isinstance(node.args[0],
                                                            ast.Constant) and node.args[0].value == "cuda":
                patterns.append({
                    "type": "torch_device",
                    "line": node.args[0].lineno,
                    "col": node.args[0].col_offset,
                })

        # torch.set_default_device("cuda") calls
        if isinstance(node, ast.Call):
            func = node.func
            is_set_default_device = (isinstance(func, ast.Attribute) and func.attr == "set_default_device"
                                     and isinstance(func.value, ast.Name) and func.value.id == "torch")
            if is_set_default_device and node.args and isinstance(node.args[0],
                                                                  ast.Constant) and node.args[0].value == "cuda":
                patterns.append({
                    "type": "torch_set_default_device",
                    "line": node.args[0].lineno,
                    "col": node.args[0].col_offset,
                })

        # torch.cuda.is_available() calls
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            is_cuda_available = (func.attr == "is_available" and isinstance(func.value, ast.Attribute)
                                 and func.value.attr == "cuda" and isinstance(func.value.value, ast.Name)
                                 and func.value.value.id == "torch")
            if is_cuda_available:
                patterns.append({
                    "type": "cuda_is_available",
                    "line": node.lineno,
                    "col": node.col_offset,
                })

        # torch.cuda.manual_seed_all() calls
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed_all"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "cuda"):
            patterns.append({
                "type": "cuda_manual_seed",
                "line": node.lineno,
                "col": node.col_offset,
            })

        # torch.cuda.Stream() calls
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Stream"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "cuda"
                and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "torch"):
            patterns.append({
                "type": "cuda_stream",
                "line": node.lineno,
                "col": node.col_offset,
            })

        # torch.cuda.get_device_capability() calls
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_device_capability"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "cuda"
                and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "torch"):
            patterns.append({
                "type": "cuda_get_device_capability",
                "line": node.lineno,
                "col": node.col_offset,
            })

        # variable = "cuda" assignments
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Name) and isinstance(node.value, ast.Constant) and node.value.value == "cuda"):
                patterns.append({
                    "type": "var_assign_cuda",
                    "line": node.value.lineno,
                    "col": node.value.col_offset,
                    "var_name": target.id,
                })

        # current_platform.is_cuda() in skipif decorators
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "is_cuda"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "current_platform"):
            patterns.append({
                "type": "is_cuda_check",
                "line": node.lineno,
                "col": node.col_offset,
            })

        # current_platform.get_device_capability() < tuple comparisons
        if isinstance(node, ast.Compare):
            if (isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute)
                    and node.left.func.attr == "get_device_capability"
                    and isinstance(node.left.func.value, ast.Name)
                    and node.left.func.value.id == "current_platform"):
                # Found: current_platform.get_device_capability() < something
                patterns.append({
                    "type": "device_capability_compare",
                    "line": node.lineno,
                    "col": node.col_offset,
                })

        # pytest.mark.skipif(current_platform.is_rocm(), ...) decorators
        # These need XPU added for platform-specific tests (e.g., Marlin = NVIDIA-only)
        if isinstance(node, ast.Call):
            # Look for pytest.mark.skipif
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "skipif"
                    and isinstance(func.value, ast.Attribute) and func.value.attr == "mark"
                    and isinstance(func.value.value, ast.Name) and func.value.value.id == "pytest"):
                # Check if first arg is is_rocm() call
                if node.args and isinstance(node.args[0], ast.Call):
                    call = node.args[0]
                    if (isinstance(call.func, ast.Attribute) and call.func.attr == "is_rocm"
                            and isinstance(call.func.value, ast.Name)
                            and call.func.value.id == "current_platform"):
                        patterns.append({
                            "type": "skipif_is_rocm",
                            "line": node.lineno,
                            "col": node.col_offset,
                        })

    return patterns


def _apply_patches(source: str, patterns: list[dict]) -> str:
    """Apply text-level patches guided by AST analysis."""
    lines = source.split("\n")

    for pattern in sorted(patterns, key=lambda p: -p["line"]):
        line_idx = pattern["line"] - 1
        line = lines[line_idx]
        ptype = pattern["type"]

        if ptype in ("device_kwarg", "torch_device", "torch_set_default_device", "device_default_param"):
            # Replace "cuda" with "xpu" in device arguments
            lines[line_idx] = line.replace('"cuda"', '"xpu"', 1)

        elif ptype == "var_assign_cuda":
            # Replace device = "cuda" with device = "xpu"
            lines[line_idx] = line.replace('"cuda"', '"xpu"', 1)

        elif ptype == "cuda_is_available":
            # Replace torch.cuda.is_available() with torch.xpu.is_available()
            lines[line_idx] = line.replace("torch.cuda.is_available()", "torch.xpu.is_available()")

        elif ptype == "cuda_manual_seed":
            # Replace torch.cuda.manual_seed_all(...) with torch.manual_seed(...)
            lines[line_idx] = re.sub(
                r"torch\.cuda\.manual_seed_all\((.+?)\)",
                r"torch.manual_seed(\1)",
                line,
            )

        elif ptype == "cuda_stream":
            # Replace torch.cuda.Stream() with torch.xpu.Stream()
            lines[line_idx] = line.replace("torch.cuda.Stream()", "torch.xpu.Stream()")

        elif ptype == "cuda_get_device_capability":
            # Replace torch.cuda.get_device_capability() with torch.xpu.get_device_capability()
            lines[line_idx] = line.replace("torch.cuda.get_device_capability()", "torch.xpu.get_device_capability()")

        elif ptype == "is_cuda_check":
            # In skipif: not current_platform.is_cuda()
            # -> not (current_platform.is_cuda() or current_platform.is_xpu())
            lines[line_idx] = line.replace(
                "current_platform.is_cuda()",
                "(current_platform.is_cuda() or current_platform.is_xpu())",
            )

        elif ptype == "device_capability_compare":
            # Handle: if current_platform.get_device_capability() < (X, Y):
            # Strategy: wrap the call in a helper that returns a safe value
            # Replace with: if (cap := current_platform.get_device_capability()) is not None and cap

            # Use walrus operator to assign and check in one line
            lines[line_idx] = re.sub(
                r'if\s+current_platform\.get_device_capability\(\)\s*(<|>|<=|>=|==|!=)',
                r'if (cap := current_platform.get_device_capability()) is not None and cap \1',
                line,
            )

        elif ptype == "skipif_is_rocm":
            # Handle: @pytest.mark.skipif(current_platform.is_rocm(), reason="...")
            # Add XPU to skip condition for platform-specific tests
            # Replace: is_rocm()
            # With: is_rocm() or current_platform.is_xpu()
            lines[line_idx] = line.replace(
                "current_platform.is_rocm()",
                "current_platform.is_rocm() or current_platform.is_xpu()",
            )

    return "\n".join(lines)


def patch_file(filepath: Path) -> bool:
    """Patch a single file. Returns True if changes were made."""
    source = filepath.read_text()
    patterns = _find_cuda_patterns(source)
    if not patterns:
        return False

    patched = _apply_patches(source, patterns)
    if patched == source:
        return False

    filepath.write_text(patched)
    for p in patterns:
        print(f"  L{p['line']:4d}: {p['type']}")
    return True


def main() -> None:
    """Entry point for the AST-guided XPU patcher."""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <vllm_root>")
        sys.exit(1)

    vllm_root = Path(sys.argv[1])
    if not vllm_root.is_dir():
        print(f"Error: {vllm_root} is not a directory")
        sys.exit(1)

    # Patch test files and source files that need CUDA->XPU transformation
    patch_dirs = [
        # Test directories
        vllm_root / "tests" / "kernels" / "attention",
        vllm_root / "tests" / "kernels" / "mamba",
        vllm_root / "tests" / "kernels" / "moe",
        vllm_root / "tests" / "kernels" / "quantization",
        vllm_root / "tests" / "v1" / "sample",
        vllm_root / "tests" / "v1" / "spec_decode",
        vllm_root / "tests" / "v1" / "worker",
        # Source directories
        vllm_root / "vllm" / "v1" / "worker",
        vllm_root / "vllm" / "model_executor" / "layers",
    ]

    total_patched = 0
    for patch_dir in patch_dirs:
        if not patch_dir.is_dir():
            continue
        # Use rglob to recursively scan subdirectories
        for py_file in sorted(patch_dir.rglob("*.py")):
            print(f"Scanning {py_file.relative_to(vllm_root)}...")
            if patch_file(py_file):
                total_patched += 1

    print(f"\nPatched {total_patched} file(s)")


if __name__ == "__main__":
    main()
