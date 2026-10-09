#!/usr/bin/env python

# pylint: disable=too-many-locals
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-boolean-expressions
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

        # torch.device("cuda"), torch.device("cuda:0"), and torch.device(f"cuda:{index}") calls
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
            if (is_torch_device and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str) and node.args[0].value.startswith("cuda:")):
                patterns.append({
                    "type": "torch_device_indexed",
                    "line": node.args[0].lineno,
                    "col": node.args[0].col_offset,
                })
            if is_torch_device and node.args and isinstance(node.args[0], ast.JoinedStr):
                for v in node.args[0].values:
                    if isinstance(v, ast.Constant) and isinstance(v.value, str) and v.value.startswith("cuda:"):
                        patterns.append({
                            "type": "torch_device_fstring",
                            "line": node.args[0].lineno,
                            "col": node.args[0].col_offset,
                        })
                        break

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

        # Tensor/device helper positional moves: tensor.to("cuda"),
        # helper.to_device("cuda").
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr in ("to", "to_device") and node.args
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == "cuda"):
                patterns.append({
                    "type": "device_posarg",
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
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_device_capability" and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "cuda" and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "torch"):
            patterns.append({
                "type": "cuda_get_device_capability",
                "line": node.lineno,
                "col": node.col_offset,
            })

        # torch.cuda.mem_get_info() calls
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "mem_get_info"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "cuda"
                and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "torch"):
            patterns.append({
                "type": "cuda_mem_get_info",
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

        # current_platform.get_device_capability() < tuple comparisons
        if isinstance(node, ast.Compare):
            if (isinstance(node.left, ast.Call) and isinstance(node.left.func, ast.Attribute)
                    and node.left.func.attr == "get_device_capability" and isinstance(node.left.func.value, ast.Name)
                    and node.left.func.value.id == "current_platform"):
                # Found: current_platform.get_device_capability() < something
                patterns.append({
                    "type": "device_capability_compare",
                    "line": node.lineno,
                    "col": node.col_offset,
                })

        # .cuda() method calls on tensors (e.g., tensor.cuda())
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "cuda":
                patterns.append({
                    "type": "tensor_cuda_method",
                    "line": func.end_lineno,
                    "col": node.col_offset,
                })

        # .is_cuda property access on tensors (e.g., tensor.is_cuda)
        if isinstance(node, ast.Attribute) and node.attr == "is_cuda":
            patterns.append({
                "type": "tensor_is_cuda_property",
                "line": node.end_lineno,
                "col": node.col_offset,
            })

    return patterns


def _is_platform_cuda_check(node: ast.AST) -> bool:
    """Match current_platform.is_cuda() and current_platform.is_cuda_alike() calls."""
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("is_cuda", "is_cuda_alike") and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "current_platform")


def _is_pytest_call(node: ast.AST, name: str) -> bool:
    """Match pytest.<name>(...) and pytest.mark.<name>(...) calls."""
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name


def _find_cuda_guard_patterns(source: str) -> list[dict]:
    """Find current_platform CUDA checks that gate a skip: skipif conditions and `if ...: pytest.skip()`."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    guards: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_pytest_call(node, "skipif"):
            guards.extend(node.args)
        elif isinstance(node, ast.If) and any(
                _is_pytest_call(call, "skip") for stmt in node.body for call in ast.walk(stmt)):
            guards.append(node.test)

    return [{
        "type": "cuda_only_skip_guard",
        "line": node.lineno,
        "col": node.col_offset,
    } for guard in guards for node in ast.walk(guard) if isinstance(node, ast.Call) and _is_platform_cuda_check(node)]


# torch.cuda runtime APIs with a torch.xpu equivalent
_CUDA_TO_XPU_RUNTIME = {
    "CUDAGraph": "XPUGraph",
    "Event": "Event",
    "Stream": "Stream",
    "_sleep": "_sleep",
    "current_stream": "current_stream",
    "graph": "graph",
    "stream": "stream",
    "synchronize": "synchronize",
}


def _to_xpu_runtime(match: re.Match[str]) -> str:
    """re.sub callback: torch.cuda.<API> -> torch.xpu.<equivalent>, other torch.cuda.* unchanged."""
    api = match.group(1)
    return f"torch.xpu.{_CUDA_TO_XPU_RUNTIME[api]}" if api in _CUDA_TO_XPU_RUNTIME else match.group(0)


def _find_cuda_runtime_patterns(source: str) -> list[dict]:
    """Find torch.cuda.<runtime API> references that have a torch.xpu equivalent."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    return [{
        "type": "cuda_runtime_api",
        "line": node.lineno,
        "col": node.col_offset,
    } for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr in _CUDA_TO_XPU_RUNTIME
            and isinstance(node.value, ast.Attribute) and node.value.attr == "cuda"
            and isinstance(node.value.value, ast.Name) and node.value.value.id == "torch"]


def _apply_patches(source: str, patterns: list[dict]) -> str:
    """Apply text-level patches guided by AST analysis."""
    lines = source.split("\n")

    for pattern in sorted(patterns, key=lambda p: -p["line"]):
        line_idx = pattern["line"] - 1
        line = lines[line_idx]
        ptype = pattern["type"]

        if ptype in ("device_kwarg", "torch_device", "torch_set_default_device", "device_default_param",
                     "device_posarg"):
            # Replace "cuda" with "xpu" in device arguments
            lines[line_idx] = line.replace('"cuda"', '"xpu"', 1)

        elif ptype == "torch_device_indexed":
            # Replace "cuda:N" with "xpu:N" in torch.device call
            lines[line_idx] = line.replace('"cuda:', '"xpu:', 1).replace("'cuda:", "'xpu:", 1)

        elif ptype == "torch_device_fstring":
            # Replace f"cuda:" with f"xpu:" in torch.device call
            lines[line_idx] = line.replace('f"cuda:', 'f"xpu:', 1).replace("f'cuda:", "f'xpu:", 1)

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

        elif ptype == "cuda_mem_get_info":
            # Replace torch.cuda.mem_get_info() with torch.xpu.mem_get_info()
            lines[line_idx] = line.replace("torch.cuda.mem_get_info()", "torch.xpu.mem_get_info()")

        elif ptype == "device_capability_compare":
            # Handle: if current_platform.get_device_capability() < (X, Y):
            # Strategy: wrap the call in a helper that returns a safe value
            # Replace with: if (cap := current_platform.get_device_capability()) is not None and cap

            # Use walrus operator to assign and check in one line
            lines[line_idx] = re.sub(
                r"if\s+current_platform\.get_device_capability\(\)\s*(<|>|<=|>=|==|!=)",
                r"if (cap := current_platform.get_device_capability()) is not None and cap \1",
                line,
            )

        elif ptype == "tensor_cuda_method":
            # Replace .cuda() with .xpu()
            lines[line_idx] = line.replace(".cuda()", ".xpu()")

        elif ptype == "tensor_is_cuda_property":
            # Replace the .is_cuda property with .is_xpu, but not .is_cuda()/.is_cuda_alike() platform calls
            lines[line_idx] = re.sub(r"\.is_cuda(?![\w(]|\s*\()", ".is_xpu", line)

        elif ptype == "cuda_only_skip_guard" and "current_platform.is_xpu()" not in line:
            # Let XPU through a CUDA-only skip: is_cuda() -> (is_cuda() or is_xpu())
            lines[line_idx] = re.sub(
                r"current_platform\.(is_cuda(?:_alike)?)\(\)",
                r"(current_platform.\1() or current_platform.is_xpu())",
                line,
            )

        elif ptype == "cuda_runtime_api":
            # Replace torch.cuda.<API> with its torch.xpu equivalent
            lines[line_idx] = re.sub(r"torch\.cuda\.(\w+)\b", _to_xpu_runtime, line)

    return "\n".join(lines)


def patch_file(filepath: Path, enable_on_xpu: bool = False) -> bool:
    """Patch a single file. Returns True if changes were made."""
    source = filepath.read_text()
    patterns = _find_cuda_patterns(source)
    if enable_on_xpu:
        patterns += _find_cuda_guard_patterns(source) + _find_cuda_runtime_patterns(source)
    if not patterns:
        return False

    patched = _apply_patches(source, patterns)
    if patched == source:
        return False

    filepath.write_text(patched)
    for p in patterns:
        line, ptype = p["line"], p["type"]
        print(f"  L{line:4d}: {ptype}")
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
        vllm_root / "tests" / "kernels",
        vllm_root / "tests" / "models" / "kimi_k3",
        vllm_root / "tests" / "v1" / "sample",
        vllm_root / "tests" / "v1" / "spec_decode",
        vllm_root / "tests" / "v1" / "worker",
        # Source directories
        vllm_root / "vllm" / "v1" / "worker",
        vllm_root / "vllm" / "model_executor" / "layers",
    ]

    # Test files whose CUDA-only skips guard Triton kernels that also run on XPU. These get the
    # CUDA->XPU replacements above, their skip guards relaxed to admit XPU, and torch.cuda
    # runtime APIs (streams, events, graphs) mapped to torch.xpu.
    cuda_guard_files = {
        vllm_root / path
        for path in (
            "tests/distributed/test_dcp_a2a.py",
            "tests/kernels/attention/test_flashmla_sparse.py",
            "tests/kernels/core/test_fused_embed_norm.py",
            "tests/kernels/core/test_fused_q_kv_rmsnorm.py",
            "tests/kernels/mamba/test_mamba_ssm.py",
            "tests/kernels/quantization/test_nvfp4_emulation.py",
            "tests/kernels/quantization/test_quantized_embedding.py",
            "tests/kernels/test_compressor_kv_cache.py",
            "tests/model_executor/test_bailing_mrope.py",
            "tests/models/inkling/test_mtp_input_fusion.py",
            "tests/models/inkling/test_qkvr_prep.py",
            "tests/models/inkling/test_sconv_metadata.py",
            "tests/v1/attention/test_dcp_a2a_pack_mask.py",
            "tests/v1/attention/test_deepseek_v4_swa_visible.py",
            "tests/v1/attention/test_indexer_dcp_localize.py",
            "tests/v1/attention/test_indexer_deepseek_v4_slot_mapping.py",
            "tests/v1/worker/test_gpu_rejection_sampler_chunking.py",
            "tests/v1/worker/test_gpu_rejection_sampler_i64.py",
            "tests/v1/worker/test_kv_block_zeroer.py",
            "tests/v1/worker/test_mamba_hybrid_model_state.py",
            "tests/watermarking/test_watermarking.py",
        )
    }

    total_patched = 0
    scanned = set()
    for patch_dir in patch_dirs:
        if not patch_dir.is_dir():
            continue
        # Use rglob to recursively scan subdirectories
        for py_file in sorted(patch_dir.rglob("*.py")):
            print(f"Scanning {py_file.relative_to(vllm_root)}...")
            scanned.add(py_file)
            if patch_file(py_file, enable_on_xpu=py_file in cuda_guard_files):
                total_patched += 1

    for py_file in sorted(cuda_guard_files - scanned):
        if not py_file.is_file():
            continue
        print(f"Scanning {py_file.relative_to(vllm_root)}...")
        if patch_file(py_file, enable_on_xpu=True):
            total_patched += 1

    print(f"\nPatched {total_patched} file(s)")


if __name__ == "__main__":
    main()
