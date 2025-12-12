from __future__ import annotations

from typing import Sequence
from functools import cache

from triton.experimental.gluon.language._layouts import DotOperandLayout
from triton.experimental.gluon.language.intel._layouts import IntelDPASLayout
from triton.experimental.gluon.language._core import builtin
from triton.language.core import constexpr, tensor

__all__ = ["dot"]


@cache
def get_dpas_capabilities():
    from triton.backends.intel.driver import XPUDriver

    driver = XPUDriver()
    target = driver.get_current_target()
    properties = target.arch

    # like annotate_module in passes
    dpas_cap = {
        "systolicDepth": 8,
        "repeatCount": 8,
        "executionSize": min(properties.get("sub_group_sizes", [16])),
        "opsChanBitWidths": 32,
        "has_subgroup_2d_block_io": properties.get("has_subgroup_2d_block_io", False),
    }

    return dpas_cap


@builtin
def dot(a, b, acc, _semantic=None):
    assert isinstance(a, tensor), "a must be a tensor"
    assert isinstance(b, tensor), "b must be a tensor"
    assert isinstance(acc, tensor), "acc must be a tensor"

    mma_layout = acc.type.layout
    assert isinstance(mma_layout, IntelDPASLayout), "acc must have a BlockedLayout"
    assert isinstance(a.type.layout, DotOperandLayout), "a must have a DotOperandLayout"
    assert isinstance(b.type.layout, DotOperandLayout), "b must have a DotOperandLayout"
    assert a.type.layout.parent == mma_layout, "a's parent layout must be the same as acc's layout"
    assert b.type.layout.parent == mma_layout, "b's parent layout must be the same as acc's layout"
    assert a.type.layout.operand_index == 0, "a's operand index must be 0"
    assert b.type.layout.operand_index == 1, "b's operand index must be 1"

    handle = _semantic.dot(a, b, acc, input_precision=None, max_num_imprecise_acc=None, out_dtype=acc.dtype).handle
    return tensor(handle, acc.type)


def is_2d_block_supported():
    capabilities = get_dpas_capabilities()
    return capabilities["has_subgroup_2d_block_io"]


def mark_2d_block_attribute(op, order, _semantic):
    if order not in ('row_major', 'column_major'):
        raise ValueError("Only row_major/column_major order is supported for 2D block")

    attr = _semantic.builder.get_string_attr(order)
    op.set_attr("ttig.block_io", attr)


@builtin
def load(desc, offsets: Sequence[constexpr | tensor], _semantic=None) -> tensor:
    return _semantic.descriptor_load(desc, offsets, "", "")


@builtin
def load_2d(desc, offsets: Sequence[constexpr | tensor], order: str = "row_major", _semantic=None) -> tensor:
    if not is_2d_block_supported():
        raise ValueError("2d block functionality is not supported for this hardware")

    op = _semantic.descriptor_load(desc, offsets, "", "")
    mark_2d_block_attribute(op.handle, order, _semantic)
    return op


@builtin
def store(desc, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
    return _semantic.descriptor_store(desc, value, offsets)


@builtin
def store_2d(desc, offsets: Sequence[constexpr | tensor], value: tensor, order: str = "row_major",
             _semantic=None) -> tensor:
    if not is_2d_block_supported():
        raise ValueError("2d block functionality is not supported for this hardware")

    op = _semantic.descriptor_store(desc, value, offsets)
    mark_2d_block_attribute(op.handle, order, _semantic)
    return op


@builtin
def prefetch(desc, offsets: Sequence[constexpr | tensor], _semantic=None):
    ptr_handle = desc.handle
    offsets_handles = [offset.handle if hasattr(offset, 'handle') else offset for offset in offsets]
    return _semantic.builder.create_prefetch(ptr_handle, offsets_handles, False)


@builtin
def prefetch_2d(desc, offsets: Sequence[constexpr | tensor], order: str = "row_major", _semantic=None):
    if not is_2d_block_supported():
        raise ValueError("2d block functionality is not supported for this hardware")

    ptr_handle = desc.handle
    offsets_handles = [offset.handle if hasattr(offset, 'handle') else offset for offset in offsets]
    op = _semantic.builder.create_prefetch(ptr_handle, offsets_handles, False)
    mark_2d_block_attribute(op, order, _semantic)
    return op
