from __future__ import annotations

from typing import List, Tuple, Sequence
from dataclasses import dataclass

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import DotOperandLayout
from triton.experimental.gluon.language.intel._layouts import IntelDPASLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.language.core import ir, constexpr, tensor_descriptor_base, block_type, tensor, tuple

# load_tensor_descriptor = builtin(tl_core.load_tensor_descriptor)
# store_tensor_descriptor = builtin(tl_core.store_tensor_descriptor)

__all__ = ["make_tensor_descriptor", "dot_fma"]


class tensor_descriptor(tensor_descriptor_base):
    """A descriptor representing a tensor in global memory."""

    def __init__(self, handle, shape: List[tensor], strides: List[tensor], block_type: block_type, layout):
        """Not called by user code."""
        # IR handle
        super().__init__(handle, block_type)
        # Global shape
        self.shape = tuple(shape)
        self.strides = tuple(strides)
        self.layout = layout

        self.type = tensor_descriptor_type(block_type, shape_type=self.shape.type, strides_type=self.strides.type,
                                           layout=self.layout,  # comment
                                           )

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)

    @builtin
    def load(self, offsets: Sequence[constexpr | tensor], _semantic=None) -> tensor:
        return _semantic.descriptor_load(self, offsets, "", "")

    def load_2d(self, offsets: Sequence[constexpr | tensor], is_2d_block=False, _semantic=None) -> tensor:
        # TODO: MaterializeBlockPointers.cpp
        # Add 2d_block_io parameter + validation to set proper attribute
        # Validation: (?)
        #   > 2 dims
        #   > stride 16 bytes aligned
        #   and others

        op = _semantic.descriptor_load(self, offsets, "", "")

        # TODO: proper handling like below test example
        # Option to set row/column major and other params
        attr = _semantic.builder.get_string_attr("row_major")
        op.handle.set_attr("ttig.block_io", attr)

        return op

    @builtin
    def store(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        return _semantic.descriptor_store(self, value, offsets)

    @builtin
    def store_2d(self, offsets: Sequence[constexpr | tensor], value: tensor, _semantic=None) -> tensor:
        op = _semantic.descriptor_store(self, value, offsets)

        attr = _semantic.builder.get_string_attr("row_major")
        op.handle.set_attr("ttig.block_io", attr)

        return op

    @builtin
    def prefetch(self, offsets: Sequence[constexpr | tensor], mask=None, cache=None, evict=None, is_volatile=False, _semantic=None):
        ptr_handle = self.handle
        offsets_handles = [offset.handle if hasattr(offset, 'handle') else offset for offset in offsets]
        return _semantic.builder.create_prefetch(ptr_handle, offsets_handles, False)

    @builtin
    def prefetch_2d(self, offsets: Sequence[constexpr | tensor], mask=None, cache=None, evict=None, is_volatile=False, _semantic=None):
        # TODO: handle other ttig.prefetch params
        # ptr is just temporary, support for tensor descriptor is needed
        # calculate offsets like tt.advance
        # maybe add support for mask, seems optional
        # also 2d block attr and others
        #return _semantic.builder.create_prefetch(ptr.handle, False)
        """
        pyton/triton/language/semantic.py @ load:1077 (TritonSemantic)
        cache_modifier: str, eviction_policy: str
        cache = self._str_to_load_cache_modifier(cache_modifier)
        eviction = self._str_to_eviction_policy(eviction_policy)
        """

        ptr_handle = self.handle
        offsets_handles = [offset.handle if hasattr(offset, 'handle') else offset for offset in offsets]
        op = _semantic.builder.create_prefetch(ptr_handle, offsets_handles, False)

        attr = _semantic.builder.get_string_attr("row_major")
        op.set_attr("ttig.block_io", attr)

        return op


@dataclass(eq=True)
class tensor_descriptor_type(ttgl.base_type):
    """The type for a tensor descriptor."""

    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: IntelDPASLayout

    def __str__(self) -> str:
        return f"tensor_descriptor<{self.block_type}, {self.layout}>"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self.block_type, self.layout)
        return value, cursor

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_layout_type(
            self.block_type.to_ir(builder),
            is_signed,
            self.layout._to_ir(builder),
        )

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}_{self.shape_type.mangle()}_{self.strides_type.mangle()}_{self.layout.mangle()}TD"


@builtin
def make_tensor_descriptor(ptr: ttgl.tensor, shape: List[int], strides: List[int], block_shape: List[int],
                           layout: IntelDPASLayout, _semantic=None) -> tensor_descriptor:
    # Unwrap constexpr if needed
    layout = _unwrap_if_constexpr(layout)

    # Get the pointer handle directly
    ptr_handle = ptr.handle

    # Convert shape and strides to IR values AND create tensor objects
    shape_handles = _semantic._convert_to_ir_values(shape, require_i64=False)
    stride_handles = _semantic._convert_to_ir_values(strides, require_i64=True)

    # Create tensor objects from the handles
    shape_tensors = [ttgl.tensor(h, ttgl.int32) for h in shape_handles]
    stride_tensors = [ttgl.tensor(h, ttgl.int64) for h in stride_handles]

    # Build type information
    block_type = ttgl.block_type(ptr.type.element_ty, block_shape)

    # TODO: this is w/a for xpu_dot_fma assertion - layout for block_type is not implemented yet
    # See: gluon/language/_core.py:19
    block_type.layout = layout

    shape_type = ttgl.tuple_type([ttgl.int32] * len(shape))
    strides_type = ttgl.tuple_type([ttgl.int64] * len(strides))

    # Pass tensor objects, not constexpr values
    shape_tuple = ttgl.tuple(shape_tensors, shape_type)
    strides_tuple = ttgl.tuple(stride_tensors, strides_type)

    desc_type = tensor_descriptor_type(block_type, shape_type, strides_type, layout)  #, shape_handles)

    # Create the descriptor
    padding = _semantic._str_to_padding_option("zero")
    desc_handle = _semantic.builder.create_make_tensor_descriptor(desc_type._to_ir(_semantic.builder), ptr_handle,
                                                                  shape_handles, stride_handles, padding)

    return tensor_descriptor(desc_handle, shape_tuple, strides_tuple, block_type, layout)


@builtin
def dot_fma(a, b, acc, _semantic=None):
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
