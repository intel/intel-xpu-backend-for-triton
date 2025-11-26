from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language.intel._layouts import IntelDPASLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr
from triton.language.core import tensor_descriptor



if TYPE_CHECKING:
    from triton._C import ir

__all__ = ["make_tensor_descriptor"]


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
        value = tensor_descriptor(handle, shape, strides, self)
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
def make_tensor_descriptor(ptr: ttgl.tensor, shape: List[int], strides: List[int],
                          block_shape: List[int], layout: IntelDPASLayout,
                          _semantic=None) -> tensor_descriptor:
    # Unwrap constexpr if needed
    layout = _unwrap_if_constexpr(layout)

    # Get the pointer handle directly
    ptr_handle = ptr.handle

    # Convert shape and strides to IR values AND create tensor objects
    shape_handles = _semantic._convert_to_ir_values(shape, require_i64=False)
    stride_handles = _semantic._convert_to_ir_values(strides, require_i64=True)

    # Create tensor objects from the handles (not constexpr!)
    shape_tensors = [ttgl.tensor(h, ttgl.int32) for h in shape_handles]
    stride_tensors = [ttgl.tensor(h, ttgl.int64) for h in stride_handles]

    # Build type information
    block_type = ttgl.block_type(ptr.type.element_ty, block_shape)
    shape_type = ttgl.tuple_type([ttgl.int32] * len(shape))
    strides_type = ttgl.tuple_type([ttgl.int64] * len(strides))

    # maybe can be partially removed?
    desc_type = tensor_descriptor_type(block_type, shape_type, strides_type, layout)

    # Create the descriptor
    padding = _semantic._str_to_padding_option("zero")
    desc_handle = _semantic.builder.create_make_tensor_descriptor(
        desc_type._to_ir(_semantic.builder),
        ptr_handle,
        shape_handles,
        stride_handles,
        padding
    )

    # Pass tensor objects, not constexpr values
    shape_tuple = ttgl.tuple(shape_tensors, shape_type)
    strides_tuple = ttgl.tuple(stride_tensors, strides_type)

    return tensor_descriptor(desc_handle, shape_tuple, strides_tuple, block_type) #desc_type)
