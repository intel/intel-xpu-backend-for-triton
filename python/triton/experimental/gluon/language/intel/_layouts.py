from __future__ import annotations

from dataclasses import dataclass
from typing import List
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import AutoLayout

__all__ = [
    "IntelDPASLayout",
]


@dataclass(frozen=True)
class IntelDPASLayout(AutoLayout):
    """
    Represents a layout for Intel DPAS (dot product accumulator) operations.

    Args:
        repeatCount (int): Number of repeats for the operation.
        systolicDepth (int): Systolic array depth.
        executionSize (int): Execution size.
        opsPerChannel (int): Operations per channel.
        warps_per_cta (List[int]): Warp layout in the block.
        repCluster (List[int]): Cluster repetition configuration.
        threadsPerWarp (int): Number of threads per warp.
    """
    repeatCount: int
    systolicDepth: int
    executionSize: int
    opsPerChannel: int
    warps_per_cta: List[int]
    repCluster: List[int]
    threadsPerWarp: int

    def __post_init__(self):
        super().__setattr__("repeatCount", _unwrap_if_constexpr(self.repeatCount))
        super().__setattr__("systolicDepth", _unwrap_if_constexpr(self.systolicDepth))
        super().__setattr__("executionSize", _unwrap_if_constexpr(self.executionSize))
        super().__setattr__("opsPerChannel", _unwrap_if_constexpr(self.opsPerChannel))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("repCluster", _unwrap_if_constexpr(self.repCluster))
        super().__setattr__("threadsPerWarp", _unwrap_if_constexpr(self.threadsPerWarp))

        self.verify()

    def _to_ir(self, builder):
        # TODO: Replace with actual Intel DPAS IR builder method
        return builder.get_intel_dpas_layout(self.repeatCount, self.systolicDepth, self.executionSize,
                                             self.opsPerChannel, self.warps_per_cta, self.repCluster, self.threadsPerWarp)

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        return f"IntelDPAS_{self.repeatCount}_{self.systolicDepth}_{self.executionSize}_{self.opsPerChannel}_{stringify(self.warps_per_cta)}_{stringify(self.repCluster)}_{self.threadsPerWarp}_IntelDPAS"

    def verify(self):
        # TODO Add implement logic
        # assert self.version >= 1 and self.version <= 4, "version must be in the [1, 4] range"
        # assert len(self.instr_shape) == 3, "instr_shape must follow the (M, N, K) format"
        # valid_shapes = [[32, 32], [16, 16], [64, 4], [4, 64]]
        # assert self.instr_shape[0:2] in valid_shapes, f"invalid intrinsic shape {self.instr_shape}"
        # assert self.element_bitwidth in [32, 64], "element bitwidth must be 32 or 64"

        # rank = len(self.warps_per_cta)
        # _realize_cta_layout(self, rank)
        # assert len(self.ctas_per_cga) == rank
        # assert len(self.cta_split_num) == rank
        # assert len(self.cta_order) == rank
        return

    def __hash__(self):
        return hash((
            self.repeatCount,
            self.systolicDepth,
            self.executionSize,
            self.opsPerChannel,
            tuple(self.warps_per_cta),
            tuple(self.repCluster),
            self.threadsPerWarp,
        ))
