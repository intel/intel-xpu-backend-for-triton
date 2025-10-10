from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from triton.language.core import _unwrap_if_constexpr

from triton.experimental.gluon.language._layouts import DistributedLayout

__all__ = [
    "IntelDPASLayout",
]


@dataclass(frozen=True)
class IntelDPASLayout(DistributedLayout):
    """
    Represents a layout for Intel DPAS (dot product accumulator) operations.

    Args:
        repeatCount (int): Number of repeats for the operation.
        systolic_depth (int): Systolic array depth.
        execution_size (int): Execution size.
        ops_per_chan (int): Operations per channel.
        warps_per_cta (List[int]): Warp layout in the block.
        rep_cluster (List[int]): Cluster repetition configuration.
        threads_per_warp (int): Number of threads per warp.
    """

    repeatCount: int
    systolic_depth: int
    execution_size: int
    ops_per_chan: int
    warps_per_cta: List[int]
    rep_cluster: List[int]
    threads_per_warp: int
    cta_order: Optional[List[int]] = None

    def __post_init__(self):
        super().__setattr__("repeatCount", _unwrap_if_constexpr(self.repeatCount))
        super().__setattr__("systolic_depth", _unwrap_if_constexpr(self.systolic_depth))
        super().__setattr__("execution_size", _unwrap_if_constexpr(self.execution_size))
        super().__setattr__("ops_per_chan", _unwrap_if_constexpr(self.ops_per_chan))
        super().__setattr__("warps_per_cta", _unwrap_if_constexpr(self.warps_per_cta))
        super().__setattr__("rep_cluster", _unwrap_if_constexpr(self.rep_cluster))
        super().__setattr__("threads_per_warp", _unwrap_if_constexpr(self.threads_per_warp))
        # Compute cta_order as reversed range of warps_per_cta length, if not provided
        super().__setattr__("cta_order", list(reversed(range(len(self.warps_per_cta)))))

        self.verify()

    def _to_ir(self, builder):
        # TODO: Replace with actual Intel DPAS IR builder method
        return builder.get_intel_dpas_layout(
            self.repeatCount,
            self.systolic_depth,
            self.execution_size,
            self.ops_per_chan,
            self.warps_per_cta,
            self.rep_cluster,
            self.threads_per_warp,
        )

    def mangle(self) -> str:

        def stringify(x):
            if x is None:
                return ""
            return "_".join(map(str, x))

        return f"IntelDPAS_{self.repeatCount}_{self.systolic_depth}_{self.execution_size}_{self.ops_per_chan}_{stringify(self.warps_per_cta)}_{stringify(self.rep_cluster)}_{self.threads_per_warp}_IntelDPAS"

    def verify(self):
        # TODO Do we need verify?
        return

    def __hash__(self):
        return hash((
            self.repeatCount,
            self.systolic_depth,
            self.execution_size,
            self.ops_per_chan,
            tuple(self.warps_per_cta),
            tuple(self.rep_cluster),
            self.threads_per_warp,
            tuple(self.cta_order),
        ))
