# SPDX-License-Identifier: Apache-2.0
"""Cache-cleared graph timing for comparison with CI benchmarks."""

import math
import statistics
from contextlib import contextmanager

import torch


@contextmanager
def cold_graph_timer(fn, device, warmup_ms, rep_ms):
    """Evict before each replay; time only the single-call attention graph."""
    backend = getattr(torch, device.type)
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=device)
    attention, eviction = backend.XPUGraph(), backend.XPUGraph()
    try:
        fn()
        cache.zero_()
        backend.synchronize(device)
        with backend.graph(attention):
            fn()
        with backend.graph(eviction):
            cache.zero_()
        eviction.replay()
        attention.replay()
        backend.synchronize(device)

        def batch(repeats):
            # Fresh events avoid re-recording profiling tags across replay batches.
            pairs = [(backend.Event(enable_timing=True), backend.Event(enable_timing=True)) for _ in range(repeats)]
            boundary_start, boundary_end = backend.Event(enable_timing=True), backend.Event(enable_timing=True)
            boundary_start.record()
            for start, end in pairs:
                eviction.replay()
                start.record()
                attention.replay()
                end.record()
            boundary_end.record()
            backend.synchronize(device)
            samples = [start.elapsed_time(end) for start, end in pairs]
            total_ms = boundary_start.elapsed_time(boundary_end)
            if not all(math.isfinite(value) and value > 0 for value in samples + [total_ms]):
                raise RuntimeError("Invalid cold graph timing")
            # Bound work using total time, including eviction, but exclude it from the result.
            return statistics.median(samples), total_ms

        _, estimate = batch(3)
        per_call_ms = estimate / 3
        warmups = max(1, min(256, math.ceil(warmup_ms / per_call_ms)))
        batch(warmups)
        repeats = max(1, min(256, math.ceil(rep_ms / per_call_ms)))

        def elapsed():
            return batch(repeats)[0]

        yield attention, elapsed
    finally:
        attention.reset()
        eviction.reset()
