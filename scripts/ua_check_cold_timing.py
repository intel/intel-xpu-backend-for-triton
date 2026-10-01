"""Reject material graph/profiler ranking reversals before the long CI sweep."""
import json
import math
from pathlib import Path
import statistics
import sys

report = json.loads(Path(sys.argv[1]).read_text())
assert report['complete'] and report['timing']['clear_cache']
for case in report['cases']:
    arms = case['arms']
    for name, arm in arms.items():
        graph_samples, profiler_samples = arm['graph_ms'], arm['profiler_ms']
        if not graph_samples or not profiler_samples or not all(
                math.isfinite(value) and value > 0 for value in graph_samples + profiler_samples):
            raise RuntimeError(f"Invalid timing samples: {case['id']}/{name}")
        if max(graph_samples) > 2 * min(graph_samples):
            raise RuntimeError(f"Unstable graph timings: {case['id']}/{name}")
        if statistics.median(graph_samples) < 0.5 * statistics.median(profiler_samples):
            raise RuntimeError(f"Graph timing undercounts profiler duration: {case['id']}/{name}")
    graph = statistics.median(arms['profile']['graph_ms']) / statistics.median(arms['main']['graph_ms'])
    profiler = statistics.median(arms['profile']['profiler_ms']) / statistics.median(arms['main']['profiler_ms'])
    print(f"{case['id']}: profile/main graph={graph:.3f}, profiler={profiler:.3f}")
    if (graph > 1.05 and profiler < 0.95) or (graph < 0.95 and profiler > 1.05):
        raise RuntimeError(f"Material timing-method ranking reversal: {case['id']}")
