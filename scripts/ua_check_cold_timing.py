"""Reject material graph/profiler ranking reversals before the long CI sweep."""
import json
from pathlib import Path
import statistics
import sys

report = json.loads(Path(sys.argv[1]).read_text())
assert report['complete'] and report['timing']['clear_cache']
for case in report['cases']:
    arms = case['arms']
    graph = statistics.median(arms['profile']['graph_ms']) / statistics.median(arms['main']['graph_ms'])
    profiler = statistics.median(arms['profile']['profiler_ms']) / statistics.median(arms['main']['profiler_ms'])
    print(f"{case['id']}: profile/main graph={graph:.3f}, profiler={profiler:.3f}")
    if (graph > 1.05 and profiler < 0.95) or (graph < 0.95 and profiler > 1.05):
        raise RuntimeError(f"Material timing-method ranking reversal: {case['id']}")
