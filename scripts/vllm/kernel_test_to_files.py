#!/usr/bin/env python3

# Transform a kernel->tests map into a kernel->test-files map.

import json
import sys

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} INPUT OUTPUT")

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump(
        {kernel: sorted({test.split("::", 1)[0]
                         for test in tests})
         for kernel, tests in data.items()},
        f,
        indent=4,
        sort_keys=True,
    )
    f.write("\n")
