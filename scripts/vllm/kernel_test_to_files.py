#!/usr/bin/env python3

import json
import os
import sys

if len(sys.argv) != 3:
    sys.exit(f"Usage: {os.path.basename(sys.argv[0])} INPUT OUTPUT")

try:
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
except Exception as e:
    sys.exit(f"Error: {e}")

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