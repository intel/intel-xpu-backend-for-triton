"""Verify portable benchmark inputs before starting GPU measurements."""
import hashlib
import json
import os
from pathlib import Path


def verify():
    root = Path(os.environ['GEMM_CI_REPORTS'])
    protocol = json.loads((root / 'protocol.json').read_text())
    for info in protocol['families'].values():
        for field in ('source', 'kernel_source'):
            path = Path(info[field])
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != info[field + '_sha256']:
                raise RuntimeError(f'Changed benchmark input: {path}')
