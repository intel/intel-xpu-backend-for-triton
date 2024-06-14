"""
Block Pointer (Experimental)
============================
Runs Block Pointer tutorial with TRITON_INTEL_ENABLE_BLOCK_PTR=1.
"""

import os.path
import runpy

if __name__ == '__main__':
    os.environ['TRITON_INTEL_ENABLE_BLOCK_PTR'] = '1'
    dirname = os.path.dirname(__file__)
    runpy.run_path(f'{dirname}/10-experimental-block-pointer.py')
