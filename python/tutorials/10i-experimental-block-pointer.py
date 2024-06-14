"""
Block Pointer (Experimental)
============================
Runs Block Pointer tutorial with TRITON_INTEL_ENABLE_BLOCK_PTR=1.
"""

import os.path
import runpy


if __name__ == '__main__':
    os.environ['TRITON_INTEL_ENABLE_BLOCK_PTR'] = '1'
    basename = os.path.basename(__file__)
    runpy.run_path(f'{basename}/10-experimental-block-pointer.py')
