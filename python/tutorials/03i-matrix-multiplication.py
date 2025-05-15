"""
Matrix Multiplication
=====================
Runs Matrix Multiplication tutorial with TRITON_INTEL_RAISE_BLOCK_POINTER=1.
"""

import os.path
import runpy

if __name__ == '__main__':
    os.environ['TRITON_INTEL_RAISE_BLOCK_POINTER'] = '1'
    dirname = os.path.dirname(__file__)
    runpy.run_path(f'{dirname}/03-matrix-multiplication.py')
