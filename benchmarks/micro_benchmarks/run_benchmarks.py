from conversion import float_conversion
from core_ops import dot_scaled

if __name__ == '__main__':
    for mod in (float_conversion, dot_scaled):
        mod.run_benchmarks()
