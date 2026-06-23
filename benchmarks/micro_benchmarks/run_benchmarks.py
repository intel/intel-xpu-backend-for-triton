from conversion import float_conversion
from core_ops import dot_scaled
from core_ops import descriptor_load

if __name__ == '__main__':
    for mod in (float_conversion, dot_scaled, descriptor_load):
        mod.run_benchmarks()
