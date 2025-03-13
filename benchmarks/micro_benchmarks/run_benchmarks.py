import argparse

from conversion import float_conversion
from core_ops import dot_scaled

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reports',
        type=str,
        default='',
        help='directory to save reports',
    )
    args = parser.parse_args()
    float_conversion.benchmark.run(print_data=True, save_path=args.reports)
    dot_scaled.benchmark.run(print_data=True, save_path=args.reports)
