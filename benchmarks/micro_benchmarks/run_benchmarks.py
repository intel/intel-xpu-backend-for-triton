import argparse

from conversion import float_conversion

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--reports',
        type=str,
        default='',
        help='directory to save reports',
    )
    args = parser.parse_args()
    float_conversion.benchmark.run(print_data=True, save_path=args.reports)
