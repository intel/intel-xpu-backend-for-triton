from defusedxml.ElementTree import parse
import argparse


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str, help='input XML file')
    parser.add_argument('output_file', type=str, help='output TXT file')
    return parser


def extract_failed_from_xml(in_file: str, out_file: str):
    """Process XML log file and output failed cases."""
    root = parse(in_file).getroot()
    failed = []

    for testcase in root.findall('.//testcase'):
        for child in testcase:
            if child.tag == 'error' or child.tag == 'failure':
                classname = testcase.get('classname').replace('.', '/') + '.py'
                case = testcase.get('name')
                result = f'{classname}::{case}'
                failed.append(result)

    if len(failed) == 0:
        return

    with open(out_file, 'w') as f:
        for result in failed:
            f.write(result + '\n')


def main():
    """Main."""
    args = create_argument_parser().parse_args()
    extract_failed_from_xml(args.input_file, args.output_file)


if __name__ == '__main__':
    main()
