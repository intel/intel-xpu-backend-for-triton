#!/usr/bin/env python3

import argparse
import re
import sys


def build_config_map(file_paths):
    config_map = {}
    pattern = re.compile(
        r'^(?P<name>\S+).*?--l=(?P<l>\d+)\s+--m=(?P<m>\d+)\s+--k=(?P<k>\d+)\s+--n=(?P<n>\d+)'
    )

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = pattern.match(line.strip())
                    if match:
                        name = match.group('name')
                        l = int(match.group('l'))
                        m = int(match.group('m'))
                        k = int(match.group('k'))
                        n = int(match.group('n'))
                        config_map[(l, m, n, k)] = name
        except IOError as e:
            print(f'Error reading {path}: {e}', file=sys.stderr)

    return config_map


def main():
    parser = argparse.ArgumentParser(
        description='Parse GEMM benchmark files and generate C++ table.'
    )
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('--name', required=True, help='Name identifier for logging or grouping')
    parser.add_argument('inputs', nargs='+', help='Input file(s) with GEMM benchmark data')

    args = parser.parse_args()

    config_map = build_config_map(args.inputs)

    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            outfile.write('// This file was auto-generated, do not edit!\n\n')
            outfile.write(f'static constexpr std::array<std::pair<Dim, GemmRunPtr>, {len(config_map)}> {args.name} = {{{{\n')
            for (l, m, n, k), name in config_map.items():
                outfile.write(f'{{ {{ {l}, {m}, {n}, {k} }}, &gemm_run<{name}> }},\n')
            outfile.write('}};\n')
    except IOError as e:
        print(f'Error writing output file: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
