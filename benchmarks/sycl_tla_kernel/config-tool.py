#!/usr/bin/env python3

import argparse
import re
import sys

PATTERNS = {
    'lmnk': re.compile(r'^(?P<name>\S+).*?--l=(?P<l>\d+)\s+--m=(?P<m>\d+)\s+--k=(?P<k>\d+)\s+--n=(?P<n>\d+)'),
    'mnk': re.compile(r'^(?P<name>\S+).*?--m=(?P<m>\d+)\s+--k=(?P<k>\d+)\s+--n=(?P<n>\d+)'),
}

DEFAULTS = {
    'lmnk': {'dim_type': 'Dim', 'ptr_type': 'GemmRunPtr', 'run_func': 'gemm_run'},
    'mnk': {'dim_type': 'SplitKDim', 'ptr_type': 'GemmSplitKRunPtr', 'run_func': 'gemm_splitk_run'},
}


def build_config_map(file_paths, dims):
    config_map = {}
    pattern = PATTERNS[dims]

    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = pattern.match(line.strip())
                    if match:
                        name = match.group('name')
                        m = int(match.group('m'))
                        k = int(match.group('k'))
                        n = int(match.group('n'))
                        if dims == 'lmnk':
                            config_map[(int(match.group('l')), m, n, k)] = name
                        else:
                            config_map[(m, n, k)] = name
        except IOError as e:
            print(f'Error reading {path}: {e}', file=sys.stderr)

    return config_map


def main():
    parser = argparse.ArgumentParser(description='Parse GEMM benchmark files and generate C++ table.')
    parser.add_argument('-o', '--output', required=True, help='Output file path')
    parser.add_argument('--name', required=True, help='Name identifier for the config array')
    parser.add_argument('--dims', choices=['lmnk', 'mnk'], default='lmnk',
                        help='Dimension format: lmnk (batch GEMM) or mnk (split-K)')
    parser.add_argument('--dim-type', help='C++ dimension tuple type (auto-detected from --dims)')
    parser.add_argument('--ptr-type', help='C++ function pointer type (auto-detected from --dims)')
    parser.add_argument('--run-func', help='C++ run function template (auto-detected from --dims)')
    parser.add_argument('inputs', nargs='+', help='Input file(s) with GEMM benchmark data')

    args = parser.parse_args()

    defaults = DEFAULTS[args.dims]
    dim_type = args.dim_type or defaults['dim_type']
    ptr_type = args.ptr_type or defaults['ptr_type']
    run_func = args.run_func or defaults['run_func']

    config_map = build_config_map(args.inputs, args.dims)

    try:
        with open(args.output, 'w', encoding='utf-8') as outfile:
            outfile.write('// This file was auto-generated, do not edit!\n\n')
            outfile.write(f'static constexpr std::array<std::pair<{dim_type}, {ptr_type}>, {len(config_map)}>'
                          f' {args.name} = {{{{\n')
            for key, name in config_map.items():
                dims_str = ', '.join(str(v) for v in key)
                outfile.write(f'{{ {{ {dims_str} }}, &{run_func}<{name}> }},\n')
            outfile.write('}};\n')
    except IOError as e:
        print(f'Error writing output file: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
