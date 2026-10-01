"""Materialize balanced CI tuning parts without splitting profile files."""
import argparse
import json
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DTYPES = ('bfloat16', 'float8_e4m3fn')


def profile_group(case):
    # CI uses TD and unquantized/native-FP8 caches for every input.
    return (case['block_size'], case['head_size'], case['q_heads'] // case['kv_heads'],
            case['dtype'], case['out_dtype'], case['q_layout'], case['kv_layout'],
            bool(case['softcap']), case['sliding_window'])


def partition(manifests, assignment):
    cases = [case for dtype in DTYPES for case in manifests[dtype]]
    expected = [case['id'] for case in cases]
    assigned = [name for names in assignment.values() for name in names]
    if set(assignment) != {'1', '2'} or any(not names for names in assignment.values()):
        raise ValueError('Expected two nonempty tuning parts')
    if len(set(expected)) != len(expected) or len(set(assigned)) != len(assigned) or set(assigned) != set(expected):
        raise ValueError('Each CI input must appear in exactly one part')
    owners = {name: part for part, names in assignment.items() for name in names}
    groups = {}
    for case in cases:
        owner = owners[case['id']]
        key = profile_group(case)
        if groups.setdefault(key, owner) != owner:
            raise ValueError(f"Profile group split across parts: {case['id']}")
    result = {part: {dtype: [case for case in manifests[dtype] if owners[case['id']] == part]
                     for dtype in DTYPES} for part in assignment}
    if any(not cases for part in result.values() for cases in part.values()):
        raise ValueError('Each tuning part must contain both dtypes')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=('1', '2'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = {dtype: json.loads((SCRIPTS / f'ua-ci-inputs-{dtype}.json').read_text()) for dtype in DTYPES}
    assignment = json.loads((SCRIPTS / 'ua-ci-tuning-shards.json').read_text())
    parts = partition(manifests, assignment)
    args.output.mkdir(parents=True, exist_ok=True)
    for dtype, cases in parts[args.part].items():
        (args.output / f'{dtype}.json').write_text(json.dumps(cases, indent=2) + '\n')
        print(f'Part {args.part}: {len(cases)} {dtype} inputs', flush=True)


if __name__ == '__main__':
    main()
