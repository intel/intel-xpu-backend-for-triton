import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate end-to-end test results')
    parser.add_argument('--input-dir', '-i', type=str, required=True, help='Input directory containing test results')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory for aggregated results')
    return parser.parse_args()


def build_suite_report(combined_df, output_path):
    print('=======================================')
    print('=           SUMMARY REPORT            =')
    print('=======================================')
    assert combined_df.groupby(['suite', 'mode', 'dtype', 'batch_size',
                                'name']).count().max().max() == 1, 'Discovered unexpected duplicates in results!'

    def fn(df):
        results = df['accuracy'].value_counts().to_dict()
        errors = df[~df['accuracy'].str.startswith('pass')]
        errors = errors.groupby('accuracy')['name'].apply(';'.join).to_dict()

        return results, errors

    agg = combined_df.groupby(['suite', 'mode', 'dtype']).apply(fn, include_groups=False)

    for index, row in agg.items():
        n_pass = sum(c for k, c in row[0].items() if k.startswith('pass'))
        n_total = sum(row[0].values())

        join_parts = []
        for k, v in row[0].items():
            if 'pass' in k:
                join_parts.append(f'{k}={v}')
            else:
                join_parts.append(f'{k}={v}[{row[1][k]}]')

        txt = f'suite={index[0]},mode={index[1]},dtype={index[2]},' + \
        f'passrate={n_pass / n_total if n_total > 0 else 0:.1%},' + \
        ','.join(join_parts)

        print(txt)

    # Unpack errors and failed models into new columns
    agg = agg.apply(lambda x: pd.Series({**x[0], **{k + '_models': v for k, v in x[1].items()}}))
    agg = agg.reset_index().fillna(0)

    agg.to_csv(output_path / 'summary_agg.csv', index=False)


def drop_duplicates(df, suite, mode):
    """ Some (name, dtype) groups can have duplicates, let's print them """
    group_counts = df.groupby(['name', 'dtype']).size()
    duplicates = group_counts[group_counts > 1]

    if not duplicates.empty:
        print(f'Found {len(duplicates)} duplicate groups for {suite} {mode}:')
        for (name, dtype), _ in duplicates.items():
            print(df[df['name'].eq(name) & df['dtype'].eq(dtype)])
            print()
    return df.groupby(['name', 'dtype'], as_index=False).first()


def build_pytorch_report(combined_df, output_path):
    print('====================\nBuiling pytorch report\n====================')
    cols = ['name', 'float32', 'bfloat16', 'float16', 'amp_bf16', 'amp_fp16']

    torch_report_dir = output_path / 'torch_format_report'
    torch_report_dir.mkdir(parents=True, exist_ok=True)
    for suite, mode in combined_df[['suite', 'mode']].drop_duplicates().values:
        df_subset = combined_df[combined_df['suite'].eq(suite)
                                & combined_df['mode'].eq(mode)][['dtype', 'name', 'accuracy']]

        df_subset = drop_duplicates(df_subset, suite, mode)
        pivoted_df = df_subset.pivot(index='name', columns='dtype', values='accuracy')

        # Reset index to make 'name' a regular column
        pivoted_df = pivoted_df.reset_index()

        # Fill NaN values if some dtype/name combinations don't exist
        pivoted_df = pivoted_df.fillna('')

        pivoted_df = pivoted_df[[c for c in cols if c in pivoted_df.columns]]

        pivoted_df.to_csv(torch_report_dir / f'inductor_{suite}_{mode}.csv', index=False)


def parse_mode(report_path):
    """
    Parse report file path to extract `mode`.

    Expected filename: 'inductor_{suite}_{dtype}_{mode}_xpu_accuracy.csv', where mode can contain `-` characters
    and dtype can contain `_` characters (e.g., `amp_bf16`).

    Returns:
        mode (str): Extracted mode from the filename
        error (str or None): Error message if parsing fails, otherwise None
    """
    parts = report_path.name.split('_')

    # Check if it follows the expected pattern
    if len(parts) < 6 or parts[0] != 'inductor' or parts[-1] != 'accuracy.csv':
        txt = f'Unexpected filename format: {report_path.name}, parsed parts: {parts}'
        print(txt)
        return None, txt
    return parts[-3], None


def load_reports(input_path):
    dfs = []
    problems = []
    for suite_path in filter(Path.is_dir, input_path.iterdir()):
        suite = suite_path.name

        for dtype_path in filter(Path.is_dir, suite_path.iterdir()):
            dtype = dtype_path.name

            for report_path in dtype_path.glob('inductor_*_xpu_accuracy.csv'):
                print(f'Reading {report_path}')
                mode, problem = parse_mode(report_path)
                if mode is None:
                    problems.append(problem)
                    continue
                df = pd.read_csv(report_path)
                df['suite'] = suite
                df['mode'] = mode
                df['dtype'] = dtype
                dfs.append(df)
    return dfs, problems


def main(input_dir, output_dir):
    """
    Main function to aggregate end-to-end test results.

    Args:
        input_dir (str): Path to input directory containing test results
        output_dir (str): Path to output directory for aggregated results
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f'Input directory does not exist: {input_path}')

    output_path.mkdir(parents=True, exist_ok=True)

    print(f'Processing results from: {input_path}')
    print(f'Output will be saved to: {output_path}')

    dfs, problems = load_reports(input_path)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['suite', 'mode', 'dtype'])

    # Artifacts
    # 1. Simple concat of all with added suite, mode, dtype
    combined_df.to_csv(output_path / 'combined_results.csv', index=False)
    # 2. torch format report, 9 items (suite, mode), dtype stored as column
    build_pytorch_report(combined_df, output_path=output_path)
    # 3. Agg report with 45 rows (suite, mode, dtype, passed, failed_REASON, failed_REASON model list)
    build_suite_report(combined_df, output_path=output_path)

    if problems:
        print('Problems found during parsing:')
        for problem in problems:
            print(problem)
        raise RuntimeError('Errors found during parsing, see above')


if __name__ == '__main__':
    args = parse_args()
    main(args.input_dir, args.output_dir)
