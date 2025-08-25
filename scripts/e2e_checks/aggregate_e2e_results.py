import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate end-to-end test results')
    parser.add_argument('--input-dir', '-i', type=str, required=True, help='Input directory containing test results')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory for aggregated results')
    return parser.parse_args()


def parse_folder_name(folder_name):
    """
    Parse folder name to extract suite and dtype.

    Expected format: logs-{suite}-{dtype}-{mode}-accuracy, where mode can contain `-` characters
    Examples:
    - logs-torchbench-float32-inference-accuracy -> suite=torchbench, dtype=float32
    - logs-huggingface-amp_bf16-training-accuracy -> suite=huggingface, dtype=amp_bf16
    """
    parts = folder_name.split('-')

    # Check if it follows the expected pattern
    if len(parts) < 4 or parts[0] != 'logs' or parts[-1] != 'accuracy':
        return None, None, None

    suite = parts[1]
    dtype = parts[2]
    # Extract mode, can include dashes
    mode = '-'.join(parts[3:-1])

    return suite, dtype, mode


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

    dfs = []
    for item_path in input_path.iterdir():
        name = item_path.name
        if not item_path.is_dir():
            continue

        suite, dtype, mode = parse_folder_name(name)
        if suite is None:
            print(f'Folder name \'{name}\' does not match expected pattern, skipping')
            continue
        filepath = item_path / suite / dtype / f'inductor_{suite}_{dtype}_{mode}_xpu_accuracy.csv'
        df = pd.read_csv(filepath)
        df['suite'] = suite
        df['mode'] = mode
        df['dtype'] = dtype
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(['suite', 'mode', 'dtype'])

    # Artifacts
    # 1. Simple concat of all with added suite, mode, dtype
    combined_df.to_csv(output_path / 'combined_results.csv', index=False)
    # 2. torch format report, 9 items (suite, mode), dtype stored as column
    build_pytorch_report(combined_df, output_path=output_path)
    # 3. Agg report with 45 rows (suite, mode, dtype, passed, failed_REASON, failed_REASON model list)
    build_suite_report(combined_df, output_path=output_path)


if __name__ == '__main__':
    args = parse_args()
    main(args.input_dir, args.output_dir)
