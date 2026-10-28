"""Audit complete scope and report both stock-relative and matched-control effects."""
import argparse
import json
import math
from pathlib import Path
import statistics


def effect(control,values):
    if len(control)!=len(values) or len(values)!=3:
        raise ValueError('Expected three paired rounds')
    return math.expm1(statistics.mean(math.log(y['mean_ms']/x['mean_ms']) for x,y in zip(control,values)))*100


def summarize(reports,quiet=False):
    manifest = json.loads((reports/'manifest.json').read_text())
    expected = {c['id'] for c in manifest['cases']}
    files = list((reports/'cases').glob('*/result.json'))
    found = {p.parent.name for p in files}
    if expected!=found or len(files)!=84:
        raise ValueError(f'Missing/extra cases: {expected^found}')
    rows = []
    no_ops = 0
    for path in sorted(files):
        data = json.loads(path.read_text())
        if not data.get('complete') or not data['inputs_unchanged']:
            raise ValueError('Incomplete or mutated-input case')
        variants = data['variants']
        reference = variants['stock_driver']['measurement_ms']
        for name,record in variants.items():
            if record.get('no_op'):
                no_ops+=1;continue
            if not record['before_validation']['full_output_stock_contract'] or not record['after_validation']['full_output_stock_contract']:
                raise ValueError('Correctness not established')
            record['stock_relative_time_change_pct'] = effect(reference,record['measurement_ms'])
            if name.endswith('__remove_A'):
                control = variants[record['matched_native_control']]['measurement_ms']
                record['matched_native_time_change_pct'] = effect(control,record['measurement_ms'])
        chosen = variants[data['selected_policy']]
        row = {'id':data['case']['id'],'case':data['case'],'selected':data['selected_policy'],
            'stock_config':data['stock_config'],'stock_ms':statistics.median(v['mean_ms'] for v in reference),
            'selected_ms':statistics.median(v['mean_ms'] for v in chosen['measurement_ms']),
            'selected_time_change_pct':chosen['stock_relative_time_change_pct'],
            'matched_native_time_change_pct':chosen.get('matched_native_time_change_pct'),
            'native_baseline_vs_driver_pct':variants['stock_config__baseline']['stock_relative_time_change_pct'],
            'study_preparation_and_selection_seconds':data['pre_measurement_seconds'],
            'variants':variants}
        saved_ms = row['stock_ms']-row['selected_ms']
        row['study_break_even_calls'] = math.ceil(row['study_preparation_and_selection_seconds']*1000/saved_ms) if saved_ms>0 else None
        rows.append(row)
    summary = {}
    for label,subset in [('contiguous',[r for r in rows if r['case']['layout']=='contiguous']),
                         ('layout_stress',[r for r in rows if r['case']['layout']!='contiguous'])]:
        values = [r['selected_time_change_pct'] for r in subset]
        summary[label] = {'cases':len(values),'geometric_mean_time_change_pct':math.expm1(statistics.mean(math.log1p(v/100) for v in values))*100,
            'wins_over_10pct':sum(v <= -10+1e-9 for v in values),'regressions_over_3pct':sum(v>3+1e-9 for v in values),
            'worst_time_change_pct':max(values),'selected_A_removal':sum(r['selected']!='stock_driver' for r in subset),
            'wins_10pct_with_matched_native_gain_over_2pct':sum(r['selected_time_change_pct']<=-10+1e-9 and r['matched_native_time_change_pct'] is not None and r['matched_native_time_change_pct'] < -2 for r in subset),
            'native_stock_path_difference_over_3pct':sum(abs(r['native_baseline_vs_driver_pct'])>3 for r in subset)}
    result = {'complete':True,'case_count':84,'no_op_configuration_policies':no_ops,'summary':summary,'cases':rows,
        'limits':'Selection and measurement samples are separate. Three paired rounds are a screening result, not independent allocation confirmation. Native baseline versus default driver-SPIR-V differences are explicit controls; removal attribution also requires the matched native effect. Break-even uses measured study preparation including evidence and validation overhead, not an estimate of production tuner cost.'}
    (reports/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['| Case | Selected | Stock ms | Selected ms | Change vs stock | Change vs matched native | Native baseline vs driver |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        matched = f"{r['matched_native_time_change_pct']:+.2f}%" if r['matched_native_time_change_pct'] is not None else '—'
        lines.append(f"| {r['id']} | {r['selected']} | {r['stock_ms']:.5f} | {r['selected_ms']:.5f} | {r['selected_time_change_pct']:+.2f}% | {matched} | {r['native_baseline_vs_driver_pct']:+.2f}% |")
    (reports/'results.md').write_text('\n'.join(lines)+'\n')
    if not quiet:
        print(json.dumps({'cases':84,'no_ops':no_ops,'summary':summary},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--reports',type=Path,required=True)
    summarize(parser.parse_args().reports.resolve())
