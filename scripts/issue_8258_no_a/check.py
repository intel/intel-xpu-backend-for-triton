"""CPU guards for provenance, no-ops, unsupported calls, manifest and reporting."""
import argparse
import json
from pathlib import Path
import tempfile

from transform import inspect,remove_a

ROOT = Path(__file__).resolve().parent
FIXTURE = '''define spir_kernel void @kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, ptr addrspace(1) %scratch) {
  %ai = ptrtoint ptr addrspace(1) %A to i64
  %ab = and i64 %ai, -64
  %bi = ptrtoint ptr addrspace(1) %B to i64
  %bb = and i64 %bi, -64
  tail call spir_func void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %ab, i32 4)
  tail call spir_func void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %bb, i32 4)
  ret void
}
'''
SPIRV_FIXTURE = '''define spir_kernel void @kernel(ptr addrspace(1) %A, ptr addrspace(1) %B, ptr addrspace(1) %C, ptr addrspace(1) %scratch) {
  %ai = ptrtoint ptr addrspace(1) %A to i64
  %ab = and i64 %ai, -64
  %23 = inttoptr i64 %ab to ptr addrspace(1)
  %bi = ptrtoint ptr addrspace(1) %B to i64
  %bb = and i64 %bi, -64
  %bp = inttoptr i64 %bb to ptr addrspace(1)
  tail call spir_func void @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i(i32 2, i32 32, i32 8, i32 1, ptr addrspace(1) nonnull %23, i32 %26, i32 512, i32 4096, <2 x i32> %29) #7, !spirv.DecorationCacheControlINTEL !53
  tail call spir_func void @_Z36__spirv_Subgroup2DBlockPrefetchINTELiiiiPU3AS1viiiDv2_i(i32 2, i32 32, i32 8, i32 1, ptr addrspace(1) nonnull %bp, i32 %26, i32 2048, i32 8192, <2 x i32> %29) #7, !spirv.DecorationCacheControlINTEL !53
  ret void
}
'''


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--fixtures-root',type=Path);args=parser.parse_args()
    mixed = SPIRV_FIXTURE.replace('  ret void',
        '  tail call spir_func void @llvm.genx.GenISA.LSC2DBlockPrefetch.isVoid(i64 %ab, i32 4)\n  ret void')
    for fixture,a_count in [(FIXTURE,1),(SPIRV_FIXTURE,1),(mixed,2)]:
        modified,audit = remove_a(fixture)
        assert audit['before']['counts']=={'A':a_count,'B':1} and audit['after']['counts']=={'A':0,'B':1}
        assert remove_a(modified)[1]['no_op']
        assert all(c['text'] in modified for c in audit['before']['calls'] if c['owner']=='B')
    for bad in [FIXTURE.replace('i64 %ab, i32 4','i64 %unknown, i32 4'),
                FIXTURE.replace('LSC2DBlockPrefetch.isVoid','OtherPrefetch'),
                FIXTURE.replace('%ab = and i64 %ai, -64','%ab = and i64 %ai, %bi'),
                FIXTURE.replace('ptrtoint ptr addrspace(1) %A','ptrtoint ptr addrspace(1) %scratch'),
                SPIRV_FIXTURE.replace('nonnull %23,','nonnull %unknown,'),
                SPIRV_FIXTURE.replace('nonnull %23,','nonnull %C,'),
                SPIRV_FIXTURE.replace('nonnull %23,','nonnull %scratch,'),
                SPIRV_FIXTURE.replace('%ab = and i64 %ai, -64','%ab = and i64 %ai, %bi'),
                SPIRV_FIXTURE.replace('PrefetchINTELiiiiPU3AS1viiiDv2_i','PrefetchINTEL_unsupported')]:
        try:remove_a(bad)
        except ValueError:pass
        else:raise AssertionError('Unsafe provenance was accepted')
    count = 0
    if args.fixtures_root:
        for path in args.fixtures_root.glob('*/*/kernel.llir'):
            remove_a(path.read_text());count+=1
        assert count>0
    import run
    study = run.manifest()
    assert study['logical_shape_count']==54 and len(study['cases'])==84
    from analyze import summarize
    with tempfile.TemporaryDirectory() as tmp:
        reports=Path(tmp);(reports/'manifest.json').write_text(json.dumps(study))
        for case in study['cases']:
            folder=reports/'cases'/case['id'];folder.mkdir(parents=True)
            variants={}
            for key,time in [('stock_driver',10),('stock_config__baseline',10.2),('stock_config__remove_A',9)]:
                if key=='stock_config__remove_A' and case==study['cases'][0]:
                    time=11
                variants[key]={'measurement_ms':[{'mean_ms':time}]*3,'before_validation':{'full_output_stock_contract':True},
                    'after_validation':{'full_output_stock_contract':True},'matched_native_control':'stock_config__baseline'}
            variants['fixed_stage2__remove_A']={'no_op':True,'alias':'stock_config__baseline'}
            data={'complete':True,'inputs_unchanged':True,'case':case,'stock_config':{},'variants':variants,
                'selected_policy':'stock_config__remove_A','stock_autotune_seconds':1,'compile_seconds':1,'policy_selection_seconds':1,'pre_measurement_seconds':3}
            (folder/'result.json').write_text(json.dumps(data))
        summarize(reports,quiet=True)
        result=json.loads((reports/'summary.json').read_text());assert result['case_count']==84
        loss=next(c for c in result['cases'] if c['id']==study['cases'][0]['id'])
        assert abs(loss['selected_time_change_pct']-10)<1e-9 and loss['study_break_even_calls'] is None
        assert result['no_op_configuration_policies']==84
    print(f'CPU checks passed; {count} additional saved LLVM fixtures. No GPU libraries imported.')


if __name__=='__main__':
    main()
