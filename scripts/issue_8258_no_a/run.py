"""Scoped baseline/A-prefetch-removal CI study; describe mode is CPU-only."""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

from transform import inspect,remove_a,sha

ROOT = Path(__file__).resolve().parent
FIXED = {'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':4,
         'grf_mode':'256','num_warps':32,'num_stages':2}


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(data,indent=2)+'\n')
    pending.replace(path)


def manifest():
    data = json.loads((ROOT/'manifest.json').read_text())
    if data['policies']!=['stock_prefetch','remove_A_prefetch'] or len(data['cases'])!=84:
        raise ValueError('Unexpected study scope')
    if len({c['id'] for c in data['cases']})!=84:
        raise ValueError('Duplicate cases')
    return data


def translate(intel,source,is_lts):
    try:
        return intel.translate_to_spirv(source,is_lts)
    except TypeError as error:
        if 'incompatible function arguments' not in str(error):
            raise
        # CPU fixtures can use the older one-argument translator ABI.
        return intel.translate_to_spirv(source)


def compile_native(backend,original,metadata,spv,folder):
    native_meta = copy.deepcopy(metadata)
    native_meta['target'] = original.metadata.target
    native_meta['cache_dir'] = str(folder)
    options = backend.parse_options(dict(metadata,generate_native_code=True))
    native = backend.make_zebin(spv,native_meta,options)
    native_meta['target'] = metadata['target']
    return native,native_meta


def worker(args,case,study):
    import torch
    import triton
    import triton.testing
    from triton._C.libtriton import intel
    from triton.compiler.compiler import CompiledKernel,make_backend
    import triton_kernels_benchmark as suite
    if not suite.BENCHMARKING_CONFIG['verify']:
        raise RuntimeError('VERIFY=1 is required')
    if 'B580' not in torch.xpu.get_device_name():
        raise RuntimeError('Expected B580')
    torch.set_grad_enabled(False)
    torch.manual_seed(8258)
    out = args.reports/'cases'/case['id']
    out.mkdir(parents=True)
    source_path = args.checkout/'benchmarks/triton_kernels_benchmark/gemm_benchmark.py'
    if sha(source_path.read_bytes())!=study['kernel_source_sha256']:
        raise RuntimeError('Pinned stock kernel source changed')
    spec = importlib.util.spec_from_file_location('triton_kernels_benchmark.gemm_benchmark',source_path)
    gemm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gemm
    spec.loader.exec_module(gemm)
    stock = gemm.matmul_kernel_with_tensor_descriptors
    stock.cache.clear()
    stock.cache_results = False
    jit = stock.fn
    if jit.arg_names[:3]!=['a_ptr','b_ptr','c_ptr']:
        raise RuntimeError('Unexpected stock user-pointer ABI')
    m,n,k = case['M'],case['N'],case['K']
    pa,pb = case['a_row_stride'],case['b_row_stride']
    a0 = torch.randn((m,k),device='xpu',dtype=torch.bfloat16)
    b0 = torch.randn((k,n),device='xpu',dtype=torch.bfloat16)
    ast = torch.zeros(m*pa,device='xpu',dtype=torch.bfloat16)
    bst = torch.zeros(k*pb+case['b_base_offset_bytes']//2,device='xpu',dtype=torch.bfloat16)
    a = ast.as_strided((m,k),(pa,1))
    b = bst.as_strided((k,n),(pb,1),storage_offset=case['b_base_offset_bytes']//2)
    a.copy_(a0);b.copy_(b0)
    c = torch.zeros((m,n),device='xpu',dtype=torch.float32)
    reference = torch.mm(a0,b0,out_dtype=torch.float32)
    rows = torch.linspace(0,m-1,16,device='xpu').long()
    cols = torch.linspace(0,n-1,16,device='xpu').long()
    cpu_reference = a0[rows].cpu().double()@b0[:,cols].cpu().double()
    kwargs = dict(M=m,N=n,K=k,stride_am=pa,stride_ak=1,stride_bk=pb,stride_bn=1,
                  stride_cm=n,stride_cn=1,transpose_a=False,transpose_b=False)
    autotune_grid = lambda meta:(triton.cdiv(m,meta['BLOCK_SIZE_M'])*triton.cdiv(n,meta['BLOCK_SIZE_N']),)
    started = time.monotonic()
    stock[autotune_grid](a,b,c,**kwargs)
    torch.xpu.synchronize()
    tune_seconds = time.monotonic()-started
    selected = stock.best_config
    stock_config = dict(selected.kwargs,num_warps=selected.num_warps,num_stages=selected.num_stages)
    configs = [('stock_config',stock_config)]
    if stock_config!=FIXED:
        configs.append(('fixed_stage2',FIXED))
    result = {'case':case,'torch':torch.__version__,'triton':triton.__version__,'triton_path':triton.__file__,
        'device':str(torch.xpu.get_device_properties(0)),'python':platform.python_version(),
        'kernel_source_sha256':sha(source_path.read_bytes()),'stock_autotune_seconds':tune_seconds,
        'stock_config':stock_config,'stock_candidates':[str(c) for c in gemm.get_matmul_autotune_configs()],
        'addresses':[a.data_ptr(),b.data_ptr(),c.data_ptr()],'variants':{},'compile_seconds':0,
        'selection_orders':[],'measurement_orders':[],'atol':1e-4,'rtol':1e-3,
        'timing':'Raw XPU GEMM activities, synchronized, preallocated output; 256MiB flush excluded.'}
    functions,keepalive = {},[]

    def loaded(folder,payload,metadata,extension):
        folder.mkdir(parents=True,exist_ok=True)
        metadata = copy.deepcopy(metadata)
        metadata.update(hash=sha(payload),cache_dir=str(folder),binary_ext=extension,generate_native_code=extension=='zebin')
        save(folder/'kernel.json',metadata)
        (folder/('kernel.'+extension)).write_bytes(payload)
        source = SimpleNamespace(constants={},signature={0:'*bf16',1:'*bf16',2:'*fp32'})
        compiled = CompiledKernel(source,{'kernel.json':str(folder/'kernel.json'),
            'kernel.'+extension:str(folder/('kernel.'+extension))},sha(payload))
        grid = (triton.cdiv(m,metadata['study_block_m'])*triton.cdiv(n,metadata['study_block_n']),1,1)
        launch = compiled[grid]
        keepalive.append(compiled)
        return compiled,lambda:launch(a,b,c)

    for label,config in configs:
        grid = (triton.cdiv(m,config['BLOCK_SIZE_M'])*triton.cdiv(n,config['BLOCK_SIZE_N']),1,1)
        original = jit.warmup(a,b,c,grid=grid,**kwargs,**config)
        metadata_file = next(Path(p) for p in original.metadata_group.values() if str(p).endswith('.json'))
        metadata = json.loads(metadata_file.read_text())
        metadata.update(study_block_m=config['BLOCK_SIZE_M'],study_block_n=config['BLOCK_SIZE_N'])
        backend = make_backend(original.metadata.target)
        is_lts = backend.is_lts(original.metadata.target.arch.get('driver_version'))
        baseline_ir = original.asm['llir']
        evidence = out/label
        evidence.mkdir(exist_ok=True)
        (evidence/'original.llir').write_text(baseline_ir)
        treatment_ir,audit = remove_a(baseline_ir)
        baseline_spv,entry = translate(intel,baseline_ir,is_lts)
        if baseline_spv!=original.asm['spv'] or entry!=original.name:
            raise RuntimeError('Unchanged LLVM roundtrip does not reproduce stock SPIR-V')
        if label=='stock_config':
            compiled,fn = loaded(out/'stock_driver',baseline_spv,metadata,'spv')
            functions['stock_driver'] = fn
            result['variants']['stock_driver'] = {'config':config,'policy':'stock_prefetch',
                'source':'Default driver-SPIR-V reference','binary_sha256':sha(baseline_spv),
                'selection_ms':[],'measurement_ms':[]}
        for policy,ir in [('baseline',baseline_ir),('remove_A',treatment_ir)]:
            key = label+'__'+policy
            if policy=='remove_A' and audit['no_op']:
                result['variants'][key] = {'config':config,'policy':'remove_A_prefetch','no_op':True,
                    'alias':label+'__baseline','prefetch_audit':audit}
                continue
            folder = out/key
            folder.mkdir()
            (folder/'kernel.llir').write_text(ir)
            save(folder/'prefetch-audit.json',audit if policy=='remove_A' else inspect(ir))
            spv,name = translate(intel,ir,is_lts)
            if name!=entry:
                raise RuntimeError('Transformed entry point changed')
            (folder/'kernel.spv').write_bytes(spv)
            start = time.monotonic()
            native,native_meta = compile_native(backend,original,metadata,spv,folder)
            result['compile_seconds'] += time.monotonic()-start
            (folder/'kernel.zebin').write_bytes(native)
            command = ['ocloc','disasm','-file',str(folder/'kernel.zebin'),'-dump',str(folder/'disassembly')]
            disasm = subprocess.run(command,capture_output=True,text=True,timeout=120)
            (folder/'ocloc.stdout').write_text(disasm.stdout)
            (folder/'ocloc.stderr').write_text(disasm.stderr)
            if disasm.returncode:
                raise RuntimeError('Native disassembly failed')
            assembly = (folder/'disassembly'/('.text.'+entry+'.asm')).read_text()
            prefetch_lines = [line for line in assembly.splitlines() if 'send.' in line and 'null:0' in line and 'load_block2d' in line and 'rd:0' in line]
            if inspect(ir)['counts']['B'] and not prefetch_lines:
                raise RuntimeError('Requested B prefetch absent from native code')
            compiled,fn = loaded(folder,native,native_meta,'zebin')
            functions[key] = fn
            result['variants'][key] = {'config':config,'policy':'stock_prefetch' if policy=='baseline' else 'remove_A_prefetch',
                'binary_sha256':sha(native),'spv_sha256':sha(spv),'prefetch_counts':inspect(ir)['counts'],
                'native_prefetch_lines':prefetch_lines,'loaded_grf_count':compiled.n_regs,'loaded_spill_size':compiled.n_spills,
                'selection_ms':[],'measurement_ms':[],'matched_native_control':label+'__baseline'}
        save(out/'result.json',result)

    def validate(key):
        c.fill_(float('nan'))
        functions[key]()
        torch.xpu.synchronize()
        if not torch.isfinite(c).all().item():
            raise RuntimeError('Non-finite or unwritten GEMM output')
        suite.assert_close(lambda:c,lambda:reference,atol=1e-4,rtol=1e-3,err_msg=key)
        sample = c[rows[:,None],cols[None,:]].cpu().double()
        triton.testing.assert_close(sample,cpu_reference,atol=1e-4,rtol=1e-3)
        return {'full_output_stock_contract':True,'cpu_fp64_sample_stock_contract':True,
                'sample_max_abs_error':(sample-cpu_reference).abs().max().item()}

    for key in functions:
        result['variants'][key]['before_validation'] = validate(key)
        for _ in range(2):
            functions[key]()
        torch.xpu.synchronize()
    flush = torch.empty(256*1024*1024//4,device='xpu',dtype=torch.int32)
    # All measured GEMM events must match the entry and launch count.
    def measure(fn,repeats):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.XPU]) as prof:
            for _ in range(repeats):
                flush.zero_();torch.xpu.synchronize()
                fn();torch.xpu.synchronize()
        events = [e for e in prof.events() if e.device_type==torch.autograd.DeviceType.XPU and entry in e.name]
        if len(events)!=repeats:
            raise RuntimeError(f'Expected {repeats} GEMM events, got {len(events)}')
        return {'mean_ms':statistics.mean(e.time_range.elapsed_us() for e in events)/1000,
                'launches':repeats,'kernel_names':sorted({e.name for e in events})}

    rng = random.Random(8258)
    selection_start = time.monotonic()
    for _ in range(2):
        order = list(functions);rng.shuffle(order);result['selection_orders'].append(order)
        for key in order:
            result['variants'][key]['selection_ms'].append(measure(functions[key],3))
    candidates = ['stock_driver']+[key for key in functions if key.endswith('__remove_A')]
    selected_policy = min(candidates,key=lambda key:statistics.median(v['mean_ms'] for v in result['variants'][key]['selection_ms']))
    result['selected_policy'] = selected_policy
    result['policy_selection_seconds'] = time.monotonic()-selection_start
    result['pre_measurement_seconds'] = time.monotonic()-started
    for round_index in range(3):
        order = list(functions);rng.shuffle(order);result['measurement_orders'].append(order)
        for key in order:
            result['variants'][key]['measurement_ms'].append(measure(functions[key],7))
        save(out/'result.json',result)
        print(case['id'],'ROUND',round_index+1,flush=True)
    for key in functions:
        result['variants'][key]['after_validation'] = validate(key)
    if not torch.equal(a,a0) or not torch.equal(b,b0):
        raise RuntimeError('Read-only inputs changed')
    result.update(complete=True,inputs_unchanged=True)
    save(out/'result.json',result)
    print('COMPLETE',case['id'],'selected',selected_policy,flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--describe',action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--reports',type=Path,default=Path('reports'))
    parser.add_argument('--checkout',type=Path,default=Path.cwd())
    args = parser.parse_args()
    args.reports = args.reports.resolve();args.checkout = args.checkout.resolve()
    study = manifest()
    if args.describe:
        print(json.dumps({k:v for k,v in study.items() if k!='cases'},indent=2));return
    if args.worker:
        matches = [c for c in study['cases'] if c['id']==args.worker]
        if len(matches)!=1:
            raise ValueError('Worker case is outside manifest')
        worker(args,matches[0],study);return
    args.reports.mkdir(parents=True,exist_ok=True)
    save(args.reports/'manifest.json',study)
    provenance = {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.checkout,text=True).strip(),
                  'manifest_sha256':sha((ROOT/'manifest.json').read_bytes()),'python':platform.python_version()}
    save(args.reports/'provenance.json',provenance)
    completed = []
    for case in study['cases']:
        log = args.reports/(case['id']+'.log')
        command = [sys.executable,str(Path(__file__).resolve()),'--worker',case['id'],
                   '--reports',str(args.reports),'--checkout',str(args.checkout)]
        with log.open('w') as output:
            try:
                subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,check=True,timeout=1800)
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                output.flush()
                print(f'Worker failed: {case["id"]}; last 40 log lines:',file=sys.stderr,flush=True)
                print('\n'.join(log.read_text().splitlines()[-40:]),file=sys.stderr,flush=True)
                raise
        completed.append(case['id'])
        save(args.reports/'progress.json',{'completed':completed,'total':84})
        print('CASE',len(completed),'/84',case['id'],flush=True)
    subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--reports',str(args.reports)],check=True)


if __name__=='__main__':
    main()
