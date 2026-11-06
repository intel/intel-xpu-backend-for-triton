"""Authorized matched-input correctness, independent selection and fixed-config timing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import traceback
from types import SimpleNamespace
from compile import save
from gpu_guard import acquire,assert_exclusive
from freeze_inputs import verify
from native_image import fingerprint

ROOT=Path(os.environ['GEMM_CI_REPORTS']).resolve()
POLICIES=['baseline','intervention','learned']

def sha(data):return hashlib.sha256(data).hexdigest()
def med(rounds):return statistics.median(statistics.median(row) for row in rounds)
def sequence_id(record):
    return sha(json.dumps([(fingerprint(Path(kernel['folder'])/'kernel.zebin'),kernel['launch'],kernel['runtime_signature'],kernel['pointer_names'],kernel['execution_metadata']) for kernel in record['kernels']],sort_keys=True).encode())

def allocation(torch,triton,key,case,audit,protocol,skew,output,winners=None):
    from triton.compiler.compiler import CompiledKernel
    family=case['baseline']['family'];m,n,k=case['baseline']['shape']
    torch.manual_seed(protocol['seed'])
    guard=torch.zeros(skew*1024*1024,device='xpu',dtype=torch.uint8)
    a_pad=case['baseline']['a_pad'];b_pad=case['baseline']['b_pad']
    a_storage=torch.randint(-4,5,(k,m+a_pad) if family=='transpose_a' else (m,k+a_pad),device='xpu',dtype=torch.int16).to(torch.bfloat16)/8
    a=a_storage[:,:m].T if family=='transpose_a' else a_storage[:,:k]
    b_storage=torch.randint(-4,5,(k,n+b_pad),device='xpu',dtype=torch.int16).to(torch.bfloat16)/8
    b=b_storage[:,:n]
    c=torch.empty((m,n),device='xpu',dtype=torch.float32)
    d=torch.randint(-4,5,(m,n),device='xpu',dtype=torch.int16).float()/8 if family=='add' else None
    reference_a=torch.exp(a.float()).to(torch.bfloat16) if family=='exp' else a
    reference=torch.bmm(reference_a[None],b[None],out_dtype=torch.float32)[0]
    if family=='gelu':reference=torch.nn.functional.gelu(reference,approximate='tanh')
    if family=='add':reference=reference+d
    rows=torch.linspace(0,m-1,min(m,16),device='xpu').long();cols=torch.linspace(0,n-1,min(n,16),device='xpu').long()
    sampled_a=a[rows,:].cpu().double();sampled_b=b[:,cols].cpu().double()
    sample_effective=torch.exp(sampled_a.float()).to(torch.bfloat16).double() if family=='exp' else sampled_a
    cpu_reference=sample_effective@sampled_b
    if family=='gelu':cpu_reference=.5*cpu_reference*(1+torch.tanh(math.sqrt(2/math.pi)*(cpu_reference+.044715*cpu_reference**3)))
    if family=='add':cpu_reference=cpu_reference+d[rows[:,None],cols[None,:]].cpu().double()
    flush=torch.empty(256*1024*1024//4,device='xpu',dtype=torch.int32)
    result={'complete':False,'started':time.time(),'pid':os.getpid(),
            'provenance':{name:sha((ROOT/name).read_bytes()) for name in ['protocol.json','suite.json','compiler-manifest.json','audit.json','run.py']},'case':key,'family':family,'shape':[m,n,k],'skew_mib':skew,
            'strides':{'A':list(a.stride()),'B':list(b.stride())},'a_pad':a_pad,'b_pad':b_pad,
            'addresses':{'guard':guard.data_ptr(),'A':a.data_ptr(),'B':b.data_ptr(),'C':c.data_ptr(),'D':d.data_ptr() if d is not None else None},
            'input_sample_sha256':sha(sampled_a.numpy().tobytes()+sampled_b.numpy().tobytes()),
            'mapping':{},'variants':{},'valid_configs':[],'selection_orders':[],'confirmation_orders':[],
            'fixed_config_indices':[], 'failures':[]}
    mapping=result['mapping'];functions={};objects=[];aliases={}
    changed=audit['changed_configs']
    indices=audit['common_compile_configs'] if winners is None else sorted(set(changed+[winners[p] for p in POLICIES]+winners.get('_controls',[])),key=int)
    pointers={'a_ptr':a,'b_ptr':b,'c_ptr':c,'d_ptr':d,'A':a,'B':b,'C':c}

    def persist():save(output,result)
    def validate(identifier):
        c.fill_(float('nan'));functions[identifier]();torch.xpu.synchronize()
        for start in range(0,m,128):
            torch.testing.assert_close(c[start:start+128],reference[start:start+128],atol=protocol['atol'],rtol=protocol['rtol'])
        actual=c[rows[:,None],cols[None,:]].cpu().double()
        torch.testing.assert_close(actual,cpu_reference,atol=protocol['atol'],rtol=protocol['rtol'])
        return {'full_output_close':True,'cpu_fp64_sample_max_abs_error':(actual-cpu_reference).abs().max().item()}

    for index in indices:
        good=True
        for policy in POLICIES:
            identifier=f'{index}:{policy}';record=case[policy]['configs'][index]
            entry=result['variants'][identifier]={'status':'pending'}
            if record['status']!='compiled':good=False;entry.update(status='compile_failed');continue
            identity=sequence_id(record)
            if identity in aliases:
                mapping[identifier]=aliases[identity];entry.update(status='valid',alias=aliases[identity]);continue
            launches=[]
            try:
                for kernel in record['kernels']:
                    folder=Path(kernel['folder']);native=(folder/'kernel.zebin').read_bytes()
                    if sha(native)!=kernel['sha256']:raise RuntimeError('Native artifact hash mismatch')
                    source=SimpleNamespace(constants={},signature={int(i):v for i,v in kernel['runtime_signature'].items()})
                    compiled=CompiledKernel(source,{'kernel.json':str(folder/'kernel.json'),'kernel.zebin':str(folder/'kernel.zebin')},kernel['sha256']);objects.append(compiled)
                    grid=tuple(kernel['launch']['grid'])
                    if math.prod(grid):launches.append((compiled[grid],[pointers[name] for name in kernel['pointer_names']]+kernel['launch']['scalars']))
                def execute(launches=launches):
                    if family in ['splitk','streamk']:c.zero_()
                    for launch,args in launches:launch(*args)
                functions[identifier]=execute;mapping[identifier]=identifier
                checked=validate(identifier)
                aliases[identity]=identifier;entry.update(status='valid',before=checked,image_fingerprint=identity)
            except Exception as error:
                good=False;entry.update(status='invalid',error=repr(error),traceback=traceback.format_exc())
                result['failures'].append({'config':index,'policy':policy,'error':repr(error)})
                functions.pop(identifier,None);mapping.pop(identifier,None)
                print('INVALID',key,index,policy,repr(error),flush=True)
            persist()
        if good:result['valid_configs'].append(index)
    valid=result['valid_configs']
    if not valid:raise RuntimeError('No common correctness-valid configuration')
    warm=mapping[f'{valid[0]}:baseline'];functions[warm]();torch.xpu.synchronize()
    warm_start,warm_end=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
    warm_start.record();functions[warm]();warm_end.record();warm_end.synchronize()
    warm_repeats=max(1,min(4096,math.ceil(50/max(warm_start.elapsed_time(warm_end),.001))))
    rng=random.Random(protocol['seed']+skew)
    result['frequency_samples']=[]

    def measure(keys,rounds,field):
        result[field+'_ms']={identifier:[] for identifier in keys}
        result.setdefault(field+'_orders',[])
        for rnd in range(rounds):
            for _ in range(warm_repeats):functions[warm]()
            torch.xpu.synchronize()
            order=list(keys);rng.shuffle(order);result[field+'_orders'].append(order)
            try:freq=Path('/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq').read_text().strip()
            except OSError:freq=None
            result['frequency_samples'].append({'phase':field,'round':rnd,'mhz':freq})
            for identifier in order:
                if field=='probe':
                    times=[]
                    for _ in range(2):
                        flush.zero_();torch.xpu.synchronize()
                        begin,end=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
                        begin.record();functions[identifier]();end.record();end.synchronize();times.append(begin.elapsed_time(end))
                else:times=triton.testing.do_bench(functions[identifier],return_mode='all')
                if not times or any(not math.isfinite(value) or value<=0 for value in times):raise RuntimeError('Invalid timing')
                result[field+'_ms'][identifier].append(times)
            assert_exclusive();persist()
            print(field.upper(),key,skew,rnd+1,flush=True)

    if winners is None:
        keys=sorted({mapping[f'{index}:{policy}'] for index in valid for policy in POLICIES})
        measure(keys,1,'probe');best={p:min(med(result['probe_ms'][mapping[f'{index}:{p}']]) for index in valid) for p in POLICIES}
        retained=list(valid)
        result['pruning']={'factor':protocol['prune_factor'],'retained':retained,'excluded':[], 'disabled':True}
        keys=sorted({mapping[f'{index}:{policy}'] for index in retained for policy in POLICIES})
        measure(keys,protocol['selection_rounds'],'selection')
        winners={policy:min(retained,key=lambda index:(med(result['selection_ms'][mapping[f'{index}:{policy}']]),int(index))) for policy in POLICIES}
        from oracle import select_oracle
        winners['_oracle']=select_oracle(valid,result['selection_ms'],mapping)
        unchanged=[index for index in retained if index in audit['unchanged_configs']]
        winners['_controls']=[min(unchanged,key=lambda index:med(result['selection_ms'][mapping[f'{index}:baseline']]))] if unchanged else []
    if any(winners[policy] not in valid for policy in POLICIES):raise RuntimeError('Selected winner failed correctness in another allocation')
    result['winners']=winners
    fixed=sorted(set([index for index in changed if index in valid]+[winners[p] for p in POLICIES]+winners.get('_controls',[])),key=int)
    result['fixed_config_indices']=fixed
    result['selected']={policy:f'{winners[policy]}:{policy}' for policy in POLICIES}
    result['selected']['oracle']=winners['_oracle']
    keys=sorted({mapping[f'{index}:{policy}'] for index in fixed for policy in POLICIES})
    measure(keys,protocol['confirmation_rounds'],'confirmation')
    for identifier in keys:result.setdefault('after_validation',{})[identifier]=validate(identifier)
    torch.testing.assert_close(a[rows,:].cpu().double(),sampled_a,atol=0,rtol=0)
    torch.testing.assert_close(b[:,cols].cpu().double(),sampled_b,atol=0,rtol=0)
    assert_exclusive();result.update(complete=True,unchanged_input_samples=True);persist()
    print('DONE',key,skew,winners,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',required=True);parser.add_argument('--skew',type=int,required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--winners')
    args=parser.parse_args();output=Path(args.output);output.parent.mkdir(exist_ok=True,parents=True)
    protocol=json.loads((ROOT/'protocol.json').read_text())
    audit=json.loads((ROOT/'audit.json').read_text())
    manifests={p:json.loads((ROOT/f'compiled-{p}.json').read_text()) for p in POLICIES}
    assert all(m['complete'] and m['protocol_sha256']==sha((ROOT/'protocol.json').read_bytes()) for m in manifests.values())
    if args.skew not in protocol['allocation_arrangements']:raise ValueError('Unregistered allocation history')
    winners=json.loads(Path(args.winners).read_text()) if args.winners else None
    verify()
    with acquire(output.with_suffix('.availability.json')):
        os.environ.update(TRITON_CACHE_DIR=str(ROOT/'launcher-cache'),TRITON_BACKENDS_IN_TREE='1')
        import torch,triton
        import triton.testing
        torch.set_grad_enabled(False);torch.set_num_threads(4)
        if torch.xpu.device_count()!=1 or 'B580' not in torch.xpu.get_device_name(0):raise RuntimeError('Expected one B580')
        library=Path(triton.__file__).parent/'_C/libtriton.so'
        manifest=json.loads((ROOT/'compiler-manifest.json').read_text())
        if sha(library.read_bytes())!=manifest['compilers']['baseline']['sha256']:raise RuntimeError('Runtime provenance mismatch')
        result=allocation(torch,triton,args.case,{p:manifests[p]['cases'][args.case] for p in POLICIES},audit['cases'][args.case],protocol,args.skew,output,winners)
        if result['failures']:raise RuntimeError('Correctness failures prevent a completed measurement')
        result.update(pid=os.getpid(),device=str(torch.xpu.get_device_properties(0)),torch=torch.__version__,finished=time.time())
        save(output,result)

if __name__=='__main__':main()
