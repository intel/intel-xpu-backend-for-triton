"""CPU-only compilation of original benchmark kernels under each saved compiler."""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(os.environ['GEMM_CI_REPORTS']).resolve()

def sha(data):return hashlib.sha256(data).hexdigest()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending');pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(path)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

def main():
    parser=argparse.ArgumentParser();parser.add_argument('policy',choices=['baseline','intervention','learned']);parser.add_argument('--families');parser.add_argument('--case',required=True);parser.add_argument('--cohorts',default='development,anchors');args=parser.parse_args()
    protocol=json.loads((ROOT/'protocol.json').read_text());paired=json.loads(Path(protocol['paired_build_manifest']).read_text())
    os.environ.update(ZE_AFFINITY_MASK='-1',TRITON_INTEL_DEVICE_ARCH='bmg',TRITON_XPU_GEN_NATIVE_CODE='1',TRITON_BACKENDS_IN_TREE='1',TRITON_CACHE_DIR=str(ROOT/'cache'/args.policy))
    import torch
    torch.xpu.is_available=lambda:False;torch.xpu.device_count=lambda:0
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource,make_backend
    library=Path(triton.__file__).parent/'_C/libtriton.so'
    assert sha(library.read_bytes())==paired['compilers'][args.policy]['sha256']
    target=GPUTarget(**json.loads(Path(protocol['target_source']).read_text())['target']);backend=make_backend(target)
    stock=load('generalization_stock_configs',ROOT/'stock_configs.py')
    result={'policy':args.policy,'complete':False,'protocol_sha256':sha((ROOT/'protocol.json').read_bytes()),'libtriton_sha256':sha(library.read_bytes()),'cases':{},'started':time.time()}
    path=ROOT/f'compiled-{args.policy}.json'
    families=args.families.split(',') if args.families else list(protocol['families'])
    for family in families:
        info=protocol['families'][family];source=Path(info['source']);assert sha(source.read_bytes())==info['source_sha256']
        kernelpath=Path(info['kernel_source']);assert sha(kernelpath.read_bytes())==info['kernel_source_sha256']
        module=load('generalization_'+family,kernelpath)
        ns={'triton':triton,'DEVICE':'xpu','List':list}
        if family in ['plain','pointer','transpose_a']:configs=stock.get_matmul_autotune_configs()
        elif family in ['splitk','streamk']:
            tree=ast.parse(source.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_autotune_configs')
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),ns);configs=ns['get_autotune_configs']()
        else:configs=eval(info['configs_expression'],ns)
        configs=[dict(c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages) for c in configs]
        for spec in [c for c in json.loads((ROOT/'suite.json').read_text())['cases'] if c['family']==family and c['id']==args.case and c['cohort'] in args.cohorts.split(',')]:
            m,n,k=spec['shape'];key=spec['id']
            case={**spec,'configs':{}};result['cases'][key]=case
            for index,cfg in enumerate(configs):
                rec={'config':cfg,'status':'failed','kernels':[]};case['configs'][str(index)]=rec
                try:
                    if shutil.disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk below 5GiB compilation guard')
                    bm=cfg.get('BLOCK_SIZE_M',cfg.get('BLOCK_M'));bn=cfg.get('BLOCK_SIZE_N',cfg.get('BLOCK_N'));bk=cfg.get('BLOCK_SIZE_K',cfg.get('BLOCK_K'))
                    constants=dict(M=m,N=n,K=k,stride_am=1 if family=='transpose_a' else k+spec['a_pad'],stride_ak=m+spec['a_pad'] if family=='transpose_a' else 1,stride_bk=n+spec['b_pad'],stride_bn=1,stride_cm=n,stride_cn=1,stride_dm=n,stride_dn=1,transpose_a=family=='transpose_a',transpose_b=False,ACCUMULATOR_DTYPE=tl.float32,acc_dtype=tl.float32)
                    constants.update({name:value for name,value in cfg.items() if name not in ['num_warps','num_stages','grf_mode']})
                    tiles=math.ceil(m/bm)*math.ceil(n/bn)
                    functions=[info['functions'][-1]]
                    launchspec={functions[0]:{'grid':[tiles,1,1],'scalars':[]}}
                    if family=='splitk':launchspec[functions[0]]['grid'][1]=cfg['SPLIT_K']
                    if family=='streamk':
                        programs=target.arch['gpu_subslice_count'];iters=math.ceil(k/bk);sk_tiles=tiles%programs
                        if tiles-sk_tiles>programs:sk_tiles+=programs
                        blocking=tiles-sk_tiles;sk_iters=sk_tiles*iters
                        functions=['first_wave','full_tiles']
                        launchspec={'first_wave':{'grid':[programs,1,1],'scalars':[sk_iters//programs,sk_iters%programs,iters]},'full_tiles':{'grid':[blocking,1,1],'scalars':[sk_tiles]}}
                    for name in functions:
                        kernel=getattr(module,name);lane=ROOT/'binaries'/args.policy/key/f'config-{index:02d}'/name;lane.mkdir(parents=True,exist_ok=True)
                        local_constants={name:constants[name] for name in kernel.arg_names if name in constants}
                        pointer_names=[name for name in kernel.arg_names if name in ['a_ptr','b_ptr','c_ptr','d_ptr','A','B','C']]
                        signature={name:'constexpr' if name in local_constants else '*bf16' if name in ['a_ptr','b_ptr','A','B'] else '*fp32' if name in pointer_names else 'i32' for name in kernel.arg_names}
                        attrs={(kernel.arg_names.index(name),):[('tt.divisibility',16)] for name in pointer_names}
                        compiled=triton.compile(ASTSource(kernel,signature,local_constants,attrs),target=target,options={**{name:cfg[name] for name in ['num_warps','num_stages','grf_mode']},'gemm_policy':args.policy,'generate_native_code':True})
                        metadata=json.loads(next(Path(p) for p in compiled.metadata_group.values() if str(p).endswith('.json')).read_text())
                        spv=compiled.asm['spv'];native=compiled.asm['zebin'];meta=copy.deepcopy(metadata)
                        meta.update(hash=sha(native),binary_ext='zebin',generate_native_code=True)
                        (lane/'kernel.zebin').write_bytes(native);save(lane/'kernel.json',meta)
                        (lane/'kernel.ttir').write_text(compiled.asm['ttir']);(lane/'kernel.llir').write_text(compiled.asm['llir']);(lane/'kernel.ttgir').write_text(compiled.asm['ttgir'])
                        rec['kernels'].append({'name':name,'folder':str(lane),'sha256':sha(native),'spv_sha256':sha(spv),'llir_sha256':sha(compiled.asm['llir'].encode()),'launch':launchspec[name], 'pointer_names':pointer_names,'runtime_signature':{i:signature[name] for i,name in enumerate([name for name in kernel.arg_names if name not in local_constants])},'execution_metadata':{field:meta.get(field) for field in ['num_warps', 'threads_per_warp', 'warp_size', 'num_ctas', 'shared', 'global_scratch_size', 'global_scratch_align', 'profile_scratch_size', 'profile_scratch_align', 'cluster_dims', 'launch_cooperative_grid', 'grf_mode']}})
                    rec['status']='compiled'
                except Exception as error:
                    import traceback
                    rec.update(error=repr(error),traceback=traceback.format_exc())
                save(path,result)
                print(args.policy,key,index,rec['status'],rec.get('error',''),flush=True)
    result.update(complete=True,finished=time.time(),free_bytes=shutil.disk_usage(ROOT).free);save(path,result)
    print('COMPILE COMPLETE',args.policy,len(result['cases']),flush=True)

if __name__=='__main__':main()
