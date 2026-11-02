"""Compile the unmodified stock kernel/configurations with each actual compiler."""
import argparse,ast,copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(os.environ['GEMM_CI_REPORTS']).resolve()

def sha(data):return hashlib.sha256(data).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.with_suffix('.pending').write_text(json.dumps(x,indent=2)+'\n');p.with_suffix('.pending').replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--compiler',required=True,choices=['baseline','candidate']);p.add_argument('--group',default='slow',choices=['slow','bulk','all']);p.add_argument('--configs',default='all');p.add_argument('--output-tag',default='actual');p.add_argument('--transpose-b',action='store_true');p.add_argument('--kernel-source-file');p.add_argument('--shapes-json');p.add_argument('--dtype',choices=['bf16','fp16'],default='bf16');args=p.parse_args()
 out=ROOT/args.output_tag;out.mkdir(parents=True,exist_ok=True)
 os.environ.update(ZE_AFFINITY_MASK='-1',TRITON_INTEL_DEVICE_ARCH='bmg',TRITON_CACHE_DIR=str(out/'compiler-cache'/args.compiler),TRITON_XPU_GEN_NATIVE_CODE='0',TRITON_BACKENDS_IN_TREE='1')
 import torch
 torch.xpu.is_available=lambda:False;torch.xpu.device_count=lambda:0
 import triton
 from triton.backends.compiler import GPUTarget
 from triton.compiler.compiler import ASTSource,make_backend
 protocol=json.loads((ROOT/'manifest.json').read_text())
 source=Path(protocol['source']);assert sha(source.read_bytes())==protocol['source_sha256'];tree=ast.parse(source.read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_kernel_with_tensor_descriptors')
 fn.decorator_list=[ast.Attribute(value=ast.Name(id='triton',ctx=ast.Load()),attr='jit',ctx=ast.Load())]
 batched_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='matmul_kernel_with_tensor_descriptors_batched')
 batched_fn.decorator_list=copy.deepcopy(fn.decorator_list)
 kernelpath=Path(args.kernel_source_file).resolve() if args.kernel_source_file else out/'kernel.py';kernel_text='import triton\nimport triton.language as tl\n\n'+ast.unparse(ast.fix_missing_locations(fn))+'\n'+ast.unparse(ast.fix_missing_locations(batched_fn))+'\n'
 if args.kernel_source_file:assert kernelpath.exists()
 elif not kernelpath.exists():kernelpath.write_text(kernel_text)
 else:assert kernelpath.read_text()==kernel_text
 spec=importlib.util.spec_from_file_location('actual_stock_kernel',kernelpath);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);kernel=module.matmul_kernel_with_tensor_descriptors
 configfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_matmul_autotune_configs');ns=dict(triton=triton,DEVICE='xpu',List=list);exec(compile(ast.Module(body=[configfn],type_ignores=[]),str(source),'exec'),ns)
 configs=[dict(c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages) for c in ns['get_matmul_autotune_configs']()];assert len(configs)==16
 batch_configfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_matmul_batched_autotune_configs');exec(compile(ast.Module(body=[batch_configfn],type_ignores=[]),str(source),'exec'),ns)
 batch_configs=[dict(c.kwargs,num_warps=c.num_warps,num_stages=c.num_stages) for c in ns['get_matmul_batched_autotune_configs']()]
 indices=list(range(len(configs))) if args.configs=='all' else list(map(int,args.configs.split(',')))
 slow=[[8192,6144,12288],[8192,6144,24576],[2048,6144,24576]]
 shapes=slow if args.group=='slow' else protocol['shapes'] if args.group=='bulk' else protocol['all_CI_shapes']
 if args.shapes_json:shapes=json.loads(Path(args.shapes_json).read_text())
 target=GPUTarget(**json.loads(Path(protocol['saved_target']).read_text())['target']);backend=make_backend(target)
 path=out/f'{args.group}-{args.compiler}.json'
 library=Path(triton.__file__).parent/'_C/libtriton.so'
 result={'compiler':args.compiler,'runtime':triton.__file__,'libtriton_sha256':sha(library.read_bytes()),'source_sha256':sha(source.read_bytes()),'configurations':configs,'batched_configurations':batch_configs,'transpose_b':args.transpose_b,'dtype':args.dtype,'cases':{},'memory_skips':[],'complete':False}
 for shape in shapes:
  batch,m,n,k=([1]+shape) if len(shape)==3 else shape
  required=batch*(2*m*k+2*n*k+8*m*n)
  if required>=target.arch['total_memory']:
   result['memory_skips'].append({'shape':[batch,m,n,k],'required_bytes':required,'device_bytes':target.arch['total_memory']});continue
  key=f'{m}x{n}x{k}' if batch==1 else f'B{batch}_{m}x{n}x{k}'
  rec={'shape':[m,n,k],'batch':batch,'transpose_b':args.transpose_b,'configs':{}};result['cases'][key]=rec
  local_configs=configs if batch==1 else batch_configs
  local_indices=indices if batch==1 or args.configs!='all' else list(range(len(local_configs)))
  kernel=module.matmul_kernel_with_tensor_descriptors if batch==1 else module.matmul_kernel_with_tensor_descriptors_batched
  for index in local_indices:
   cfg=local_configs[index];lane=out/'binaries'/args.compiler/key/f'config-{index:02d}';lane.mkdir(parents=True,exist_ok=True)
   try:
    constants=dict(M=m,N=n,K=k,stride_am=k,stride_ak=1,stride_bk=1 if args.transpose_b else n,stride_bn=k if args.transpose_b else 1,stride_cm=n,stride_cn=1,transpose_a=False,transpose_b=args.transpose_b,**{s:v for s,v in cfg.items() if s not in ['num_warps','num_stages','grf_mode']})
    if batch!=1:constants.update(B=batch,stride_az=m*k,stride_bz=k*n,stride_cz=m*n)
    signature={s:'constexpr' if s in constants else '*'+args.dtype if s in ['a_ptr','b_ptr'] else '*fp32' for s in kernel.arg_names};attrs={(kernel.arg_names.index(s),):[('tt.divisibility',16)] for s in ['a_ptr','b_ptr','c_ptr']}
    compiled=triton.compile(ASTSource(kernel,signature,constants,attrs),target=target,options={s:cfg[s] for s in ['num_warps','num_stages','grf_mode']})
    metadata=json.loads(next(Path(p) for p in compiled.metadata_group.values() if str(p).endswith('.json')).read_text())
    (lane/'kernel.llir').write_text(compiled.asm['llir']);(lane/'kernel.ttgir').write_text(compiled.asm['ttgir'])
    spv=compiled.asm['spv'];meta=copy.deepcopy(metadata);meta.update(target=target,cache_dir=str(lane));native=backend.make_zebin(spv,meta,backend.parse_options(dict(metadata,generate_native_code=True)))
    meta.update(target=metadata['target'],hash=sha(native),binary_ext='zebin',generate_native_code=True);binary=lane/'kernel.zebin';binary.write_bytes(native);save(lane/'kernel.json',meta)
    pref=[];asm=lane/'not-disassembled'
    rec['configs'][str(index)]={'status':'compiled','zebin':str(binary),'sha256':sha(native),'metadata':str(lane/'kernel.json'),'assembly':str(asm),'config':cfg,'native_prefetch_sends':pref,'spv_sha256':sha(spv)}
   except Exception as e:rec['configs'][str(index)]={'status':'failed','error':repr(e)}
   save(path,result);print(args.compiler,key,index,rec['configs'][str(index)]['status'],rec['configs'][str(index)].get('error',''),flush=True)
 result['complete']=True;save(path,result)
if __name__=='__main__':main()
