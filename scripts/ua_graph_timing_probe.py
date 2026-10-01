from pathlib import Path
import sys,time,json,statistics,random
import torch
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
P=args.output;P.mkdir(parents=True,exist_ok=True)
(P/'environment.json').write_text(json.dumps(dict(torch=torch.__version__,device=torch.xpu.get_device_name()),indent=2))
def fence():
 e=torch.xpu.Event();e.record();e.wait()

def measure(mode,loops=8,repeats=12,profile_first=False):
 a=torch.randn(2048,2048,device='xpu',dtype=torch.bfloat16);b=torch.randn_like(a);out=torch.empty_like(a)
 cache=torch.empty(256*1024*1024//4,device='xpu',dtype=torch.int32)
 def fn():
  for _ in range(loops):torch.mm(a,b,out=out)
 fn();cache.zero_();torch.xpu.synchronize()
 g=torch.xpu.XPUGraph();ev=torch.xpu.XPUGraph();stream=torch.xpu.current_stream()
 with torch.xpu.graph(g):fn()
 with torch.xpu.graph(ev):cache.zero_()
 for _ in range(3):ev.replay();g.replay()
 torch.xpu.synchronize()
 pairs=[(torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)) for _ in range(repeats)]
 for s,e in pairs:s.record();e.record()
 torch.xpu.synchronize()
 result=[]
 for rnd in range(4):
  if profile_first:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU]) as prof:
    for _ in range(3):ev.replay();g.replay()
    torch.xpu.synchronize()
  samples=[];begin=time.perf_counter()
  for s,e in pairs:
   if mode=='fresh_events':s,e=torch.xpu.Event(enable_timing=True),torch.xpu.Event(enable_timing=True)
   ev.replay()
   if mode=='barriers':fence()
   if mode=='synchronized':torch.xpu.synchronize()
   s.record()
   if mode=='synchronized':s.synchronize()
   if mode=='barriers':fence()
   g.replay()
   if mode=='barriers':fence()
   if mode=='synchronized':torch.xpu.synchronize()
   e.record()
   if mode in ('per_replay_sync','synchronized'):e.synchronize()
   samples.append((s,e))
  torch.xpu.synchronize()
  wall=(time.perf_counter()-begin)*1000
  values=[s.elapsed_time(e) for s,e in samples]
  result.append(dict(event_ms=values,wall_per_call_ms=wall/repeats))
 # Independent throughput reference including eviction, no timing events.
 start=time.perf_counter()
 for _ in range(repeats):ev.replay();g.replay()
 torch.xpu.synchronize();wall=(time.perf_counter()-start)*1000/repeats
 g.reset();ev.reset()
 return dict(mode=mode,loops=loops,profile_first=profile_first,rounds=result,wall_reference_including_eviction_ms=wall)
rows=[]
with torch.inference_mode():
 for profile_first in (False,True):
  for loops in (1,64):
   for mode in ('original','fresh_events','per_replay_sync','barriers','synchronized'):
    row=measure(mode,loops,profile_first=profile_first);rows.append(row)
    (P/'probe.json').write_text(json.dumps(rows,indent=2))
    print(mode,loops,'profile_first',profile_first,'events',[round(statistics.median(r['event_ms']),4) for r in row['rounds']],'wall',round(row['wall_reference_including_eviction_ms'],4),flush=True)
print('COMPLETE',flush=True)
