import sys,runpy,torch,statistics
sys.argv=["capture.py",sys.argv[1],"/tmp/repro/out/_tmp"]
g=runpy.run_path("/tmp/repro/scripts/capture.py")
call=g["call"]
for _ in range(10): call()
torch.xpu.synchronize(); ts=[]
for _ in range(60):
    s=torch.xpu.Event(enable_timing=True); e=torch.xpu.Event(enable_timing=True); s.record(); call(); e.record(); torch.xpu.synchronize(); ts.append(s.elapsed_time(e))
print("triton full-call p50 ms", round(statistics.median(ts),4), "min", round(min(ts),4))
