"""
Reproducer for sporadic hang in torch.tensor() on XPU after fork().
No Triton dependency.
"""
import os
import sys
import signal
import torch

TIMEOUT = 30
ITERATIONS = 10

for i in range(ITERATIONS):
    print(f"--- iteration {i + 1}/{ITERATIONS} ---")
    pid = os.fork()
    if pid == 0:
        signal.alarm(TIMEOUT)
        print(f"child PID={os.getpid()} creating tensors on XPU...")
        x = torch.tensor([-2**31], dtype=torch.int32, device='xpu')
        print(f"x created: {x}")
        y = torch.tensor([-1], dtype=torch.int32, device='xpu')
        print(f"y created: {y}")
        result = x + y
        torch.xpu.synchronize()
        print(f"result: {result}")
        os._exit(0)
    else:
        _, status = os.waitpid(pid, 0)
        if os.WIFSIGNALED(status):
            print(f"HANG: child killed by signal {os.WTERMSIG(status)}")
            sys.exit(1)
        if os.WEXITSTATUS(status) != 0:
            print(f"FAIL: child exited with code {os.WEXITSTATUS(status)}")
            sys.exit(1)

print(f"OK: all {ITERATIONS} iterations passed")
