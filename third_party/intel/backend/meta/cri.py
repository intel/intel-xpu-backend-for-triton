import atexit
import functools
import os
import re
import signal
import subprocess
import sys
import threading
import torch

from triton.backends.compiler import BaseBackend
from .utils import echo, thread_dump
from .timeout_watchdog import timed, TimeoutWatchdog


class XPUBackendMeta(type(BaseBackend)):

    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        cls.parse_target = mcls.wrap_parse_target(cls.parse_target)
        return cls

    @staticmethod
    def wrap_parse_target(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            dev_prop = fn(*args, **kwargs)
            dev_prop['has_shader_atomic_bfloat16'] = True
            dev_prop['has_support_block_scale_dpas'] = True
            # TODO: Change to True when supported by the driver
            dev_prop['has_f4_conversions'] = False
            return dev_prop

        return wrapper


class Simulator:
    _proc = None

    @staticmethod
    def can_start():
        return os.path.exists("/crisim/crisim-env.sh")

    @classmethod
    def start(cls):
        msg = ["Starting simulator ...\n"]
        export_pattern = re.compile(r"^\s*export\s+(\w+)=\s*(.*)\s*$")
        with open("/crisim/crisim-env.sh", "r") as f:
            for line in f:
                if (m := export_pattern.match(line)):
                    if (name := m.group(1)) == "TbxPort":
                        continue
                    val = m.group(2)
                    if val.startswith('"') or val.startswith("'"):
                        val = val[1:-1]
                    os.environ[name] = val
                    msg.append(f"export {name}={val}\n")

        echo("".join(msg), sys.stdout)
        os.environ["TbxPort"] = "0"
        cls._proc = cls._start_sim()
        port_set = threading.Event()
        threading.Thread(target=cls._sim_thread, args=(cls._proc, port_set), name="crisim", daemon=True).start()
        port_set.wait(10)
        if os.environ["TbxPort"] == "0":
            raise RuntimeError("Failed to start the simulator")
        else:
            echo(f"Simulator started on port {os.environ['TbxPort']}\n", sys.stdout)
        return cls._proc.pid

    @classmethod
    def stop(cls):
        if (proc := cls._proc) is not None:
            cls._proc = None
            try:
                echo("Terminating the simulator ...\n", sys.stdout)
                proc.terminate()
                proc.wait(5)
            except Exception as err:
                echo(f"Failed to terminate the simulator due to {err}. Killing ...\n")
                proc.kill()
                proc.wait(1)

    @staticmethod
    def _start_sim():
        cmd = [
            "/crisim/xesim/AubLoad", "-device",
            ":config/fleur_de_lis/devices/cri.1tx1x4x8x8.a0.fused_2x8.gt.noAts.map.xml", "-sim_mode", "perf_mpu",
            "-enableFeature", "criBootromUpdate", "-attr", "MEM_IGNORE_UNINITIALIZED_PTE", "true", "-pageFaultDebug",
            "tile.m_gt_core.mempipe_module.fabric.m_nodex_loopback", "1", "-cb_cfg", "multithread_mode", "disabled",
            "-enableDcGpgpuPrintMessage", "-socket", "tcp:0", "keep_alive"
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

    @staticmethod
    def _sim_thread(proc, port_set):
        port_pattern = re.compile(r"Listening on port:\s*(\d+)")
        with proc.stdout as stdout:
            for line in stdout:
                if line is None:
                    break
                if port_pattern is not None:
                    if (port := port_pattern.search(line)):
                        os.environ["TbxPort"] = port.group(1)
                        port_set.set()
                        port_set = None
                        port_pattern = None
                echo(f"[crisim] {line}", sys.stdout)
        ec = proc.wait()
        if ec not in (0, -signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
            echo(f"The simulator exited with code {ec} and can not be safely " +
                 "restarted. Terminating the current process ...\n")
            TimeoutWatchdog.stop()
            os._exit(1)


def on_exit():
    Simulator.stop()
    TimeoutWatchdog.stop()


def on_signal(signum, _):
    if signum == signal.SIGUSR1:
        thread_dump()
    on_exit()
    os._exit(signum)


if (os.getenv("TRITON_INTEL_FORCE_DISABLE_WRAPPERS", "").lower() not in ("true", "1", "yes", "on")
        and Simulator.can_start()):
    from . import torch_wrappers as wrappers

    sim_pid = Simulator.start()
    TimeoutWatchdog.start([sim_pid])

    atexit.register(on_exit)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)

    wrappers.wrap_launch = timed(wrappers.wrap_launch)
    torch.xpu.synchronize = timed(torch.xpu.synchronize)
    torch.xpu.device_count = timed(torch.xpu.device_count, 10)
    os.fork = lambda: (_ for _ in ()).throw(Exception("os.fork is prohibited"))

    try:
        import pytest

        pytest.Function.runtest = timed(pytest.Function.runtest)
        pytest.mark.forked = lambda fn: fn  # Disable tests forking
    except ImportError:
        pass  # Ignore if pytest is not installed
