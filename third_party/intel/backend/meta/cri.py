import atexit
import functools
import os
import re
import signal
import subprocess
import sys
import time
import torch

from triton.backends.compiler import BaseBackend
from .utils import echo, thread_dump
from .timeout_watchdog import timed, TimeoutWatchdog

simulator_keep_alive = os.getenv("TRITON_INTEL_SIMULATOR_KEEPALIVE", "False").lower() in ("true", "1", "yes", "on")


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
            dev_prop['has_16bit_atomics'] = True
            dev_prop['has_subgroup_scaled_matrix_multiply_accumulate'] = True
            dev_prop['has_f4_conversion'] = True
            dev_prop['has_f8_conversion'] = True
            dev_prop['has_256b_prefetch'] = True
            dev_prop['has_bfloat16_arithmetic'] = True
            dev_prop['has_predicated_io'] = True
            dev_prop['has_subgroup_matrix_multiply_accumulate_bfloat8'] = True
            return dev_prop

        return wrapper


class Simulator:
    _proc = None

    @staticmethod
    def can_start():
        return os.path.exists("/simulator/simulator-env.sh")

    @classmethod
    def start(cls):
        msg = ["Starting simulator ...\n"]
        export_pattern = re.compile(r"^\s*export\s+(\w+)=\s*(.*)\s*$")
        with open("/simulator/simulator-env.sh", "r") as f:
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
                echo("Interruprting the simulator ...\n", sys.stderr)
                proc.send_signal(signal.SIGINT)
                proc.wait(10)
            except Exception:
                try:
                    echo("Terminating the simulator ...\n", sys.stderr)
                    proc.terminate()
                    proc.wait(5)
                except Exception as err:
                    echo(f"Failed to terminate the simulator due to {err}. Killing ...\n")
                    proc.kill()
                    proc.wait(1)

    @staticmethod
    def _start_sim():
        start_dir = os.getenv("TRITON_INTEL_SIMULATOR_START_DIR", ".")
        start_dir_prefix = "fulsim_logs_"
        if current_test := os.getenv("PYTEST_CURRENT_TEST"):
            start_dir = os.path.join(start_dir, start_dir_prefix + current_test.replace(":", "-"))
            # This directory may exist already in case of test rerun
            os.makedirs(start_dir, exist_ok=True)
            start_dir = os.path.join(start_dir, str(os.getpid()))
        else:
            start_dir = os.path.join(start_dir, start_dir_prefix + str(os.getpid()))
        os.mkdir(start_dir)
        os.environ["TRITON_INTEL_SIMULATOR_EFFECTIVE_START_DIR"] = start_dir

        cmd = ["/simulator/run-simulator.sh"]
        if simulator_keep_alive:
            cmd.append("keep_alive")
        if extra_args := os.getenv("TRITON_INTEL_SIMULATOR_EXTRA_ARGS"):
            cmd += extra_args.split()
        output_file_path = os.path.join(start_dir or ".", "simulator-" + str(os.getpid()) + ".log")
        with open(output_file_path, "w") as out_file:
            proc = subprocess.Popen(
                cmd,
                cwd=start_dir,
                stdout=out_file,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
            )
        # Check that output file exists before starting to read it
        while not os.path.exists(output_file_path):
            time.sleep(0.1)

        def follow_file(filepath):
            # Open the file and seek to the end
            with open(filepath, "r") as infile:
                infile.seek(0, os.SEEK_END)
                while True:
                    # Read new lines
                    line = infile.readline()
                    if not line:
                        # If no new line, wait a bit and check again
                        time.sleep(0.5)
                        continue
                    # Yield the new line for processing
                    yield line

        port_pattern = re.compile(r"Listening on port:\s*(\d+)")
        for line in follow_file(output_file_path):
            if line is None:
                break
            if port_pattern is not None:
                if (port := port_pattern.search(line)):
                    os.environ["TbxPort"] = port.group(1)
                    break
        return proc


def on_exit():
    if simulator_keep_alive:
        Simulator.stop()
    TimeoutWatchdog.stop()


def on_signal(signum, _):
    if signum == signal.SIGUSR1:
        thread_dump()
    on_exit()
    os._exit(signum)


USE_WRAPPERS = os.getenv("TRITON_INTEL_FORCE_DISABLE_WRAPPERS", "").lower() not in ("true", "1", "yes", "on")

if USE_WRAPPERS:
    from . import torch_wrappers as wrappers

if (os.getenv("TRITON_INTEL_ENABLE_SIMULATOR_WRAPPER", "False").lower() in ("true", "1", "yes", "on")
        and Simulator.can_start()):
    sim_pid = Simulator.start()
    if simulator_keep_alive:
        TimeoutWatchdog.start([sim_pid])

if USE_WRAPPERS:
    atexit.register(on_exit)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)

    wrappers.wrap_launch = timed(wrappers.wrap_launch)
    torch.xpu.synchronize = timed(torch.xpu.synchronize)
    torch.xpu.device_count = timed(torch.xpu.device_count, 120)

    try:
        import pytest

        pytest.Function.runtest = timed(pytest.Function.runtest)
        pytest.mark.forked = lambda fn: fn  # Disable tests forking
    except ImportError:
        pass  # Ignore if pytest is not installed

# Disable block IO for all layouts, which is currently not working correctly in the simulator and causes hang.
# FIXME: Remove once the issue is fixed in the simulator.
os.environ["TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS"] = "0"
