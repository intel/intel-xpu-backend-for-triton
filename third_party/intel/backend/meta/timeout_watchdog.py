import cmd
import fcntl
import functools
import os
import signal
import subprocess
import sys
import threading
import time

try:
    from setproctitle import setproctitle
except ImportError:

    def setproctitle(*_):
        pass


def timed(fn, timeout=float(os.environ.get("PYTEST_TIMEOUT", "600")) + 2.):

    @functools.wraps(fn)
    def timed_wrapper(*args, **kwargs):
        timer = TimeoutWatchdog.start_timer(timeout, f"Function \"{fn.__name__}\"")
        try:
            return fn(*args, **kwargs)
        finally:
            TimeoutWatchdog.cancel_timer(timer)

    return timed_wrapper


class TimeoutWatchdog:
    _lock = threading.Lock()
    _counter = 0
    _proc = None

    @classmethod
    def start_timer(cls, timeout, what):
        with cls._lock:
            if (proc := cls._proc) is None:
                cls.start()
                proc = cls._proc
            tid = cls._counter
            cls._counter += 1
            proc.stdin.write(f"0|{tid}|{timeout}|{what}\n")
            proc.stdin.flush()
        return tid

    @classmethod
    def cancel_timer(cls, tid):
        with cls._lock:
            if (proc := cls._proc) is not None:
                proc.stdin.write(f"1|{tid}\n")
                proc.stdin.flush()

    @classmethod
    def start(cls, kill_pids=[]):

        def watchdog_watchdog(proc):
            proc.wait()
            with cls._lock:
                if cls._proc is proc:
                    print("Watchdog process died! Exiting ...", file=sys.stderr)
                    os._exit(1)

        with cls._lock:
            assert cls._proc is None
            cls._proc = proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__)] + [str(pid) for pid in kill_pids],
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=watchdog_watchdog, args=(proc, ), daemon=True).start()

    @classmethod
    def stop(cls):
        with cls._lock:
            if (proc := cls._proc) is None:
                return
            cls._proc = None
        proc.stdin.write("3\n")  # Shutdown
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait()

    @staticmethod
    def _worker():
        ppid = os.getppid()

        def parent_watchdog():
            while True:
                time.sleep(10)
                if os.getppid() != ppid:
                    try:
                        os.kill(os.getpid(), signal.SIGINT)
                    except Exception:
                        ...
                    break

        def report_timeout(timeout, what, report_file=os.environ.get("TRITON_TEST_REPORTS_DIR", ".") + "/timeout.txt"):
            try:
                with open(f"/proc/{ppid}/cmdline", "r") as f:
                    cmd = f.read()[:-1]
                with open(report_file, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    f.write(f"pid={ppid}: {cmd}\n")
                    fcntl.flock(f, fcntl.LOCK_UN)
                print(f"{what} has not been completed in {timeout} seconds!", file=sys.stderr)
            except Exception:
                ...

        def kill_parent():
            try:
                print(f"Killing {ppid}: {cmd} ... ", file=sys.stderr)
                os.kill(ppid, signal.SIGUSR1)  # Dump threads
                time.sleep(2)
            except Exception:
                ...
            try:
                os.kill(ppid, signal.SIGTERM)
                time.sleep(2)
            except Exception:
                ...
            try:
                os.kill(ppid, signal.SIGKILL)
            except Exception:
                ...

        def sighandler(signum, frame):
            kill_parent()
            sys.stdin.close()

        def timer_handler(timeout, what):
            report_timeout(timeout, what)
            kill_parent()
            os.kill(os.getpid(), signal.SIGINT)

        timers = {}
        proc_title = f"[python-watchdog-{ppid}]"
        setproctitle(proc_title)
        threading.Thread(target=parent_watchdog, daemon=True).start()
        signal.signal(signal.SIGINT, sighandler)
        signal.signal(signal.SIGTERM, sighandler)

        for line in sys.stdin:
            line = line[:-1].split("|")  # action|timer_id|timeout|what

            if line[0] == "0":  # Start timer
                timeout = float(line[2])
                timer = threading.Timer(timeout, timer_handler, args=(timeout, line[3]))
                timer.daemon = True
                timers[line[1]] = timer
                timer.start()
            elif line[0] == "1":  # Cancel timer
                timers.pop(line[1]).cancel()
            elif timers:  # Shutdown
                timer = next(reversed(timers.values()))
                report_timeout(*timer.args)
                break

            title = [proc_title]
            for timer in timers.values():
                title.append(f"[waiting {timer.args[0]}s for {timer.args[1]}]")
            setproctitle(" ".join(title))


if __name__ == "__main__":
    try:
        TimeoutWatchdog._worker()
    except Exception:
        ...

    for pid in map(int, sys.argv[1:]):
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
        except Exception:
            ...
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            ...
