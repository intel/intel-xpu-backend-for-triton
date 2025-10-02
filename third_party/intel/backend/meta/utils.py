import os
import sys
import traceback
import threading
from datetime import datetime


def echo(msg, file=sys.stderr):
    if not file.closed:
        if not msg.endswith('\n'):
            msg += '\n'
        ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        os.write(file.fileno(), f"[{ts}] {msg}".encode())


def thread_dump(*_):
    with open(f"/proc/{os.getpid()}/cmdline", "r") as f:
        cmdline = f.read()[:-1]
    msg = [f"Command line: '{cmdline}'."]
    thread_names = {t.ident: t.name for t in threading.enumerate()}
    frames = sys._current_frames()
    frames.pop(threading.current_thread().ident, None)
    for ident, frame in sys._current_frames().items():
        thread_name = thread_names.get(ident, "unknown")
        msg.append(f"\n~~~~~~~~~~ Stack of {thread_name} ({ident}) ~~~~~~~~~~\n")
        msg.extend(traceback.format_stack(frame))
    msg.append("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    echo("".join(msg))
