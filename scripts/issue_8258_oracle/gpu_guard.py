"""Acquire the CI benchmark lock and require complete DRM ownership scans."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

SCAN = r'''
import json, os
from pathlib import Path
owners = []
for process in Path('/proc').glob('[0-9]*'):
    try:
        for fd in (process / 'fd').iterdir():
            target = os.readlink(fd)
            if target.startswith('/dev/dri/'):
                owners.append({'pid': int(process.name), 'device': target,
                               'command': (process / 'comm').read_text().strip()})
    except (FileNotFoundError, ProcessLookupError):
        pass
    except PermissionError:
        raise RuntimeError('Root-visible GPU ownership scan was incomplete')
print(json.dumps(owners))
'''


def scan():
    return json.loads(subprocess.check_output(
        ['sudo', '-n', 'python3', '-c', SCAN], text=True, timeout=30))


def assert_exclusive():
    foreign = [owner for owner in scan() if owner['pid'] != os.getpid()]
    if foreign:
        raise RuntimeError(f'Foreign GPU owner appeared: {foreign}')


@contextmanager
def acquire(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 600
    checks = []
    with Path('/tmp/xe-forge-issue-8258-compiler-gpu.lock').open('a') as lock:
        while True:
            held = False
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                for _ in range(2):
                    owners = scan()
                    checks.append({'time': time.time(), 'owners': owners})
                    path.write_text(json.dumps(checks, indent=2) + '\n')
                    if owners:
                        break
                    time.sleep(0.5)
                else:
                    break
            except BlockingIOError:
                pass
            if held:
                fcntl.flock(lock, fcntl.LOCK_UN)
            if time.monotonic() >= deadline:
                raise RuntimeError('GPU remained busy; no benchmark started')
            print('Waiting for a free GPU', flush=True)
            time.sleep(10)
        yield checks
