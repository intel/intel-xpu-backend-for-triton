"""Fail when an upstream file that the backend keeps a copy of has changed since the copy recorded its hash.

A copy names its original in two comment lines, anywhere in the file:

    copied-from: <path of the original>
    copied-from-sha256: <sha256 of the original's content>

When the original changes, read its upstream change, port it to the copy or decide that it does not apply, then run
`python3 scripts/check_copied_sources.py --update` to record the new hash. The check hashes the staged files, so
unstaged edits and line-ending conversion in the checkout do not affect it.
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys

SCRIPT = "scripts/check_copied_sources.py"
HEADER = re.compile(rb"copied-from: *(\S+)[ \t]*\r?\n[^\n]*copied-from-sha256: *([0-9a-f]{64})")


def staged(path):
    result = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def digest(path):
    content = staged(path)
    return None if content is None else hashlib.sha256(content).hexdigest()


def copies():
    result = subprocess.run(["git", "grep", "--cached", "-l", "-z", "copied-from-sha256:", "--", ".", f":!{SCRIPT}"],
                            capture_output=True, check=False)
    if result.returncode > 1:  # 1 means no file matches
        sys.exit(result.stderr.decode())
    return [path for path in result.stdout.decode().split("\0") if path]


def update(copy):
    """Record the current hashes in the working-tree copy, keeping its unstaged changes."""

    def rehash(match):
        current = digest(match.group(1).decode())
        if current is None:
            return match.group(0)
        return match.group(0)[:match.start(2) - match.start(0)] + current.encode()

    with open(copy, "rb") as file:
        content = file.read()
    with open(copy, "wb") as file:
        file.write(HEADER.sub(rehash, content))
    print(f"updated {copy}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update", action="store_true", help="record the current hashes of the originals")
    args = parser.parse_args()
    os.chdir(
        subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True,
                       text=True).stdout.strip())
    failed = False
    for copy in copies():
        content = staged(copy)
        headers = HEADER.findall(content)
        if len(headers) != content.count(b"copied-from-sha256:"):
            print(f"{copy}: every copied-from-sha256 line needs a copied-from: <path> line right above it")
            failed = True
        stale = False
        for path, recorded in headers:
            original = path.decode()
            current = digest(original)
            if current is None:
                print(f"{copy}: {original} does not exist")
                failed = True
            elif current != recorded.decode():
                stale = True
                if not args.update:
                    print(f"{original} has changed since {copy} recorded its hash: port the change to the copy or "
                          f"decide that it does not apply, then run python3 {SCRIPT} --update")
                    failed = True
        if args.update and stale:
            update(copy)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
