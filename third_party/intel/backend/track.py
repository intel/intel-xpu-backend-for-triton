import os
from typing import Any, Callable, Dict, TextIO, Union

# A tracking utility for gathering the compile and/or runtime time, size,
# profiling and other statistics.
# To enable the tracking, set the environment variable ``TRITON_TRACK_DUMP``
# to either ``1``, ``true``, ``yes``, ``on``, ``y`` or a path to a directory
# where the tracking reports will be dumped.
# To add the profiling statistics to the reports, set the ``TRITON_TRACK_PROFILE``
# environment variable.
# To track the kernel launches, set the ``TRITON_TRACK_RUN`` environment variable.


def _tr_on(val: str, on_vals=("1", "true", "yes", "on", "y")) -> bool:
    return val.lower() in on_vals


def _tr_env(name: str, default: str = "", type: Any = str) -> Any:
    return type(os.environ.get("TRITON_TRACK_" + name, default).strip())


def _tr_env_on(name: str, default: bool = False) -> bool:
    return _tr_on(_tr_env(name)) or default


_TR_DUMP = _tr_env("DUMP")
if _TR_DUMP.lower() in ("", "0", "false", "off", "no", "n"):
    _TR_DUMP = None
elif _tr_on(_TR_DUMP):
    _TR_DUMP = lambda tr: tr.dump()
else:
    import pathlib
    _TR_DUMP = lambda tr, dir=pathlib.Path(_TR_DUMP): tr.dump(dir)
if _TR_DUMP is not None:
    import cProfile
    import functools
    import json
    import pathlib
    import pstats
    import sys
    import time
    from tempfile import NamedTemporaryFile as unique_file

    class Track:
        _stack: "Track" = None

        def __init__(self, name: str):
            self.name = name
            self.metrics: Dict[str, Union[int, float, dict]] = {}

        def __enter__(self):
            self.parent = Track._stack
            Track._stack = self
            if self.parent:
                self.parent[self.name] = self.metrics
            self["time"] = time.time()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self["time"] = time.time() - self["time"]
            Track._stack = self.parent
            if (on_exit := getattr(self, "on_exit", None)) is not None:
                on_exit(self)
            if Track._stack is None:
                _TR_DUMP(self)

        def __getitem__(self, key: str):
            return self.metrics[key]

        def __contains__(self, key):
            return key in self.metrics

        def __setitem__(self, key: str, value: Union[int, float, dict]):
            self.metrics[key] = value

        def callback(self, name: str) -> Callable:
            cnt = 1
            n = name
            while n in self:
                n = f"{name}_{cnt}"
                cnt += 1
            self[n] = m = {}
            stack = [m]

            def cb(name, time, type):
                if type == 2:
                    m = stack.pop()
                    n = stack.pop()
                    if len(m) == 1:
                        stack[-1][n] = m["time"]
                    return

                cnt = 1
                n = name
                m = stack[-1]
                while n in m:
                    n = f"{name}_{cnt}"
                    cnt += 1

                if type == 0:
                    m[n] = time
                else:
                    m[n] = {"time": time}
                    stack.append(n)
                    stack.append(m[n])

            return cb

        def dump(self, dirOrStream: Union[str, pathlib.Path, TextIO] = sys.stdout):
            if isinstance(dirOrStream, str):
                dirOrStream = pathlib.Path(dirOrStream)
            if isinstance(dirOrStream, pathlib.Path):
                dir = dirOrStream
                name = self.name
                slash = self.name.rfind("/")
                if slash != -1:
                    dir = dir / name[:slash]
                    name = name[slash + 1:]
                dir.mkdir(parents=True, exist_ok=True)
                if (file := dir / (name + ".json")).exists():
                    file = unique_file(dir=dir, prefix=name + "_", suffix=".json", mode='w', delete=False)
                else:
                    file = open(file, "w")
                with file as f:
                    _, value = Track._to_value(self.metrics)
                    json.dump(value if isinstance(value, dict) else self.metrics, f, indent=2)
            else:
                json.dump({self.name: Track._to_value(self.metrics)[1]}, dirOrStream, indent=2)

        @staticmethod
        def _to_value(values: Dict):
            if len(values) == 1 and (time := values.get("time", None)) is not None:
                return time, time
            return 0., {k: Track._to_value(v) if isinstance(v, dict) else v for k, v in values.items()}

    if _tr_env_on("SORT", True):  # Sort results by the total time

        def _to_value(values: Dict):
            if len(values) == 1 and (time := values.get("time", None)) is not None:
                return time, time

            time = 0.
            items = []
            no_sort = []
            total = None
            for k, v in values.items():
                if k == "time":
                    total = v
                    continue
                if isinstance(v, float):
                    t = v
                elif isinstance(v, dict):
                    t, v = Track._to_value(v)
                else:
                    no_sort.append((0., k, v))
                    continue
                time += t
                items.append((t, k, v))
            if time != 0.:
                items = sorted(items, key=lambda x: x[0], reverse=True)
            if total is not None:
                time = total
                items.insert(0, (total, "time", total))
            items += no_sort
            return time, {k: v for _, k, v in items}

        Track._to_value = _to_value

    class TrackAndProfile(Track):

        class NullStream:

            def write(self, data):
                pass

            def flush(self):
                pass

        _DEVNULL = NullStream()
        _LIMIT = _tr_env("PROFILE_LIMIT", "5", int)

        def __init__(self, name: str):
            super().__init__(name)
            self.pr = cProfile.Profile()

        def __enter__(self):
            if isinstance(st := Track._stack, TrackAndProfile):
                st.pr.disable()
            super().__enter__()
            self.pr.enable()
            return self

        @staticmethod
        def on_exit(self):
            self.pr.disable()
            st = pstats.Stats(self.pr, stream=TrackAndProfile._DEVNULL)
            for fn in list(st.stats.keys()):
                if fn[0] == __file__:
                    del st.stats[fn]
            st.strip_dirs()
            st.sort_stats("cumulative")
            self["pstats"] = sm = {}
            for fn in st.get_print_list([TrackAndProfile._LIMIT])[1]:
                _, nc, tt, ct, _ = st.stats[fn]
                sm[pstats.func_std_string(fn)] = fm = {}
                fm["ncalls"] = nc
                fm["tottime"] = tt
                fm["cumtime"] = ct

            if isinstance(st := Track._stack, TrackAndProfile):
                st.pr.enable()

        def __setattr__(self, name, value):
            if name == "on_exit":

                def chain(self, cur=self.on_exit):
                    try:
                        value(self)
                    finally:
                        cur(self)

                super().__setattr__(name, chain)
            else:
                super().__setattr__(name, value)

    def track(funcOrName: Union[str, Callable] = None, *, profile: bool = _tr_env_on("PROFILE"),
              name: str = None) -> Union[Callable, Track]:
        type = TrackAndProfile if profile else Track

        if isinstance(funcOrName, str):
            return type(funcOrName)

        def decorator(fn):

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with type(name or fn.__qualname__):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator(funcOrName) if callable(funcOrName) else decorator

    # This ugly hook is used to decorate the upstream functions and avoid circular imports.
    def _tr_import_hook(name, *args, orig_import=__builtins__["__import__"], decorate_jit=[True], decorate_pm=[True],
                        **kwargs):
        module = orig_import(name, *args, **kwargs)
        if decorate_jit[0] and name == "triton.runtime.jit":
            LS = "\r" if os.linesep == "\r" else "\n"
            TRACK_RUN = _tr_env_on("RUN", False)

            def on_compile_exit(tr):
                kernel = tr.kernel
                kname = kernel.name
                tr.name = f"{kname}/{tr.name}"
                cnt = [0]

                if kernel.asm:  # Add the kernel IR and binary sizes
                    tr["asm"] = asm = {}
                    for k, v in kernel.asm.items():
                        if isinstance(v, str):
                            asm[k] = v.count(LS) + 0 if v[-1] == LS else v.count(LS) + 1
                        else:
                            asm[k] = len(v)

                if TRACK_RUN:  # Track the kernel runs

                    @functools.wraps(kfn := kernel.run)
                    def kernel_run(*args, **kwargs):
                        if tr.parent is None:
                            tname = f"{kname}/kernel.run_{cnt[0]}"
                            cnt[0] += 1
                        else:
                            tname = kname
                        with track(tname):
                            return kfn(*args, **kwargs)

                    kernel.run = kernel_run

            @functools.wraps(fn := module.JITFunction._do_compile)
            def compile(*args, **kwargs):
                with track("JITFunction._do_compile") as tr:
                    tr.kernel = k = fn(*args, **kwargs)
                    tr.on_exit = on_compile_exit
                    return k

            module.JITFunction._do_compile = compile
            decorate_jit[0] = False
        elif decorate_pm[0] and name == "triton._C.libtriton":

            @functools.wraps(fn := module.ir.pass_manager.run)
            def pm_run(*args, **kwargs):
                with track("pm.run") as tr:
                    args[0].enable_timing(tr.callback("passes"))
                    fn(*args, **kwargs)

            module.ir.pass_manager.run = pm_run
            decorate_pm[0] = False
        if not decorate_jit[0] and not decorate_pm[0]:
            __builtins__["__import__"] = orig_import
        return module

    __builtins__["__import__"] = _tr_import_hook
else:

    class Track:

        def __enter__(self):
            return self

        def __exit__(self, *excinfo):
            pass

        def callback(self, name: str) -> Union[Callable, None]:
            return None

    def track(funcOrName: Union[str, Callable] = None, *, profile: bool = False, name: str = None,
              notrack=Track()) -> Union[Callable, Track]:
        return notrack if isinstance(funcOrName, str) else funcOrName if callable(funcOrName) else lambda fn: fn
