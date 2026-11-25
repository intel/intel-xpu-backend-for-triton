import cProfile
import functools
import json
import os
import pathlib
import pstats
import sys
import time

from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile as unique_file
from typing import Any, Callable, Dict, TextIO, Union

# A tracking utility for gathering the compile and/or runtime time, size,
# profiling and other statistics.
# To enable the tracking, set the environment variable ``TRITON_TRACK_DUMP``
# to either ``1``, ``true``, ``yes``, ``on``, ``y`` or a path to a directory
# where the tracking reports will be dumped.
# To add the profiling statistics to the reports, set the ``TRITON_TRACK_PROFILE``
# environment variable.
# To track the kernel launches, set the ``TRITON_TRACK_RUN`` environment variable.
# To sort the results by the commulative time, set the ``TRITON_TRACK_SORT``
# environment variable.


def _tr_on(val: str, on_vals=("1", "true", "yes", "on", "y")) -> bool:
    return val.lower() in on_vals


def _tr_env(name: str, default: str = "", cls: Any = str) -> Any:
    return cls(os.environ.get(name, default).strip())


def _tr_env_on(name: str, default: bool = False) -> bool:
    return _tr_on(_tr_env(name)) or default


_TR_DUMP = _tr_env("TRITON_TRACK_DUMP")
if _TR_DUMP.lower() in ("", "0", "false", "off", "no", "n"):
    _TR_DUMP = None
elif _tr_on(_TR_DUMP):
    _TR_DUMP = lambda tr: tr.dump()
else:
    _TR_DUMP = lambda tr, dir=pathlib.Path(_TR_DUMP): tr.dump(dir)


class Track(ABC):

    @abstractmethod
    def __enter__(self) -> "Track":
        ...

    @abstractmethod
    def __exit__(self, *excinfo) -> None:
        ...

    @abstractmethod
    def callback(self, name: str) -> Callable | None:
        ...


class NoTrack(Track):

    def __enter__(self) -> "NoTrack":
        return self

    def __exit__(self, *excinfo) -> None:
        pass

    def callback(self, name: str) -> None:
        return None


NoTrack.INSTANCE = NoTrack()


class TrackImpl:
    _stack: "Track" = None

    def __init__(self, name: str):
        self.name = name
        self.metrics: Dict[str, Union[int, float, dict]] = {}

    def __enter__(self):
        self.parent = TrackImpl._stack
        TrackImpl._stack = self
        if self.parent:
            self.parent[self.name] = self.metrics
        self["time"] = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self["time"] = time.time() - self["time"]
        TrackImpl._stack = self.parent
        if (on_exit := getattr(self, "on_exit", None)) is not None:
            on_exit(self)
        if TrackImpl._stack is None:
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
                _, value = TrackImpl._to_value(self.metrics)
                json.dump(value if isinstance(value, dict) else self.metrics, f, indent=2)
        else:
            json.dump({self.name: TrackImpl._to_value(self.metrics)[1]}, dirOrStream, indent=2)

    @staticmethod
    def _to_value(values: Dict, sort: bool = _tr_env_on("TRITON_TRACK_SORT", True)):
        if len(values) == 1 and (time := values.get("time", None)) is not None:
            return time, time

        if not sort:
            return 0., {k: TrackImpl._to_value(v) if isinstance(v, dict) else v for k, v in values.items()}

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
                t, v = TrackImpl._to_value(v)
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


class TrackAndProfile(TrackImpl):
    _LIMIT = _tr_env("TRITON_TRACK_PROFILE_LIMIT", "5", int)

    def __init__(self, name: str):
        super().__init__(name)
        self.pr = cProfile.Profile()

    def __enter__(self) -> "TrackAndProfile":
        if isinstance(st := TrackImpl._stack, TrackAndProfile):
            st.pr.disable()
        super().__enter__()
        self.pr.enable()
        return self

    @staticmethod
    def on_exit(self):
        self.pr.disable()
        st = pstats.Stats(self.pr)
        st.strip_dirs()
        self["pstats"] = sm = {}
        stats_profile = st.get_stats_profile()
        file_name = os.path.basename(__file__)
        # function-name -> FunctionProfile
        items = [(n, p) for n, p in stats_profile.func_profiles.items() if p.file_name != file_name]
        items.sort(key=lambda kv: kv[1].cumtime, reverse=True)
        for n, p in items[:TrackAndProfile._LIMIT]:
            sm[n] = {
                "ncalls": p.ncalls,
                "tottime": p.tottime,
                "cumtime": p.cumtime,
            }

        if isinstance(st := TrackImpl._stack, TrackAndProfile):
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


def track(funcOrName: Union[str, Callable] = None, *, profile: bool = _tr_env_on("TRITON_TRACK_PROFILE"),
          name: str = None) -> Union[Callable, Track]:
    if _TR_DUMP is None:
        return NoTrack.INSTANCE if isinstance(funcOrName,
                                              str) else funcOrName if callable(funcOrName) else lambda fn: fn

    cls = TrackAndProfile if profile else TrackImpl

    if isinstance(funcOrName, str):
        return cls(funcOrName)

    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with cls(name or fn.__qualname__):
                return fn(*args, **kwargs)

        return wrapper

    return decorator(funcOrName) if callable(funcOrName) else decorator


# This ugly hook is used to decorate the upstream functions and avoid circular imports.
def _tr_import_hook(name, *args, orig_import=__builtins__["__import__"], decorate_jit=[True], decorate_pm=[True],
                    **kwargs):
    module = orig_import(name, *args, **kwargs)
    if decorate_jit[0] and name == "triton.runtime.jit":
        LS = "\r" if os.linesep == "\r" else "\n"
        TRACK_RUN = _tr_env_on("TRITON_TRACK_RUN", False)

        def on_compile_exit(tr):
            kernel = tr.kernel
            kname = kernel.name
            if tr.constexprs:
                kname += f"_{'_'.join(str(e) for e in tr.constexprs.values())}"
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

                @functools.wraps(fn := kernel._init_handles)
                def init_handles(*args, **kwargs):
                    fn(*args, **kwargs)
                    run = kernel._run

                    def kernel_run(*args, **kwargs):
                        sfx = f"_run_m{args[0]}_n{args[1]}_k{args[2]}"
                        if tr.parent is None:
                            tname = f"{kname}/kernel{sfx}_{cnt[0]}"
                            cnt[0] += 1
                        else:
                            tname = kname + sfx
                        with track(tname):
                            return run(*args, **kwargs)

                    kernel._run = kernel_run

                kernel._init_handles = init_handles

        @functools.wraps(fn := module.JITFunction._do_compile)
        def compile(*args, **kwargs):
            with track("JITFunction._do_compile") as tr:
                tr.kernel = k = fn(*args, **kwargs)
                tr.constexprs = args[4]
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


if _TR_DUMP is not None:
    __builtins__["__import__"] = _tr_import_hook
