from __future__ import annotations
from typing import Callable, Optional
from concurrent.futures import Executor, as_completed, Future
from contextvars import ContextVar

active_mode: ContextVar[Optional[AsyncCompileMode]] = ContextVar("async_compile_active_mode", default=None)


class FutureKernel:

    def __init__(self, future: Future):
        # Assigned first: __getattr__ reads private state, so every private
        # attribute must exist before anything else can trigger a lookup.
        self._state = "pending"
        # Several warmups can share one pending compile by cache key. Keep
        # per-waiter callbacks so every JIT cache entry is finalized or cleaned.
        self._finalize_callbacks: list[Callable] = []
        self._cleanup_callbacks: list[Callable] = []
        self._error_type_name = ""
        self._error_message = ""
        self._error_is_exception = False
        self.kernel = None
        self.future = future

    def add_callbacks(self, finalize_compile: Callable, cleanup_compile: Callable):
        self._finalize_callbacks.append(finalize_compile)
        self._cleanup_callbacks.append(cleanup_compile)

    def _run_finalize_callbacks(self, kernel):
        # Detach before invoking: a callback may register further waiters, and
        # every waiter must run even when an earlier one raises. The first
        # error is reported, the rest are dropped.
        first_error = None
        # Looping rather than draining one batch: a same-key submit() appends
        # here without queueing a new future, so no other code path would ever
        # reach the callbacks it adds. A callback that unconditionally
        # re-submits its own key is an infinite loop in the caller's own logic;
        # looping is more honest than silently leaking its cache entry.
        while self._finalize_callbacks:
            callbacks, self._finalize_callbacks = self._finalize_callbacks, []
            for finalize_compile in callbacks:
                try:
                    finalize_compile(kernel)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        return first_error

    def _run_cleanup_callbacks(self):
        # Looped for the same reason as _run_finalize_callbacks: a callback can
        # append another cleanup that nothing else would ever run.
        #
        # Asymmetric with _run_finalize_callbacks on purpose. A plain Exception
        # here is swallowed so a cleanup failure can never mask the compilation
        # error the caller is about to receive, but an interpreter-level unwind
        # is returned instead, for the caller to re-raise once the terminal
        # transition is complete.
        first_error = None
        while self._cleanup_callbacks:
            callbacks, self._cleanup_callbacks = self._cleanup_callbacks, []
            for cleanup_compile in callbacks:
                try:
                    cleanup_compile(self)
                except Exception:
                    pass
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        return first_error

    def _fail(self, error_type_name: str, error_message: str, error_is_exception: bool):
        # The single terminal-failure transition, and the only one that never
        # waits on the future. Dropping the future and evicting the JIT cache
        # entry is what releases a compile: with nothing left referencing this
        # object, the worker's eventual result -- or its exception traceback,
        # and every compiler frame behind it -- becomes garbage. Only plain
        # strings and a bool are kept, never the exception itself.
        if self._state != "pending":
            return None
        self._error_type_name = error_type_name
        self._error_message = error_message
        self._error_is_exception = error_is_exception
        self._state = "failed"
        self.future = None
        cleanup_error = self._run_cleanup_callbacks()
        self._finalize_callbacks = []
        return cleanup_error

    def result(self, ignore_errors: bool = False):
        if self._state == "failed":
            # Waiters that joined after the failure still need their cache
            # entry cleaned; their finalize callbacks are moot and dropped.
            cleanup_error = self._run_cleanup_callbacks()
            self._finalize_callbacks = []
            if cleanup_error is not None:
                raise cleanup_error
            # Only a compilation error stays ignorable: an interrupted compile
            # must keep raising on every later resolution too, not just the
            # first one.
            if ignore_errors and self._error_is_exception:
                return None
            raise RuntimeError("Async compilation for this kernel previously failed with "
                               f"{self._error_type_name}: {self._error_message}")
        if self._state == "succeeded":
            # Waiters that joined after the compile resolved never saw a
            # finalize callback run, so drain theirs before handing back.
            callback_error = self._run_finalize_callbacks(self.kernel)
            self._cleanup_callbacks = []
            if callback_error is not None:
                raise callback_error
            return self.kernel

        try:
            kernel = self.future.result()
        except BaseException as exc:
            # BaseException is caught too: a worker cancelled or interrupted
            # mid-compile would otherwise leave the future -- and its traceback
            # -- cached.
            cleanup_error = self._fail(type(exc).__name__, str(exc), isinstance(exc, Exception))
            if cleanup_error is not None:
                raise cleanup_error
            # ignore_errors is an affordance for compilation errors; something
            # that is not an Exception is unwinding the interpreter and must
            # never be swallowed.
            if ignore_errors and isinstance(exc, Exception):
                return None
            raise
        callback_error = self._run_finalize_callbacks(kernel)
        self._cleanup_callbacks = []
        self._state = "succeeded"
        self.future = None
        self.kernel = kernel
        if callback_error is not None:
            raise callback_error
        return kernel

    def __getitem__(self, item):
        # Implicit dunder lookups go to the type, never through __getattr__, so
        # the kernel[grid](...) launch idiom needs a real method here for a
        # warmed-up kernel to be usable at all.
        return self.result()[item]

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            # copy/pickle/repr probe for dunders like __deepcopy__ and expect
            # AttributeError; resolving a compile here would raise the wrong
            # error, and on a pending compile would block instead. Only dunders
            # are refused: CompiledKernel's own _init_handles/_run have to keep
            # forwarding through the proxy. __del__ is deliberately not proxied
            # either -- forwarding a destructor would tie the compiled kernel's
            # finalizer to the proxy's lifetime, which is not the same object.
            raise AttributeError(name)
        if "_state" not in self.__dict__:
            # Allocated without __init__ (copy/pickle via __new__): there is
            # nothing to resolve, and forwarding would recurse through
            # result() reading the very attribute that is missing.
            raise AttributeError(name)
        # Defer to the compiled kernel so users can interact with this object
        # like a normal CompiledKernel without needing to call result() first.
        return getattr(self.result(), name)


class AsyncCompileMode:

    def __init__(self, executor: Executor, *, ignore_errors=False):
        self.executor = executor
        self.ignore_errors = ignore_errors
        self.raw_futures = []
        self.future_kernels = {}

    def submit(self, key, compile_fn, finalize_fn, cleanup_fn):
        future = self.future_kernels.get(key)
        if future is not None:
            future.add_callbacks(finalize_fn, cleanup_fn)
            return future

        future = self.executor.submit(compile_fn)
        future._key = key
        self.raw_futures.append(future)
        future_kernel = FutureKernel(future)
        future_kernel.add_callbacks(finalize_fn, cleanup_fn)
        self.future_kernels[key] = future_kernel
        return future_kernel

    def __enter__(self):
        if active_mode.get() is not None:
            raise RuntimeError("Another AsyncCompileMode is already active")
        active_mode.set(self)
        return self

    def _abandon_pending(self, exc: BaseException):
        # Give up on every compile still in flight without waiting for it:
        # evicting the cache entries is what releases them. A cleanup callback
        # that unwinds in here is ignored on purpose, because the exception
        # already propagating takes precedence and the sweep has to stay
        # exhaustive.
        #
        # One pass is not enough. _fail() runs the cleanup callbacks, and a
        # cleanup callback can submit(), inserting a key that a snapshot taken
        # up front would never visit -- leaving it cached and still pending.
        # Re-scan until a scan finds nothing pending. A callback that
        # unconditionally submits a new key loops here rather than silently
        # leaking, the same trade the callback runners make.
        while True:
            pending = [
                future_kernel for future_kernel in self.future_kernels.values() if future_kernel._state == "pending"
            ]
            if not pending:
                return
            for future_kernel in pending:
                future_kernel._fail(type(exc).__name__, str(exc), isinstance(exc, Exception))

    def __exit__(self, exc_type, exc_value, traceback):
        active_mode.set(None)
        # Finalize any outstanding compiles
        first_error = None
        try:
            if exc_type is not None and not issubclass(exc_type, Exception):
                # The body is unwinding the interpreter, so abandon the compiles
                # in flight instead of making a Ctrl-C wait for all of them.
                self._abandon_pending(exc_value)
            else:
                # A callback can submit further compiles, and as_completed() only
                # walks the futures it was handed, so keep draining until the
                # queue stays empty.
                while self.raw_futures:
                    draining, self.raw_futures = self.raw_futures, []
                    for future in as_completed(draining):
                        try:
                            self.future_kernels[future._key].result(self.ignore_errors)
                        except Exception as e:
                            # Keep draining: stopping here would leave the
                            # remaining compiles unresolved and cached, and only
                            # the first error is worth reporting to the caller.
                            if first_error is None:
                                first_error = e
        except BaseException as e:
            # Attached to the whole drain rather than to a single result(): an
            # interrupt most often arrives while the thread is parked inside
            # as_completed(), which raises from the for statement.
            self._abandon_pending(e)
            raise
        finally:
            # Completed futures can still point back to compile frames so need
            # to drop them to avoid resource leakage.
            self.raw_futures = []
            self.future_kernels = {}
        if first_error is not None:
            raise first_error
