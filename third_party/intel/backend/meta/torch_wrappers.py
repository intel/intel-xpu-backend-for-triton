import ctypes
import importlib.abc
import importlib.machinery
import sys
import torch

from weakref import WeakValueDictionary

# This dictionary is used for the case, when a tensor is passed as a pointer to the kernel.
_ptr_to_tensor = WeakValueDictionary()

_tensor_to = torch.Tensor.to
_as_strided = torch.as_strided
# Used to check if there is a test expecting an exception to be raised.
_pytest_raises_counter = 0


def _copy_by_ptr(src, dst, numel):
    nbytes = src.element_size() * numel
    src_buf = (ctypes.c_byte * nbytes).from_address(src.data_ptr())
    dst_buf = (ctypes.c_byte * nbytes).from_address(dst.data_ptr())
    ctypes.memmove(dst_buf, src_buf, nbytes)


def _tensor_to_dev(tensor: torch.Tensor, device) -> torch.Tensor:
    if tensor.data_ptr() == 0:
        return _tensor_to(tensor, device)
    numel = sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride()) if d > 0) + 1
    if numel == tensor.numel():
        return _tensor_to(tensor, device)
    dst = torch.empty(numel, dtype=tensor.dtype, device="cpu")
    _copy_by_ptr(tensor, dst, numel)
    dst = _tensor_to(dst, device)
    return _as_strided(dst, tuple(tensor.size()), tuple(tensor.stride()))


def wrap_launch(launcher, *args):
    """
    Copy tensors to the device before launching and back after launching.
    """
    tensors = {}
    new_args = list(args[:9])

    def process_arg(arg):
        if isinstance(arg, tuple):
            return tuple(process_arg(a) for a in arg)

        is_ptr = False
        if isinstance(arg, int):
            if (tensor := _ptr_to_tensor.get(arg, None)) is not None:
                arg = tensor
                is_ptr = True
            else:
                return arg

        if isinstance(tensor := getattr(arg, "base", None), torch.Tensor):  # triton.runtime.jit.TensorWrapper
            arg = tensor

        if not isinstance(arg, torch.Tensor):
            return arg

        # If the same tensor is passed to the kernel in multiple args, return the previously processed.
        if (tensor := tensors.get(arg, None)) is not None:
            return tensor.data_ptr() if is_ptr else tensor

        if arg.is_xpu:
            tensor = arg
        elif (dev := getattr(arg, "_xpu_dev", None)) is None or dev.type != "xpu":
            # If the tensor does not have the `_xpu_dev` attribute (it could be missing after some operations),
            # we still copy it to the xpu device, unless there is a test expecting an exception to be raised.
            tensor = arg if _pytest_raises_counter else _tensor_to_dev(arg, "xpu")
        else:
            tensor = _tensor_to_dev(arg, dev)
        tensors[arg] = tensor
        return tensor.data_ptr() if is_ptr else tensor

    for arg in args[9:]:
        new_args.append(process_arg(arg))

    launcher(tuple(new_args))

    while tensors:
        tensor, dev_tensor = tensors.popitem()
        if tensor is dev_tensor:
            continue
        try:
            dev_tensor = _tensor_to(dev_tensor, "cpu")
        except RuntimeError as err:  # Fails to copy strided tensors
            if (base := getattr(dev_tensor, "_base", None)) is None:
                raise err
            else:
                base = _tensor_to(base, "cpu")
                dev_tensor = _as_strided(base, tuple(dev_tensor.size()), tuple(dev_tensor.stride()))
        if not tensor.equal(dev_tensor):
            try:
                tensor.copy_(dev_tensor)
            except RuntimeError as err:
                if not torch._debug_has_internal_overlap(tensor) or 0 not in tensor.stride():
                    raise err

                # A workaround for expanded tensors. Copying with data_ptr.
                orig_shape = tuple(1 if s == 0 else d for d, s in zip(tensor.shape, tensor.stride()))
                numel = torch.prod(torch.tensor(orig_shape)).item()
                _copy_by_ptr(dev_tensor, tensor, numel)


class _DeviceWrapper:
    """
    A device wrapper, that overrides the comparison operators.
    """

    def __init__(self, device):
        assert device.type == "xpu" or device.type == "cpu"
        self._device = device

    def __getattr__(self, name):
        return getattr(self._device, name)

    def __eq__(self, other):
        return isinstance(other, (_DeviceWrapper, torch.device)) and (other.type == "xpu" or other.type == "cpu")

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return repr(self._device)

    def __str__(self):
        return str(self._device)


def _get_dev_arg(kwargs):
    if isinstance((device := kwargs.get("device", None)), _DeviceWrapper):
        device = device._device
        kwargs["device"] = device
    return device


def _attach_device(tensor, device):
    """
    Save the device in the tensor's `_xpu_dev` attribute.
    """
    if isinstance(tensor, torch.Tensor) and not hasattr(tensor, "_xpu_dev"):
        if isinstance(device, str):
            device = torch.device(device, 0)
        elif isinstance(device, _DeviceWrapper):
            device = device._device
        tensor._xpu_dev = device
        _ptr_to_tensor[tensor.data_ptr()] = tensor


def _is_xpu_device(device):
    if isinstance(device, _DeviceWrapper):
        device = device._device
    return device == "xpu" or getattr(device, "type", None) == "xpu"


def _device_arg_decorator(func):
    """
    If the function has a `device` argument and the type is "xpu", then remove the argument, call the function and
    attach the device to the resulting tensor.
    """

    def wrapper(*args, **kwargs):
        if (device := _get_dev_arg(kwargs)) is not None:
            if _is_xpu_device(device):
                kwargs.pop("device")
                tensor = func(*args, **kwargs)
                _attach_device(tensor, device)
                return tensor
        return func(*args, **kwargs)

    return wrapper


def _xpu_dev_decorator(func, idx):
    """
    If the argument at the specified index has the `_xpu_dev` attribute, then attach it to the resulting tensor.
    """

    def wrapper(*args, **kwargs):
        tensor = func(*args, **kwargs)
        if len(args) > idx and (dev := getattr(args[idx], "_xpu_dev", None)):
            _attach_device(tensor, dev)
        return tensor

    return wrapper


def _xpu_dev_property_decorator(prop):
    """
    A property decorator, that returns a `_DeviceWrapper` for xpu and cpu devices.
    """

    @property
    def wrapper(self):
        if dev := getattr(self, "_xpu_dev", None):
            return _DeviceWrapper(dev)
        dev = prop.__get__(self)
        return _DeviceWrapper(dev) if dev.type in ("xpu", "cpu") else dev

    return wrapper


def _ignore_err_decorator(func):
    """
    Silently ignore errors in the decorated function.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            pass

    return wrapper


# The following functions are decorated to return a cpu tensor with the `_xpu_dev` attribute,
# if the "xpu" device is specified in the function arguments.
for name in (
        "arange",
        "as_tensor",
        "asarray",
        "bartlett_window",
        "blackman_window",
        "empty",
        "empty_like",
        "empty_permuted",
        "empty_strided",
        "eye",
        "full",
        "full_like",
        "hamming_window",
        "hann_window",
        "kaiser_window",
        "linspace",
        "logspace",
        "ones",
        "ones_like",
        "rand",
        "rand_like",
        "randint",
        "randint_like",
        "randn",
        "randn_like",
        "randperm",
        "range",
        "sparse_bsc_tensor",
        "sparse_bsr_tensor",
        "sparse_compressed_tensor",
        "sparse_coo_tensor",
        "sparse_csc_tensor",
        "sparse_csr_tensor",
        "tensor",
        "tril_indices",
        "triu_indices",
        "zeros",
        "zeros_like",
):
    setattr(torch, name, _device_arg_decorator(getattr(torch, name)))

# The following functions are decorated to propagate the `_xpu_dev` attribute from the input tensor to the output.
for name in (
        "as_tensor",
        "asarray",
        "empty_like",
        "full_like",
        "ones_like",
        "rand_like",
        "randint_like",
        "randn_like",
        "zeros_like",
):
    setattr(torch, name, _xpu_dev_decorator(getattr(torch, name), 0))


def _tensor_to_decorator(func):
    """"
    If the function is called with 'xpu' device argument, return a cpu tensor with the `_xpu_dev` attribute.
    """

    def wrapper(*args, **kwargs):
        if len(args) == 2 and isinstance(args[1], _DeviceWrapper):
            args = (args[0], args[1]._device)
        if len(args) == 2 and _is_xpu_device(device := args[1]):
            tensor = func(args[0], "cpu", **kwargs)
        elif _is_xpu_device(device := _get_dev_arg(kwargs)):
            tensor = func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
        _attach_device(tensor, device)
        return tensor

    return wrapper


torch.Tensor.to = _tensor_to_decorator(_tensor_to)
# Override Tensor.device to return a `_DeviceWrapper`.
torch.Tensor.device = _xpu_dev_property_decorator(torch.Tensor.device)

# The Event.record() function fails with UR_RESULT_ERROR_UNSUPPORTED_FEATURE
torch.xpu.Event.record = _ignore_err_decorator(torch.xpu.Event.record)


class _LoaderWrapper(importlib.abc.Loader):

    def __init__(self, loader, wrapper):
        self.loader = loader
        self.wrapper = wrapper

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.wrapper(module)


class _PathFinderWrapper(importlib.abc.MetaPathFinder):
    """
    This hook allows to avoid circular imports, by wrapping modules when they are loaded.
    """

    def __init__(self):
        self.wrappers = {}
        self.wrap("triton.backends.intel.driver", self.wrap_driver)
        self.wrap("triton.testing", self.wrap_triton_testing)
        self.wrap("pytest", self.wrap_pytest)

    def wrap(self, name, wrapper):
        if name in sys.modules:
            wrapper(sys.modules[name])
        else:
            self.wrappers[name] = wrapper

    @staticmethod
    def wrap_driver(driver):
        orig_getattr = driver.TritonLauncher.__getattribute__

        def wrapper(self, name):
            attr = orig_getattr(self, name)
            if name == "launch":
                return lambda args: wrap_launch(attr, *args)
            return attr

        driver.TritonLauncher.__getattribute__ = wrapper

    @staticmethod
    def wrap_triton_testing(mod):
        # Benchmarking takes to much time, when running on simulator. Disabling it.
        mod.Mark.run = lambda *_, **__: print("Benchmarking on simulator is disabled due to low performance.")

    @staticmethod
    def wrap_pytest(pytest):

        class RaisesContextWrapper:
            """
            Wrapper for the `pytest.raises` context, that increments/decrements the _pytest_raises_counter counter.
            """

            def __init__(self, context):
                self.context = context

            def __enter__(self):
                global _pytest_raises_counter
                obj = self.context.__enter__()
                _pytest_raises_counter += 1
                return obj

            def __exit__(self, exc_type, exc_value, traceback):
                global _pytest_raises_counter
                _pytest_raises_counter -= 1
                return self.context.__exit__(exc_type, exc_value, traceback)

        _pytest_raises = pytest.raises
        pytest.raises = lambda *args, **kwargs: RaisesContextWrapper(_pytest_raises(*args, **kwargs))

    def find_spec(self, fullname, path, target=None):
        if (spec := importlib.machinery.PathFinder.find_spec(fullname, path)) is None:
            return None
        if (wrapper := self.wrappers.pop(fullname, None)) is not None:
            spec.loader = _LoaderWrapper(spec.loader, wrapper)
            if not self.wrappers:
                sys.meta_path.remove(self)
        return spec


sys.meta_path.insert(0, _PathFinderWrapper())
