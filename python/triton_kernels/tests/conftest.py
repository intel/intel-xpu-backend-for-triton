import pytest
import torch


def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cuda")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


def get_device_capability(device):
    """Get device capability for the given device type."""
    if device == "cuda":
        return torch.cuda.get_device_capability()
    elif device == "xpu":
        return (9, )
    else:
        return (0, )


@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl
    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()
