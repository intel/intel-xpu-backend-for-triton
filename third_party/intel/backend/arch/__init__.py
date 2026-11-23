"""Per-chip backend modules.

`XPUBackend.__new__` imports `arch.<name>` for the chip it compiles for, where `<name>` is the name `arch_parser.c`
reports (or `TRITON_INTEL_DEVICE_ARCH`), and uses the module's `XPUBackendImpl` class. A chip with no module gets a
plain subclass of `XPUBackend`. A module changes the chip's properties by overriding `parse_target`::

    from ..compiler import XPUBackend


    class XPUBackendImpl(XPUBackend):

        def parse_target(self, tgt_prop):
            dev_prop = super().parse_target(tgt_prop)
            # Set or clear this chip's entries in `dev_prop` here.
            return dev_prop
"""
