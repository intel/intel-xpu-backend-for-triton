"""Per-SKU memory specs for Intel GPUs, keyed by PCI device ID."""

# (aggregate_bus_width_bits, clock_multiplier)
#
# bus_width  — total memory bus width across all channels.
# multiplier — transfers per clock cycle for the memory technology:
#              1 for HBM2e (L0 clock already accounts for DDR),
#              8 for GDDR6 (quad-pump × dual-channel).
_PCI_ID_SPECS = {
    0x0BD5: (8192, 1),  # PVC Max 1550 — 8 HBM2e stacks × 1024b
    0x0BDA: (3072, 1),  # PVC Max 1100 — 3 HBM2e stacks × 1024b
    0xE20B: (192, 8),  # BMG B580 — 3 × 64b GDDR6
    0xE20C: (128, 8),  # BMG B570 — 2 × 64b GDDR6
}


def get_aggregate_bus_width(pci_device_id):
    """Return aggregate bus width in bits, or -1 if unknown."""
    spec = _PCI_ID_SPECS.get(pci_device_id)
    return spec[0] if spec else -1


def get_dram_gbps(pci_device_id, mem_clock_rate_khz):
    """Return peak DRAM bandwidth in GB/s, or -1 if unknown."""
    spec = _PCI_ID_SPECS.get(pci_device_id)
    if spec is None:
        return -1
    bus_width, multiplier = spec
    return bus_width * multiplier * mem_clock_rate_khz / 1e6 / 8
