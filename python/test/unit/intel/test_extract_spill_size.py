"""Tests for `extract_spill_size_from_zebin` in the Intel backend.

Regression coverage for https://github.com/intel/intel-xpu-backend-for-triton/issues/6901:
a missing `.ze_info` section must not raise; it must warn and return 0.

Regression coverage for https://github.com/intel/intel-xpu-backend-for-triton/issues/6941:
a degenerate zebin (no `.text.<kernel>` and no `.symtab`) must be raised as
`IntelGPUError` so the existing 256-GRF retry path catches it.
"""
import struct
import warnings

import pytest

from triton.backends.intel.compiler import extract_spill_size_from_zebin
from triton.runtime.errors import IntelGPUError


def _build_elf64(sections):
    """Build a minimal valid ELF64 little-endian object file.

    `sections` is a list of (name, data_bytes). A `.shstrtab` is appended
    automatically. Returns the raw ELF bytes.
    """
    names = [b""] + [s[0].encode() for s in sections] + [b".shstrtab"]
    shstrtab = b"\x00".join(names) + b"\x00"
    name_offsets = {}
    offset = 0
    for n in names:
        name_offsets[n] = offset
        offset += len(n) + 1

    ehsize = 64
    shentsize = 64
    section_data = [s[1] for s in sections] + [shstrtab]
    section_names = [s[0].encode() for s in sections] + [b".shstrtab"]

    data_offsets = []
    cursor = ehsize
    for d in section_data:
        data_offsets.append(cursor)
        cursor += len(d)
    sh_offset = cursor

    num_sections = 1 + len(section_data)
    shstrndx = num_sections - 1

    elf_header = struct.pack("<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + b"\x00" * 9,  # e_ident
                             1,  # e_type = ET_REL
                             0,  # e_machine
                             1,  # e_version
                             0,  # e_entry
                             0,  # e_phoff
                             sh_offset,  # e_shoff
                             0,  # e_flags
                             ehsize,  # e_ehsize
                             0,  # e_phentsize
                             0,  # e_phnum
                             shentsize,  # e_shentsize
                             num_sections,  # e_shnum
                             shstrndx,  # e_shstrndx
                             )

    section_headers = [struct.pack("<IIQQQQIIQQ", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    SHT_PROGBITS = 1
    SHT_STRTAB = 3
    for name, data, off in zip(section_names, section_data, data_offsets):
        sh_type = SHT_STRTAB if name == b".shstrtab" else SHT_PROGBITS
        section_headers.append(
            struct.pack("<IIQQQQIIQQ", name_offsets[name],  # sh_name
                        sh_type,  # sh_type
                        0,  # sh_flags
                        0,  # sh_addr
                        off,  # sh_offset
                        len(data),  # sh_size
                        0,  # sh_link
                        0,  # sh_info
                        1,  # sh_addralign
                        0,  # sh_entsize
                        ))

    return elf_header + b"".join(section_data) + b"".join(section_headers)


def _write_elf(tmp_path, sections):
    path = tmp_path / "kernel.zebin"
    path.write_bytes(_build_elf64(sections))
    return str(path)


def test_missing_ze_info_warns_and_returns_zero(tmp_path):
    zebin = _write_elf(tmp_path, [(".text.kernel", b"\x00\x00\x00\x00"), (".symtab", b"\x00")])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = extract_spill_size_from_zebin(zebin)

    assert result == 0
    assert any(".ze_info" in str(w.message) for w in caught), \
        f"expected a warning mentioning .ze_info, got: {[str(w.message) for w in caught]}"


def test_degenerate_zebin_raises(tmp_path):
    zebin = _write_elf(tmp_path, [(".note.intelgt.compat", b"\x00")])

    with pytest.raises(IntelGPUError):
        extract_spill_size_from_zebin(zebin)


def test_ze_info_with_spill_size_returns_value(tmp_path):
    zebin = _write_elf(tmp_path, [(".ze_info", b"kernels:\n  - spill_size: 1234\n")])
    assert extract_spill_size_from_zebin(zebin) == 1234


def test_ze_info_without_spill_size_returns_zero(tmp_path):
    zebin = _write_elf(tmp_path, [(".ze_info", b"kernels:\n  - name: foo\n")])
    assert extract_spill_size_from_zebin(zebin) == 0
