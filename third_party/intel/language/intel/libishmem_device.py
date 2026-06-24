################################################################################
#
# Copyright (c) 2025 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Intel SHMEM Device Library Bindings for Triton-Distributed

This module provides Python bindings for Intel SHMEM (ISHMEM) device-callable APIs,
enabling GPU-initiated communication on Intel Data Center GPU Max Series.

Reference: https://oneapi-src.github.io/ishmem/
Paper: https://arxiv.org/abs/2409.20476
"""

from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call
import sys

# Pointer types
pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
pi_i64_t = tl.core.pointer_type(tl.core.dtype("int64"))
void_ptr = core.pointer_type(core.void)


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    """Patch hash method for pointer types to enable dictionary lookups"""
    elem_dtype_list = tl.core.dtype.SINT_TYPES + tl.core.dtype.UINT_TYPES + tl.core.dtype.FP_TYPES + tl.core.dtype.OTHER_TYPES
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


# Apply monkey patch at module load
patch_hash_method_for_pointer_type()

# ============================================================================
# Constants
# ============================================================================

# Comparison operators for signal_wait_until
ISHMEM_CMP_EQ = 1  # Equal
ISHMEM_CMP_NE = 2  # Not equal
ISHMEM_CMP_GT = 3  # Greater than
ISHMEM_CMP_GE = 4  # Greater or equal
ISHMEM_CMP_LT = 5  # Less than
ISHMEM_CMP_LE = 6  # Less or equal
ISHMEM_CMP_SENTINEL = sys.maxsize

# Team constants
ISHMEM_TEAM_INVALID = -1
ISHMEM_TEAM_WORLD = 0
ISHMEM_TEAM_SHARED = 1

# ============================================================================
# Setup and Query Operations
# ============================================================================


@core.extern
def my_pe(_semantic=None):
    """Get current PE (Processing Element) ID / rank"""
    # is_pure=False: ishmem_my_pe reads from the global_info device global
    # which is initialized at runtime by module_init. LLVM LTO will optimize
    # away the call if marked pure (global_info is zeroinitializer in bitcode).
    return extern_call(
        "libishmem_device",
        "",
        [],
        {
            (): ("ishmem_my_pe", core.dtype("int32")),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    """Get total number of PEs in the job"""
    # is_pure=False: same reason as my_pe — reads from runtime-initialized global
    return extern_call(
        "libishmem_device",
        "",
        [],
        {
            (): ("ishmem_n_pes", core.dtype("int32")),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def pe_accessible(pe, _semantic=None):
    """Check if the specified PE is accessible"""
    return extern_call(
        "libishmem_device",
        "",
        [tl.cast(pe, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("ishmem_pe_accessible", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


# ============================================================================
# Team Management Operations
# ============================================================================


@core.extern
def team_my_pe(team, _semantic=None):
    """Get PE ID within a team"""
    return extern_call(
        "libishmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("ishmem_team_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def team_n_pes(team, _semantic=None):
    """Get number of PEs in a team"""
    return extern_call(
        "libishmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("ishmem_team_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


# ============================================================================
# Memory Operations - Remote Pointer
# ============================================================================


@core.extern
def _ptr_wrapper(local_ptr, pe, _semantic=None):
    """Internal wrapper for ishmem_ptr"""
    return extern_call(
        "libishmem_device",
        "",
        [local_ptr, pe],
        {(core.pointer_type(core.void), core.dtype("int32")): (
             "ishmem_ptr", core.pointer_type(core.void),
         )},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ptr(local_ptr, pe, _semantic=None):
    """
    Get local pointer to symmetric object on remote PE.
    Returns address that can be used for direct load/store to remote memory.
    """
    tl.static_assert(
        local_ptr.dtype.is_ptr(),
        "ptr(local_ptr, pe) local_ptr should be a pointer",
        _semantic=_semantic,
    )
    tl.static_assert(
        pe.dtype.is_int(),
        "ptr(local_ptr, pe) pe should be an integer",
        _semantic=_semantic,
    )
    return tl.cast(
        _ptr_wrapper(
            tl.cast(local_ptr, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            _semantic=_semantic,
        ),
        local_ptr.dtype,
        _semantic=_semantic,
    )


# Alias for consistency with NVSHMEM naming
remote_ptr = ptr


# ============================================================================
# RMA Operations - Blocking Put/Get
# ============================================================================


@core.extern
def putmem(dest, source, nbytes, pe, _semantic=None):
    """Blocking byte-level put to remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                "ishmem_putmem",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def getmem(dest, source, nbytes, pe, _semantic=None):
    """Blocking byte-level get from remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                "ishmem_getmem",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    """Single int32 put (low latency, blocking)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(
            core.pointer_type(core.dtype("int32")),
            core.dtype("int32"),
            core.dtype("int32"),
        ): ("ishmem_int_p", ()),
         },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int_g(source, pe, _semantic=None):
    """Single int32 get (low latency, blocking)"""
    return extern_call(
        "libishmem_device",
        "",
        [source, pe],
        {(
            core.pointer_type(core.dtype("int32")),
            core.dtype("int32"),
        ): ("ishmem_int_g", core.dtype("int32")),
         },
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# RMA Operations - Non-Blocking Put/Get (NBI)
# ============================================================================


@core.extern
def putmem_nbi(dest, source, nbytes, pe, _semantic=None):
    """Non-blocking implicit put. Must call quiet() or barrier to complete."""
    return extern_call(
        "libishmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                "ishmem_putmem_nbi",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def getmem_nbi(dest, source, nbytes, pe, _semantic=None):
    """Non-blocking implicit get. Must call quiet() or barrier to complete."""
    return extern_call(
        "libishmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                "ishmem_getmem_nbi",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def put_nbi(dest, source, nelems, pe, _semantic=None):
    """Non-blocking typed put. Must call quiet() or barrier to complete."""
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_put_nbi", (),
         )
         for core_dtype in ["int", "long", "longlong", "float", "double",
                           "int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def get_nbi(dest, source, nelems, pe, _semantic=None):
    """Non-blocking typed get. Must call quiet() or barrier to complete."""
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_get_nbi", (),
         )
         for core_dtype in ["int", "long", "longlong", "float", "double",
                           "int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Signaling Operations
# ============================================================================


@core.extern
def signal_fetch(sig_addr, _semantic=None):
    """Fetch current value of local signal variable"""
    return extern_call(
        "libishmem_device",
        "",
        [sig_addr],
        {
            (pi_u64_t, ): ("ishmem_signal_fetch", tl.uint64),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def signal_set(sig_addr, signal, pe, _semantic=None):
    """Set remote signal variable to value (atomic)"""
    return extern_call(
        "libishmem_device",
        "",
        [
            sig_addr,
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_u64_t, tl.uint64, tl.int32): (
                "ishmemx_signal_set",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def signal_add(sig_addr, signal, pe, _semantic=None):
    """Add to remote signal variable (atomic)"""
    return extern_call(
        "libishmem_device",
        "",
        [
            sig_addr,
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_u64_t, tl.uint64, tl.int32): (
                "ishmemx_signal_add",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def signal_wait_until(sig_addr, cmp, cmp_value, _semantic=None):
    """
    Wait until local signal satisfies comparison condition.

    Args:
        sig_addr: Local signal address (uint64*)
        cmp: Comparison operator (ISHMEM_CMP_EQ, ISHMEM_CMP_GE, etc.)
        cmp_value: Value to compare against

    Returns:
        Signal value when condition satisfied
    """
    return extern_call(
        "libishmem_device",
        "",
        [
            sig_addr,
            tl.cast(cmp, tl.int32, _semantic=_semantic),
            tl.cast(cmp_value, tl.uint64, _semantic=_semantic),
        ],
        {
            (pi_u64_t, tl.int32, tl.uint64): (
                "ishmem_signal_wait_until",
                tl.uint64,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def test(ivar, cmp, cmp_value, _semantic=None):
    """
    Non-blocking test of local variable against condition.

    Returns:
        1 if condition satisfied, 0 otherwise
    """
    return extern_call(
        "libishmem_device",
        "",
        [ivar, cmp, cmp_value],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype("int32"),
          core.dtype(core_dtype)): (
             "ishmem_" + core_dtype + "_test", core.dtype("int32"),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def wait_until(ivar, cmp, cmp_value, _semantic=None):
    """
    Wait until local variable satisfies comparison condition.
    Similar to signal_wait_until but for regular variables.
    """
    return extern_call(
        "libishmem_device",
        "",
        [ivar, cmp, cmp_value],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype("int32"),
          core.dtype(core_dtype)): (
             "ishmem_" + core_dtype + "_wait_until", (),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Atomic Memory Operations (AMO)
# ============================================================================

# Atomic Fetch Operations (return old value)


@core.extern
def atomic_fetch(dest, pe, _semantic=None):
    """Atomically fetch value from remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_fetch", core.dtype(core_dtype),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64", "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_inc(dest, pe, _semantic=None):
    """Atomically fetch and increment value on remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_fetch_inc", core.dtype(core_dtype),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_add(dest, value, pe, _semantic=None):
    """Atomically fetch and add value to remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_fetch_add", core.dtype(core_dtype),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_and(dest, value, pe, _semantic=None):
    """Atomically fetch and bitwise AND value on remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_fetch_and", core.dtype(core_dtype),
         )
         for core_dtype in ["uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_or(dest, value, pe, _semantic=None):
    """Atomically fetch and bitwise OR value on remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_fetch_or", core.dtype(core_dtype),
         )
         for core_dtype in ["uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_xor(dest, value, pe, _semantic=None):
    """Atomically fetch and bitwise XOR value on remote PE"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_fetch_xor", core.dtype(core_dtype),
         )
         for core_dtype in ["uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_swap(dest, value, pe, _semantic=None):
    """Atomically swap value with remote PE, return old value"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_swap", core.dtype(core_dtype),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64", "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_compare_swap(dest, cond, value, pe, _semantic=None):
    """
    Atomically compare and swap value on remote PE.
    If *dest == cond, set *dest = value, return old value.
    """
    return extern_call(
        "libishmem_device",
        "",
        [dest, cond, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_compare_swap", core.dtype(core_dtype),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64", "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


# Atomic Non-Fetch Operations (no return value)


@core.extern
def atomic_set(dest, value, pe, _semantic=None):
    """Atomically set value on remote PE (no return value)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_set", (),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64", "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_inc(dest, pe, _semantic=None):
    """Atomically increment value on remote PE (no return value)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_inc", (),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_add(dest, value, pe, _semantic=None):
    """Atomically add value to remote PE (no return value)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_add", (),
         )
         for core_dtype in ["int32", "int64", "uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_and(dest, value, pe, _semantic=None):
    """Atomically bitwise AND value on remote PE (no return value)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_and", (),
         )
         for core_dtype in ["uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_or(dest, value, pe, _semantic=None):
    """Atomically bitwise OR value on remote PE (no return value)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_or", (),
         )
         for core_dtype in ["uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_xor(dest, value, pe, _semantic=None):
    """Atomically bitwise XOR value on remote PE (no return value)"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, value, pe],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.dtype(core_dtype),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_atomic_xor", (),
         )
         for core_dtype in ["uint32", "uint64"]},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Synchronization Operations
# ============================================================================


@core.extern
def barrier_all(_semantic=None):
    """Global barrier across all PEs"""
    return extern_call(
        "libishmem_device",
        "",
        [],
        {
            (): ("ishmem_barrier_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def barrier(team, _semantic=None):
    """Barrier for PEs in specified team (calls ishmem_team_sync)"""
    return extern_call(
        "libishmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("ishmem_team_sync", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def sync_all(_semantic=None):
    """Synchronize all PEs (deprecated, use barrier_all)"""
    return extern_call(
        "libishmem_device",
        "",
        [],
        {
            (): ("ishmem_sync_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def sync(team, _semantic=None):
    """Synchronize PEs in team (calls ishmem_team_sync)"""
    return extern_call(
        "libishmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("ishmem_team_sync", tl.int32),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence(_semantic=None):
    """
    Memory ordering fence. Ensures all prior put/get operations
    are ordered before subsequent operations.
    """
    return extern_call(
        "libishmem_device",
        "",
        [],
        {
            (): ("ishmem_fence", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def quiet(_semantic=None):
    """
    Wait for completion of all outstanding put/get operations.
    Must be called after NBI operations to ensure completion.
    """
    return extern_call(
        "libishmem_device",
        "",
        [],
        {
            (): ("ishmem_quiet", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Collective Operations
# ============================================================================


@core.extern
def broadcast(dest, source, nelems, PE_root, team, _semantic=None):
    """Broadcast data from root PE to all PEs in team"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, PE_root, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_broadcast", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fcollect(dest, source, nelems, team, _semantic=None):
    """
    Concatenate fixed-size data from all PEs.
    AllGather-like operation where each PE contributes same size.
    """
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_fcollect", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def alltoall(dest, source, nelems, team, _semantic=None):
    """
    All-to-all exchange: each PE sends data to all PEs.
    dest and source are split into npes chunks.
    """
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_alltoall", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def sum_reduce(dest, source, nelems, team, _semantic=None):
    """
    Reduction with SUM operation across team.
    Each PE contributes source data, result in dest on all PEs.
    """
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_sum_reduce", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def prod_reduce(dest, source, nelems, team, _semantic=None):
    """Reduction with PRODUCT operation across team"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_prod_reduce", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def min_reduce(dest, source, nelems, team, _semantic=None):
    """Reduction with MIN operation across team"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_min_reduce", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def max_reduce(dest, source, nelems, team, _semantic=None):
    """Reduction with MAX operation across team"""
    return extern_call(
        "libishmem_device",
        "",
        [dest, source, nelems, team],
        {(core.pointer_type(core.dtype(core_dtype)),
          core.pointer_type(core.dtype(core_dtype)),
          core.dtype("uint64"),
          core.dtype("int32")): (
             "ishmem_" + core_dtype + "_max_reduce", tl.int32,
         )
         for core_dtype in ["int8", "int16", "int32", "int64",
                           "uint8", "uint16", "uint32", "uint64",
                           "float32", "float64"]},
        is_pure=False,
        _semantic=_semantic,
    )


# ============================================================================
# Work-Group Collaborative Operations (Future Enhancement)
# ============================================================================
# Note: Work-group variants require SYCL group parameter and are more efficient
# when all work-items in a work-group cooperate. These will be added in Phase 9.
#
# Examples:
# - ishmemx_putmem_work_group()
# - ishmemx_getmem_work_group()
# - ishmemx_barrier_all_work_group()
#
# These operations enable better hardware utilization by having all threads
# in a work-group collaborate on the communication operation.
