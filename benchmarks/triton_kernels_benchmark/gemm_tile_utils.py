"""Matrix tile loads and stores for the GEMM benchmarks.

They use a tensor descriptor when the strides allow one, and a masked tensor of pointers otherwise, so the kernels
accept strided, transposed and unaligned matrices without copying them. A matrix with a zero stride, such as an addmm
bias broadcast along M, is loaded once along the other dimension and broadcast.
"""
import triton
import triton.language as tl


@triton.constexpr_function
def is_transposed(M, N, stride_m, stride_n, transposed):
    """Return whether the matrix is stored transposed, i.e. contiguous along M.

    The strides cannot tell when M or N is 1, so a caller that knows the layout passes `transposed`. It must agree with
    the strides whenever they can tell.
    """
    inferred = stride_m == 1 and stride_n != 1
    if transposed is None:
        return inferred
    assert M == 1 or N == 1 or transposed == inferred, \
        f'transposed={transposed} disagrees with strides ({stride_m}, {stride_n}) of a {M}x{N} matrix'
    return transposed


@triton.jit
def make_matrix_tile_ptr(ptr, M: tl.constexpr, N: tl.constexpr, stride_m: tl.constexpr, stride_n: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr = None):
    """Build a descriptor or pointer tensor before the reduction loop."""
    TRANSPOSE: tl.constexpr = is_transposed(M, N, stride_m, stride_n, TRANSPOSED)
    if TRANSPOSE:
        ROWS: tl.constexpr = N
        COLS: tl.constexpr = M
        STRIDE_ROW: tl.constexpr = stride_n
        STRIDE_COL: tl.constexpr = stride_m
        BLOCK_ROWS: tl.constexpr = BLOCK_N
        BLOCK_COLS: tl.constexpr = BLOCK_M
    else:
        ROWS: tl.constexpr = M
        COLS: tl.constexpr = N
        STRIDE_ROW: tl.constexpr = stride_m
        STRIDE_COL: tl.constexpr = stride_n
        BLOCK_ROWS: tl.constexpr = BLOCK_M
        BLOCK_COLS: tl.constexpr = BLOCK_N
    ITEMSIZE: tl.constexpr = ptr.dtype.element_ty.primitive_bitwidth // 8
    USE_DESCRIPTOR: tl.constexpr = (STRIDE_COL == 1 and STRIDE_ROW >= COLS and STRIDE_ROW * ITEMSIZE % 16 == 0
                                    and BLOCK_COLS * ITEMSIZE >= 16)
    if USE_DESCRIPTOR:
        handle = tl.make_tensor_descriptor(ptr, (ROWS, COLS), (STRIDE_ROW, STRIDE_COL), (BLOCK_ROWS, BLOCK_COLS))
    else:
        # Along a zero stride every element is the same, so load one.
        LOAD_ROWS: tl.constexpr = 1 if STRIDE_ROW == 0 else BLOCK_ROWS
        LOAD_COLS: tl.constexpr = 1 if STRIDE_COL == 0 else BLOCK_COLS
        rr = tl.arange(0, LOAD_ROWS).to(tl.int64)
        cc = tl.arange(0, LOAD_COLS).to(tl.int64)
        handle = ptr + rr[:, None] * STRIDE_ROW + cc[None, :] * STRIDE_COL
    return handle


@triton.jit
def load_matrix_tile(matrix_ptr, M: tl.constexpr, N: tl.constexpr, stride_m: tl.constexpr, stride_n: tl.constexpr, row,
                     col, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, TRANSPOSED: tl.constexpr = None):
    """Load a tile from a prebuilt descriptor or pointer tensor."""
    handle = matrix_ptr
    TRANSPOSE: tl.constexpr = is_transposed(M, N, stride_m, stride_n, TRANSPOSED)
    if TRANSPOSE:
        ROWS: tl.constexpr = N
        COLS: tl.constexpr = M
        STRIDE_ROW: tl.constexpr = stride_n
        STRIDE_COL: tl.constexpr = stride_m
        BLOCK_ROWS: tl.constexpr = BLOCK_N
        BLOCK_COLS: tl.constexpr = BLOCK_M
        off_row, off_col = col, row
    else:
        ROWS: tl.constexpr = M
        COLS: tl.constexpr = N
        STRIDE_ROW: tl.constexpr = stride_m
        STRIDE_COL: tl.constexpr = stride_n
        BLOCK_ROWS: tl.constexpr = BLOCK_M
        BLOCK_COLS: tl.constexpr = BLOCK_N
        off_row, off_col = row, col
    if isinstance(handle, tl.tensor_descriptor):
        tile = handle.load([off_row, off_col])
    else:
        LOAD_ROWS: tl.constexpr = 1 if STRIDE_ROW == 0 else BLOCK_ROWS
        LOAD_COLS: tl.constexpr = 1 if STRIDE_COL == 0 else BLOCK_COLS
        rr = (off_row + tl.arange(0, LOAD_ROWS)).to(tl.int64)
        cc = (off_col + tl.arange(0, LOAD_COLS)).to(tl.int64)
        tile = tl.load(handle + tl.cast(off_row, tl.int64) * STRIDE_ROW + tl.cast(off_col, tl.int64) * STRIDE_COL,
                       mask=(rr < ROWS)[:, None] & (cc < COLS)[None, :], other=0)
        if LOAD_ROWS != BLOCK_ROWS or LOAD_COLS != BLOCK_COLS:
            # The broadcast also fills indices past ROWS or COLS. Zero them as the masked load would, since a dot's K
            # tail relies on that padding.
            rr = (off_row + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
            cc = (off_col + tl.arange(0, BLOCK_COLS)).to(tl.int64)
            tile = tl.where((rr < ROWS)[:, None] & (cc < COLS)[None, :],
                            tl.broadcast_to(tile, (BLOCK_ROWS, BLOCK_COLS)), 0)
    if TRANSPOSE:
        tile = tile.T
    return tile


@triton.jit
def store_matrix_tile(ptr, tile, M: tl.constexpr, N: tl.constexpr, stride_m: tl.constexpr, stride_n: tl.constexpr, row,
                      col, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Store a tile without requiring aligned output rows or padded storage."""
    ITEMSIZE: tl.constexpr = ptr.dtype.element_ty.primitive_bitwidth // 8
    USE_DESCRIPTOR: tl.constexpr = (stride_n == 1 and stride_m >= N and stride_m * ITEMSIZE % 16 == 0
                                    and BLOCK_N * ITEMSIZE >= 16)
    if USE_DESCRIPTOR:
        desc = tl.make_tensor_descriptor(ptr, (M, N), (stride_m, stride_n), (BLOCK_M, BLOCK_N))
        desc.store([row, col], tile)
    else:
        rr = (row + tl.arange(0, BLOCK_M)).to(tl.int64)
        cc = (col + tl.arange(0, BLOCK_N)).to(tl.int64)
        tl.store(ptr + rr[:, None] * stride_m + cc[None, :] * stride_n, tile,
                 mask=(rr < M)[:, None] & (cc < N)[None, :])
