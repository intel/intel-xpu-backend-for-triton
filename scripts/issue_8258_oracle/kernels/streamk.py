import triton
import triton.language as tl

@triton.jit
def swizzle_tile(tile_id, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + tile_id % group_size
    pid_n = tile_id % width // group_size
    return (pid_m, pid_n)

@triton.jit
def linear_tile(tile_id, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid_m = tile_id // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_SIZE_N)
    return (pid_m, pid_n)

@triton.jit
def mac_loop(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, iters_per_tile, start_iter, end_iter, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    tile_id = start_iter // iters_per_tile
    remain_iters = start_iter % iters_per_tile
    if GROUP_SIZE_M > 0:
        (pid_m, pid_n) = swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    else:
        (pid_m, pid_n) = linear_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = (iters_per_tile - remain_iters - (end_iter - start_iter)) * BLOCK_SIZE_K
    for _ in range(start_iter, end_iter):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        acc = tl.dot(a, b, acc)
        off_k += BLOCK_SIZE_K
    if remain_iters == 0 and end_iter % iters_per_tile == 0:
        c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
        c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], acc)
    else:
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptr_ = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(c_ptr_, acc, mask=mask, sem='relaxed')

@triton.jit
def first_wave(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, full_tiles, partial_tiles, iters_per_tile, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    start_iter = pid * full_tiles + tl.minimum(pid, partial_tiles)
    last_iter = (pid + 1) * full_tiles + tl.minimum(pid + 1, partial_tiles)
    while start_iter < last_iter:
        end_iter = start_iter + (iters_per_tile - start_iter % iters_per_tile)
        end_iter = tl.minimum(end_iter, last_iter)
        mac_loop(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, iters_per_tile, start_iter, end_iter, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
        start_iter = end_iter

@triton.jit
def full_tiles(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, streamk_tiles, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    tile_id = tl.program_id(axis=0) + streamk_tiles
    if GROUP_SIZE_M > 0:
        (pid_m, pid_n) = swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    else:
        (pid_m, pid_n) = linear_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        acc = tl.dot(a, b, acc)
        off_k += BLOCK_SIZE_K
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], acc)
