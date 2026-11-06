import triton
import triton.language as tl

@triton.jit
def _kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, acc_dtype: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % width // group_size
    a_desc = tl.make_tensor_descriptor(base=A, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(base=B, shape=(K, N), strides=(stride_bk, stride_bn), block_shape=(BLOCK_K, BLOCK_N))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=acc_dtype)
    off_k = pid_z * BLOCK_K
    for _ in range(0, K, BLOCK_K * SPLIT_K):
        a = a_desc.load([pid_m * BLOCK_M, off_k])
        b = b_desc.load([off_k, pid_n * BLOCK_N])
        acc = tl.dot(a, b, acc, out_dtype=acc_dtype)
        off_k += BLOCK_K * SPLIT_K
    acc = acc.to(C.dtype.element_ty)
    if SPLIT_K == 1:
        c_desc = tl.make_tensor_descriptor(base=C, shape=(M, N), strides=(stride_cm, stride_cn), block_shape=(BLOCK_M, BLOCK_N))
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], acc)
    else:
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(C, acc, mask=mask, sem='relaxed')
