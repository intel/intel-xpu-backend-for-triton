import triton
import triton.language as tl

@triton.jit
def matmul_kernel_with_tensor_descriptors(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr, transpose_a: tl.constexpr, transpose_b: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    if transpose_a:
        A_DESC_ALIGNED: tl.constexpr = stride_ak * (a_ptr.dtype.element_ty.primitive_bitwidth // 8) % 16 == 0
        if A_DESC_ALIGNED:
            a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(K, M), strides=(stride_ak, stride_am), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_M))
    else:
        a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    if transpose_b:
        b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(N, K), strides=(stride_bn, stride_bk), block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K))
    else:
        b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        if transpose_a:
            if A_DESC_ALIGNED:
                a = a_desc.load([off_k, pid_m * BLOCK_SIZE_M]).T
            else:
                a_ptrs = a_ptr + (off_k + offs_k)[:, None] * stride_ak + offs_am[None, :] * stride_am
                a_mask = ((off_k + offs_k)[:, None] < K) & (offs_am[None, :] < M)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0).T
        else:
            a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        if transpose_b:
            b = b_desc.load([pid_n * BLOCK_SIZE_N, off_k]).T
        else:
            b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator = tl.dot(a, b, accumulator)
        off_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float32)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
