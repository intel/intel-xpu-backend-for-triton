import triton
import triton.language as tl

@triton.jit
def matmul_kernel_with_tensor_descriptors(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr, stride_cn: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + pid % num_pid_in_group % group_size_m
    pid_n = pid % num_pid_in_group // group_size_m
    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    off_k = 0
    for _ in range(0, K, BLOCK_SIZE_K):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, off_k])
        a = a.to(tl.float32)
        a = tl.math.exp(a)
        a = a.to(tl.bfloat16)
        b = b_desc.load([off_k, pid_n * BLOCK_SIZE_N])
        accumulator = tl.dot(a, b, accumulator)
        off_k += BLOCK_SIZE_K
    c = accumulator.to(tl.float32)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], c)
