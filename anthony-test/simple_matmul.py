import torch
import triton
import triton.language as tl


@triton.jit
def matmul_2d_block(x_ptr, y_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr,
                    stride_ak: tl.constexpr, stride_bk: tl.constexpr, stride_bn: tl.constexpr, stride_cm: tl.constexpr,
                    stride_cn: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_block_ptr = tl.make_block_ptr(base=x_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(m_start, 0),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))

    b_block_ptr = tl.make_block_ptr(base=y_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, n_start),
                                    block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_data = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_data = tl.load(b_block_ptr, boundary_check=(0, 1))

        acc += tl.dot(a_data, b_data)

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c_block_ptr = tl.make_block_ptr(base=z_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(m_start, n_start), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))

    tl.store(c_block_ptr, acc, boundary_check=(0, 1))


M = 500
K = 500
N = 500

BLOCK_SIZE_M = 64
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 64

X = torch.randn((M, K), device='xpu', dtype=torch.float32)
Y = torch.randn((K, N), device='xpu', dtype=torch.float32)

grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

xstride0, xstride1 = X.stride(0), X.stride(1)
ystride0, ystride1 = Y.stride(0), Y.stride(1)

Z = torch.empty((M, N), device='xpu', dtype=torch.float32)
matmul_2d_block[grid](X, Y, Z, M, N, K, xstride0, xstride1, ystride0, ystride1, Z.stride(0), Z.stride(1), BLOCK_SIZE_M,
                      BLOCK_SIZE_N, BLOCK_SIZE_K)

triton_result = Z
torch_result = torch.matmul(X, Y)
print("Triton and Torch results are close:", torch.allclose(triton_result, torch_result, rtol=1e-02, atol=1e-02))
